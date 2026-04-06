import re
import logging
import numpy as np
from config import *

log = logging.getLogger("nba_api")


class ActiveSearchEngine:
    """
    Handles user-initiated queries with text and/or image.

    Pipeline:
      0. BM25 keyword search  (built at init from metadata_df titles/authors)
         Returns (hits, kw_confidence) where confidence is 0.0-1.0.
      1. BLaIR FAISS search   (text)  -> Top-50 semantic matches
      2. CLIP  FAISS search   (image) -> Top-50 visual matches
      3. Adaptive Weighted RRF
           weight_kw       = kw_confidence          (0.0 -> 1.0)
           weight_semantic = 1.0 - kw_weight * 0.8  (min 0.2)
           weight_visual   = 1.0 + BM25_VISUAL_BOOST if image present else 0.0
      4. Metadata filter  -> drop ASINs absent from the parquet DB
      5. BGE Reranker     -> cross-encoder re-scores top-20 (text queries only)

    Degrades gracefully:
      - BM25 not installed      -> kw_conf=0, pure semantic fallback
      - No image                -> visual_weight=0, text-only fusion
      - Encoder offline         -> BM25-only result if confidence is high
      - Concept query           -> BM25 near-zero, BLaIR takes full control
    """

    def __init__(self, retriever, profile_manager, reranker=None, metadata_df=None):
        self.retriever       = retriever
        self.profile_manager = profile_manager
        self.reranker        = reranker
        self.metadata_df     = metadata_df

        # BM25 index -- built once at startup from the metadata parquet
        self.bm25_index = None   # BM25Okapi instance or None
        self.bm25_asins = []     # ASINs aligned to bm25 corpus rows
        self._build_bm25_index()

    # ---- BM25 index construction --------------------------------------------

    def _build_bm25_index(self):
        """
        Tokenises every book's `title + author_name` and fits a BM25Okapi index.
        Uses vectorised pandas string ops (no iterrows) -- fast even on 1.73M rows.
        Called once at __init__ time. Falls back silently if rank_bm25 is missing.
        """
        if self.metadata_df is None or len(self.metadata_df) == 0:
            log.warning("BM25: metadata_df empty -- keyword channel disabled.")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            log.warning(
                "BM25: 'rank_bm25' not installed -- keyword channel disabled. "
                "Run: pip install rank-bm25"
            )
            return

        log.info("BM25: building keyword index (vectorised)...")
        meta = self.metadata_df

        # 1. Filter to ASINs present in the FAISS index (single set lookup)
        faiss_set  = set(self.retriever.asin_to_idx.keys())
        meta_valid = meta[meta.index.isin(faiss_set)]

        if meta_valid.empty:
            log.warning("BM25: no overlap between metadata and FAISS index -- disabled.")
            return

        # 2. Build combined title + author Series (no Python loop)
        title_col  = "title"       if "title"       in meta_valid.columns else None
        author_col = "author_name" if "author_name" in meta_valid.columns else (
                     "author"      if "author"      in meta_valid.columns else None)

        if title_col:
            combined = meta_valid[title_col].fillna("").astype(str)
        else:
            combined = meta_valid.index.to_series().str[:0]  # empty Series

        if author_col:
            authors = meta_valid[author_col].fillna("").astype(str)
            # Blank out placeholder strings before concat
            bad_mask = authors.str.lower().isin({"unknown author", "nan", ""})
            authors  = authors.where(~bad_mask, other="")
            if title_col:
                combined = (combined + " " + authors).str.strip()
            else:
                combined = authors.str.strip()
        else:
            combined = combined.str.strip()

        # 3. Drop empty rows
        non_empty   = combined.str.len() > 0
        combined    = combined[non_empty]
        valid_asins = combined.index.tolist()

        if not valid_asins:
            log.warning("BM25: corpus empty after filtering -- keyword channel disabled.")
            return

        # 4. Vectorised tokenisation: lower -> strip punctuation -> split
        tokenised = (
            combined
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
            .str.split()
            .tolist()
        )

        # Drop rows that became empty after tokenisation
        pairs = [(a, toks) for a, toks in zip(valid_asins, tokenised) if toks]
        if not pairs:
            log.warning("BM25: all tokens empty after normalisation -- disabled.")
            return

        corpus_asins, corpus_docs = zip(*pairs)
        self.bm25_index = BM25Okapi(list(corpus_docs))
        self.bm25_asins = list(corpus_asins)
        log.info(f"BM25 index built: {len(self.bm25_asins):,} documents indexed.")

    # ---- Text normalisation -------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list:
        """
        Lowercase + strip all non-alphanumeric chars, then split.
        Examples:
          "Jojo's Bizarre Adventure" -> ["jojos", "bizarre", "adventure"]
          "A Court of Thorns & Roses" -> ["a", "court", "of", "thorns", "roses"]
          "Spider-Man: No Way Home"   -> ["spiderman", "no", "way", "home"]
        """
        if not text:
            return []
        cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
        return [tok for tok in cleaned.split() if tok]

    # ---- BM25 search --------------------------------------------------------

    def _bm25_search(self, query: str) -> tuple:
        """
        Run the BM25 query. Returns (hits, confidence).

        hits       : list of (asin, bm25_score) sorted desc, capped at BM25_TOP_N.
        confidence : float in [0.0, 1.0].
                     High = clear title/author match (keyword channel should dominate).
                     Low  = no clear keyword match, fall back to semantic.
        """
        if self.bm25_index is None or not query:
            return [], 0.0

        tokens = self._tokenize(query)
        if not tokens:
            return [], 0.0

        scores = self.bm25_index.get_scores(tokens)

        top_score = float(scores.max()) if len(scores) else 0.0
        if top_score < BM25_MIN_SCORE:
            return [], 0.0

        # Confidence = absolute score heuristic. 
        # On a 1.7M row corpus, exact word matches yield roughly +5.0 to +25.0 score.
        # Rare words ("jojo"): ~19.6
        # Common words ("fantasy", "magic"): ~7.0
        # We calculate confidence based on average score per query token. By requiring ~15.0 pts
        # per word, only highly specific/rare title words trigger high confidence.
        BM25_EXPECTED_SCORE_PER_WORD = 15.0
        expected_score = len(tokens) * BM25_EXPECTED_SCORE_PER_WORD
        
        confidence = min(1.0, top_score / expected_score)

        # Collect top-N ASIN hits, filtered to metadata index
        top_indices = np.argsort(scores)[::-1][:BM25_TOP_N * 2]
        hits = []
        meta_ok = self.metadata_df is not None
        for idx in top_indices:
            if len(hits) >= BM25_TOP_N:
                break
            asin  = self.bm25_asins[idx]
            score = float(scores[idx])
            if score < BM25_MIN_SCORE:
                break
            if meta_ok and asin not in self.metadata_df.index:
                continue
            hits.append((asin, score))

        return hits, confidence

    # ---- Adaptive Weighted RRF ----------------------------------------------

    def _adaptive_rrf(self, channels: list, k: float = RRF_K) -> list:
        """
        Weighted Reciprocal Rank Fusion across all retrieval channels.

        channels: list of (modality_name, ranked_items, weight)
          modality_name : 'bm25' | 'blair' | 'clip'
          ranked_items  : list of (asin, raw_score) sorted desc
          weight        : float >= 0 -- scales this channel's RRF contribution

        Returns list of (asin, data_dict) sorted by fused_score desc.
        data_dict keys: score, text_sim, img_sim, bm25_score.
        """
        scores = {}

        for modality, ranked_items, weight in channels:
            if weight <= 0 or not ranked_items:
                continue
            for rank, (asin, raw_score) in enumerate(ranked_items):
                if asin not in scores:
                    scores[asin] = {
                        "score":      0.0,
                        "text_sim":   0.0,
                        "img_sim":    0.0,
                        "bm25_score": 0.0,
                    }
                scores[asin]["score"] += weight * (1.0 / (k + rank + 1))

                if modality == "blair":
                    scores[asin]["text_sim"]   = float(raw_score)
                elif modality == "clip":
                    scores[asin]["img_sim"]    = float(raw_score)
                elif modality == "bm25":
                    scores[asin]["bm25_score"] = float(raw_score)

        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [(asin, data) for asin, data in sorted_items]

    # ---- Main search entry point --------------------------------------------

    def search(
        self,
        user_id,
        text_query_vec=None,
        image_query_vec=None,
        text_query: str = "",
        top_k=TOP_K,
        include_timings=False,
    ):
        """
        Unified hybrid search with adaptive weighting.

        All three retrieval channels (BM25, BLaIR, CLIP) feed into a single
        Adaptive Weighted RRF pass whose weights are computed on-the-fly.
        """
        import time
        timings = {}
        t_start = time.perf_counter()

        has_text  = text_query_vec  is not None
        has_image = image_query_vec is not None

        # Step 0: BM25 keyword search
        t0 = time.perf_counter()
        bm25_hits, kw_conf = self._bm25_search(text_query) if text_query else ([], 0.0)
        timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 1: BLaIR semantic search
        t1 = time.perf_counter()
        blair_results = []
        if has_text:
            D, I = self.retriever.blair_index.search(
                text_query_vec.astype("float32").reshape(1, -1), 50
            )
            blair_results = [
                (self.retriever.asins[i], float(D[0][idx]))
                for idx, i in enumerate(I[0]) if i != -1
            ]
        timings["blair_search_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        # Step 2: CLIP image search
        t2 = time.perf_counter()
        clip_results = []
        if has_image:
            D, I = self.retriever.clip_index.search(
                image_query_vec.astype("float32").reshape(1, -1), 50
            )
            clip_results = [
                (self.retriever.asins[i], float(D[0][idx]))
                for idx, i in enumerate(I[0]) if i != -1
            ]
        timings["clip_search_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        # Step 3: Compute adaptive channel weights
        kw_weight       = kw_conf
        semantic_weight = 1.0 - kw_weight * 0.8  # always at least 0.2
        visual_weight   = (1.0 + BM25_VISUAL_BOOST) if has_image else 0.0

        log.info(
            f"Search weights -- BM25:{kw_weight:.2f}  "
            f"BLaIR:{semantic_weight:.2f}  "
            f"CLIP:{visual_weight:.2f}  "
            f"(conf={kw_conf:.2f}, image={'yes' if has_image else 'no'})"
        )

        # Step 4: Adaptive Weighted RRF
        t3 = time.perf_counter()
        channels = [
            ("bm25",  bm25_hits,     kw_weight),
            ("blair", blair_results, semantic_weight),
            ("clip",  clip_results,  visual_weight),
        ]

        # Guard: nothing produced results at all
        if not any(items for _, items, w in channels if w > 0):
            log.warning("Search: all retrieval channels empty -- returning []")
            if include_timings: return [], timings
            return []

        # Edge-case: encoder offline but BM25 confident -- use BM25 alone
        if not blair_results and not clip_results and bm25_hits:
            log.info("Search: encoder unavailable, using BM25-only result.")
            final_ranking = [
                (asin, {"score": 1.0 / (i + 1), "text_sim": 0.0,
                         "img_sim": 0.0, "bm25_score": score})
                for i, (asin, score) in enumerate(bm25_hits)
            ]
        else:
            final_ranking = self._adaptive_rrf(channels)
        timings["rrf_ms"] = round((time.perf_counter() - t3) * 1000, 2)

        # Step 5: Metadata filter -- drop ghost ASINs not in parquet
        t4 = time.perf_counter()
        if self.metadata_df is not None:
            final_ranking = [
                (asin, data) for asin, data in final_ranking
                if asin in self.metadata_df.index
            ]
        timings["meta_filter_ms"] = round((time.perf_counter() - t4) * 1000, 2)

        # Step 6: BGE Reranker (text-query path only)
        t5 = time.perf_counter()
        if (
            self.reranker is not None
            and self.reranker.is_ready
            and text_query
            and has_text
        ):
            candidates_for_rerank = final_ranking[:20]
            final_ranking = self.reranker.rerank(
                query=text_query,
                candidates=candidates_for_rerank,
                metadata_df=self.metadata_df,
                top_k=top_k,
            )
        else:
            final_ranking = final_ranking[:top_k]
        timings["reranker_ms"] = round((time.perf_counter() - t5) * 1000, 2)

        # Log to user profile
        result_items = [asin for asin, _ in final_ranking]
        self.profile_manager.log_search(
            user_id,
            "SIMULATED_TEXT"  if has_text  else None,
            "SIMULATED_IMAGE" if has_image else None,
            result_items,
        )

        if include_timings:
            timings["total_search_engine_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return final_ranking, timings

        return final_ranking
