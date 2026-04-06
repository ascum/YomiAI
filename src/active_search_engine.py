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

        # Tantivy Rust Index -- loaded from disk
        self.tantivy_index   = None
        self.tantivy_searcher = None
        self._load_tantivy_index()

    # ---- Tantivy index loading --------------------------------------------

    def _load_tantivy_index(self):
        """Loads the pre-built Tantivy Rust index from disk."""
        import os
        try:
            import tantivy
            base_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(base_dir, "data", "tantivy_index")
            
            if not os.path.exists(index_path):
                log.warning(f"Tantivy: Index not found at {index_path}. Keyword channel disabled.")
                return

            self.tantivy_index = tantivy.Index.open(index_path)
            self.tantivy_searcher = self.tantivy_index.searcher()
            log.info("Tantivy Rust keyword index loaded ✓")
        except Exception as e:
            log.error(f"Tantivy: failed to load: {e}")

    @staticmethod
    def _tokenize(text: str) -> list:
        """
        Lowercase + strip all non-alphanumeric chars, then split.
        """
        import re
        if not text:
            return []
        cleaned = re.sub(r"[^a-z0-9\s]", "", str(text).lower())
        return [tok for tok in cleaned.split() if tok]

    # ---- BM25 search --------------------------------------------------------

    def _bm25_search(self, query: str) -> tuple:
        """
        Run the high-speed Tantivy Rust query. Returns (hits, confidence).
        """
        if self.tantivy_searcher is None:
            return [], 0.0
        if not query:
            return [], 0.0

        import re
        # Clean and keep only meaningful words (length > 1)
        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        tokens = [t for t in clean_q.split() if len(t) > 1]
        if not tokens: tokens = clean_q.split()
        
        # USE "OR" LOGIC: This is the critical fix for book titles.
        # "jojo bizarre" -> "jojo OR bizarre"
        or_query_str = " OR ".join(tokens)

        try:
            query_parser = self.tantivy_index.parse_query(or_query_str, ["title", "author"])
            search_results = self.tantivy_searcher.search(query_parser, limit=BM25_TOP_N)
            
            if not search_results.hits:
                return [], 0.0

            hits = []
            top_score = float(search_results.hits[0][0])
            
            for score, doc_address in search_results.hits:
                doc = self.tantivy_searcher.doc(doc_address)
                # Raw string from Tantivy (identical to Parquet after map(str))
                asin_str = str(doc["asin"][0])
                
                # Direct string lookup confirmed by forensic script
                if self.metadata_df is not None and asin_str not in self.metadata_df.index:
                    continue
                    
                hits.append((asin_str, float(score)))

            if not hits:
                return [], 0.0

            # Confidence Logic
            num_tokens = len(tokens)
            EXPECTED_PER_WORD = 10.0
            confidence = min(1.0, top_score / (num_tokens * EXPECTED_PER_WORD))

            log.info(f"[SUCCESS] Tantivy: Found {len(hits)} valid hits | Top: {top_score:.2f} | Conf: {confidence:.2f}")
            return hits, confidence

        except Exception as e:
            print(f"[ERROR] Tantivy search error: {e}")
            return [], 0.0

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
