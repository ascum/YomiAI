"""
BGE Reranker Wrapper
====================
A thin wrapper around BAAI/bge-reranker-v2-m3 using sentence-transformers CrossEncoder.

Compatible with transformers >= 5.x (unlike FlagEmbedding which requires < 5.x).

Usage:
    reranker = BGEReranker()
    reranked = reranker.rerank(query, candidates, metadata_df, top_k=10)
"""

import logging

log = logging.getLogger("nba_api")

# Max chars fed to the cross-encoder per (query, passage) pair
_MAX_TEXT_LEN = 512


class BGEReranker:
    """
    Wraps BAAI/bge-reranker-v2-m3 as a Cross-Encoder post-processor.

    Accepts the top-N RRF candidates and re-scores each one by reading
    the query and the book's text (title + description) together via
    full self-attention — much higher precision than cosine similarity.

    Falls back to pass-through (no reranking) if the model fails to load.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            # max_length controls the tokenizer — keep at 512 to match the model's window
            self._model = CrossEncoder(model_name, max_length=512)
            log.info(f"BGE Reranker loaded: {model_name} ✓")
        except Exception as e:
            log.warning(
                f"BGE Reranker failed to load ({e}) — reranking disabled, "
                "falling back to RRF order."
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: list,          # list of (asin, data_dict) from rrf_fusion
        metadata_df,               # pandas DataFrame indexed by parent_asin
        top_k: int = 10,
    ) -> list:
        """
        Re-ranks candidates by scoring (query, book_text) pairs with the cross-encoder.

        Args:
            query:        The raw text query string from the user.
            candidates:   list of (asin, data_dict) — output of rrf_fusion.
            metadata_df:  DataFrame indexed by parent_asin; must have 'title' and 'description'.
            top_k:        Number of candidates to return after reranking.

        Returns:
            list of (asin, data_dict) sorted by reranker_score descending.
            Each data_dict gains a 'reranker_score' key (float in [0,1]).
        """
        if not self.is_ready or not query or not candidates:
            return candidates[:top_k]

        # Build [query, document_text] pairs
        pairs = []
        for asin, _ in candidates:
            book_text = self._get_book_text(asin, metadata_df)
            pairs.append([query, book_text])

        try:
            # CrossEncoder.predict returns a list/array of raw logit scores
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            log.warning(f"BGE Reranker predict failed: {e} — returning RRF order.")
            return candidates[:top_k]

        # Inject reranker_score into each candidate's data dict
        reranked = []
        for (asin, data), score in zip(candidates, raw_scores):
            enriched_data = dict(data)
            
            # sentence-transformers already applies sigmoid for 1-label classification
            val = float(score)
            # In case a different model without auto-sigmoid is loaded and returns logits:
            if val < 0.0 or val > 1.0:
                import math
                val = 1.0 / (1.0 + math.exp(-val))
                
            enriched_data["reranker_score"] = val
            reranked.append((asin, enriched_data))

        # Sort by reranker score descending
        reranked.sort(key=lambda x: x[1]["reranker_score"], reverse=True)

        log.info(
            f"BGE Reranker — reranked {len(candidates)} → {min(top_k, len(reranked))} "
            f"candidates (top score: {reranked[0][1]['reranker_score']:.4f})"
        )
        return reranked[:top_k]

    def _get_book_text(self, asin: str, metadata_df) -> str:
        """Build the text representation of a book for the cross-encoder."""
        try:
            if metadata_df is not None and asin in metadata_df.index:
                row = metadata_df.loc[asin]
                title = str(row.get("title") or "").strip()
                description = str(row.get("description") or "").strip()
                if description:
                    combined = f"Title: {title}. Description: {description}"
                    return combined[:_MAX_TEXT_LEN]
                elif title:
                    return f"Title: {title}"
        except Exception:
            pass
        return asin  # last resort: just the ASIN
