"""
app/repository/faiss_repo.py — FAISS + Cleora index access layer.

Index file names are driven by settings (app/config.py) so swapping to a new
text encoder only requires updating TEXT_ENCODER_MODEL + TEXT_INDEX_* there.
"""
import os
import faiss
import numpy as np
import pandas as pd

from app.config import settings


class Retriever:
    """
    Loads and provides access to FAISS indices (text encoder, CLIP) and the
    Cleora behavioral embedding space.

    Public attributes:
        text_index  — HNSW index used for ANN search (fast)
        text_flat   — Flat index used for exact reconstruction (RL pipeline)
        clip_index  — CLIP visual HNSW/flat index
        cleora_index — Behavioural embedding index
    """

    def __init__(self, base_dir: str, cleora_data=None):
        self.asins = pd.read_csv(os.path.join(base_dir, "asins.csv"), header=None)[0].tolist()
        self.asin_to_idx = {asin: i for i, asin in enumerate(self.asins)}

        self.text_index, self.text_flat = self._load_text_index(base_dir)
        self.clip_index = self._load_clip_index(base_dir)

        self.cleora_index = None
        self.cleora_asins = []
        self.asin_to_cleora_idx = {}
        if cleora_data is not None:
            self.cleora_asins = list(cleora_data["asins"])
            self.cleora_index = self._build_index(cleora_data["embeddings"])
            self.asin_to_cleora_idx = {asin: i for i, asin in enumerate(self.cleora_asins)}

    # ── Private index loaders ─────────────────────────────────────────────────

    def _load_text_index(self, base_dir: str):
        """
        Load the text (semantic) FAISS index.

        Priority:
          1. Primary HNSW  (fast ANN, non-zero only)         — TEXT_INDEX_HNSW
          2. Primary flat  (exact, fallback when HNSW absent) — TEXT_INDEX_FLAT
          3. Legacy HNSW   (older build)                      — TEXT_INDEX_HNSW_LEGACY
          4. Legacy flat   (oldest fallback)                  — TEXT_INDEX_FLAT_LEGACY

        Returns (search_index, flat_index).  flat_index == search_index when
        only one file is available (reconstruction still works for flat types).
        """
        p_hnsw = os.path.join(base_dir, settings.TEXT_INDEX_HNSW)
        p_flat = os.path.join(base_dir, settings.TEXT_INDEX_FLAT)
        p_hnsw_legacy = os.path.join(base_dir, settings.TEXT_INDEX_HNSW_LEGACY)
        p_flat_legacy = os.path.join(base_dir, settings.TEXT_INDEX_FLAT_LEGACY)

        if os.path.exists(p_hnsw):
            print(f"Loading text HNSW index: {p_hnsw}")
            search = faiss.read_index(p_hnsw, faiss.IO_FLAG_MMAP)
            flat   = faiss.read_index(p_flat, faiss.IO_FLAG_MMAP) if os.path.exists(p_flat) else search
            return search, flat

        if os.path.exists(p_flat):
            print(f"Loading text flat index (HNSW not found): {p_flat}")
            idx = faiss.read_index(p_flat, faiss.IO_FLAG_MMAP)
            return idx, idx

        if os.path.exists(p_hnsw_legacy):
            print(f"Loading legacy text HNSW index: {p_hnsw_legacy}")
            idx = faiss.read_index(p_hnsw_legacy, faiss.IO_FLAG_MMAP)
            return idx, idx

        print(f"Loading legacy text flat index: {p_flat_legacy}")
        idx = faiss.read_index(p_flat_legacy, faiss.IO_FLAG_MMAP)
        return idx, idx

    def _load_clip_index(self, base_dir: str):
        p_hnsw = os.path.join(base_dir, settings.CLIP_INDEX_HNSW)
        p_flat = os.path.join(base_dir, settings.CLIP_INDEX_FLAT)

        if os.path.exists(p_hnsw):
            print(f"Loading CLIP HNSW index: {p_hnsw}")
            return faiss.read_index(p_hnsw, faiss.IO_FLAG_MMAP)

        print(f"Loading CLIP flat index: {p_flat}")
        return faiss.read_index(p_flat, faiss.IO_FLAG_MMAP)

    # ── Index builder (Cleora) ────────────────────────────────────────────────

    def _build_index(self, embeddings):
        X = embeddings.astype("float32")
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        return index

    # ── Public retrieval helpers ──────────────────────────────────────────────

    def get_behavioral_candidates(self, query_asin: str, top_n: int = 50) -> list:
        """Layer 1: Nearest neighbours in Cleora behavioural space."""
        if self.cleora_index is None or query_asin not in self.asin_to_cleora_idx:
            return []
        idx = self.asin_to_cleora_idx[query_asin]
        q = self.cleora_index.reconstruct(idx).reshape(1, -1)
        D, I = self.cleora_index.search(q, top_n + 1)
        results = [self.cleora_asins[i] for i in I[0]]
        return [asin for asin in results if asin != query_asin][:top_n]

    def score_candidates(self, candidates: list, query_asin: str) -> pd.DataFrame:
        """Layer 2: Text + CLIP similarity scores for a list of candidates."""
        valid_asins = [a for a in candidates if a in self.asin_to_idx]
        if not valid_asins or query_asin not in self.asin_to_idx:
            return pd.DataFrame()

        q_idx  = self.asin_to_idx[query_asin]
        q_text = self.text_flat.reconstruct(q_idx).reshape(1, -1)
        q_clip = self.clip_index.reconstruct(q_idx).reshape(1, -1)

        scores = []
        for asin in valid_asins:
            c_idx  = self.asin_to_idx[asin]
            c_text = self.text_flat.reconstruct(c_idx).reshape(1, -1)
            c_clip = self.clip_index.reconstruct(c_idx).reshape(1, -1)
            scores.append({
                "asin":       asin,
                "text_score": float((q_text @ c_text.T)[0, 0]),
                "clip_score": float((q_clip @ c_clip.T)[0, 0]),
            })

        return pd.DataFrame(scores)

    def get_asin_vec(self, asin: str):
        """Return (text_vec, clip_vec) tuple for the given ASIN, or None."""
        if asin in self.asin_to_idx:
            idx = self.asin_to_idx[asin]
            return (
                self.text_flat.reconstruct(idx),
                self.clip_index.reconstruct(idx),
            )
        return None

    def get_content_candidates(self, query_vector, top_n: int = 200,
                                exclude_asins: set = None) -> list:
        """
        Personal Pipeline candidate generation via HNSW KNN search.

        Uses self.text_index (HNSW when bge_index_hnsw.faiss is present) to find
        nearest neighbours to the query vector. COMPLETELY INDEPENDENT of Cleora.

        Used by the DIF-SASRec personal pipeline ("You Might Like") tab.
        The query_vector is typically the user's text_profile — a 1024-dim
        weighted average of their clicked items' BGE-M3 embeddings.

        Args:
            query_vector:  np.ndarray [1024] — user's BGE-M3 profile vector
            top_n:         number of candidates to return
            exclude_asins: set of ASINs to exclude (already-seen items)
        Returns:
            list of ASIN strings ordered by cosine similarity (best first)
        """
        if exclude_asins is None:
            exclude_asins = set()

        q = query_vector.reshape(1, -1).astype("float32")

        # Over-fetch to compensate for excluded items
        fetch_n = top_n + len(exclude_asins) + 50
        D, I = self.text_index.search(q, fetch_n)

        results = []
        for faiss_i in I[0]:
            if faiss_i < 0 or faiss_i >= len(self.asins):
                continue
            asin = self.asins[faiss_i]
            if asin in exclude_asins:
                continue
            results.append(asin)
            if len(results) >= top_n:
                break

        return results
