"""
app/repository/faiss_repo.py — FAISS + Cleora index access layer.

Moved from src/retriever.py. Logic unchanged; import path updated.
"""
import os
import faiss
import numpy as np
import pandas as pd


class Retriever:
    """
    Loads and provides access to FAISS indices (BLaIR, CLIP) and the
    Cleora behavioral embedding space.
    """

    def __init__(self, base_dir: str, cleora_data=None):
        """
        Args:
            base_dir:     Directory where indices live (src/data/).
            cleora_data:  npz dict with 'asins' and 'embeddings' keys, or None.
        """
        self.asins = pd.read_csv(os.path.join(base_dir, "asins.csv"), header=None)[0].tolist()
        self.asin_to_idx = {asin: i for i, asin in enumerate(self.asins)}

        blair_hnsw_bge = os.path.join(base_dir, "blair_index_bge_hnsw.faiss")
        blair_flat_bge = os.path.join(base_dir, "blair_index_bge_flat.faiss")
        blair_hnsw     = os.path.join(base_dir, "blair_index_hnsw.faiss")
        clip_hnsw      = os.path.join(base_dir, "clip_index_hnsw.faiss")

        # Priority: BGE HNSW (fast, non-zero only) → BGE flat (exact fallback) → legacy BLaIR
        if os.path.exists(blair_hnsw_bge):
            print(f"Loading BGE HNSW index: {blair_hnsw_bge}")
            self.blair_index = faiss.read_index(blair_hnsw_bge, faiss.IO_FLAG_MMAP)
            self.blair_flat  = faiss.read_index(blair_flat_bge, faiss.IO_FLAG_MMAP) \
                               if os.path.exists(blair_flat_bge) else self.blair_index
        elif os.path.exists(blair_flat_bge):
            print(f"Loading BGE flat index (HNSW not found): {blair_flat_bge}")
            self.blair_index = faiss.read_index(blair_flat_bge, faiss.IO_FLAG_MMAP)
            self.blair_flat  = self.blair_index
        elif os.path.exists(blair_hnsw):
            print(f"Loading HNSW BLaIR index: {blair_hnsw}")
            self.blair_index = faiss.read_index(blair_hnsw, faiss.IO_FLAG_MMAP)
            self.blair_flat  = self.blair_index
        else:
            self.blair_index = faiss.read_index(
                os.path.join(base_dir, "blair_index.faiss"), faiss.IO_FLAG_MMAP
            )
            self.blair_flat  = self.blair_index

        if os.path.exists(clip_hnsw):
            print(f"Loading HNSW CLIP index: {clip_hnsw}")
            self.clip_index = faiss.read_index(clip_hnsw, faiss.IO_FLAG_MMAP)
        else:
            self.clip_index = faiss.read_index(
                os.path.join(base_dir, "clip_index.faiss"), faiss.IO_FLAG_MMAP
            )

        self.cleora_index = None
        self.cleora_asins = []
        self.asin_to_cleora_idx = {}
        if cleora_data is not None:
            self.cleora_asins = list(cleora_data["asins"])
            self.cleora_index = self._build_index(cleora_data["embeddings"])
            self.asin_to_cleora_idx = {asin: i for i, asin in enumerate(self.cleora_asins)}

    def _build_index(self, embeddings):
        X = embeddings.astype("float32")
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        return index

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
        """Layer 2: BLaIR + CLIP similarity scores for a list of candidates."""
        valid_asins = [a for a in candidates if a in self.asin_to_idx]
        if not valid_asins or query_asin not in self.asin_to_idx:
            return pd.DataFrame()

        q_idx = self.asin_to_idx[query_asin]
        q_blair = self.blair_flat.reconstruct(q_idx).reshape(1, -1)
        q_clip  = self.clip_index.reconstruct(q_idx).reshape(1, -1)

        scores = []
        for asin in valid_asins:
            c_idx  = self.asin_to_idx[asin]
            c_blair = self.blair_flat.reconstruct(c_idx).reshape(1, -1)
            c_clip  = self.clip_index.reconstruct(c_idx).reshape(1, -1)
            scores.append({
                "asin":        asin,
                "blair_score": float((q_blair @ c_blair.T)[0, 0]),
                "clip_score":  float((q_clip  @ c_clip.T)[0, 0]),
            })

        return pd.DataFrame(scores)

    def get_asin_vec(self, asin: str):
        """Return (blair_vec, clip_vec) tuple for the given ASIN, or None."""
        if asin in self.asin_to_idx:
            idx = self.asin_to_idx[asin]
            return (
                self.blair_flat.reconstruct(idx),
                self.clip_index.reconstruct(idx),
            )
        return None
