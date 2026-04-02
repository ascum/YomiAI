import faiss
import numpy as np
import pandas as pd
import os

class Retriever:
    def __init__(self, base_dir, cleora_data=None):
        """
        base_dir: Directory where indices are stored (src/data/)
        cleora_data: dict with 'asins' and 'embeddings'
        """
        # Load ASIN mapping
        self.asins = pd.read_csv(os.path.join(base_dir, "asins.csv"), header=None)[0].tolist()
        self.asin_to_idx = {asin: i for i, asin in enumerate(self.asins)}
        
        # Load Content Indices with Memory Mapping to save RAM
        self.blair_index = faiss.read_index(os.path.join(base_dir, "blair_index.faiss"), faiss.IO_FLAG_MMAP)
        self.clip_index = faiss.read_index(os.path.join(base_dir, "clip_index.faiss"), faiss.IO_FLAG_MMAP)
        
        # Behavioral Index
        self.cleora_index = None
        self.cleora_asins = []
        if cleora_data is not None:
            self.cleora_asins = list(cleora_data['asins'])
            # Build an in-memory index for Cleora as it's small/dynamic
            self.cleora_index = self._build_index(cleora_data['embeddings'])
            self.asin_to_cleora_idx = {asin: i for i, asin in enumerate(self.cleora_asins)}

    def _build_index(self, embeddings):
        X = embeddings.astype("float32")
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        return index

    def get_behavioral_candidates(self, query_asin, top_n=50):
        """Layer 1: Get candidates from Cleora behavioral space"""
        if self.cleora_index is None or query_asin not in self.asin_to_cleora_idx:
            return []
        
        idx = self.asin_to_cleora_idx[query_asin]
        q = self.cleora_index.reconstruct(idx).reshape(1, -1)
        D, I = self.cleora_index.search(q, top_n + 1)
        
        results = [self.cleora_asins[i] for i in I[0]]
        return [asin for asin in results if asin != query_asin][:top_n]

    def score_candidates(self, candidates, query_asin):
        """Layer 2: Score candidates using multimodal content"""
        # Filter candidates to only those in our large index
        valid_asins = [asin for asin in candidates if asin in self.asin_to_idx]
        if not valid_asins:
            return pd.DataFrame()

        # Get query indices
        if query_asin not in self.asin_to_idx:
            return pd.DataFrame()
        
        q_idx = self.asin_to_idx[query_asin]
        cand_idxs = [self.asin_to_idx[asin] for asin in valid_asins]

        # Prepare query embeddings (reconstruct from index)
        q_blair = self.blair_index.reconstruct(q_idx).reshape(1, -1)
        q_clip = self.clip_index.reconstruct(q_idx).reshape(1, -1)

        # Prepare candidate embeddings (reconstruct from index)
        scores = []
        for asin, c_idx in zip(valid_asins, cand_idxs):
            c_blair = self.blair_index.reconstruct(c_idx).reshape(1, -1)
            c_clip = self.clip_index.reconstruct(c_idx).reshape(1, -1)
            
            blair_score = (q_blair @ c_blair.T)[0,0]
            clip_score = (q_clip @ c_clip.T)[0,0]
            
            scores.append({
                'asin': asin,
                'blair_score': blair_score,
                'clip_score': clip_score
            })
        
        return pd.DataFrame(scores)
    
    def get_asin_vec(self, asin):
        """
        Returns a (blair_vec, clip_vec) tuple for the given ASIN.
        Both vectors are reconstructed from their respective FAISS indices.
        Returns None if the ASIN is not in the catalog.
        """
        if asin in self.asin_to_idx:
            idx = self.asin_to_idx[asin]
            blair_vec = self.blair_index.reconstruct(idx)  # [1024]
            clip_vec  = self.clip_index.reconstruct(idx)   # [512]
            return blair_vec, clip_vec
        return None
