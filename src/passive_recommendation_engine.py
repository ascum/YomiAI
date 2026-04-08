import numpy as np
import faiss
from config import *
from rl_collaborative_filter import RLCollaborativeFilter
import os

class PassiveRecommendationEngine:
    """
    System-initiated recommendations based on user behavior profile.
    3-layer funnel: Cleora (behavioral) → Content veto (BLaIR+CLIP) → RRF + RL re-rank.
    """
    def __init__(self, retriever, profile_manager):
        self.retriever = retriever
        self.profile_manager = profile_manager

        # RL-based collaborative filter
        # state_dim = BLaIR profile (1024) + CLIP profile (512) = 1536
        # item_dim  = projected+fused item repr = RL_ITEM_PROJ_DIM * 2 = 512
        self.rl_cf = RLCollaborativeFilter(
            state_dim=BLAIR_DIM + CLIP_DIM,
            item_dim=RL_ITEM_PROJ_DIM * 2
        )

    # ─── Per-user RL weight persistence ──────────────────────────────────────
    def _dqn_path(self, data_dir: str, user_id: str) -> str:
        profiles_dir = os.path.join(data_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return os.path.join(profiles_dir, f"{safe_id}_dqn.pt")

    def save_rl_weights(self, user_id: str, data_dir: str):
        path = self._dqn_path(data_dir, user_id)
        self.rl_cf.save(path)

    def load_rl_weights(self, user_id: str, data_dir: str):
        path = self._dqn_path(data_dir, user_id)
        if os.path.exists(path):
            self.rl_cf.load(path)

    # ─── Main recommendation entry point ──────────────────────────────────────
    async def recommend_for_user(self, user_id, top_k=TOP_K):
        """Generate personalized recommendations separated into two distinct pools."""
        profile = await self.profile_manager.get_profile(user_id)

        # Cold start — not enough data yet
        if len(profile.clicks) < COLD_START_THRESHOLD:
            return None

        # Layer 1: Cleora behavioral candidate generation
        candidates = self.collaborative_filter(profile, top_n=BEHAVIORAL_CANDIDATES)
        if not candidates:
            return None

        # Layer 2: Content sanity check (BLaIR + CLIP vs user profile)
        verified_candidates = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile
        )
        if not verified_candidates:
            return None

        # Tab 1: People Also Buy -> Pure retrieval (BLaIR / CLIP fusion)
        # We sort simply by the highest similarity score the item achieved in either modality
        pab_ranked = sorted(
            verified_candidates, 
            key=lambda x: max(x['text_score'], x['visual_score']), 
            reverse=True
        )
        people_also_buy = [
            (item['asin'], float(max(item['text_score'], item['visual_score'])), "Retrieval") 
            for item in pab_ranked[:top_k]
        ]

        # Tab 2: You Might Like -> Pure RL
        candidate_asins = [item['asin'] for item in verified_candidates]
        rl_scores = self.rl_cf.get_candidate_scores(profile, candidate_asins, self.retriever)
        y_ranked = sorted(
            [{'asin': a, 'score': s} for a, s in rl_scores.items()], 
            key=lambda x: x['score'], 
            reverse=True
        )
        you_might_like = [
            (item['asin'], float(item['score']), "RL-DQN") 
            for item in y_ranked[:top_k]
        ]

        return {
            "people_also_buy": people_also_buy,
            "you_might_like": you_might_like
        }

    # ─── Layer 1: Behavioral candidate generation ─────────────────────────────
    def collaborative_filter(self, profile, top_n=50):
        """
        Dual-mode retrieval:
        1. Query Cleora index using the user's aggregated cleora_profile vector.
        2. Query using the 5 most recent interactions as seeds.
        """
        all_candidates = set()

        # 1. Vector-based search (The "Fast Lane" projection)
        if profile.cleora_profile is not None and self.retriever.cleora_index is not None:
            # Query the FAISS index with the user's average behavioral vector
            query_vec = profile.cleora_profile.reshape(1, -1).astype("float32")
            faiss.normalize_L2(query_vec)
            D, I = self.retriever.cleora_index.search(query_vec, top_n)
            for idx in I[0]:
                if idx != -1:
                    all_candidates.add(self.retriever.cleora_asins[idx])

        # 2. Seed-based search (The "Legacy" path)
        seeds = list(profile.recent_interactions)[-5:]
        for item_id in seeds:
            neighbors = self.retriever.get_behavioral_candidates(item_id, top_n=top_n)
            all_candidates.update(neighbors)

        # Fallback: if seeds aren't in Cleora graph, grab popular nodes
        if not all_candidates:
            import random
            pool = [a for a in self.retriever.cleora_asins if a in self.retriever.asin_to_idx]
            all_candidates.update(random.sample(pool, min(top_n, len(pool))))

        # Exclude already-seen items
        seen_items = {c['item_id'] for c in profile.clicks}
        return list(all_candidates - seen_items)

    # ─── Layer 2: Content veto ────────────────────────────────────────────────
    def content_verify(self, candidates, user_text_profile, user_visual_profile):
        """BLaIR + CLIP sanity check against user's aggregated embedding profile."""
        verified = []
        valid_candidates = [a for a in candidates if a in self.retriever.asin_to_idx]

        for asin in valid_candidates:
            idx = self.retriever.asin_to_idx[asin]
            item_blair = self.retriever.blair_index.reconstruct(idx)
            item_clip  = self.retriever.clip_index.reconstruct(idx)

            text_sim   = float(user_text_profile   @ item_blair)
            visual_sim = float(user_visual_profile @ item_clip)

            if text_sim >= SIMILARITY_THRESHOLD or visual_sim >= SIMILARITY_THRESHOLD:
                verified.append({
                    'asin': asin,
                    'text_score': text_sim,
                    'visual_score': visual_sim,
                })

        return verified

    # ─── Layer 3: RRF + RL fusion ─────────────────────────────────────────────
    def rrf_fusion(self, verified_candidates, profile, k=RRF_K):
        """Fuse BLaIR, CLIP, and RL scores via Reciprocal Rank Fusion."""
        candidate_asins = [item['asin'] for item in verified_candidates]
        rl_scores = self.rl_cf.get_candidate_scores(profile, candidate_asins, self.retriever)

        text_ranked   = sorted(verified_candidates, key=lambda x: x['text_score'],   reverse=True)
        visual_ranked = sorted(verified_candidates, key=lambda x: x['visual_score'], reverse=True)
        rl_ranked     = sorted(
            [{'asin': a, 'rl_score': s} for a, s in rl_scores.items()],
            key=lambda x: x['rl_score'],
            reverse=True,
        )

        scores = {}
        def add_score(asin, rank, layer_name):
            if asin not in scores:
                scores[asin] = {'score': 0.0, 'best_layer': '', 'best_comp': 0.0}
            comp = 1.0 / (k + rank + 1)
            scores[asin]['score'] += comp
            if comp > scores[asin]['best_comp']:
                scores[asin]['best_comp'] = comp
                scores[asin]['best_layer'] = layer_name

        for rank, item in enumerate(text_ranked):
            add_score(item['asin'], rank, 'Cleora + BLaIR')
        for rank, item in enumerate(visual_ranked):
            add_score(item['asin'], rank, 'Cleora + CLIP')
        for rank, item in enumerate(rl_ranked):
            add_score(item['asin'], rank, 'RL-DQN')

        sorted_items = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        return [(asin, data['score'], data['best_layer']) for asin, data in sorted_items]

    # ─── RL training hook ─────────────────────────────────────────────────────
    def train_rl(self, user_profile, item_asin, reward, next_profile=None):
        """
        Train the RL agent with one interaction.
        Called from api.py /interact.

        Args:
            user_profile: UserProfile BEFORE the interaction (s_t)
            item_asin:    Item the user interacted with
            reward:       Scalar reward signal
            next_profile: UserProfile AFTER profile_manager.record_interaction()
                          (s_t+1). Pass None if unavailable (terminal treatment).
        """
        return self.rl_cf.train_step(
            user_profile, item_asin, reward, self.retriever,
            next_profile=next_profile
        )

    # ─── Disks Persistence ────────────────────────────────────────────────────
    def load_rl_weights(self, user_id: str, data_dir: str):
        """Load the user's specific DQN weights if they exist on disk."""
        import os
        path = os.path.join(data_dir, "profiles", f"{user_id}_dqn.pt")
        if os.path.exists(path):
            self.rl_cf.load(path)

    def save_rl_weights(self, user_id: str, data_dir: str):
        """Save the user's specific DQN weights to disk."""
        import os
        path = os.path.join(data_dir, "profiles", f"{user_id}_dqn.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.rl_cf.save(path)
