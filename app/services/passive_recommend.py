"""
app/services/passive_recommend.py — Mode 2: 3-Layer NBA Recommendation Funnel.

Moved from src/passive_recommendation_engine.py.
Imports updated: from app.config / app.services.rl_filter
"""
import os

import faiss
import numpy as np

from app.config import settings
from app.services.rl_filter import RLSequentialFilter

TOP_K                  = settings.TOP_K
COLD_START_THRESHOLD   = settings.COLD_START_THRESHOLD
BEHAVIORAL_CANDIDATES  = settings.BEHAVIORAL_CANDIDATES
SIMILARITY_THRESHOLD   = settings.SIMILARITY_THRESHOLD
RRF_K                  = settings.RRF_K


class PassiveRecommendationEngine:
    """
    System-initiated recommendations based on user behaviour profile.
    3-layer funnel: Cleora (behavioural) → Content veto (Text+CLIP) → RRF + RL re-rank.
    """

    def __init__(self, retriever, profile_manager):
        self.retriever       = retriever
        self.profile_manager = profile_manager
        self.rl_cf           = RLSequentialFilter(retriever)

    # ── Per-user RL weight persistence ────────────────────────────────────────

    def _dqn_path(self, data_dir: str, user_id: str) -> str:
        profiles_dir = os.path.join(data_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return os.path.join(profiles_dir, f"{safe_id}_seq_dqn.pt")

    def save_rl_weights(self, user_id: str, data_dir: str):
        self.rl_cf.save(self._dqn_path(data_dir, user_id))

    def load_rl_weights(self, user_id: str, data_dir: str):
        path = self._dqn_path(data_dir, user_id)
        if os.path.exists(path):
            self.rl_cf.load(path)

    # ── Main recommendation entry point ───────────────────────────────────────

    async def recommend_for_user(self, user_id: str, top_k: int = TOP_K):
        """Generate personalised recommendations split into two pools."""
        profile = await self.profile_manager.get_profile(user_id)

        if len(profile.clicks) < COLD_START_THRESHOLD:
            return None

        candidates = self.collaborative_filter(profile, top_n=BEHAVIORAL_CANDIDATES)
        if not candidates:
            return None

        verified_candidates = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile,
        )
        if not verified_candidates:
            return None

        # Tab 1: People Also Buy — pure retrieval score
        pab_ranked = sorted(
            verified_candidates,
            key=lambda x: max(x["text_score"], x["visual_score"]),
            reverse=True,
        )
        people_also_buy = [
            (item["asin"], float(max(item["text_score"], item["visual_score"])), "Retrieval")
            for item in pab_ranked[:top_k]
        ]

        # Tab 2: You Might Like — GRU-Sequential RL
        candidate_asins = [item["asin"] for item in verified_candidates]
        click_seq       = await self.profile_manager.get_click_sequence(user_id)
        rl_scores       = self.rl_cf.get_candidate_scores(click_seq, candidate_asins)
        y_ranked        = sorted(
            [{"asin": a, "score": s} for a, s in rl_scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )
        you_might_like = [
            (item["asin"], float(item["score"]), "RL-SeqDQN")
            for item in y_ranked[:top_k]
        ]

        return {"people_also_buy": people_also_buy, "you_might_like": you_might_like}

    # ── Layer 1: Behavioural candidate generation ─────────────────────────────

    def collaborative_filter(self, profile, top_n: int = 50) -> list:
        all_candidates = set()

        if profile.cleora_profile is not None and self.retriever.cleora_index is not None:
            query_vec = profile.cleora_profile.reshape(1, -1).astype("float32")
            faiss.normalize_L2(query_vec)
            D, I = self.retriever.cleora_index.search(query_vec, top_n)
            for idx in I[0]:
                if idx != -1:
                    all_candidates.add(self.retriever.cleora_asins[idx])

        seeds = list(profile.recent_interactions)[-5:]
        for item_id in seeds:
            all_candidates.update(
                self.retriever.get_behavioral_candidates(item_id, top_n=top_n)
            )

        if not all_candidates:
            import random
            pool = [a for a in self.retriever.cleora_asins if a in self.retriever.asin_to_idx]
            all_candidates.update(random.sample(pool, min(top_n, len(pool))))

        seen_items = {c["item_id"] for c in profile.clicks}
        return list(all_candidates - seen_items)

    # ── Layer 2: Content veto ──────────────────────────────────────────────────

    def content_verify(self, candidates: list, user_text_profile, user_visual_profile) -> list:
        if user_text_profile is None or user_visual_profile is None:
            return []

        verified       = []
        valid_candidates = [a for a in candidates if a in self.retriever.asin_to_idx]

        for asin in valid_candidates:
            idx        = self.retriever.asin_to_idx[asin]
            item_text  = self.retriever.text_flat.reconstruct(idx)
            item_clip  = self.retriever.clip_index.reconstruct(idx)

            text_sim   = float(user_text_profile   @ item_text)
            visual_sim = float(user_visual_profile @ item_clip)

            if text_sim >= SIMILARITY_THRESHOLD or visual_sim >= SIMILARITY_THRESHOLD:
                verified.append({"asin": asin, "text_score": text_sim, "visual_score": visual_sim})

        return verified

    # ── Layer 3: RRF + RL fusion ───────────────────────────────────────────────

    async def rrf_fusion(self, verified_candidates: list, user_id: str, k: int = RRF_K):
        candidate_asins = [item["asin"] for item in verified_candidates]
        click_seq       = await self.profile_manager.get_click_sequence(user_id)
        rl_scores       = self.rl_cf.get_candidate_scores(click_seq, candidate_asins)

        text_ranked   = sorted(verified_candidates, key=lambda x: x["text_score"],   reverse=True)
        visual_ranked = sorted(verified_candidates, key=lambda x: x["visual_score"], reverse=True)
        rl_ranked     = sorted(
            [{"asin": a, "rl_score": s} for a, s in rl_scores.items()],
            key=lambda x: x["rl_score"],
            reverse=True,
        )

        scores = {}
        def add_score(asin, rank, layer_name):
            if asin not in scores:
                scores[asin] = {"score": 0.0, "best_layer": "", "best_comp": 0.0}
            comp = 1.0 / (k + rank + 1)
            scores[asin]["score"] += comp
            if comp > scores[asin]["best_comp"]:
                scores[asin]["best_comp"]  = comp
                scores[asin]["best_layer"] = layer_name

        for rank, item in enumerate(text_ranked):
            add_score(item["asin"], rank, "Cleora + Text")
        for rank, item in enumerate(visual_ranked):
            add_score(item["asin"], rank, "Cleora + CLIP")
        for rank, item in enumerate(rl_ranked):
            add_score(item["asin"], rank, "RL-SeqDQN")

        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [(asin, data["score"], data["best_layer"]) for asin, data in sorted_items]

    # ── RL training hook ──────────────────────────────────────────────────────

    def train_rl(self, user_id: str, item_asin: str, reward: float,
                 click_seq_before: list = None, click_seq_after: list = None) -> float | None:
        return self.rl_cf.train_step(
            click_seq_before or [],
            item_asin,
            reward,
            click_seq_after or [],
        )
