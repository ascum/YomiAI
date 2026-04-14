"""
app/services/passive_recommend.py — Mode 2: Dual-Pipeline NBA Recommendation Funnel.

Pipeline A — "People Also Buy"  : Cleora behavioural → Content veto → similarity rank
Pipeline B — "You Might Like"   : BGE-M3 HNSW KNN   → Content veto → DIF-SASRec score

Pipeline B has ZERO dependency on Cleora. If Cleora is unavailable, Pipeline A
returns an empty list while Pipeline B continues to work via HNSW retrieval.

Agent ownership: this engine no longer owns a DIFSASRecAgent. Routes borrow an
agent from AgentPool, call agent.load_user() to load per-user weights, then pass
the agent into the engine methods. This gives true per-request isolation.
"""
import faiss
import numpy as np

from app.config import settings

TOP_K                 = settings.TOP_K
COLD_START_THRESHOLD  = settings.COLD_START_THRESHOLD
BEHAVIORAL_CANDIDATES = settings.BEHAVIORAL_CANDIDATES
SIMILARITY_THRESHOLD  = settings.SIMILARITY_THRESHOLD
RRF_K                 = settings.RRF_K
PERSONAL_CANDIDATES   = settings.PERSONAL_CANDIDATES


class PassiveRecommendationEngine:
    """
    System-initiated recommendations split into two independent pipelines.

    Tab 1 "People Also Buy"  — Cleora collaborative filtering (unchanged).
    Tab 2 "You Might Like"   — DIF-SASRec sequential model, zero Cleora dependency.

    Routes supply a DIFSASRecAgent (borrowed from AgentPool) to every method
    that needs one. The engine itself holds no model state.
    """

    def __init__(self, retriever, profile_manager, category_encoder=None):
        self.retriever        = retriever
        self.profile_manager  = profile_manager
        self.category_encoder = category_encoder

    # ── Main recommendation entry point ───────────────────────────────────────

    async def recommend_for_user(self, user_id: str, agent, top_k: int = TOP_K):
        """
        Generate personalised recommendations split into two pools.

        Returns None when the user has fewer than COLD_START_THRESHOLD clicks
        and both pipelines produce nothing.
        """
        profile = await self.profile_manager.get_profile(user_id)

        if len(profile.clicks) < COLD_START_THRESHOLD:
            return None

        # ── Pipeline A: People Also Buy (Cleora-dependent) ───────────────────
        pab_candidates = self.collaborative_filter(profile, top_n=BEHAVIORAL_CANDIDATES)
        if pab_candidates:
            verified_pab = self.content_verify(
                pab_candidates,
                user_text_profile=profile.text_profile,
                user_visual_profile=profile.visual_profile,
            )
            pab_ranked = sorted(
                verified_pab,
                key=lambda x: max(x["text_score"], x["visual_score"]),
                reverse=True,
            )
            people_also_buy = [
                (item["asin"], float(max(item["text_score"], item["visual_score"])), "Retrieval")
                for item in pab_ranked[:top_k]
            ]
        else:
            people_also_buy = []

        # ── Pipeline B: You Might Like (DIF-SASRec, zero Cleora) ─────────────
        you_might_like = await self._personal_recommend(profile, user_id, agent, top_k)

        if not people_also_buy and not you_might_like:
            return None

        return {"people_also_buy": people_also_buy, "you_might_like": you_might_like}

    # ── Pipeline B implementation ─────────────────────────────────────────────

    async def _personal_recommend(self, profile, user_id: str, agent, top_k: int) -> list:
        """
        DIF-SASRec intent → HNSW KNN → content veto → DIF-SASRec scoring.

        Uses the user's text_profile (1024-dim BGE-M3 weighted average) for
        HNSW retrieval — no Cleora index required.
        """
        if profile.text_profile is None:
            return []

        # Step 1: HNSW KNN retrieval anchored on the user's BGE-M3 text profile
        seen = {c["item_id"] for c in profile.clicks}
        candidates = self.retriever.get_content_candidates(
            profile.text_profile,
            top_n=PERSONAL_CANDIDATES,
            exclude_asins=seen,
        )
        if not candidates:
            return []

        # Step 2: Content veto (same threshold as Pipeline A)
        verified = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile,
        )
        if not verified:
            return []

        # Step 3: DIF-SASRec scoring
        asins, cat_ids = await self.profile_manager.get_click_sequence_with_categories(
            user_id
        )
        candidate_asins = [item["asin"] for item in verified]
        scores = agent.get_candidate_scores(asins, cat_ids, candidate_asins)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (asin, float(score), "DIF-SASRec")
            for asin, score in ranked[:top_k]
        ]

    # ── Layer 1: Behavioural candidate generation (Pipeline A) ───────────────

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

    # ── Layer 2: Content veto (shared by both pipelines) ─────────────────────

    def content_verify(self, candidates: list, user_text_profile,
                       user_visual_profile) -> list:
        if user_text_profile is None or user_visual_profile is None:
            return []

        verified         = []
        valid_candidates = [a for a in candidates if a in self.retriever.asin_to_idx]

        for asin in valid_candidates:
            idx       = self.retriever.asin_to_idx[asin]
            item_text = self.retriever.text_flat.reconstruct(idx)
            item_clip = self.retriever.clip_index.reconstruct(idx)

            text_sim   = float(user_text_profile   @ item_text)
            visual_sim = float(user_visual_profile @ item_clip)

            if text_sim >= SIMILARITY_THRESHOLD or visual_sim >= SIMILARITY_THRESHOLD:
                verified.append({
                    "asin":        asin,
                    "text_score":  text_sim,
                    "visual_score": visual_sim,
                })

        return verified

    # ── DIF-SASRec online training hook ──────────────────────────────────────

    def train_personal(self, user_id: str, item_asin: str, agent,
                       click_seq_before: list = None) -> float | None:
        """Train the DIF-SASRec model on the latest click event."""
        if not click_seq_before:
            return None
        cat_id = (self.category_encoder.get_category_id(item_asin)
                  if self.category_encoder else 1)
        all_asins = list(self.retriever.asin_to_idx.keys())
        return agent.train_step(
            click_seq_before,
            item_asin,
            cat_id,
            all_asins,
        )

    # ── RRF fusion (preserved for future use) ─────────────────────────────────

    async def rrf_fusion(self, verified_candidates: list, user_id: str, agent,
                         k: int = RRF_K):
        candidate_asins = [item["asin"] for item in verified_candidates]
        asins, cat_ids  = await self.profile_manager.get_click_sequence_with_categories(user_id)
        sasrec_scores   = agent.get_candidate_scores(asins, cat_ids, candidate_asins)

        text_ranked   = sorted(verified_candidates, key=lambda x: x["text_score"],   reverse=True)
        visual_ranked = sorted(verified_candidates, key=lambda x: x["visual_score"], reverse=True)
        sasrec_ranked = sorted(
            [{"asin": a, "score": s} for a, s in sasrec_scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )

        scores: dict = {}

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
        for rank, item in enumerate(sasrec_ranked):
            add_score(item["asin"], rank, "DIF-SASRec")

        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        return [(asin, data["score"], data["best_layer"]) for asin, data in sorted_items]
