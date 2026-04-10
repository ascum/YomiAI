"""
app/repository/profile_repo.py — User profile management.

Moved from src/user_profile_manager.py.
Import path for database updated: from app.infrastructure.database import db
"""
import asyncio
import json
import os
from collections import Counter, deque
from datetime import datetime

import numpy as np

from app.config import settings
from app.infrastructure.database import db


class UserBehaviorProfile:
    def __init__(self, user_id: str):
        self.user_id   = user_id
        self.searches  = []
        self.clicks    = []
        self.recommendations = []
        self.purchases = []
        self.text_profile    = None
        self.visual_profile  = None
        self.cleora_profile  = None
        self.recent_interactions = deque(maxlen=settings.MAX_RECENT_INTERACTIONS)
        self.preferred_categories = Counter()


class UserProfileManager:
    """
    Manages all user profiles: creates, updates, and persists them.
    Primary store is MongoDB; local JSON is used for migration fallback.
    """

    def __init__(self, retriever=None, data_dir: str = None):
        self.retriever = retriever
        self.data_dir  = data_dir
        self._profiles_dir = os.path.join(data_dir, "profiles") if data_dir else None
        if self._profiles_dir:
            os.makedirs(self._profiles_dir, exist_ok=True)

        self._cache: dict[str, UserBehaviorProfile] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    # ── Profile access ────────────────────────────────────────────────────────

    async def get_profile(self, user_id: str) -> UserBehaviorProfile:
        async with self._get_user_lock(user_id):
            if user_id in self._cache:
                return self._cache[user_id]

            profile = UserBehaviorProfile(user_id)
            self._cache[user_id] = profile

            mongo_data = await db.fetch_profile(user_id)
            if mongo_data:
                self._load_from_dict(profile, mongo_data)
                print(f"[UserProfileManager] Loaded profile for '{user_id}' from MongoDB")
            else:
                migrated = await self._migrate_from_disk(user_id, profile)
                if migrated:
                    print(f"[UserProfileManager] Migrated profile for '{user_id}' from Disk to MongoDB")
                    await self._save_to_mongo(user_id, profile)

            return profile

    # ── Click sequence for GRU-Sequential DQN ────────────────────────────────

    async def get_click_sequence(self, user_id: str,
                                  max_len: int = settings.MAX_RECENT_INTERACTIONS) -> list:
        profile = await self.get_profile(user_id)
        asins = [c["item_id"] for c in profile.clicks
                 if c.get("action", "click") in ("click", "cart")]
        return asins[-max_len:]

    # ── Event logging ─────────────────────────────────────────────────────────

    async def log_search(self, user_id: str, query_text, query_image, results):
        profile = await self.get_profile(user_id)
        profile.searches.append({
            "timestamp":   datetime.now().isoformat(),
            "query_text":  query_text,
            "query_image": query_image,
            "results":     results[:20],
            "modality":    self._detect_modality(query_text, query_image),
        })
        await self._save_to_mongo(user_id, profile)

    async def log_click(self, user_id: str, item_id: str,
                         source: str = "search", position: int = 0, action: str = "click"):
        profile = await self.get_profile(user_id)
        profile.clicks.append({
            "timestamp": datetime.now().isoformat(),
            "item_id":   item_id,
            "source":    source,
            "position":  position,
            "action":    action,
        })
        profile.recent_interactions.append(item_id)
        if self.retriever:
            self.update_aggregated_embeddings(profile)
        await self._save_to_mongo(user_id, profile)

    async def log_recommendation(self, user_id: str, item_ids: list):
        profile = await self.get_profile(user_id)
        profile.recommendations.append({
            "timestamp": datetime.now().isoformat(),
            "item_ids":  item_ids,
        })
        if len(profile.recommendations) > 50:
            profile.recommendations = profile.recommendations[-50:]
        await self._save_to_mongo(user_id, profile)

    # ── Embedding updates (synchronous) ───────────────────────────────────────

    def update_aggregated_embeddings(self, profile: UserBehaviorProfile):
        all_items = [e["item_id"] for e in profile.clicks]
        if not all_items:
            return

        text_vecs, clip_vecs, cleora_vecs = [], [], []
        for item_id in all_items:
            if item_id in self.retriever.asin_to_idx:
                idx = self.retriever.asin_to_idx[item_id]
                text_vecs.append(self.retriever.text_flat.reconstruct(idx))
                clip_vecs.append(self.retriever.clip_index.reconstruct(idx))
            if (self.retriever.cleora_index is not None
                    and item_id in self.retriever.asin_to_cleora_idx):
                c_idx = self.retriever.asin_to_cleora_idx[item_id]
                cleora_vecs.append(self.retriever.cleora_index.reconstruct(c_idx))

        if not text_vecs:
            return
        weights = self._compute_order_weights(len(text_vecs))
        profile.text_profile   = np.average(text_vecs, axis=0, weights=weights)
        profile.visual_profile = np.average(clip_vecs, axis=0, weights=weights)
        if cleora_vecs:
            c_weights = self._compute_order_weights(len(cleora_vecs))
            profile.cleora_profile = np.average(cleora_vecs, axis=0, weights=c_weights)

    # ── Persistence helpers ───────────────────────────────────────────────────

    async def _save_to_mongo(self, user_id: str, profile: UserBehaviorProfile):
        history = []
        for c in profile.clicks:
            history.append({"item_id": c["item_id"], "action": c.get("action", "click"),
                             "timestamp": c.get("timestamp")})
        for p in profile.purchases:
            history.append({"item_id": p["item_id"], "action": p.get("action", "skip"),
                             "timestamp": p.get("timestamp")})
        history.sort(key=lambda x: x["timestamp"] or "", reverse=True)

        payload = {
            "user_id":              user_id,
            "recent_history":       history[:50],
            "recent_searches":      profile.searches[-20:],
            "recent_recs":          profile.recommendations[-20:],
            "text_profile":         profile.text_profile.tolist() if profile.text_profile is not None else [],
            "visual_profile":       profile.visual_profile.tolist() if profile.visual_profile is not None else [],
            "cleora_profile":       profile.cleora_profile.tolist() if profile.cleora_profile is not None else [],
            "recent_interactions":  list(profile.recent_interactions),
            "last_updated":         datetime.now().isoformat(),
        }
        await db.upsert_profile(user_id, payload)

    async def save_profile(self, user_id: str):
        profile = await self.get_profile(user_id)
        await self._save_to_mongo(user_id, profile)

    def _load_from_dict(self, profile: UserBehaviorProfile, data: dict):
        profile.text_profile   = np.array(data["text_profile"])   if data.get("text_profile")   else None
        profile.visual_profile = np.array(data["visual_profile"]) if data.get("visual_profile") else None
        profile.cleora_profile = np.array(data["cleora_profile"]) if data.get("cleora_profile") else None

        profile.searches       = data.get("recent_searches", [])
        profile.recommendations = data.get("recent_recs", [])

        history = data.get("recent_history", [])
        profile.clicks, profile.purchases = [], []
        for h in reversed(history):
            if h["action"] in ("click", "cart"):
                profile.clicks.append({**h, "source": "web_ui", "position": 0})
            else:
                profile.purchases.append(h)

        for item_id in data.get("recent_interactions", []):
            profile.recent_interactions.append(item_id)

    # ── Disk migration ────────────────────────────────────────────────────────

    def _profile_path(self, user_id: str) -> str | None:
        if not self._profiles_dir:
            return None
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return os.path.join(self._profiles_dir, f"{safe_id}.json")

    async def _migrate_from_disk(self, user_id: str, profile: UserBehaviorProfile) -> bool:
        path = self._profile_path(user_id)
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._old_load_logic(profile, payload)
            return True
        except Exception as e:
            print(f"[UserProfileManager] Migration failed for '{user_id}': {e}")
            return False

    def _old_load_logic(self, profile: UserBehaviorProfile, payload: dict):
        for h in payload.get("history", []):
            action = h.get("action", "click")
            if action in ("click", "cart"):
                profile.clicks.append({
                    "timestamp": h.get("timestamp"), "item_id": h.get("item_id"),
                    "source": "web_ui", "position": 0, "action": action,
                })
            else:
                profile.purchases.append({
                    "timestamp": h.get("timestamp"), "item_id": h.get("item_id"),
                    "action": action,
                })
        for item_id in payload.get("recent_interactions", []):
            if item_id not in profile.recent_interactions:
                profile.recent_interactions.append(item_id)
        if self.retriever and profile.clicks:
            self.update_aggregated_embeddings(profile)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_order_weights(self, n: int) -> np.ndarray:
        weights = np.exp(settings.TEMPORAL_DECAY * np.arange(n))
        return weights / weights.sum()

    def _detect_modality(self, text, image) -> str:
        if text and image:
            return "hybrid"
        elif text:
            return "text"
        elif image:
            return "image"
        return "none"
