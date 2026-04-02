from collections import deque, Counter
import numpy as np
from datetime import datetime
import os
import json
import pandas as pd
from config import *


class UserBehaviorProfile:
    """
    Stores a user's interaction history and aggregated embedding vectors.
    Aggregated embeddings (text_profile, visual_profile) are re-computed on
    every click using exponential temporal weighting.
    """
    def __init__(self, user_id):
        self.user_id = user_id

        # Raw interaction logs
        self.searches = []        # Active search events
        self.clicks   = []        # Clicked items  [{item_id, timestamp, source, position}]
        self.recommendations = [] # System suggested items [{item_ids, timestamp}]
        self.purchases = []
        self.ratings   = []

        # Aggregated embedding vectors (updated on each click)
        self.text_profile   = None   # Weighted avg of BLaIR embeddings (1024-dim)
        self.visual_profile = None   # Weighted avg of CLIP embeddings  (512-dim)

        # Temporal feature: rolling window of recent item IDs
        self.recent_interactions = deque(maxlen=MAX_RECENT_INTERACTIONS)

        # Behavioural summary
        self.preferred_categories = Counter()
        self.diversity_score = 0.0


class UserProfileManager:
    """
    Manages all user profiles: creates, updates, and persists them.
    Profiles are stored as JSON files in  <data_dir>/profiles/<user_id>.json
    so they survive server restarts.
    """
    def __init__(self, retriever=None, data_dir: str = None):
        self.retriever = retriever
        self.data_dir  = data_dir
        self._profiles_dir = os.path.join(data_dir, "profiles") if data_dir else None
        if self._profiles_dir:
            os.makedirs(self._profiles_dir, exist_ok=True)

        self._cache: dict[str, UserBehaviorProfile] = {}  # in-memory cache

    # ─── Profile access ───────────────────────────────────────────────────────
    def get_profile(self, user_id: str) -> UserBehaviorProfile:
        """Load profile from cache, then disk, then create fresh."""
        if user_id not in self._cache:
            profile = UserBehaviorProfile(user_id)
            self._load_from_disk(user_id, profile)
            self._cache[user_id] = profile
        return self._cache[user_id]

    # ─── Event logging ────────────────────────────────────────────────────────
    def log_search(self, user_id, query_text, query_image, results):
        """Record an active search event."""
        profile = self.get_profile(user_id)
        profile.searches.append({
            'timestamp': datetime.now().isoformat(),
            'query_text': query_text,
            'query_image': query_image,
            'results': results[:20],   # store only top-20 to keep JSON lean
            'modality': self._detect_modality(query_text, query_image),
        })
        # searches are ephemeral — no disk save needed for every search

    def log_click(self, user_id: str, item_id: str, source: str = 'search', position: int = 0, action: str = 'click'):
        """Record a click, update aggregated embeddings, then persist to disk."""
        profile = self.get_profile(user_id)
        profile.clicks.append({
            'timestamp': datetime.now().isoformat(),
            'item_id': item_id,
            'source': source,
            'position': position,
            'action': action
        })
        profile.recent_interactions.append(item_id)

        if self.retriever:
            self.update_aggregated_embeddings(profile)

        # Persist after every click so restarts don't lose history
        self.save_profile(user_id)

    def log_recommendation(self, user_id, item_ids):
        """Record a system-initiated recommendation event."""
        profile = self.get_profile(user_id)
        profile.recommendations.append({
            'timestamp': datetime.now().isoformat(),
            'item_ids': item_ids
        })
        # Keep only the last 50 recommendation events to save space
        if len(profile.recommendations) > 50:
            profile.recommendations = profile.recommendations[-50:]
        
        self.save_profile(user_id)

    # ─── Embedding updates ───────────────────────────────────────────────────
    def update_aggregated_embeddings(self, profile: UserBehaviorProfile):
        """Re-compute BLaIR + CLIP profile vectors via temporally-weighted average."""
        all_items = [e['item_id'] for e in profile.clicks]
        if not all_items:
            return

        blair_vecs, clip_vecs = [], []
        for item_id in all_items:
            if item_id in self.retriever.asin_to_idx:
                idx = self.retriever.asin_to_idx[item_id]
                blair_vecs.append(self.retriever.blair_index.reconstruct(idx))
                clip_vecs.append(self.retriever.clip_index.reconstruct(idx))

        if not blair_vecs:
            return

        weights = self._compute_order_weights(len(blair_vecs))
        profile.text_profile   = np.average(blair_vecs, axis=0, weights=weights)
        profile.visual_profile = np.average(clip_vecs,  axis=0, weights=weights)

    # ─── Disk persistence ────────────────────────────────────────────────────
    def _profile_path(self, user_id: str) -> str | None:
        if not self._profiles_dir:
            return None
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return os.path.join(self._profiles_dir, f"{safe_id}.json")

    def save_profile(self, user_id: str):
        """Persist structured unified history and state to a JSON file."""
        path = self._profile_path(user_id)
        if not path:
            return
        profile = self.get_profile(user_id)
        
        # Unify clicks and skips into a chronological history list
        history = []
        for c in profile.clicks:
            history.append({
                "item_id": c["item_id"],
                "action": c.get("action", "click"),
                "timestamp": c.get("timestamp", datetime.now().isoformat())
            })
        for p in profile.purchases: # Skips are appended here
            history.append({
                "item_id": p["item_id"],
                "action": p.get("action", "skip"),
                "timestamp": p.get("timestamp", datetime.now().isoformat())
            })
        history.sort(key=lambda x: x["timestamp"])
        
        # Enforce maximum history size
        if len(history) > 500:
            history = history[-500:]

        embedding = []
        if profile.text_profile is not None and profile.visual_profile is not None:
            embedding = np.concatenate([profile.text_profile, profile.visual_profile]).tolist()

        payload = {
            "user_id": user_id,
            "history": history,
            "state": {
                "embedding": embedding,
                "preferences": dict(profile.preferred_categories),
                "last_updated": datetime.now().isoformat()
            },
            "recent_interactions": list(profile.recent_interactions),
            "recommendations": profile.recommendations,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self, user_id: str, profile: UserBehaviorProfile):
        """Restore history from JSON schema and reconstruct aggregated embeddings."""
        path = self._profile_path(user_id)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            
            history = payload.get("history", [])
            profile.clicks = []
            profile.purchases = []
            
            for h in history:
                action = h.get("action", "click")
                if action in ("click", "cart"):
                    profile.clicks.append({
                        "timestamp": h.get("timestamp"),
                        "item_id": h.get("item_id"),
                        "source": "web_ui",
                        "position": 0,
                        "action": action
                    })
                else: # Skip
                    profile.purchases.append({
                        "timestamp": h.get("timestamp"),
                        "item_id": h.get("item_id"),
                        "action": action
                    })
            
            profile.recommendations = payload.get("recommendations", [])
            recent = payload.get("recent_interactions", [])
            for item_id in recent:
                profile.recent_interactions.append(item_id)
                
            # Rebuild embedding vectors from the loaded click history
            if self.retriever and profile.clicks:
                self.update_aggregated_embeddings(profile)
            print(f"[UserProfileManager] Loaded {len(profile.clicks)} clicks and {len(profile.purchases)} skips for '{user_id}'")
        except Exception as e:
            print(f"[UserProfileManager] Could not load profile for '{user_id}': {e}")

    # ─── Helpers ─────────────────────────────────────────────────────────────
    def _compute_order_weights(self, n: int) -> np.ndarray:
        """Exponential temporal weights — most recent item gets the highest weight."""
        weights = np.exp(TEMPORAL_DECAY * np.arange(n))
        return weights / weights.sum()

    def _detect_modality(self, text, image) -> str:
        if text and image:
            return 'hybrid'
        elif text:
            return 'text'
        elif image:
            return 'image'
        return 'none'
