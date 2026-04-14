"""
app/core/container.py — Typed dependency container (replaces _state dict).

Built once in lifespan.py and injected into routes via FastAPI Depends().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch


@dataclass
class AppContainer:
    """
    Holds every runtime dependency the routes need.

    All fields are set during lifespan startup.  Routes receive this via
    Depends(get_container) — no global mutable dict, no sys.path hacks.
    """
    # Infrastructure / data layer
    retriever:        Any = None          # app.repository.faiss_repo.Retriever
    metadata_repo:    Any = None          # app.repository.metadata_repo.MetadataRepository
    profile_manager:  Any = None          # app.repository.profile_repo.UserProfileManager

    # Service layer
    search_engine:    Any = None          # app.services.active_search.ActiveSearchEngine
    recommend_engine: Any = None          # app.services.passive_recommend.PassiveRecommendationEngine
    agent_pool:       Any = None          # app.services.agent_pool.AgentPool
    category_encoder: Any = None          # app.services.category_encoder.CategoryEncoder

    # ML models
    text_encoder:     Any = None          # sentence_transformers.SentenceTransformer (BGE-M3)
    clip_model:       Any = None          # transformers.CLIPModel
    clip_processor:   Any = None          # transformers.CLIPProcessor
    device:           torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Readiness flag — routes return 503 until True
    ready: bool = False
