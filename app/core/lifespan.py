"""
app/core/lifespan.py — FastAPI lifespan: startup, background worker, shutdown.

Extracted from api.py (@asynccontextmanager lifespan + _log_worker).
Builds the AppContainer and attaches it to app.state.container.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch

from app.config import settings
from app.core.container import AppContainer
from app.core import models as model_loader
from app.infrastructure.database import db
from app.repository.faiss_repo import Retriever
from app.repository.metadata_repo import MetadataRepository
from app.repository.profile_repo import UserProfileManager
from app.services.active_search import ActiveSearchEngine
from app.services.passive_recommend import PassiveRecommendationEngine

log = logging.getLogger("nba_api")


async def _log_worker():
    """Drains the Redis queue into MongoDB in the background."""
    log.info("Background logging worker started.")
    while True:
        try:
            if db.redis:
                res = await db.redis.blpop("nba_interactions", timeout=5)
                if res:
                    _, data_json = res
                    interaction  = json.loads(data_json)
                    await db.log_interaction(interaction)
            else:
                await asyncio.sleep(5)
        except Exception as e:
            log.error(f"Logger worker error: {e}")
            await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app):
    """Load all ML models and infrastructure once at startup."""
    container = AppContainer()
    app.state.container = container

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    container.device = device
    log.info(f"Device: {device}")

    # 1. FAISS indices + Cleora embeddings
    log.info("Loading FAISS indices and Cleora embeddings…")
    try:
        cleora_path = settings.DATA_DIR + "/cleora_embeddings.npz"
        cleora_data = np.load(cleora_path)
        retriever   = Retriever(settings.DATA_DIR, cleora_data)
        container.retriever = retriever
    except Exception as e:
        log.error(f"Failed to load FAISS indices: {e}")
        yield
        return

    # 2. Item metadata (Parquet)
    container.metadata_repo = MetadataRepository(settings.DATA_DIR)

    # 3. Pipeline objects
    profile_manager  = UserProfileManager(retriever=retriever, data_dir=settings.DATA_DIR)
    recommend_engine = PassiveRecommendationEngine(retriever, profile_manager)
    container.profile_manager  = profile_manager
    container.recommend_engine = recommend_engine

    # 4. BLaIR text encoder
    log.info("Loading BLaIR text encoder (hyp1231/blair-roberta-large)…")
    container.blair_model = model_loader.load_blair(device)

    # 5. CLIP image encoder
    log.info("Loading CLIP image encoder (openai/clip-vit-base-patch32)…")
    container.clip_model, container.clip_processor = model_loader.load_clip(device)

    # 5.5 Active Search Engine
    search_engine = ActiveSearchEngine(
        retriever,
        profile_manager,
        reranker=None,
        metadata_df=container.metadata_repo.df,
        data_dir=settings.DATA_DIR,
    )
    container.search_engine = search_engine
    bm25_status = ("ready ✓ (Tantivy Rust)" if search_engine.tantivy_index is not None
                   else "disabled (index not found)")
    log.info(f"Search engine ready ✓  |  Keyword index: {bm25_status}")

    # 6. Qwen LLM — lazy-loaded on first /ask_llm call (keeps VRAM free for NLLB)
    log.info("Qwen LLM will lazy-load on first /ask_llm request.")

    # Infrastructure
    await db.connect()
    app.state.log_worker_task = asyncio.create_task(_log_worker())

    container.ready = True
    log.info("NBA API is ready!")
    yield

    log.info("Shutting down NBA API…")
    await db.disconnect()
