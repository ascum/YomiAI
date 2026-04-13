"""
app/core/lifespan.py — FastAPI lifespan: startup, background worker, shutdown.

Extracted from api.py (@asynccontextmanager lifespan + _log_worker).
Builds the AppContainer and attaches it to app.state.container.
"""
import asyncio
import json
import logging
import os
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

    # 3. Category encoder (for DIF-SASRec personal pipeline)
    from app.services.category_encoder import CategoryEncoder
    cat_encoder    = CategoryEncoder()
    cat_vocab_path = os.path.join(settings.DATA_DIR, "category_vocab.json")
    if os.path.exists(cat_vocab_path):
        cat_encoder.load(cat_vocab_path)
    else:
        log.info("Building category vocabulary from item_metadata.parquet ...")
        cat_encoder.build_from_parquet(
            os.path.join(settings.DATA_DIR, "item_metadata.parquet")
        )
        cat_encoder.save(cat_vocab_path)
    container.category_encoder = cat_encoder

    # 3b. Pipeline objects
    profile_manager  = UserProfileManager(
        retriever=retriever,
        data_dir=settings.DATA_DIR,
        category_encoder=cat_encoder,
    )
    recommend_engine = PassiveRecommendationEngine(
        retriever, profile_manager, category_encoder=cat_encoder
    )
    container.profile_manager  = profile_manager
    container.recommend_engine = recommend_engine

    # 4. Text encoder (BGE-M3)
    log.info(f"Loading text encoder ({settings.TEXT_ENCODER_MODEL})…")
    container.text_encoder = model_loader.load_text_encoder(device)
    model_loader.warmup_text_encoder()

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
    tantivy_status = ("ready ✓ (Tantivy Rust)" if search_engine.tantivy_index is not None
                      else "disabled (index not found)")
    log.info(f"Search engine ready ✓  |  Keyword index: {tantivy_status}")

    # 6. Translation warmup — load NLLB-600M and compile CUDA kernels now so
    #    the first real request doesn't pay the 274ms cold-start penalty.
    await asyncio.get_event_loop().run_in_executor(
        None,
        __import__("app.infrastructure.translation", fromlist=["warmup"]).warmup,
    )

    # 7. Qwen LLM — pre-load for faster first response
    log.info("Pre-loading Qwen2.5-1.5B-Instruct…")
    from app.services import llm as llm_service
    await asyncio.get_event_loop().run_in_executor(None, llm_service.ensure_loaded)
    log.info("Qwen LLM ready ✓")

    # Infrastructure
    await db.connect()
    app.state.log_worker_task = asyncio.create_task(_log_worker())

    container.ready = True
    log.info("NBA API is ready!")
    yield

    log.info("Shutting down NBA API…")
    await db.disconnect()
