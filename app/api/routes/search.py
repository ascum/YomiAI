"""POST /search — Mode 1: Multimodal Active Search."""
import ast
import logging
import time

import anyio
import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import require_ready
from app.api.schemas import SearchRequest
from app.core import models as ml
from app.core.container import AppContainer

router = APIRouter()
log    = logging.getLogger("nba_api")

_IGNORE_TEXTS = {
    "i am looking for books with similar cover like this",
    "find me a book like this cover",
    "books with this cover",
}


@router.post("/search")
async def search(req: SearchRequest, debug: bool = False,
                 container: AppContainer = Depends(require_ready)):
    return await _run_pipeline(req, debug, container)


async def _run_pipeline(req: SearchRequest, debug: bool, container: AppContainer):
    t_start = time.perf_counter()
    timings = {}

    if req.image_base64 and req.query.lower().strip() in _IGNORE_TEXTS:
        req.query = ""

    # Translation (VI → EN)
    t0 = time.perf_counter()
    query_for_encoding = req.query
    if req.query:
        try:
            from app.infrastructure.translation import translate_to_en
            translated = await anyio.to_thread.run_sync(translate_to_en, req.query)
            if translated != req.query:
                log.info(f"[Translation] '{req.query}' → '{translated}'")
            query_for_encoding = translated
        except Exception as e:
            log.warning(f"[Translation] Hook failed, using original query: {e}")
    timings["translate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Parallel encoding (text + CLIP)
    text_vec  = None
    image_vec = None

    async def encode_text_task():
        nonlocal text_vec
        t = time.perf_counter()
        text_vec = await anyio.to_thread.run_sync(
            ml.encode_text, query_for_encoding, container.text_encoder
        ) if query_for_encoding else None
        timings["encode_text_ms"] = round((time.perf_counter() - t) * 1000, 2)

    async def encode_image_task():
        nonlocal image_vec
        t = time.perf_counter()
        image_vec = await anyio.to_thread.run_sync(
            ml.encode_image_b64,
            req.image_base64,
            container.clip_model,
            container.clip_processor,
            container.device,
        ) if req.image_base64 else None
        timings["encode_clip_ms"] = round((time.perf_counter() - t) * 1000, 2)

    async with anyio.create_task_group() as tg:
        tg.start_soon(encode_text_task)
        tg.start_soon(encode_image_task)

    if text_vec is None and image_vec is None:
        text_vec, image_vec = ml.proxy_query_vecs(container.retriever)

    # FAISS search + RRF
    t2 = time.perf_counter()
    search_results = await container.search_engine.search(
        "web_user", text_vec, image_vec, req.query, req.top_k, True
    )
    results, engine_timings = search_results
    timings.update(engine_timings)
    timings["search_engine_total_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    # Metadata hydration
    t3 = time.perf_counter()
    enriched  = []
    meta_df   = container.metadata_repo.df
    max_score = results[0][1]["score"] if results else 1.0

    for asin, data in results:
        if meta_df is not None and asin in meta_df.index:
            row        = meta_df.loc[asin]
            raw_author = row.get("author_name", "Unknown")
            author     = str(raw_author)
            if author.startswith("{") and "name" in author:
                try:
                    author = ast.literal_eval(author).get("name", author)
                except Exception:
                    pass
            norm_score = (data["score"] / max_score) ** 0.5 if max_score > 0 else 0
            enriched.append({
                "id":        asin,
                "title":     str(row.get("title", f"Book {asin[:8]}")),
                "author":    author,
                "image_url": str(row.get("image_url", "")),
                "score":     float(norm_score),
            })
        else:
            enriched.append({"id": asin, "title": f"Book {asin[:8]}", "score": 0.5})

    timings["metadata_hydration_ms"] = round((time.perf_counter() - t3) * 1000, 2)
    timings["total_ms"]              = round((time.perf_counter() - t_start) * 1000, 2)

    response_data = {
        "results":       enriched,
        "total":         len(enriched),
        "live_encoding": text_vec is not None,
        "query":         req.query,
    }
    if debug:
        response_data["_debug_timings"] = timings

    return JSONResponse(content=response_data)
