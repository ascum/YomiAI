"""POST /ask_llm — Qwen2.5 grounded book assistant."""
import asyncio
import logging

from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.api.schemas import AskLLMRequest
from app.core.container import AppContainer
from app.services import llm as llm_service

from fastapi.responses import StreamingResponse

router = APIRouter()
log    = logging.getLogger("nba_api")


@router.post("/ask_llm")
async def ask_llm(req: AskLLMRequest,
                  debug: bool = False,
                  container: AppContainer = Depends(require_ready)):
    """
    Generates a conversational response about a book using Qwen2.5 (Sync).
    """
    log.info(f"POST /ask_llm called with debug={debug}")
    if not llm_service.ensure_loaded():
        return {"response": "The AI assistant failed to load. Please try again."}

    # Fetch local context from metadata repo
    item = container.metadata_repo.get_item(req.item_id)
    local_desc = item.get("description", "")

    try:
        start_total = asyncio.get_event_loop().time()
        loop = asyncio.get_event_loop()
        answer, timings = await loop.run_in_executor(
            None,
            llm_service.generate,
            req.title,
            req.author,
            req.user_prompt,
            local_desc
        )
        
        res = {"response": answer}
        if debug:
            total_ms = round((asyncio.get_event_loop().time() - start_total) * 1000, 2)
            timings["total_ms"] = total_ms
            res["_debug_timings"] = timings
            
        return res
    except Exception as e:
        log.error(f"LLM error: {e}")
        return {"response": "Sorry, I had trouble thinking of a response."}


@router.post("/ask_llm_stream")
async def ask_llm_stream(req: AskLLMRequest,
                         container: AppContainer = Depends(require_ready)):
    """
    Streams a conversational response about a book token-by-token.
    """
    log.info(f"POST /ask_llm_stream called for {req.title} (ID: {req.item_id})")
    
    # Fetch local context from metadata repo
    item = container.metadata_repo.get_item(req.item_id)
    local_desc = item.get("description", "")

    async def stream_generator():
        try:
            loop = asyncio.get_event_loop()
            gen = llm_service.generate_stream(req.title, req.author, req.user_prompt, local_desc)
            
            while True:
                chunk = await loop.run_in_executor(None, next, gen, None)
                if chunk is None:
                    break
                yield chunk
        except Exception as e:
            log.error(f"Streaming error: {e}")
            yield " [Error generating response] "

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
