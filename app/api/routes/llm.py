"""POST /ask_llm — Qwen2.5 grounded book assistant."""
import logging

from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.api.schemas import AskLLMRequest
from app.core.container import AppContainer
from app.services import llm as llm_service

router = APIRouter()
log    = logging.getLogger("nba_api")


@router.post("/ask_llm")
async def ask_llm(req: AskLLMRequest,
                  container: AppContainer = Depends(require_ready)):
    """
    Generates a conversational response about a book using Qwen2.5-1.5B-Instruct.
    The model is lazy-loaded on the first call (~5s warm-up), then stays warm.
    """
    if not llm_service.ensure_loaded():
        return {"response": "The AI assistant failed to load. Please try again."}

    try:
        answer = llm_service.generate(req.title, req.author, req.user_prompt)
        return {"response": answer}
    except Exception as e:
        log.error(f"LLM error: {e}")
        return {"response": "Sorry, I had trouble thinking of a response."}
