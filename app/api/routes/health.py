"""GET /health — liveness probe."""
from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.core.container import AppContainer

router = APIRouter()


@router.get("/health")
def health(container: AppContainer = Depends(get_container)):
    retriever = container.retriever
    return {
        "status":       "ready" if container.ready else "initializing",
        "catalog_size": len(retriever.asins) if retriever else 0,
        "text_encoder_live": container.text_encoder is not None,
        "clip_live":    container.clip_model   is not None,
        "device":       str(container.device),
    }
