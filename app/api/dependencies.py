"""
app/api/dependencies.py — FastAPI Depends() helpers.

Routes call Depends(get_container) to receive the AppContainer without
touching any global state directly.
"""
from fastapi import HTTPException, Request

from app.core.container import AppContainer


def get_container(request: Request) -> AppContainer:
    """Return the AppContainer built during lifespan startup."""
    return request.app.state.container


def require_ready(request: Request) -> AppContainer:
    """Return the container, raising 503 if the system is still initializing."""
    container = get_container(request)
    if not container.ready:
        raise HTTPException(
            status_code=503,
            detail="System still initializing. Try again in a moment.",
        )
    return container
