"""POST /auth/check and POST /auth/create — password-free identity endpoints."""
from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.core.container import AppContainer
from app.infrastructure.database import db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/check")
async def auth_check(body: dict, container: AppContainer = Depends(require_ready)):
    """Check whether a user_id exists in MongoDB.

    Body: { "username": str }
    Returns: { "found": bool, "user_id": str }
    """
    user_id = (body.get("username") or "").strip()
    if not user_id:
        return {"found": False, "user_id": ""}

    doc = await db.fetch_profile(user_id)
    return {"found": doc is not None, "user_id": user_id}


@router.post("/create")
async def auth_create(body: dict, container: AppContainer = Depends(require_ready)):
    """Create a new user profile in MongoDB.

    Body: { "username": str }
    Returns: { "user_id": str }
    """
    user_id = (body.get("username") or "").strip()
    if not user_id:
        return {"user_id": ""}

    profile_manager = container.profile_manager
    await profile_manager.get_profile(user_id)
    await profile_manager.save_profile(user_id)
    return {"user_id": user_id}
