"""POST /interact — log user click/skip/cart, train RL agent."""
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.api.schemas import InteractRequest
from app.config import settings
from app.core.container import AppContainer
from app.infrastructure.database import db

router = APIRouter()
log    = logging.getLogger("nba_api")


@router.post("/interact")
async def interact(req: InteractRequest,
                   container: AppContainer = Depends(require_ready)):
    profile_manager  = container.profile_manager
    recommend_engine = container.recommend_engine

    # Capture s_t BEFORE profile update
    click_seq_before = await profile_manager.get_click_sequence(req.user_id)

    if req.action == "cart":
        reward = 5.0
        await profile_manager.log_click(req.user_id, req.item_id,
                                         source="web_ui", action="cart")
    elif req.action == "click":
        reward = 1.0
        await profile_manager.log_click(req.user_id, req.item_id,
                                         source="web_ui", action="click")
    else:
        reward  = 0.0
        profile = await profile_manager.get_profile(req.user_id)
        profile.purchases.append({
            "timestamp": datetime.now().isoformat(),
            "item_id":   req.item_id,
            "action":    "skip",
        })
        await profile_manager.save_profile(req.user_id)

    # Push to Redis for background logging
    try:
        if db.redis:
            log_entry = {
                "user_id":    req.user_id,
                "asin":       req.item_id,
                "action":     req.action,
                "timestamp":  datetime.now().isoformat(),
                "session_id": req.session_id,
                "source":     req.source,
                "is_guest":   req.user_id.startswith("guest_") or req.user_id == "web_user",
            }
            await db.redis.rpush("nba_interactions", json.dumps(log_entry))
    except Exception as e:
        log.error(f"Redis queue push failed: {e}")

    # Capture s_{t+1} AFTER profile update
    click_seq_after = await profile_manager.get_click_sequence(req.user_id)

    loss = None
    if click_seq_before:
        loss = recommend_engine.train_rl(
            req.user_id, req.item_id, reward,
            click_seq_before=click_seq_before,
            click_seq_after=click_seq_after,
        )
        recommend_engine.save_rl_weights(req.user_id, settings.DATA_DIR)

    return {"status": "ok", "reward": reward, "rl_loss": loss}
