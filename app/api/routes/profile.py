"""GET /profile — user stats and hydrated recent history."""
from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.core.container import AppContainer

router = APIRouter()


@router.get("/profile")
async def get_profile(user_id: str, container: AppContainer = Depends(require_ready)):
    """Return aggregated user stats and hydrated recent history for the UI bar."""
    profile_manager = container.profile_manager
    meta_repo       = container.metadata_repo
    profile         = await profile_manager.get_profile(user_id)

    # Unify clicks and skips, sort by timestamp
    history = []
    for c in profile.clicks:
        history.append({"item_id": c["item_id"], "action": c.get("action", "click"),
                         "ts": c.get("timestamp")})
    for p in profile.purchases:
        history.append({"item_id": p["item_id"], "action": "skip", "ts": p.get("timestamp")})

    history.sort(key=lambda x: x["ts"] or "", reverse=True)

    hydrated_history = []
    for entry in history[:10]:
        details = meta_repo.get_item(entry["item_id"])
        hydrated_history.append({
            **details,
            "action":    entry["action"],
            "timestamp": entry["ts"],
        })

    total_clicks  = len(profile.clicks)
    total_searches = len(profile.searches)

    return {
        "user_id":           user_id,
        "interaction_count": total_clicks,
        "searches_count":    total_searches,
        "ctr":               total_clicks / max(1, total_clicks + total_searches),
        "recent_items":      hydrated_history,
        "has_profile":       profile.text_profile is not None,
    }
