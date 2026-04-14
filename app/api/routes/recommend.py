"""GET /recommend and GET /rl_metrics."""
import random

from fastapi import APIRouter, Depends

from app.api.dependencies import require_ready
from app.config import settings
from app.core.container import AppContainer

router = APIRouter()

COLD_START_THRESHOLD = settings.COLD_START_THRESHOLD


@router.get("/recommend")
async def recommend(user_id: str, container: AppContainer = Depends(require_ready)):
    """
    Mode 2: 3-Layer NBA Funnel.
    Cold-start users receive random catalogue items.
    Warm users go through Cleora → Veto → DIF-SASRec.
    """
    retriever        = container.retriever
    profile_manager  = container.profile_manager
    recommend_engine = container.recommend_engine

    profile = await profile_manager.get_profile(user_id)

    if len(profile.clicks) < COLD_START_THRESHOLD:
        rec_dict, mode = _cold_start(retriever)
    else:
        async with container.agent_pool.borrow() as agent:
            agent.load_user(user_id, settings.DATA_DIR)
            res = await recommend_engine.recommend_for_user(user_id, agent, top_k=5)
        if res is None:
            rec_dict, mode = _cold_start(retriever)
        else:
            rec_dict, mode = res, "personalized"

    all_rec_ids = (
        [a for a, _, _ in rec_dict["people_also_buy"]]
        + [a for a, _, _ in rec_dict["you_might_like"]]
    )
    await profile_manager.log_recommendation(user_id, all_rec_ids)

    meta_repo = container.metadata_repo

    def enrich_list(recs):
        enriched = []
        for asin, score, layer in recs:
            if meta_repo.df is not None and len(meta_repo.df) > 0 and asin not in meta_repo.df.index:
                continue
            details          = meta_repo.get_item(asin)
            details["score"] = float(score)
            details["layer"] = layer
            enriched.append(details)
        return enriched

    return {
        "people_also_buy": enrich_list(rec_dict["people_also_buy"]),
        "you_might_like":  enrich_list(rec_dict["you_might_like"]),
        "user_id":         user_id,
        "mode":            mode,
    }


def _cold_start(retriever):
    pool   = [a for a in retriever.cleora_asins if a in retriever.asin_to_idx]
    sample = random.sample(pool, min(10, len(pool)))
    rec_dict = {
        "people_also_buy": [(a, 1.0, "Discovery") for a in sample[:5]],
        "you_might_like":  [(a, 1.0, "Discovery") for a in sample[5:]],
    }
    return rec_dict, "cold_start"


@router.get("/rl_metrics")
async def rl_metrics(user_id: str, container: AppContainer = Depends(require_ready)):
    """Return real-time DIF-SASRec model metrics."""
    async with container.agent_pool.borrow() as agent:
        agent.load_user(user_id, settings.DATA_DIR)
        return {
            "user_id":      user_id,
            "loss_history": list(agent.loss_history),
            "step":         agent._step,
            "arch":         "DIF-SASRec",
        }
