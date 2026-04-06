"""
NBA Multimodal Recommendation System — FastAPI Backend
=======================================================
Endpoints:
  GET  /health       — liveness probe
  POST /search       — Mode 1: active search with live BLaIR / CLIP encoding
  GET  /recommend    — Mode 2: 3-layer NBA funnel (Cleora → Veto → RL-DQN)
  POST /interact     — log user click/skip, train RL agent, persist profile
  GET  /profile      — return user profile stats for the dashboard
"""

import os
import sys
import random
import logging
import base64
import io
import ast
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ─── Path setup so `src.*` imports work when api.py is in the project root ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import *
from retriever import Retriever
from user_profile_manager import UserProfileManager
from active_search_engine import ActiveSearchEngine
from passive_recommendation_engine import PassiveRecommendationEngine
from environment import get_reward  # kept for simulated reward fallback

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("nba_api")

DATA_DIR = os.path.join(BASE_DIR, "src", "data")

# ─── Global state ─────────────────────────────────────────────────────────────
_state: dict = {
    "ready": False,
    "retriever": None,
    "profile_manager": None,
    "search_engine": None,
    "recommend_engine": None,
    "metadata_df": None,
    "blair_model": None,     # sentence-transformers model
    "clip_model": None,      # transformers CLIP model
    "clip_processor": None,  # CLIP processor
    "device": None,
}

# ─── Lifespan: load everything on startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and indices once at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device
    log.info(f"Device: {device}")

    # 1. Load FAISS indices + Cleora embeddings
    log.info("Loading FAISS indices and Cleora embeddings (this may take a minute)…")
    try:
        cleora_path = os.path.join(DATA_DIR, "cleora_embeddings.npz")
        cleora_data = np.load(cleora_path)
        retriever = Retriever(DATA_DIR, cleora_data)
        _state["retriever"] = retriever
    except Exception as e:
        log.error(f"Failed to load FAISS indices: {e}")
        yield
        return

    # 2. Load item metadata
    log.info("Loading item metadata…")
    try:
        meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
        metadata_df = pd.read_parquet(meta_path)
        
        # CRITICAL: Force ASINs to strings. 
        # Forensic script proved simple map(str) works best for this parquet.
        if "parent_asin" in metadata_df.columns:
            metadata_df["parent_asin"] = metadata_df["parent_asin"].astype(str)
            metadata_df.set_index("parent_asin", inplace=True)
        else:
            metadata_df.index = metadata_df.index.map(str)
        
        _state["metadata_df"] = metadata_df
        log.info(f"Metadata loaded: {len(metadata_df):,} items")
    except Exception as e:
        log.warning(f"Metadata not found — falling back to stub metadata: {e}")
        _state["metadata_df"] = pd.DataFrame()

    # 3. Build pipeline objects (reranker injected after it's loaded in step 5.5)
    profile_manager  = UserProfileManager(retriever=retriever, data_dir=DATA_DIR)
    recommend_engine = PassiveRecommendationEngine(retriever, profile_manager)
    _state["profile_manager"]  = profile_manager
    _state["recommend_engine"] = recommend_engine

    # 4. Load live text encoder — BLaIR (1024-dim)
    #    The BLaIR paper's model is 'hyp1231/blair-roberta-large'.
    #    It produces 1024-dim embeddings matching the FAISS blair_index.
    log.info("Loading BLaIR text encoder (hyp1231/blair-roberta-large)…")
    try:
        from sentence_transformers import SentenceTransformer
        blair_model = SentenceTransformer("hyp1231/blair-roberta-large", device=str(device))
        _state["blair_model"] = blair_model
        log.info("BLaIR encoder ready ✓")
    except Exception as e:
        log.warning(f"BLaIR encoder failed to load — text search will use proxy mode: {e}")

    # 5. Load live image encoder — CLIP (512-dim)
    #    openai/clip-vit-base-patch32 produces 512-dim image embeddings.
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_dim        = 512
    log.info(f"Loading CLIP image encoder ({clip_model_name})…")
    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model     = CLIPModel.from_pretrained(clip_model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model.eval()
        _state["clip_model"]     = clip_model
        _state["clip_processor"] = clip_processor
        # Task 1.4: Log CLIP configuration at startup for audit trail
        log.info(
            f"CLIP model loaded: {clip_model_name} │ dim={clip_dim} │ device={device}"
        )
    except Exception as e:
        log.warning(f"CLIP encoder failed to load — image search will be disabled: {e}")

    # 5.5 Build ActiveSearchEngine (reranker removed — latency was 2–54s with no quality gain)
    search_engine = ActiveSearchEngine(
        retriever,
        profile_manager,
        reranker=None,
        metadata_df=_state["metadata_df"],
    )
    _state["search_engine"] = search_engine
    bm25_status = "ready ✓ (Tantivy Rust)" if search_engine.tantivy_index is not None else "disabled (index not found)"
    log.info(f"Search engine ready ✓  |  Keyword index: {bm25_status}")

    # 6. Qwen2.5-1.5B-Instruct — lazy-loaded on first /ask_llm call
    # Not pre-loaded at startup to keep VRAM free for NLLB translation model.
    # First /ask_llm request will trigger a ~5s one-time load, then it stays warm.
    _state["llm_pipeline"] = None
    log.info("Qwen LLM will lazy-load on first /ask_llm request.")

    _state["ready"] = True
    log.info("🚀 NBA API is ready!")
    yield
    log.info("Shutting down NBA API…")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="NBA Multimodal Recommendation API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic models ──────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = ""
    image_base64: Optional[str] = None
    top_k: int = 20

class InteractRequest(BaseModel):
    user_id: str
    item_id: str
    action: str   # "click" | "skip"

class AskLLMRequest(BaseModel):
    item_id: str
    title: str
    author: str
    user_prompt: str = "Why should I read this book? Give me a short 2-sentence pitch."

# ─── Guard helper ─────────────────────────────────────────────────────────────
def _require_ready():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="System still initializing. Try again in a moment.")

# ─── Encoding helpers ─────────────────────────────────────────────────────────
def _encode_text_query(text: str) -> np.ndarray | None:
    """Encode a text string to a 1024-dim BLaIR embedding. Returns None on failure."""
    model = _state["blair_model"]
    if model is None or not text.strip():
        return None
    try:
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype("float32")
    except Exception as e:
        log.warning(f"BLaIR encode failed: {e}")
        return None

def _encode_image_b64(image_b64: str) -> np.ndarray | None:
    """Decode a base64 image string to a 512-dim CLIP image embedding."""
    clip_model     = _state["clip_model"]
    clip_processor = _state["clip_processor"]
    device         = _state["device"]
    if clip_model is None or not image_b64:
        return None
    try:
        # Client-side base64 often loses its padding characters, crashing Python's strict b64decode.
        image_b64 += "=" * ((4 - len(image_b64) % 4) % 4)
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            if not isinstance(feat, torch.Tensor):
                # HuggingFace sometimes returns a BaseModelOutput tuple instead of a tensor.
                # pooler_output (or feat[1]) is the 512-dim visual projection array
                feat = feat.pooler_output if hasattr(feat, 'pooler_output') else feat[1]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy()
    except Exception as e:
        log.warning(f"CLIP image encode failed: {e}")
        return None

def _proxy_query_vecs() -> tuple[np.ndarray, np.ndarray]:
    """Fallback: pick a random item's vectors as a proxy query."""
    retriever = _state["retriever"]
    asin = random.choice(retriever.asins)
    idx  = retriever.asin_to_idx[asin]
    return (
        retriever.blair_index.reconstruct(idx).astype("float32"),
        retriever.clip_index.reconstruct(idx).astype("float32"),
    )


def _ensure_llm_loaded() -> bool:
    """
    Lazy-loads Qwen2.5-1.5B-Instruct on the first /ask_llm call.
    Subsequent calls return immediately (model already in _state).
    Returns True if LLM is ready, False on failure.
    """
    if _state["llm_pipeline"] is not None:
        return True
    try:
        from transformers import pipeline as hf_pipeline
        device    = 0 if torch.cuda.is_available() else -1
        log.info("[Qwen] Lazy-loading Qwen2.5-1.5B-Instruct…")
        llm = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            device=device,
            torch_dtype=torch.float16,
        )
        _state["llm_pipeline"] = llm
        log.info("[Qwen] LLM ready ✓")
        return True
    except Exception as e:
        log.warning(f"[Qwen] Failed to load: {e}")
        return False

# ─── Metadata helper ──────────────────────────────────────────────────────────
def _get_item_details(asin: str) -> dict:
    metadata_df = _state["metadata_df"]
    if metadata_df is not None and asin in metadata_df.index:
        row = metadata_df.loc[asin]
        
        # Parse author name if it's a stringified dict from HuggingFace
        author_val = row.get("author_name")
        clean_author = "Unknown Author"
        if pd.notna(author_val):
            auth_str = str(author_val)
            if auth_str.startswith("{") and auth_str.endswith("}"):
                try:
                    auth_dict = ast.literal_eval(auth_str)
                    clean_author = auth_dict.get("name", "Unknown Author")
                except:
                    clean_author = auth_str
            else:
                clean_author = auth_str

        # Truncate description to 300 chars for API response (full text stays in parquet)
        raw_desc = row.get("description", "") or ""
        description = str(raw_desc).strip()[:300] if pd.notna(raw_desc) else ""
        return {
            "id": asin,
            "title":       str(row["title"])        if pd.notna(row.get("title"))        else f"Book {asin[:8]}",
            "author":      clean_author,
            "genre":       str(row["main_category"]) if pd.notna(row.get("main_category")) else "Books",
            "image_url":   str(row["image_url"])     if pd.notna(row.get("image_url"))     else None,
            "description": description,
            "cover_color": "#" + hex(abs(hash(asin)) % 0xFFFFFF)[2:].zfill(6),
        }
    return {
        "id": asin,
        "title": f"Book {asin[:8]}",
        "author": "Unknown Author",
        "genre": "Books",
        "image_url": None,
        "cover_color": "#" + hex(abs(hash(asin)) % 0xFFFFFF)[2:].zfill(6),
    }

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    retriever = _state.get("retriever")
    return {
        "status":       "ready" if _state["ready"] else "initializing",
        "catalog_size": len(retriever.asins) if retriever else 0,
        "blair_live":   _state["blair_model"]  is not None,
        "clip_live":    _state["clip_model"]   is not None,
        "device":       str(_state.get("device", "unknown")),
    }


@app.post("/search")
async def search(req: SearchRequest, debug: bool = False):
    """
    Mode 1: Multimodal Active Search.
    Offloaded to a thread pool to avoid blocking the event loop during heavy GPU/FAISS ops.
    """
    import anyio
    _require_ready()
    
    # Run the heavy computation in a thread pool
    return await anyio.to_thread.run_sync(_run_search_pipeline, req, debug)

def _run_search_pipeline(req: SearchRequest, debug: bool):
    """Internal synchronous pipeline executed in a worker thread."""
    retriever     = _state["retriever"]
    search_engine = _state["search_engine"]

    t_start = time.perf_counter()
    timings = {}

    # ── Attempt live encoding ──
    ignore_texts = ["i am looking for books with similar cover like this", "find me a book like this cover", "books with this cover"]
    if req.image_base64 and req.query.lower().strip() in ignore_texts:
        req.query = ""

    # Task 1.3: Translation
    t_trans_start = time.perf_counter()
    query_for_encoding = req.query
    if req.query:
        try:
            from utils import translate_vi_to_en
            translated = translate_vi_to_en(req.query)
            if translated != req.query:
                log.info(f"[Translation] '{req.query}' → '{translated}'")
            query_for_encoding = translated
        except Exception as e:
            log.warning(f"[Translation] Hook failed, using original query: {e}")
    timings["translate_ms"] = round((time.perf_counter() - t_trans_start) * 1000, 2)

    # Stage 1: BLaIR text encoding
    t0 = time.perf_counter()
    text_vec  = _encode_text_query(query_for_encoding) if query_for_encoding else None
    timings["encode_blair_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 2: CLIP image encoding
    t1 = time.perf_counter()
    image_vec = _encode_image_b64(req.image_base64)   if req.image_base64 else None
    timings["encode_clip_ms"] = round((time.perf_counter() - t1) * 1000, 2)

    if text_vec is None and image_vec is None:
        text_vec, image_vec = _proxy_query_vecs()

    # Stage 3: FAISS search + adaptive RRF
    t2 = time.perf_counter()
    search_results = search_engine.search(
        "web_user",
        text_query_vec=text_vec,
        image_query_vec=image_vec,
        text_query=req.query,
        top_k=req.top_k,
        include_timings=True,
    )
    results, engine_timings = search_results
    timings.update(engine_timings)
    timings["search_engine_total_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    # Stage 4: Metadata hydration
    t3 = time.perf_counter()
    enriched = []
    for asin, data in results:
        details = _get_item_details(asin)
        details["score"]    = data['score']
        details["text_sim"] = max(0.0, float(data['text_sim']))
        details["img_sim"]  = max(0.0, data['img_sim'])
        if "reranker_score" in data:
            details["reranker_score"] = data["reranker_score"]
        enriched.append(details)
    timings["metadata_hydration_ms"] = round((time.perf_counter() - t3) * 1000, 2)
    timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

    if not enriched:
        response = {
            "results": [],
            "query":   req.query,
            "message": "No sufficiently relevant results found.",
            "total":   0,
            "live_encoding": text_vec is not None,
        }
        if debug: response["_debug_timings"] = timings
        return response

    response = {"results": enriched, "total": len(enriched), "live_encoding": text_vec is not None}
    if debug: response["_debug_timings"] = timings
    return response


@app.get("/recommend")
async def recommend(user_id: str):
    """
    Mode 2: 3-Layer NBA Funnel.
    Cold-start users (< COLD_START_THRESHOLD clicks) receive random catalog items.
    Warm users go through Cleora → Content Veto → RL-DQN.
    """
    _require_ready()
    retriever        = _state["retriever"]
    profile_manager  = _state["profile_manager"]
    recommend_engine = _state["recommend_engine"]

    profile = profile_manager.get_profile(user_id)

    if len(profile.clicks) < COLD_START_THRESHOLD:
        # Cold start — surface random items from the high-precision pool
        pool = [a for a in retriever.cleora_asins if a in retriever.asin_to_idx]
        sample = random.sample(pool, min(10, len(pool)))
        rec_dict = {
            "people_also_buy": [(a, 1.0, "Discovery") for a in sample[:5]],
            "you_might_like": [(a, 1.0, "Discovery") for a in sample[5:]]
        }
        mode = "cold_start"
    else:
        # Load per-user DQN weights if they exist
        recommend_engine.load_rl_weights(user_id, DATA_DIR)
        res = recommend_engine.recommend_for_user(user_id, top_k=5)
        if res is None:
            pool = [a for a in retriever.cleora_asins if a in retriever.asin_to_idx]
            sample = random.sample(pool, min(10, len(pool)))
            rec_dict = {
                "people_also_buy": [(a, 1.0, "Discovery") for a in sample[:5]],
                "you_might_like": [(a, 1.0, "Discovery") for a in sample[5:]]
            }
            mode = "cold_start"
        else:
            rec_dict = res
            mode = "personalized"

    # Log all shown recommendations once
    all_rec_ids = [asin for asin, _, _ in rec_dict["people_also_buy"]] + [asin for asin, _, _ in rec_dict["you_might_like"]]
    profile_manager.log_recommendation(user_id, all_rec_ids)

    metadata_df = _state["metadata_df"]
    
    def enrich_list(recs):
        enriched = []
        for asin, score, layer in recs:
            if metadata_df is not None and len(metadata_df) > 0 and asin not in metadata_df.index:
                continue
            details = _get_item_details(asin)
            details["score"] = float(score)
            details["layer"] = layer
            enriched.append(details)
        return enriched

    return {
        "people_also_buy": enrich_list(rec_dict["people_also_buy"]),
        "you_might_like": enrich_list(rec_dict["you_might_like"]),
        "user_id": user_id, 
        "mode": mode
    }


@app.post("/interact")
async def interact(req: InteractRequest):
    """
    Log a user interaction (click or skip), train the RL agent, and persist
    both the profile and the updated DQN weights.
    """
    _require_ready()
    profile_manager  = _state["profile_manager"]
    recommend_engine = _state["recommend_engine"]

    # ── Capture s_t BEFORE the profile update ────────────────────────────────
    # This must happen first so the replay buffer stores the true pre-interaction
    # state for the Bellman target computation: Q*(s_t, a) = r + γ·max Q(s_t+1, ·)
    pre_profile = profile_manager.get_profile(req.user_id)

    # ── Determine reward and update profile (s_t → s_t+1) ───────────────────
    if req.action == "cart":
        # Strongest positive signal — user intends to purchase
        reward = 5.0
        profile_manager.log_click(req.user_id, req.item_id, source="web_ui", action="cart")
    elif req.action == "click":
        # Normal positive interaction
        reward = 1.0
        profile_manager.log_click(req.user_id, req.item_id, source="web_ui", action="click")
    else:
        # Skips carry zero reward — not a penalty, just no signal
        reward = 0.0
        profile = profile_manager.get_profile(req.user_id)
        profile.purchases.append({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "item_id": req.item_id,
            "action": "skip",
        })
        profile_manager.save_profile(req.user_id)

    # ── Train the RL agent with (s_t, action, reward, s_t+1) ─────────────────
    loss = None
    if pre_profile.text_profile is not None:
        # Profile was updated above; get_profile now returns s_t+1
        post_profile = profile_manager.get_profile(req.user_id)
        loss = recommend_engine.train_rl(
            pre_profile, req.item_id, reward,
            next_profile=post_profile,
        )
        recommend_engine.save_rl_weights(req.user_id, DATA_DIR)

    return {"status": "ok", "reward": reward, "rl_loss": loss}


@app.get("/profile")
async def get_profile(user_id: str):
    """Return aggregated user stats for the frontend dashboard."""
    _require_ready()
    profile_manager = _state["profile_manager"]
    profile = profile_manager.get_profile(user_id)
    total  = len(profile.clicks)
    return {
        "user_id":           user_id,
        "interaction_count": total,
        "click_count":       total,
        "ctr":               total / max(1, total + len(profile.searches)),
        "rl_steps":          total,
        "has_profile":       profile.text_profile is not None,
    }


@app.post("/ask_llm")
async def ask_llm(req: AskLLMRequest):
    """
    Generates a conversational response about a book using Qwen2.5-1.5B-Instruct.
    The model is lazy-loaded on the first call (~5s warm-up), then stays warm in VRAM.
    """
    _require_ready()
    if not _ensure_llm_loaded():
        return {"response": "The AI assistant failed to load. Please try again."}
    llm = _state["llm_pipeline"]
    
    import urllib.request, urllib.parse, json
    
    # Advanced Wikipedia query for ground truth plot summary
    try:
        # Step 1: Find the exact Wikipedia page title for the book
        query_safe = urllib.parse.quote(f"{req.title} (novel)")
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query_safe}&utf8=&format=json&srlimit=1"
        
        req_search = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_search, timeout=3) as response:
            search_data = json.loads(response.read())
            
            if search_data.get('query', {}).get('search'):
                page_title = urllib.parse.quote(search_data['query']['search'][0]['title'])
                
                # Step 2: Extract the actual introductory plot summary for that exact page
                extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=5&explaintext=1&titles={page_title}&format=json"
                req_extract = urllib.request.Request(extract_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                with urllib.request.urlopen(req_extract, timeout=3) as extract_response:
                    extract_data = json.loads(extract_response.read())
                    pages = extract_data['query']['pages']
                    page_id = list(pages.keys())[0]
                    
                    if page_id != "-1":
                        plot_text = pages[page_id].get('extract', '')
                        ground_truth = f"Here is factual internet summary information about the book: {plot_text}\n\nUse this information to accurately summarize the plot."
                    else:
                        ground_truth = "Use your internal knowledge to summarize the plot."
            else:
                ground_truth = "Use your internal knowledge to summarize the plot."
    except Exception as e:
        log.warning(f"Internet search for {req.title} failed: {e}")
        ground_truth = "Use your internal knowledge to summarize the plot."

    messages = [
        {"role": "system", "content": f"You are an AI book assistant. The user wants detailed information about a book. Provide a short response structured EXACTLY like this using Markdown:\n\n**Genre:** [Genre]\n**Plot Summary:** [2-3 sentences summarizing the plot]\n\n{ground_truth}"},
        {"role": "user", "content": f"Tell me about the book '{req.title}' by {req.author}."}
    ]
    
    try:
        # Generate text using the proper chat template formatting
        prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Bypass 'pipeline' warning by using direct model inference
        inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.model.device)
        out_tokens = llm.model.generate(
            **inputs, 
            max_new_tokens=250, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=llm.tokenizer.eos_token_id
        )
        
        # Decode only the generated tokens (everything after the prompt)
        answer = llm.tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        return {"response": answer}
    except Exception as e:
        log.error(f"LLM error: {e}")
        return {"response": "Sorry, I had trouble thinking of a response."}


@app.get("/rl_metrics")
async def rl_metrics(user_id: str):
    """Return real-time RL loss history and buffer sizing"""
    _require_ready()
    recommend_engine = _state["recommend_engine"]
    # Ensure DQN weights are loaded into VRAM
    recommend_engine.load_rl_weights(user_id, DATA_DIR)
    agent = recommend_engine.rl_cf
    
    return {
        "user_id": user_id,
        "loss_history": agent.loss_history,
        "buffer_size": len(agent.buffer),
        "step": agent._step
    }

# ─── Dev entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


