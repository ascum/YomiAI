"""
app/core/models.py — ML model loading and encoding helpers.

Extracted from api.py (lifespan model loading blocks + _encode_* helpers).
All functions are pure: they receive a model/processor and return a result.
"""
from __future__ import annotations

import base64
import io
import logging
import random

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("nba_api")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_blair(device: torch.device):
    """Load BGE-M3 sentence-transformer (replaces BLaIR). Returns model or None on failure."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3", device=str(device), trust_remote_code=True)
        if device.type == "cuda":
            model.half()
        log.info(f"BGE-M3 encoder ready ✓ (dtype={'fp16' if device.type == 'cuda' else 'fp32'})")
        return model
    except Exception as e:
        log.warning(f"BGE-M3 encoder failed to load — text search will use proxy mode: {e}")
        return None


def load_clip(device: torch.device):
    """Load CLIP model + processor. Returns (model, processor) or (None, None)."""
    model_name = "openai/clip-vit-base-patch32"
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model     = CLIPModel.from_pretrained(model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        clip_model.eval()
        if device.type == "cuda":
            clip_model.half()
        log.info(f"CLIP model loaded: {model_name} │ dim=512 │ device={device} │ "
                 f"dtype={'fp16' if device.type == 'cuda' else 'fp32'}")
        return clip_model, clip_processor
    except Exception as e:
        log.warning(f"CLIP encoder failed to load — image search will be disabled: {e}")
        return None, None


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_text(text: str, blair_model) -> np.ndarray | None:
    """Encode a text string to a 1024-dim BLaIR embedding. Returns None on failure."""
    if blair_model is None or not text.strip():
        return None
    try:
        vec = blair_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype("float32")
    except Exception as e:
        log.warning(f"BLaIR encode failed: {e}")
        return None


def encode_image_b64(image_b64: str, clip_model, clip_processor,
                     device: torch.device) -> np.ndarray | None:
    """Decode a base64 image and return a 512-dim CLIP embedding."""
    if clip_model is None or not image_b64:
        return None
    try:
        image_b64 += "=" * ((4 - len(image_b64) % 4) % 4)
        img_bytes = base64.b64decode(image_b64)
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs    = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat[1]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy()
    except Exception as e:
        log.warning(f"CLIP image encode failed: {e}")
        return None


def proxy_query_vecs(retriever) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: pick a random item's vectors as a proxy query."""
    asin = random.choice(retriever.asins)
    idx  = retriever.asin_to_idx[asin]
    return (
        retriever.blair_flat.reconstruct(idx).astype("float32"),
        retriever.clip_index.reconstruct(idx).astype("float32"),
    )
