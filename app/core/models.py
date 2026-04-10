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
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from app.config import settings

log = logging.getLogger("nba_api")


# ── BGE-M3 Configuration ──────────────────────────────────────────────────────

_TEXT_ENCODE_MAX_LEN = 64   # search queries are ≤ 15 tokens; 64 is safe headroom
_text_encoder_ref = None    # module-level reference for lru_cache


@lru_cache(maxsize=4096)
def _cached_encode(text: str, encoder_id: str) -> tuple:
    """
    Cache wrapper around the text encoder. Keyed on (text, encoder_id) so the
    cache is automatically invalidated if the model is swapped in config.

    Returns a tuple (not ndarray) because lru_cache requires hashable return
    values — callers convert back with np.array(..., dtype='float32').
    """
    if _text_encoder_ref is None:
        raise RuntimeError("Text encoder not loaded")

    # .encode expects a list of strings. max_seq_length is already set on the model.
    vec = _text_encoder_ref.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    # Flatten and convert to tuple for hashability
    return tuple(vec.flatten().astype("float32").tolist())


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_text_encoder(device: torch.device):
    """Load the configured text encoder (BGE-M3). Returns model or None on failure."""
    global _text_encoder_ref
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.TEXT_ENCODER_MODEL, device=str(device),
                                    trust_remote_code=True)
        if device.type == "cuda":
            model.half()

            # Step 2: torch.compile() the inner transformer for fused CUDA kernels.
            # mode="reduce-overhead" optimizes for repeated same-shape inputs.
            # Skip on Windows as it requires Triton (Linux-only official support).
            import sys
            if sys.platform != "win32":
                try:
                    # model[0] is the transformer module in SentenceTransformer
                    model[0].auto_model = torch.compile(
                        model[0].auto_model, mode="reduce-overhead"
                    )
                    log.info("BGE-M3 inner transformer compiled with torch.compile ✓")
                except Exception as e:
                    log.warning(f"torch.compile skipped (non-fatal): {e}")
            else:
                log.info("torch.compile skipped (platform=win32)")

        # Set the max sequence length for encoding. BGE-M3 defaults to 8192.
        model.max_seq_length = _TEXT_ENCODE_MAX_LEN

        _text_encoder_ref = model
        log.info(f"Text encoder ready ✓  model={settings.TEXT_ENCODER_MODEL}  "
                 f"limit={_TEXT_ENCODE_MAX_LEN} tokens  "
                 f"dtype={'fp16' if device.type == 'cuda' else 'fp32'}")
        return model
    except Exception as e:
        log.warning(f"Text encoder failed to load — text search will use proxy mode: {e}")
        return None


def warmup_text_encoder():
    """Run a dummy encode to trigger torch.compile and NLLB warmup."""
    if _text_encoder_ref is None:
        return
    log.info("Warming up text encoder (BGE-M3)…")
    try:
        # One short query, one near the limit to ensure varied shapes are handled
        # (though we use static 64 padding via max_seq_length)
        encode_text("warmup query", _text_encoder_ref)
        encode_text("a slightly longer warmup query to trigger fused kernels", _text_encoder_ref)
        log.info("Text encoder warmup done ✓")
    except Exception as e:
        log.warning(f"Text encoder warmup failed (non-fatal): {e}")


def load_clip(device: torch.device):
# ... rest of file (load_clip, encode_text, etc.)
    """Load CLIP model + processor. Returns (model, processor) or (None, None)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model     = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(device)
        clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        clip_model.eval()
        if device.type == "cuda":
            clip_model.half()
        log.info(f"CLIP model loaded: {settings.CLIP_MODEL_NAME} │ dim={settings.CLIP_DIM} │ "
                 f"device={device} │ dtype={'fp16' if device.type == 'cuda' else 'fp32'}")
        return clip_model, clip_processor
    except Exception as e:
        log.warning(f"CLIP encoder failed to load — image search will be disabled: {e}")
        return None, None


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_text(text: str, text_encoder) -> np.ndarray | None:
    """Encode a text string with the text encoder. Returns None on failure."""
    if text_encoder is None or not text.strip():
        return None
    try:
        # result is a tuple (float32, ...)
        result = _cached_encode(text, settings.TEXT_ENCODER_MODEL)
        # Convert back to (1, 1024) ndarray
        return np.array(result, dtype="float32").reshape(1, -1)
    except Exception as e:
        log.warning(f"Text encode failed: {e}")
        return None


def encode_image_b64(image_b64: str, clip_model, clip_processor,
                     device: torch.device) -> np.ndarray | None:
    """Decode a base64 image and return a CLIP embedding."""
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
        retriever.text_flat.reconstruct(idx).astype("float32"),
        retriever.clip_index.reconstruct(idx).astype("float32"),
    )
