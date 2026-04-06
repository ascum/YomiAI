"""
Language detection and VI→EN translation utilities for the NBA search pipeline.

Translation model: facebook/nllb-200-distilled-1.3B
  - 200-language multilingual seq2seq
  - Lazy-loaded on first Vietnamese query (adds ~2s warm-up, then ~25-50ms/query on GPU)
  - Falls back to original query on any failure — never breaks the search pipeline

Usage:
    from utils import translate_vi_to_en
    en_query = translate_vi_to_en("tiểu thuyết trinh thám")
    # → "detective thriller novel"
"""

import logging
import sys

# Fix Windows terminal crash when printing Vietnamese characters to stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log = logging.getLogger("nba_api")

# ── Constants ─────────────────────────────────────────────────────────────────
NLLB_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
NLLB_SRC_LANG = "vie_Latn"   # Vietnamese (Latin script)
NLLB_TGT_LANG = "eng_Latn"   # English (Latin script)

# ── Lazy model holders (module-level singletons) ──────────────────────────────
_nllb_tokenizer = None
_nllb_model     = None
_nllb_device    = None


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Returns the ISO 639-1 language code for the given text ('vi', 'en', etc.).
    Returns 'en' on empty input or if langdetect fails or is not installed.

    Requires: pip install langdetect
    """
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except Exception as e:
        log.debug(f"detect_language failed (defaulting to 'en'): {e}")
        return "en"


# ── Translation ───────────────────────────────────────────────────────────────

def _load_nllb():
    """Lazy-load the NLLB-1.3B model into VRAM (called once on first VI query)."""
    global _nllb_tokenizer, _nllb_model, _nllb_device

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nllb_device = device

    log.info(
        f"[NLLB] Loading translation model: {NLLB_MODEL_ID}  "
        f"(device={device}, dtype=fp16 on GPU / fp32 on CPU)"
    )

    _nllb_tokenizer = AutoTokenizer.from_pretrained(
        NLLB_MODEL_ID,
        src_lang=NLLB_SRC_LANG,
    )

    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    _nllb_model.eval()

    log.info(f"[NLLB] Translation model ready ✓  ({NLLB_MODEL_ID} | {device})")


def translate_vi_to_en(text: str) -> str:
    """
    Translates Vietnamese text to English using NLLB-200-distilled-1.3B.

    Behaviour:
    - If text is empty → returns as-is immediately.
    - If langdetect says the text is NOT Vietnamese → returns as-is (no translation).
    - If the model isn't loaded yet → loads it lazily on this first call (~2s warm-up).
    - If translation raises any exception → falls back to original text with a warning.
      This ensures the search pipeline never breaks due to a translation failure.

    Args:
        text: Raw user query string (may be any language).

    Returns:
        English translation if input was Vietnamese, otherwise the original string.
    """
    global _nllb_tokenizer, _nllb_model

    if not text or not text.strip():
        return text

    # Only translate if actually Vietnamese — skip the model for EN queries (fast path)
    detected = detect_language(text)
    if detected != "vi":
        log.debug(f"[NLLB] detect_language='{detected}', skipping translation.")
        return text

    # Lazy-load the model on first Vietnamese query
    if _nllb_model is None:
        try:
            _load_nllb()
        except Exception as e:
            log.warning(
                f"[NLLB] Model load failed — falling back to original query. Error: {e}"
            )
            return text

    # Run translation
    try:
        import torch

        inputs = _nllb_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,      # Book queries are short; 128 tokens is generous
        ).to(_nllb_device)

        forced_bos_token_id = _nllb_tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

        with torch.no_grad():
            output_ids = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=1,          # Greedy search is much faster for short queries
                do_sample=False,
                max_new_tokens=64,    # Queries are rarely longer than this
            )

        translated = _nllb_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        log.info(f"[NLLB] VI→EN: '{text}'  →  '{translated}'")
        return translated

    except Exception as e:
        log.warning(f"[NLLB] Translation failed — falling back to original. Error: {e}")
        return text
