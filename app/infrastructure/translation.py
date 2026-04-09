"""
app/infrastructure/translation.py — VI→EN translation via NLLB-200.

Moved from src/utils.py. Logic unchanged; import path updated.
"""
import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log = logging.getLogger("nba_api")

NLLB_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
NLLB_SRC_LANG = "vie_Latn"
NLLB_TGT_LANG = "eng_Latn"

_nllb_tokenizer = None
_nllb_model     = None
_nllb_device    = None


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        log.debug(f"detect_language failed (defaulting to 'en'): {e}")
        return "en"


def _load_nllb():
    global _nllb_tokenizer, _nllb_model, _nllb_device
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nllb_device = device
    log.info(f"[NLLB] Loading {NLLB_MODEL_ID} (device={device})")

    _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID, src_lang=NLLB_SRC_LANG)
    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_ID,
        torch_dtype=__import__("torch").float16 if device == "cuda" else __import__("torch").float32,
    ).to(device)
    _nllb_model.eval()
    log.info(f"[NLLB] Translation model ready ✓")


def translate_vi_to_en(text: str) -> str:
    """Translate Vietnamese text to English. Returns original on non-VI input or failure."""
    global _nllb_tokenizer, _nllb_model

    if not text or not text.strip():
        return text

    if detect_language(text) != "vi":
        log.debug("[NLLB] Non-Vietnamese input, skipping translation.")
        return text

    if _nllb_model is None:
        try:
            _load_nllb()
        except Exception as e:
            log.warning(f"[NLLB] Model load failed — using original query. Error: {e}")
            return text

    try:
        import torch

        inputs = _nllb_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(_nllb_device)

        forced_bos_token_id = _nllb_tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)
        with torch.no_grad():
            output_ids = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=1,
                do_sample=False,
                max_new_tokens=64,
            )

        translated = _nllb_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        log.info(f"[NLLB] VI→EN: '{text}' → '{translated}'")
        return translated

    except Exception as e:
        log.warning(f"[NLLB] Translation failed — using original. Error: {e}")
        return text
