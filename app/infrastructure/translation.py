"""
app/infrastructure/translation.py — Multilingual → EN translation via NLLB-200.

Design:
  - lingua detects language (< 1ms, deterministic)
  - _ISO_TO_NLLB maps ISO 639-1 → NLLB language tag (add rows to support more)
  - _cached_translate wraps NLLB with an LRU cache — repeat queries are free
  - translate_to_en is the single public entry point for all callers
"""
import logging
import sys
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log = logging.getLogger("nba_api")

# ── Model config ──────────────────────────────────────────────────────────────

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"   # 600M: same 200 langs, half the size
NLLB_TGT_LANG = "eng_Latn"

# ── Supported source languages ────────────────────────────────────────────────
# ISO 639-1 → NLLB language tag.
# To add a language: append one line here. Nothing else needs to change.

_ISO_TO_NLLB: dict[str, str] = {
    "vi": "vie_Latn",   # Vietnamese
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ja": "jpn_Jpan",   # Japanese
    "ko": "kor_Hang",   # Korean
    "ar": "arb_Arab",   # Arabic
    "pt": "por_Latn",   # Portuguese
    "ru": "rus_Cyrl",   # Russian
    "it": "ita_Latn",   # Italian
    "th": "tha_Thai",   # Thai
    "id": "ind_Latn",   # Indonesian
    "nl": "nld_Latn",   # Dutch
    "pl": "pol_Latn",   # Polish
    "tr": "tur_Latn",   # Turkish
    "uk": "ukr_Cyrl",   # Ukrainian
    "hi": "hin_Deva",   # Hindi
    "sv": "swe_Latn",   # Swedish
}

SUPPORTED_LANGS = frozenset(_ISO_TO_NLLB)

# ── Lazy singletons ───────────────────────────────────────────────────────────

_nllb_tokenizer  = None
_nllb_model      = None
_nllb_device     = None
_lingua_detector = None


# ── Language detection ────────────────────────────────────────────────────────

def _get_lingua_detector():
    global _lingua_detector
    if _lingua_detector is None:
        from lingua import LanguageDetectorBuilder
        _lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
    return _lingua_detector


def detect_language(text: str) -> str:
    """Return ISO 639-1 code for `text` (e.g. 'vi', 'fr'). Defaults to 'en'."""
    if not text or not text.strip():
        return "en"
    try:
        lang = _get_lingua_detector().detect_language_of(text)
        return lang.iso_code_639_1.name.lower() if lang else "en"
    except Exception as e:
        log.debug(f"detect_language failed (defaulting to 'en'): {e}")
        return "en"


# ── NLLB model loading ────────────────────────────────────────────────────────

def _load_nllb():
    global _nllb_tokenizer, _nllb_model, _nllb_device
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _nllb_device = device
    log.info(f"[NLLB] Loading {NLLB_MODEL_ID} on {device}…")

    _nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_ID,
        torch_dtype=__import__("torch").float16 if device == "cuda" else __import__("torch").float32,
    ).to(device)
    _nllb_model.eval()
    log.info(f"[NLLB] {NLLB_MODEL_ID} ready ✓  (cache size: {_cached_translate.cache_info().maxsize})")


# ── Cached translation core ───────────────────────────────────────────────────

def _has_untranslated_words(source: str, translation: str) -> bool:
    """
    Return True if significant words from `source` survived into `translation`
    unchanged — a reliable signal that greedy decoding left tokens untranslated.

    Example:
        source      = "tiểu thuyết trinh thám"
        translation = "The novel trinh thám"   ← "trinh", "thám" leaked through
        → True  (retry with beam search)

        source      = "lịch sử thế giới"
        translation = "The history of the world"
        → False (clean translation, keep it)
    """
    src_words   = {w.lower() for w in source.split() if len(w) > 3}
    trans_words = {w.lower().rstrip(".,!?;:") for w in translation.split()}
    return bool(src_words & trans_words)


@lru_cache(maxsize=2048)
def _cached_translate(text: str, nllb_src_lang: str) -> str:
    """
    Translate `text` (in `nllb_src_lang`) to English.

    Strategy:
      1. Fast greedy pass (num_beams=1, ~30ms)
      2. Quality gate: if source words survived into the output, the greedy
         decoder skipped them — retry beam=4 (~60ms) for that query only.
      3. Cache whichever result we keep — repeat queries cost 0ms.
    """
    import torch

    _nllb_tokenizer.src_lang = nllb_src_lang
    inputs = _nllb_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(_nllb_device)
    forced_bos = _nllb_tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    def _generate(num_beams: int) -> str:
        with torch.no_grad():
            ids = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=num_beams,
                do_sample=False,
                max_new_tokens=64,
            )
        return _nllb_tokenizer.decode(ids[0], skip_special_tokens=True)

    result = _generate(num_beams=1)

    if _has_untranslated_words(text, result):
        log.debug(f"[NLLB] Partial translation detected (greedy='{result}') — retrying beam=4")
        result = _generate(num_beams=4)

    log.info(f"[NLLB] {nllb_src_lang}→EN: '{text}' → '{result}'")
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def translate_to_en(text: str) -> str:
    """
    Translate any supported non-English query to English.

    Returns `text` unchanged when:
      - text is empty / whitespace
      - detected language is English
      - detected language is not in _ISO_TO_NLLB (unsupported)
      - NLLB fails to load or throws during inference
    """
    if not text or not text.strip():
        return text

    lang = detect_language(text)
    if lang == "en":
        log.debug("[NLLB] English input — skipping translation.")
        return text

    nllb_src = _ISO_TO_NLLB.get(lang)
    if nllb_src is None:
        log.debug(f"[NLLB] Unsupported lang '{lang}' — passing through untranslated.")
        return text

    if _nllb_model is None:
        try:
            _load_nllb()
        except Exception as e:
            log.warning(f"[NLLB] Model load failed — using original query: {e}")
            return text

    try:
        return _cached_translate(text, nllb_src)
    except Exception as e:
        log.warning(f"[NLLB] Translation failed — using original: {e}")
        return text


def warmup():
    """
    Eagerly load NLLB and run two translation passes to compile CUDA kernels.
    Call once at server startup to eliminate the cold-start P95 spike.
    """
    log.info("[NLLB] Warming up translation model…")
    try:
        translate_to_en("sách hay")        # Vietnamese path
        translate_to_en("livre de magie")  # French path — second src_lang
        info = _cached_translate.cache_info()
        log.info(f"[NLLB] Warmup done ✓  cache={info.currsize}/{info.maxsize}")
    except Exception as e:
        log.warning(f"[NLLB] Warmup failed (non-fatal): {e}")


# Backward-compat alias — old callers continue to work without changes
translate_vi_to_en = translate_to_en
