# Vietnamese Query Handling — Technical Proposal
**Date**: April 2, 2026  
**Topic**: Fixing Vietnamese search degradation in the NBA Multimodal Recommendation System

---

## 1. The Core Problem

The search pipeline uses **BLaIR** (`hyp1231/blair-roberta-large`) as the text encoder. BLaIR
is a fine-tuned **RoBERTa-large** backbone trained on English product review data. Its tokenizer
and embedding space are almost entirely English-centric.

The audit from March 31 proved this conclusively:

| Query (VI raw) | VI genre hit rate | EN genre hit rate |
|---|---|---|
| `tiểu thuyết trinh thám` | 0/10 | 3/10 |
| `sách phát triển bản thân` | 0/10 | 0/10 |
| `fantasy phép thuật` | 0/10 | 2/10 |
| `lịch sử thế chiến` | 0/10 | 1/10 |
| `sách thiếu nhi` | 0/10 | 0/10 |

**Verdict**: 5/5 queries failed. Raw Vietnamese tokens are essentially noise in BLaIR's embedding
space — the model returned random Spanish titles with artificially high similarity scores (~0.94).

**Teacher's concern is valid**: RoBERTa-large was never designed for multilingual input.
Even if you fine-tuned BLaIR yourself on Vietnamese data, the base tokenizer's vocabulary
(50k English/Latin WordPiece tokens) would fragment Vietnamese syllables into meaningless
subword units, producing embedded garbage regardless of fine-tuning.

---

## 2. Why Simple Translation is the Right Approach

The fundamental fix is **not** to find a Vietnamese-capable embedding model. Here is why:

1. **The FAISS index is already built** on ~3M BLaIR-1024 English embeddings. Swapping the
   encoder (e.g., to `intfloat/multilingual-e5-large`) would require a complete re-embed pass
   (~10–20 GPU hours) and a full FAISS index rebuild. That is out of scope.

2. **BM25 still works natively in Vietnamese** — the keyword channel (substring matching on
   Vietnamese title/author text in the parquet) is unaffected. You only need to fix the
   *dense* retrieval channel (BLaIR).

3. **Translation is battle-tested**: The Google Translate / Helsinki-NLP / NLLB family of
   models produce high-quality VI→EN translation on the exact domain (book queries) and run
   in milliseconds on GPU.

4. **It's architecturally clean**: The translation sits as a single preprocessing step before
   BLaIR encoding. Nothing else in the pipeline changes.

---

## 3. Proposed Architecture

```
User types Vietnamese query
        │
        ▼
┌─────────────────────────┐
│    Language Detection   │  (langdetect · ~1ms)
│    detect_language(q)   │
└─────────┬───────────────┘
          │ is_vi = True
          ▼
┌─────────────────────────┐
│   VI→EN Translation     │  (see Model Options below)
│   translate_vi_to_en(q) │
└─────────┬───────────────┘
          │ "detective mystery novels"
          ▼
┌─────────────────────────┐     ┌───────────────────────────┐
│   BLaIR Encoding        │     │   BM25 Keyword Scan       │
│   (translated EN query) │     │   (ORIGINAL VI query)     │
│   1024-dim FAISS search │     │   title/author substring  │
└─────────┬───────────────┘     └─────────────┬─────────────┘
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
              Adaptive Weighted RRF
              BGE Reranker (EN query)
                         ▼
                   Final Results
```

Key insight: **BM25 always receives the original Vietnamese text**, because BM25 does
substring/token matching on the raw metadata strings — Vietnamese titles and author names in the
parquet are stored in Unicode, so VI-token matching still works. Only BLaIR (the semantic dense
retrieval channel) needs the translated English version.

---

## 4. Translation Model Options

Three viable candidates, ranked by recommendation:

### Option A — `facebook/nllb-200-distilled-600M` ✅ RECOMMENDED

| Property | Value |
|---|---|
| Type | NLLB (No Language Left Behind) — Meta AI |
| Languages | 200 languages including Vietnamese (`vie_Latn`) |
| Size | 600M params (~1.2 GB in fp16) |
| Latency (GPU) | ~15–40 ms per short query |
| Latency (CPU) | ~150–400 ms per short query |
| Quality | State-of-the-art for VI→EN on general text |
| License | CC-BY-NC-4.0 (thesis OK) |
| HF model | `facebook/nllb-200-distilled-600M` |

**Why NLLB over MarianMT**: The Helsinki-NLP `opus-mt-vi-en` model was originally planned,
but NLLB-600M has been shown to produce significantly more natural, context-aware translations
on informal and domain-specific short queries (book titles, genre descriptions). MarianMT's
VI→EN pair is a smaller, older model (77M params) trained on parallel corpus data that skews
toward formal text.

**Trade-off**: NLLB is ~8× heavier than MarianMT. With lazy loading (load only on first VI
query), this adds ~0.5–1s to the *first* VI query per server session, then is essentially free
(model stays in VRAM, ~15–40ms per query thereafter).

### Option B — `Helsinki-NLP/opus-mt-vi-en` (MarianMT)

| Property | Value |
|---|---|
| Type | MarianMT seq2seq |
| Size | ~77M params (~150 MB) |
| Latency (GPU) | ~10–20 ms |
| Quality | Good for formal text, weaker on short informal queries |
| License | Apache 2.0 |

This is the originally planned model. Acceptable if VRAM is very constrained (e.g., < 2 GB
free), but NLLB is better quality for the use case.

### Option C — `google/madlad-400-3b-mt` (Overkill, for reference)

A 3B parameter model. Translation quality is exceptional but adds ~3 GB VRAM and
~100–200ms latency. Not recommended for this thesis within the current resource constraints.

---

## 5. Recommended Implementation Plan

### 5.1 Create `src/utils.py`

The file was designed in Session 1 but **`src/utils.py` does not yet exist** in the repository.
It needs to be created. It should contain:

```python
# src/utils.py
"""
Language detection and translation utilities for the NBA search pipeline.
"""
import logging
import sys

# Fix Windows terminal crash on Vietnamese characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = logging.getLogger("nba_api")

# ── Lazy model holders ────────────────────────────────────────────────────────
_langdetect_ready = False
_nllb_tokenizer   = None
_nllb_model       = None

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_SRC_LANG = "vie_Latn"
NLLB_TGT_LANG = "eng_Latn"


def detect_language(text: str) -> str:
    """
    Returns ISO 639-1 language code ('vi', 'en', etc.) or 'en' on failure.
    Uses langdetect under the hood (pip install langdetect).
    """
    global _langdetect_ready
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect
        _langdetect_ready = True
        return detect(text)
    except Exception:
        return "en"  # safe fallback


def translate_vi_to_en(text: str) -> str:
    """
    Translates Vietnamese text to English using NLLB-200-distilled-600M.
    - Lazy-loads the model on first call (warm-up ~1s on GPU).
    - Returns original text if language is not detected as Vietnamese,
      or if translation fails for any reason (safe fallback contract).
    """
    global _nllb_tokenizer, _nllb_model

    if not text or not text.strip():
        return text

    # Only translate if actually Vietnamese
    detected = detect_language(text)
    if detected != "vi":
        log.debug(f"detect_language={detected}, skipping translation.")
        return text

    # Lazy-load NLLB on first Vietnamese query
    if _nllb_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            log.info(f"Loading NLLB translation model: {NLLB_MODEL_ID}")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            _nllb_tokenizer = AutoTokenizer.from_pretrained(
                NLLB_MODEL_ID, src_lang=NLLB_SRC_LANG
            )
            _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            _nllb_model.eval()
            log.info(f"NLLB translation model ready ✓  (device={device})")

        except Exception as e:
            log.warning(f"NLLB load failed — falling back to original query: {e}")
            return text

    # Translate
    try:
        import torch
        device = next(_nllb_model.parameters()).device

        inputs = _nllb_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        forced_bos_token_id = _nllb_tokenizer.lang_code_to_id[NLLB_TGT_LANG]

        with torch.no_grad():
            output_ids = _nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                num_beams=4,
                max_new_tokens=128,
            )

        translated = _nllb_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        log.info(f"VI→EN: '{text}' → '{translated}'")
        return translated

    except Exception as e:
        log.warning(f"NLLB translation failed — falling back to original: {e}")
        return text
```

### 5.2 Wire Into `api.py` (Already Partially Done)

The hook in `api.py` (lines 337–343) already calls `translate_vi_to_en`. Once `src/utils.py`
exists, this will work automatically. The only fix is the import path — it currently uses
`from src.utils import ...` which may fail depending on `sys.path`. Should be:

```python
# api.py — search endpoint, pre-encoding block
query_for_encoding = req.query
if req.query:
    try:
        from utils import translate_vi_to_en   # src/ is already on sys.path
        translated = translate_vi_to_en(req.query)
        if translated != req.query:
            log.info(f"Query translated: '{req.query}' → '{translated}'")
        query_for_encoding = translated
    except Exception as e:
        log.warning(f"Translation hook failed: {e}")
        # always fall back to original query — never break search
```

### 5.3 Keep BM25 on Original VI String

In `active_search_engine.py`, the `_bm25_search(text_query)` call uses `text_query`
(the **raw original** string). This is correct — do NOT change it. Vietnamese book titles,
translated author names, etc., in the parquet metadata will still be searched
as-is by the BM25 tokenizer.

### 5.4 Keep Reranker on Original VI String (with Translated Context)

The BGE Reranker cross-encoder receives `text_query=req.query` (the raw VI string).
This is acceptable for now since the reranker pairs the query against the retrieved
English book titles — the cross-encoder (`BAAI/bge-reranker-v2-m3`) is multilingual
and handles VI↔EN cross-lingual pairs reasonably well.

**Future improvement**: Pass `query_for_encoding` (the translated EN string) to the
reranker instead. This would likely improve reranker scores significantly. This can be
done as a follow-up without any structural changes.

---

## 6. Dependencies

```bash
pip install langdetect              # language detection (~fast, no model download)
pip install sentencepiece           # required by NLLB tokenizer
# transformers is already installed (CLIP/BLaIR use it)
# The NLLB model will be downloaded automatically on first VI query
# HF cache size: ~1.2 GB for nllb-200-distilled-600M (fp16)
```

---

## 7. Expected Outcomes After Fix

Based on the March 31 audit results, after the translation hook is live:

| Metric | Before (raw VI) | Expected After (VI→EN→BLaIR) |
|---|---|---|
| Genre hit rate (avg) | 0/10 | ~2–4/10 (approaching EN baseline) |
| Top-5 relevance | Spanish books, random noise | Thematically relevant EN books |
| Translation latency | — | 15–40 ms on GPU (NLLB warm) |
| Total search latency | ~2,800 ms | ~2,850 ms (+~40ms) |

The EN baseline itself is 0–3/10 on this test set, which reflects BLaIR's general weakness
on abstract genre concept queries (not a VI-specific problem). Post-translation VI should
match EN performance.

---

## 8. Verification Plan

After implementing, run the existing audit script:

```bash
python scripts/audit_vietnamese_search.py
```

Compare:
- `VI (raw)` results → should still be in the output for reference
- `VI (translated)` results → new column showing post-translation equivalent  
- Genre hit rates should increase from 0/10 toward the EN baseline

Also run the benchmark to confirm translation adds < 50ms:
```bash
python scripts/benchmark_search.py
```

Check the `Vietnamese query (VI→EN translation path)` row. Target: FAISS+RRF+Rerank < 600ms,
total < 1000ms (note: the ~2800ms FAISS number from March 31 is the BGE Reranker bottleneck,
which is a separate problem to be fixed in the next session).

---

## 9. Open Questions / Decisions Needed

1. **NLLB vs MarianMT**: Do you prefer NLLB-distilled-600M (better quality, 1.2 GB VRAM) or
   `opus-mt-vi-en` (smaller, 150 MB, slightly worse quality)? Given that VRAM on your system
   already has BLaIR + CLIP + BGE Reranker loaded, VRAM budget is a real concern.
   - BLaIR: ~1.4 GB VRAM (fp32)
   - CLIP ViT-B/32: ~0.6 GB VRAM
   - BGE Reranker v2-m3: ~0.5 GB VRAM
   - NLLB-600M fp16: ~1.2 GB VRAM
   - **Total with NLLB**: ~3.7 GB — check `nvidia-smi` before deciding.

2. **Should we pass the translated query to the BGE Reranker** (instead of the raw VI string)?
   This would be a clean improvement and should be done at the same time.

3. **Should audit_vietnamese_search.py be updated** to show a three-column comparison
   (VI raw vs VI translated vs EN equiv) instead of two columns?

---

*Last updated: April 2, 2026*
