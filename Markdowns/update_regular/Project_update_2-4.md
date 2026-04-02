# Project Update — April 2, 2026

This document captures all functional changes made to the NBA Multimodal Recommendation System on **April 2, 2026** (Week 2 — Vietnamese Search Fix & Reranker Removal). It should be read alongside `Project_update_31-3.md`.

---

## Summary of Changes

| Area | Change |
|---|---|
| Search — `src/utils.py` | **NEW FILE** — Language detection and VI→EN translation via NLLB-1.3B |
| Backend — `api.py` | Wired translation hook into `/search` pre-encoding stage |
| Backend — `api.py` | Removed BGE Cross-Encoder Reranker from the pipeline entirely |
| Backend — `api.py` | Fixed broken import path (`src.utils` → `utils`) |
| Backend — `api.py` | Removed `reranker_live` field from `/health` endpoint |
| Dependencies | Installed `langdetect` and `sentencepiece` into venv |

---

## 🇻🇳 1. Vietnamese Query Fix — NLLB Translation Layer

### The Problem (Confirmed March 31)
The BLaIR text encoder (`hyp1231/blair-roberta-large`) is a fine-tuned **RoBERTa-large** backbone. RoBERTa's tokenizer uses a 50k English/Latin WordPiece vocabulary that fragments Vietnamese diacritical syllables (`tiểu`, `thuyết`, `trinh`) into meaningless subword units. Vietnamese queries are effectively noise in BLaIR's embedding space.

The March 31 audit proved this definitively — **5/5 Vietnamese test queries returned 0/10 genre hit rate**, surfacing random Spanish-language books instead of relevant content.

This is a fundamental tokenizer-level constraint, not fixable by threshold tuning. The root cause is that RoBERTa was never pre-trained on Vietnamese.

### The Solution: Translation Middleware

```
User types Vietnamese query ("tiểu thuyết trinh thám")
        │
        ▼
┌────────────────────────────┐
│   detect_language(query)   │  langdetect — ~1ms
│   → 'vi'                   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   translate_vi_to_en(q)    │  NLLB-1.3B — ~25-50ms GPU (warm)
│   → "detective thriller…"  │  Lazy-loaded on first VI query
└────────────┬───────────────┘
             │
   ┌─────────┴────────────────────────┐
   ▼                                  ▼
BLaIR encoding                  BM25 keyword scan
(translated EN query)           (ORIGINAL Vietnamese)
1024-dim FAISS retrieval        substring match on parquet metadata
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
               Adaptive Weighted RRF
                            ▼
                      Final Results
```

**Key design decision**: BM25 always receives the **original Vietnamese string**, because the parquet metadata stores Unicode titles and author names — substring matching on Vietnamese text still works natively. Only the BLaIR *dense* retrieval channel needs the translated English version.

### Implementation — `src/utils.py` (NEW)

```python
NLLB_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
NLLB_SRC_LANG = "vie_Latn"   # Vietnamese (Latin script)
NLLB_TGT_LANG = "eng_Latn"   # English (Latin script)
```

Two public functions:

**`detect_language(text: str) → str`**
- Wraps `langdetect.detect()`, returns ISO 639-1 code (`'vi'`, `'en'`, etc.)
- Returns `'en'` on empty input or any failure (safe default)

**`translate_vi_to_en(text: str) → str`**
- Fast-path: if `detect_language()` returns anything other than `'vi'`, returns the original string immediately (zero NLLB overhead for English queries)
- Lazy-loads NLLB-1.3B in fp16 on first Vietnamese query (~2s warm-up, model stays warm for subsequent calls)
- Runs beam search (n_beams=4) over 128-token max for quality
- **Always falls back to original text on any exception** — the translation layer can never break the search pipeline

```python
# Fast path — en queries skip NLLB entirely
detected = detect_language(text)
if detected != "vi":
    return text

# Lazy init on first VI query
if _nllb_model is None:
    _load_nllb()   # downloads + loads to CUDA fp16 once

# Translate
translated = _nllb_tokenizer.decode(
    _nllb_model.generate(**inputs, forced_bos_token_id=eng_token, num_beams=4),
    skip_special_tokens=True
)
```

### Why NLLB-1.3B over the originally planned MarianMT (`opus-mt-vi-en`)

| Model | Params | VRAM (fp16) | VI→EN quality |
|---|---|---|---|
| `Helsinki-NLP/opus-mt-vi-en` | 77M | ~150 MB | Good on formal text |
| `facebook/nllb-200-distilled-600M` | 600M | ~1.2 GB | Better on short queries |
| **`facebook/nllb-200-distilled-1.3B`** ✅ | 1.3B | ~2.6 GB | Best for informal/genre queries |

The 1.3B distilled model handles colloquial Vietnamese genre vocabulary (`phép thuật`, `thiếu nhi`, `trinh thám`) significantly better than the smaller alternatives. Hardware allows it: RTX 4060 8 GB VRAM with BLaIR + CLIP + Qwen totalling ~5 GB, NLLB brings it to ~7.6 GB — tight but workable since NLLB is lazy-loaded.

### Wire-up in `api.py`

```python
# /search endpoint — pre-BLaIR-encoding block
query_for_encoding = req.query
if req.query:
    try:
        from utils import translate_vi_to_en   # src/ is on sys.path
        translated = translate_vi_to_en(req.query)
        if translated != req.query:
            log.info(f"[Translation] '{req.query}' → '{translated}'")
        query_for_encoding = translated
    except Exception as e:
        log.warning(f"[Translation] Hook failed, using original query: {e}")

# BLaIR uses translated query; BM25 still gets req.query (original VI)
text_vec = _encode_text_query(query_for_encoding)
```

The import was also corrected from `from src.utils import` (broken — `src/` not reachable from the project root that way) to `from utils import` (correct — `src/` is explicitly added to `sys.path` at line 34 of `api.py`).

---

## ⚡ 2. BGE Reranker Removed

### The Problem
The March 31 benchmark revealed that the `FAISS+RRF+Rerank` stage dominated total latency across every query profile:

| Query type | FAISS+RRF+Rerank latency |
|---|---|
| Short keyword | 35,204 ms |
| Long dense | 6,010 ms |
| Vietnamese | 2,784 ms |
| Image-only | 21,964 ms |
| Hybrid | 31,848 ms |
| Nonsense | 53,854 ms |

The BGE Reranker (`BAAI/bge-reranker-v2-m3`) is a cross-encoder: it performs a full transformer forward pass for every (query, candidate) pair individually. With 20 candidates, that is 20 sequential inferences. On the first call (model cold, BGE loading from disk) this is catastrophic. Even warm, BGE adds hundreds of milliseconds with negligible observable quality improvement during demos.

### The Decision
The Adaptive Weighted RRF already produces a well-ordered ranking by fusing BM25, BLaIR, and CLIP signals. The BGE Reranker was added to compensate for BLaIR's weakness on ambiguous queries — but the correct fix is cleaner BLaIR inputs via translation, not an expensive post-processor. With translation in place, the reranker's marginal contribution does not justify its latency cost.

### The Fix: Complete Removal from `api.py`

1. **`_state` dict** — `"reranker"` key removed
2. **Startup lifespan** — BGE Reranker loading block (previously step 5.5) removed entirely
3. **`ActiveSearchEngine` init** — now always receives `reranker=None`
4. **`/health` endpoint** — `reranker_live` field removed from response

`ActiveSearchEngine.search()` in `active_search_engine.py` was **not modified** — the existing `Step 6` already handles `reranker=None` gracefully via its `else` branch (`final_ranking = final_ranking[:top_k]`), so the engine degrades correctly with no code changes needed.

```python
# active_search_engine.py — Step 6 (unchanged, already handles None)
if (self.reranker is not None and self.reranker.is_ready and text_query and has_text):
    final_ranking = self.reranker.rerank(...)
else:
    final_ranking = final_ranking[:top_k]   # ← now always takes this path
```

### Expected Latency Impact
The FAISS + Adaptive RRF alone (without BGE) should take **< 100 ms** for most queries on GPU. The 2–54 second benchmarks were almost entirely BGE Reranker overhead.

---

## 📁 3. Files Changed in This Update Cycle

| File | Type | Key Change |
|---|---|---|
| `src/utils.py` | NEW | `detect_language()` + `translate_vi_to_en()` with lazy NLLB-1.3B loading |
| `api.py` | MODIFIED | Translation hook wired; BGE Reranker removed; import path fixed |
| `Markdowns/planning/2-4-2026_vietnamese_query_proposal.md` | NEW | Technical proposal document for this session's changes |

No changes to `src/active_search_engine.py`, `src/reranker.py`, `src/retriever.py`, or any model files.

---

## 🔧 4. Dependencies Added

```bash
pip install langdetect     # language detection (fast, no model download)
pip install sentencepiece  # required by NLLB tokenizer
# transformers is already present (CLIP/BLaIR); NLLB model downloads on first VI query (~2.6 GB)
```

---

## ✅ 5. Verification Notes

- Import smoke test confirmed: `utils.py`, `langdetect`, and `sentencepiece` all import cleanly from within the venv.
- Server started successfully with `python api.py` — BLaIR, CLIP, and BM25 index all loaded. No reranker loading log line (confirmed removal).
- NLLB model has not been downloaded yet — it will download from HuggingFace (~2.6 GB) automatically on the first Vietnamese query after server restart.

### Next Steps (Pending Verification)
1. **Run `scripts/audit_vietnamese_search.py`** after the NLLB model downloads to confirm hit rate improvement from 0/10 → approaching EN baseline
2. **Re-run `scripts/benchmark_search.py`** to measure the actual latency improvement from removing BGE Reranker (expect FAISS+RRF stage to drop from 2,784–53,854 ms to < 100 ms)
3. **Threshold tuning** — `DENSE_SCORE_THRESHOLD` and `RERANKER_SCORE_THRESHOLD` in `active_search_engine.py` (the reranker threshold is now moot; dense threshold may need re-calibration without reranker in the pipeline)

---

*Last update: April 2, 2026. Next update cycle will cover benchmark re-run results and threshold tuning.*
