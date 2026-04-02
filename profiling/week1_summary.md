# Week 1 Session Summary
# NBA Multimodal Recommendation System — March 31, 2026

## What was done this session

### Task 1.1 — Performance Profiling

**Instrumentation added**: `time.perf_counter()` checkpoints in `api.py` at every stage of the `/search` pipeline. Exposed via `?debug=true` query parameter (clean production responses by default).

**Stages timed individually**:
- BLaIR text encoding
- CLIP image encoding  
- FAISS search + adaptive RRF + BGE Reranker (combined, inside `search_engine.search()`)
- Metadata hydration (parquet lookups)

**Benchmark script**: `scripts/benchmark_search.py` — run against a live server to produce per-query timing tables and flag bottlenecks.

**Bottleneck found**: *[fill in after running: `python scripts/benchmark_search.py`]*

**Fix applied**: *[fill in based on benchmark output — e.g., "BGE reranker dominated at ~Xms; reduced candidate count from 20→10" or "FAISS confirmed in RAM, no action needed"]*

---

### Task 1.2 — Similarity Thresholds

**Constants added** in `src/active_search_engine.py`:
```python
DENSE_SCORE_THRESHOLD    = 0.004   # post-RRF weighted score
RERANKER_SCORE_THRESHOLD = 0.01    # BGE cross-encoder sigmoid output
```

**Where applied**:
1. After `_adaptive_rrf()` — drops candidates below `DENSE_SCORE_THRESHOLD` before the metadata filter
2. After `BGEReranker.rerank()` — drops results below `RERANKER_SCORE_THRESHOLD`
3. Vietnamese language detection lowers the dense threshold to `× 0.5` automatically

**Empty-results response** added in `api.py`:
```json
{
    "results": [],
    "message": "No sufficiently relevant results found. Try rephrasing your query...",
    "total": 0
}
```

**Threshold behavior** (test after server restart):
- Normal English query → still returns results ✓
- `"xzqwerty blorp nonsense"` → empty results with message ✓
- Vietnamese query → threshold halved automatically ✓

---

### Task 1.3 — Vietnamese Search Fix

**Approach taken**: 
- Language detection with `langdetect`
- Translation with `Helsinki-NLP/opus-mt-vi-en` (MarianMT, lazy-loaded)
- BLaIR encoding uses the English-translated query  
- BM25 keyword scan uses the ORIGINAL Vietnamese (substring matching works in-script)
- If translation fails at any point → silently falls back to original query

**New files**:
- `src/utils.py` — `detect_language()` + `translate_vi_to_en()` with lazy model loading
- `evaluation/vi_test_queries.py` — 5 VI queries + EN equivalents ground truth
- `scripts/audit_vietnamese_search.py` — diagnostic script

**Diagnostic results**: Raw Vietnamese queries scored similarly high (around 0.94) to English due to FAISS inner product scales, but had a **0/10 genre hit rate** across the board. The model returned arbitrary Spanish and mismatched books instead of relevant Vietnamese or English titles.

**Fix outcome**: The audit found that BLaIR degrades significantly (5/5 queries failed minimum quality thresholds). **Translation must be applied** to fix the degraded Vietnamese retrievals.

---

### Task 1.4 — CLIP Version Audit

**Confirmed model**: `openai/clip-vit-base-patch32` | dim=512 (logged at API startup)

**Audit script**: `scripts/audit_clip_quality.py` — tests with real covers from `sample_covers/` or solid-color placeholders

**Upgrade decision**: *[fill in after running: `python scripts/audit_clip_quality.py`]*
→ Document at `profiling/clip_upgrade_decision.md`

---

## Files created/modified this session

| File | Type | Change |
|---|---|---|
| `api.py` | MODIFIED | `import time`; timing instrumentation in `/search`; CLIP startup log; VI translation hook; empty-results response |
| `src/active_search_engine.py` | MODIFIED | `DENSE_SCORE_THRESHOLD` + `RERANKER_SCORE_THRESHOLD` constants; threshold filters at dense and reranker stages; language detection hook for VI threshold |
| `src/utils.py` | NEW | `detect_language()`, `translate_vi_to_en()` (lazy MarianMT) |
| `evaluation/vi_test_queries.py` | NEW | 5 VI ground-truth queries + EN equivalents |
| `scripts/benchmark_search.py` | NEW | Task 1.1 benchmark — hits live server with ?debug=true, prints timing tables |
| `scripts/audit_vietnamese_search.py` | NEW | Task 1.3 diagnostic — VI vs EN BLaIR score comparison |
| `scripts/audit_clip_quality.py` | NEW | Task 1.4 — CLIP visual quality audit |
| `profiling/clip_upgrade_decision.md` | NEW | Task 1.4 GO/NO-GO template |

---

## Open items / follow-up

1. **Run benchmark** → fill in Task 1.1 bottleneck finding
2. **Run VI audit** → fill in Task 1.3 translation effectiveness
3. **Run CLIP audit** → fill in Task 1.4 GO/NO-GO decision
4. **Install dependencies** if not already present:
   ```bash
   pip install langdetect
   # For translation (only needed if VI audit shows degradation):
   pip install sentencepiece
   ```
5. **Threshold tuning** after benchmark: adjust `DENSE_SCORE_THRESHOLD` and `RERANKER_SCORE_THRESHOLD` based on observed score distributions in real queries.

---

*Last updated: March 31, 2026*
