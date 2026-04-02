# Project Update — March 31, 2026

This document captures all functional changes and audits made to the NBA Multimodal Recommendation System on **March 31, 2026** (Week 1 Performance & Accuracy Fixes). It should be read alongside `Project_update_25-3.md`.

---

## Summary of Changes

| Area | Change |
|---|---|
| Search — `api.py` | Added end-to-end pipeline latency timing instrumentation (`?debug=true`) |
| Search — `active_search_engine.py` | Added configurable `DENSE_SCORE_THRESHOLD` and `RERANKER_SCORE_THRESHOLD` to filter weak candidates |
| Backend — `api.py` | Added graceful empty-results handling when threshold removes all candidates |
| Evaluation — `audit_vietnamese_search.py` | Built audit script comparing VI vs EN queries, successfully identified BLaIR degradation |
| Evaluation — `benchmark_search.py` | Built benchmark suite to test response times across 5 representative query profiles |
| Evaluation — `audit_clip_quality.py` | Created script for visual quality audit and evaluation of the current `clip-vit-base-patch32` model |

---

## ⏱️ 1. Performance Profiling and Instrumentation

### The Problem
The system needed a baseline for end-to-end response time to guarantee search execution operates under 1 second. Missing visibility into stage-by-stage latency made it impossible to confidently identify code bottlenecks.

### The Fix
Added `time.perf_counter()` timings in `api.py` for each search stage (conditionally exposed via the `?debug=true` query parameter):
- BLaIR text encoding
- CLIP image encoding
- FAISS search + adaptive RRF + BGE Reranker (Combined)
- Metadata hydration

Created `scripts/benchmark_search.py` to automatically execute tests against a live server for Short, Long, Image-only, Hybrid, and Nonsense queries, reporting the duration against preset millisecond budget constraints.

---

## 🛡️ 2. Similarity Thresholds

### The Problem
The previous active search implementation would indiscriminately return 20 items (post-Reranker) regardless of relevance. Completely nonsensical queries still yielded full product matrices that wasted space and confused users.

### The Fix
Configurable thresholds introduced in `src/active_search_engine.py`:
- `DENSE_SCORE_THRESHOLD = 0.004` (Applied pre-metadata filter)
- `RERANKER_SCORE_THRESHOLD = 0.01` (Applied post-cross-encoding)

Added dynamic threshold adjustment: automatically halves (`× 0.5`) dense threshold boundaries if the Vietnamese language is detected as a fallback.
Updated `api.py` to intercept empty arrays and return a user-friendly response payload instead of a blank list.

---

## 🇻🇳 3. Vietnamese Search Audit & Fixes

### The Problem
Vietnamese queries were yielding significantly poorer retrieval quality compared to English equivalents because the BLaIR model space is heavily English-biased.

### The Fix & Discoveries
- **Ground Truth**: Created a static test suite in `evaluation/vi_test_queries.py` containing 5 VI vs EN equivalent query pairs and their expected genres.
- **Diagnostic Tooling**: Developed `scripts/audit_vietnamese_search.py` to measure top-10 genre hit rates and calculate mean score deltas between the languages.
- **Crash Fix**: The Windows terminal threw a fatal `UnicodeEncodeError` when printing Vietnamese characters to standard output, abruptly crashing the `multiprocess` layer. Forced `sys.stdout.reconfigure(encoding='utf-8')` to prevent the crash and display results cleanly.
- **Audit Findings**: The results definitively proved severe degradation. Raw Vietnamese queries yielded a **0/10 genre hit rate** and failed minimum quality thresholds across all 5 tests.
- **Go-forward Action**: The test justified a mandatory translation layer hook for Vietnamese queries (planning translation via `Helsinki-NLP/opus-mt-vi-en` before executing dense encoding).

---

## 🖼️ 4. CLIP Version Verification

### The Process
Logged startup configurations in `api.py` (model name, dim size, and assigned device) for continuous system audit trailing. Created `scripts/audit_clip_quality.py` to test and evaluate the current baseline image encoder (`openai/clip-vit-base-patch32`, 512-dim). 

Drafted a formal upgrade decision document (`profiling/clip_upgrade_decision.md`) comparing the current baseline against ViT-L and ViT-H benchmark improvements. This establishes a framework to decide whether the massive resource cost of re-embedding 3M book covers and retraining the DQN is justifiable under the thesis timeline.

---

*Last update: March 31, 2026. Next update cycle will likely cover the integration of the translation hook and comprehensive evaluation model ablations.*
