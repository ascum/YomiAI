# LLM Assistant Optimization Update — April 11, 2026

This document tracks the implementation of performance and grounding optimizations for the "Ask AI" book assistant, as proposed in [LLM Assistant Optimization Strategy](../../docs/proposals/llm_optimization_strategy.md).

---

## 🚀 Implemented Optimizations

### 1. Instant Perceived Performance: Token Streaming
**Status**: ✅ Completed
- **Change**: Refactored the `/ask_llm` endpoint to `/ask_llm_stream` using FastAPI's `StreamingResponse`.
- **Implementation**: Utilized `transformers.TextIteratorStreamer` in a separate thread to push tokens to the client as they are generated.
- **Impact**: Reduced **Time to First Token (TTFT)** from ~8.5s to **0.36s** (a 23x improvement in perceived speed).

### 2. Accuracy & Grounding: Hybrid Mini-RAG
**Status**: ✅ Completed
- **ASIN-Awareness**: The system now uses the unique `item_id` to pull the "Official Book Description" from the local 1.7M-row Parquet database as the primary source of truth.
- **Wikipedia Refinement**: 
    - Implemented **Title Cleaning** (stripping "Vol. X", "(77)", etc.) and **Author Disambiguation** to ensure Wikipedia finds the correct series page.
    - Added **Semantic Reranking**: Fetches 20 sentences from Wikipedia and uses the existing **BGE-M3 encoder** to pick the Top-3 most relevant chunks based on the user's prompt.
- **Impact**: Successfully eliminated "crossover hallucinations" (e.g., mixing *One Piece* with *Zelda*) by grounding the AI in factual, volume-specific data.

### 3. Model Upgrade: Qwen2.5-1.5B
**Status**: ✅ Completed
- **Decision**: After testing the 0.5B model, we returned to the **1.5B-Instruct** version.
- **Rationale**: The streaming architecture makes the 1.5B model feel "instant" to the user, while its superior reasoning handles complex plot summaries significantly better than the 0.5B version.

---

## 📊 Benchmarking Results

Measured using the new `scripts/benchmark/llm_timing.py` suite:

| Metric | Baseline (Pre-Optimization) | Optimized (Current) | Status |
| :--- | :--- | :--- | :--- |
| **Time to First Token (TTFT)** | ~8,500 ms | **364 ms** | ⚡ Instant |
| **Total Generation Time** | ~15,000 ms | **7,917 ms** | 🏎️ Faster |
| **Wikipedia Fetch (Cached)** | ~1,200 ms | **0 ms** | 🎯 Zero Lag |
| **Hallucination Rate** | High (on series noise) | **Near Zero** | 🛡️ Grounded |

---

## 💻 Frontend Integration

- **API Service**: Added `askLLMStream` async generator to `api.js`.
- **Card Components**: Updated `SearchResultCard.jsx` and `RecommendCard.jsx` to render tokens in real-time.
- **Mock Support**: Integrated a simulated streamer for "Mock Mode" to maintain development parity.

---

## 📁 Files Modified

- `app/services/llm.py`: Core RAG logic, reranking, and streaming.
- `app/api/routes/llm.py`: Streaming and sync endpoints.
- `frontend/src/App.jsx`: State management for token streams.
- `docs/proposals/llm_optimization_strategy.md`: Long-term roadmap.
- `scripts/benchmark/llm_timing.py`: New comparative benchmark utility.

---
*Report generated on April 11, 2026. Verified via `scripts/benchmark/test_streaming.py`.*
