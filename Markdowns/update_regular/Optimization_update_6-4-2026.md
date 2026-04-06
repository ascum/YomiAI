# Search Optimization Update — April 6, 2026

This document tracks the progress of Phase 1.1 (Performance Benchmarking & Optimization) as outlined in the [31-3 Planning Roadmap](../../Markdowns/planning/31-3%20planning.md).

---

## 🚀 Stage 1: Threading & Async Response Optimization

### Status: ✅ Completed

**Goal**: Resolve the "Ghost Latency" (2-second gap between internal API processing and end-to-end wall clock time) by offloading synchronous GPU and FAISS operations to a worker thread pool.

### Technical Implementation
The `/search` endpoint in `api.py` was refactored to use `anyio.to_thread.run_sync`. This prevents the heavy BLaIR/CLIP encoding and FAISS/BM25 search passes from blocking the FastAPI event loop, ensuring the server remains responsive and handles I/O (like sending the final response) without queuing delays.

### Benchmarking Results (Baseline vs. Stage 1)

| Metric | Baseline (Flat Index) | Stage 1 (Threading) | Delta |
| :--- | :--- | :--- | :--- |
| **Internal Pipeline (`total_ms`)** | 2,210 ms | 2,279 ms | +69 ms |
| **Full E2E Cycle (`e2e_wall_clock_ms`)** | 4,240 ms | 4,308 ms | +68 ms |
| **The "Ghost Gap"** | ~2,030 ms | ~2,029 ms | **No Change** |

**Observation**: While the server is now async-safe and handles concurrency better, the 2-second gap between internal and external timing persists. This indicates the overhead is likely due to JSON serialization of large metadata payloads or network stack overhead on localhost, rather than event loop blocking.

---

## 🔍 Current Bottlenecks (Top 3)

Based on the latest benchmark (`benchmark_stage_1_threading_20260406_111504.json`):

1.  **BM25 Keyword Search (1,519 ms)**: The pure-Python keyword scanner is the primary bottleneck, consuming ~66% of the internal processing time.
2.  **FAISS Text Search (623 ms)**: The BLaIR semantic search is using an exact `IndexFlatIP`, causing high latency on the 1.7M document catalog.
3.  **BLaIR Encoding (74 ms)**: Text encoding is efficient but contributes to the total.

---

## 🎯 Next Steps: 48-Hour Sprint

| Stage | Optimization | Target | Est. Gain |
| :--- | :--- | :--- | :--- |
| **Stage 2** | **HNSW Index Rebuild** | < 50ms | ~570ms |
| **Stage 3** | **Tantivy/Rust BM25** | < 100ms | ~1,400ms |
| **Stage 4** | **FP16 Model Casting** | < 50ms | ~25ms |

**Target End-of-Week Latency**: **< 800ms E2E** (Meeting the advisor's 1s requirement).

---
*Report generated on April 6, 2026. Benchmarks saved in `profiling/benchmark_stage_1_threading_*.json`.*
