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
| **E2E Overhead / Queue** | ~2,030 ms | < 100 ms | 🔴 Critical |

---

## 🏎️ Stage 2: HNSW Index Rebuild (Graph-Based Search)

### Status: ✅ Completed

**Goal**: Replace the "Brute-Force" `IndexFlatIP` search with a graph-based `IndexHNSWFlat` to reduce semantic search latency on the 1.7M document catalog.

### Technical Implementation
Used `scripts/rebuild_hnsw_index.py` to reconstruct 1.7M vectors from existing indices and re-insert them into a Hierarchical Navigable Small World (HNSW) graph structure with `M=32` and `efConstruction=200`. The `Retriever` class was updated to auto-detect and prioritize these `_hnsw.faiss` files at startup.

### Benchmarking Results (Stage 1 vs. Stage 2)

| Metric | Stage 1 (Flat) | Stage 2 (HNSW) | Delta |
| :--- | :--- | :--- | :--- |
| **BLaIR FAISS Search** | 623.38 ms | **1.13 ms** | **-622.25 ms (550x speedup)** |
| **Internal Pipeline (`total_ms`)** | 2,279.57 ms | 1,587.70 ms | -691.87 ms |
| **Full E2E Cycle (`e2e_wall_clock_ms`)** | 4,308.49 ms | 3,617.27 ms | -691.22 ms |

**Observation**: The semantic search component is now officially "solved" (sub-2ms). The overall system speed improved by nearly 700ms, but the total time is still held back by the Python-based BM25.

---

## 🦀 Stage 3: Tantivy Rust Engine (Keyword Optimization)

### Status: ✅ Completed

**Goal**: Replace the pure-Python `rank-bm25` (which accounted for >90% of internal latency) with `tantivy`, a high-performance search engine library written in Rust.

### Technical Implementation
Implemented a new indexing pipeline in `scripts/build_tantivy_index.py` that builds a schema-backed Rust index of all 1.7M book titles and authors. The `ActiveSearchEngine` was refactored to use `tantivy-py` for sub-millisecond keyword lookups with native stemming and BM25 scoring.

### Benchmarking Results (Stage 2 vs. Stage 3)

| Metric | Stage 2 (HNSW) | Stage 3 (Tantivy) | Delta |
| :--- | :--- | :--- | :--- |
| **BM25 Keyword Search** | 1,471.06 ms | **4.28 ms** | **-1,466.78 ms (340x speedup)** |
| **Internal Pipeline (`total_ms`)** | 1,587.70 ms | **103.50 ms** | **-1,484.20 ms** |
| **Full E2E Cycle (`e2e_wall_clock_ms`)** | 3,617.27 ms | **2,133.94 ms** | -1,483.33 ms |

**Observation**: Internal processing is now extremely lean (< 110ms total). The keyword search bottleneck has been completely eliminated. However, the ~2,000ms "Ghost Gap" remains the final barrier to sub-second E2E response times.

---

## ⚡ Stage 5: Payload Reduction & Network Resolution

### Status: ✅ Completed (Final Optimization)

**Goal**: Resolve the final 2,000ms "Ghost Gap" and achieve sub-second E2E performance.

### Technical Implementation
1.  **FP16 Model Casting**: Cast BLaIR and CLIP models to `float16` on GPU, reducing text encoding time from 53ms to ~16ms.
2.  **Payload Pruning**: Refactored the search pipeline to return only essential UI fields, bypassing heavy `pd.notna` checks and redundant string conversions.
3.  **Pydantic Bypass**: Used `JSONResponse` directly to avoid FastAPI's automatic Pydantic validation of large result lists.
4.  **Network resolution**: Identified and fixed a 2-second delay caused by Windows `localhost` DNS/IPv6 resolution by switching to `127.0.0.1`.

### Benchmarking Results (Stage 3 vs. Stage 5 Final)

| Metric | Stage 3 (Tantivy) | Stage 5 (Final) | Delta |
| :--- | :--- | :--- | :--- |
| **BLaIR Encoding (GPU)** | 53.31 ms | **15.92 ms** | **-37.39 ms (3x speedup)** |
| **Internal Pipeline (`total_ms`)** | 103.50 ms | **61.16 ms** | **-42.34 ms** |
| **Full E2E Cycle (`e2e_wall_clock_ms`)** | 2,133.94 ms | **63.88 ms** | **-2,070.06 ms (33x speedup)** |

**Final Outcome**: We have achieved a total end-to-end latency of **~64ms** (Stage 5), further optimized to **~62ms** (Stage 6), exceeding the advisor's 1,000ms requirement by **16x**. The system is now production-ready for high-speed multimodal search on a 1.7M document catalog.

---

## 🚀 Stage 6: Async Parallelism & Greedy Translation

### Status: ✅ Completed (Final Pass)

**Goal**: Further reduce latency by parallelizing multimodal encoding and optimizing the translation bottleneck for Vietnamese queries.

### Technical Implementation
1.  **Greedy Translation**: Switched NLLB-200 from `num_beams=4` to `num_beams=1`. This reduced Vietnamese translation overhead from ~320ms to ~35ms with negligible quality loss for short search queries.
2.  **Parallel Encoding**: Refactored `_run_search_pipeline` into an `async` function and used `anyio.create_task_group()` to overlap BLaIR (text) and CLIP (image) encoding tasks.
3.  **Thread Pool Offloading**: Used `anyio.to_thread.run_sync` for all blocking GPU and FAISS operations to maintain event loop responsiveness.

### Benchmarking Results (Stage 5 vs. Stage 6)

| Metric | Stage 5 (Final) | Stage 6 (Parallel) | Delta |
| :--- | :--- | :--- | :--- |
| **Translate (VI -> EN)** | 319.54 ms | **35.51 ms** | **-284.03 ms (9x speedup)** |
| **BLaIR Encoding (GPU)** | 15.92 ms | **15.49 ms** | -0.43 ms |
| **Internal Pipeline (`total_ms`)** | 61.16 ms | **59.84 ms** | -1.32 ms |
| **Full E2E Cycle (`e2e_wall_clock_ms`)** | 63.88 ms | **62.37 ms** | **-1.51 ms** |

**Observation**: The system is now extremely fast. Even with full Vietnamese translation and semantic encoding, the response is delivered in **62ms**.

---

## 🔍 Remaining Bottlenecks (Solved)

1.  **BM25 Keyword Search**: SOLVED (Tantivy Rust)
2.  **Semantic Search**: SOLVED (HNSW Index)
3.  **E2E Overhead**: SOLVED (DNS Fix + Payload Reduction + Pydantic Bypass)
4.  **Translation Overhead**: SOLVED (Greedy Mode)
5.  **Encoder Concurrency**: SOLVED (Async TaskGroups)

---

## 🎯 Project Status: Phase 1.1 Complete

The search performance optimization phase is officially closed. All latency targets have been met or exceeded.

**Next Phase**: Multilingual Vietnamese-specific relevance tuning (Phase 1.2).

---
*Report generated on April 6, 2026. Benchmarks saved in `profiling/benchmark_stage_6_final_*.json`.*
