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

## 🔍 Remaining Bottlenecks (Updated)

Based on the latest benchmark (`benchmark_stage_2_hnsw_20260406_121421.json`):

1.  **BM25 Keyword Search (1,471 ms)**: Now accounts for **~92%** of internal processing time.
2.  **BLaIR Encoding (67 ms)**: Stable and within target limits.
3.  **E2E Overhead (~2,029 ms)**: Disconnect between API and Client remains constant.

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
