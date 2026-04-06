# Search Performance Optimization Strategy

Based on the benchmark results from April 2, 2026, the current Mode 1 pipeline averages **2,210ms** internally, with a total wall-clock time of **~4.2s**. To meet the advisor's requirement of **< 1,000ms**, the following technical interventions are required.

## 📊 Performance Breakdown (Current)

| Component | Current (ms) | Target (ms) | Impact |
| :--- | :--- | :--- | :--- |
| **BM25 Keyword Search** | **1,501ms** | < 100ms | 🔴 Critical |
| **BLaIR FAISS Search** | **613ms** | < 100ms | 🔴 Critical |
| **BLaIR Encoding (GPU)** | 93ms | < 80ms | 🟢 Low |
| **Metadata Hydration** | 2ms | < 5ms | 🟢 Optimized |
| **E2E Overhead / Queue** | **~2,000ms** | < 100ms | 🔴 Critical |

---

## 🚀 Solution 1: Replace Python BM25 (The "1.5s Leak")
The current `rank-bm25` implementation is pure Python. Calculating scores for 1.73M documents in a Python loop is the single largest bottleneck.

*   **Option A (Fastest): Use a Rust-based Keyword Engine.**
    Integrate `tantivy-py` or `pyserini`. These use compiled engines (Rust/Lucene) that can handle 1.7M documents in < 50ms.
*   **Option B (Minimal Change): Hybrid Sparse-Dense FAISS.**
    Instead of a separate BM25 library, use **FAISS `IndexLSH`** or a Sparse vector index (like SPLADE) to handle keywords within the same C++ environment as your semantic search.
*   **Option C (The "Lazy" Fix): BM25 Sampled Search.**
    Run BM25 only on the top 10,000 documents returned by BLaIR rather than the full 1.7M corpus.

## 🚀 Solution 2: Approximate Nearest Neighbor (ANN) Search
The `blair_search_ms` of **613ms** suggests you are using `IndexFlatIP` (Exact Search). While 100% accurate, it is $O(N)$ and scales poorly.

*   **Intervention:** Switch to **`IndexIVFFlat`** or **`HNSW`**.
    *   **IVF (Inverted File):** Clusters the 1.7M vectors into ~4,096 cells. Search only checks the closest 64 cells (`nprobe=64`).
    *   **HNSW (Hierarchical Navigable Small World):** The industry standard for speed. Offers < 10ms search time at the cost of higher RAM usage.
*   **Action:** Rebuild the `.faiss` indices using an `IVF4096,Flat` factory string.

## 🚀 Solution 3: Resolve the 2-Second "Ghost" Latency
There is a ~2,000ms gap between the internal `total_ms` and the client's `e2e_wall_clock_ms`. This usually indicates **Event Loop Blocking**.

*   **The Issue:** `blair_model.encode()` is a synchronous, CPU/GPU-intensive operation. In FastAPI, this blocks the entire worker, preventing it from handling other requests or even sending the response back to the socket promptly.
*   **Intervention:** 
    1.  **Thread Pooling:** Wrap the encoding and search calls in `run_in_threadpool`.
    2.  **Gunicorn/Uvicorn Workers:** Increase workers to `--workers 4` to allow concurrency while one worker is "frozen" by a GPU calculation.

## 🚀 Solution 4: GPU Inference Batching & Quantization
*   **Half-Precision (FP16):** Ensure BLaIR and CLIP are running in `torch.float16`. This can cut encoding time by 40-50% on modern RTX/Tesla GPUs.
*   **ONNX Runtime:** Export the BLaIR (RoBERTa) model to ONNX. This often yields a 2x-3x speedup on CPU fallbacks and 20% on GPU.

---

## 🎯 Short-Term Implementation Roadmap (Next 48 Hours)

1.  **Immediate (Task 1.1b):** Switch FAISS indices to `IVFFlat` (est. gain: **~500ms**).
2.  **Immediate (Task 1.1c):** Implement `numba` or a limited corpus for BM25 (est. gain: **~1,400ms**).
3.  **Validation:** Rerun `benchmark_search_timing.py`. If E2E is < 800ms, the thesis requirement is met.
