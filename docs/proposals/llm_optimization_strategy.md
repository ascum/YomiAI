# LLM Assistant Optimization Strategy

## Current State (Post-Streaming Update)

After switching to **Qwen2.5-0.5B-Instruct** and implementing **StreamingResponse**, the "Ask AI" feature has reached a high level of perceived responsiveness:

| Metric | Pre-Optimization (1.5B) | Current (0.5B + Stream) |
|---|---|---|
| **Time to First Token (TTFT)** | ~8,000ms - 10,000ms | **~350ms** |
| **Total Generation Time** | ~15,000ms | **~7,000ms** |
| **VRAM Usage** | ~3.2 GB | **~1.1 GB** |
| **Wikipedia Fetch** | Synchronous (Slow) | Cached (Fast) |

---

## Proposed Optimizations

### 1. Semantic Context Selection (Reranked RAG)
**Priority: High | Effort: Medium**

Currently, `fetch_wikipedia_summary` returns the first 3-5 sentences of a Wikipedia page. For many books, this is just biographical info about the author or publishing history, missing the actual plot summary.

*   **Improvement**: Fetch a larger extract (e.g., 20 sentences).
*   **Method**: Use the existing `text_encoder` (BGE-M3) to embed the chunks and the user's prompt. Perform a cosine similarity check to pick the 3 most relevant chunks.
*   **Result**: Higher quality, more "grounded" answers without increasing the LLM's context window.

### 2. Result Caching (Persistence Layer)
**Priority: High | Effort: Low**

Many users will ask about the same popular books (e.g., "1984", "The Great Gatsby").
*   **Method**: Implement a cache (Redis or MongoDB) keyed by `hash(title + author + user_prompt)`.
*   **Result**: **0ms latency** for repeat questions. We can serve the full response immediately from the DB.

### 3. Model Quantization (4-bit AWQ/GPTQ)
**Priority: Medium | Effort: High**

Even at 0.5B, the model uses ~1.1GB of VRAM in FP16. In a production environment with many concurrent users, this adds up.
*   **Method**: Convert the model to **4-bit AWQ**.
*   **Result**: VRAM usage drops to **~300-400MB**. This allows the LLM to reside comfortably on even low-end 4GB GPUs alongside the Search and CLIP models.

### 4. Speculative Decoding (Self-Speculation)
**Priority: Low | Effort: High**

*   **Method**: Use a tiny "draft" model (or a smaller version of Qwen) to predict the next few tokens, which the main 0.5B model then verifies in parallel.
*   **Result**: Potential 1.5x - 2x speedup in raw generation throughput (tokens/sec).

---

## Next Steps

1.  **Implement Step 1 (Semantic RAG)**: Leverage the BGE-M3 model already in memory to improve the Wikipedia grounding quality.
2.  **Add MongoDB Caching**: Store generated responses in the existing `interactions` or a new `llm_cache` collection.
3.  **Update UI**: Add a "Copy to Clipboard" or "Save to Profile" button for AI responses now that they are more reliable.

*Written: April 11, 2026*
