# Dynamic Cleora Integration & User Logging Plan

**Date**: April 6, 2026  
**Topic**: Transitioning Cleora from a static behavioral snapshot to a dynamic, production-grade learning system.

---

## 1. Architectural Overview

The current system uses Cleora embeddings as a static "lookup table." This plan introduces a **Feedback Loop** where live user interactions (clicks, carts, skips) are logged, queued, and used to periodically re-generate the behavioral graph.

```
[ Frontend ] ──(Interaction)──▶ [ FastAPI /interact ]
                                      │
                                      ▼
[ Redis Queue ] ◀────────────── [ Background Task ]
      │
      ▼
[ MongoDB ] ──(Batch Fetch)──▶ [ Refresh Script ] ──▶ [ Cleora Engine ]
                                      │                      │
[ Updated Retriever ] ◀────────(Reload .npz) ◀───────────────┘
```

---

## 2. Deep Dive: The Current Cleora System

The behavioral layer currently operates as a high-speed "Discovery Engine" that identifies hidden product relationships based on global Amazon purchase patterns.

### 2.1 Data Pipeline & Filtering (`prepare_cleora_data.py`)
1.  **Ingestion**: Fetches the McAuley-Lab Amazon Reviews 2023 (Books) benchmark.
2.  **K-Core Density**: Applies a **5-Core filter**. This is critical for graph embedding quality; it recursively prunes the graph until every user has at least 5 interactions and every book has at least 5 sales. 
    - *Why*: Sparse graphs produce "noisy" embeddings. High-degree nodes provide the "anchor points" for the behavioral space.
3.  **Hyperedge Mapping**: Interactions are grouped by `user_id`. Each user is represented as a **Hyperedge** (a set of items). 
    - *Format*: A text file where each line is a user: `ASIN_1 ASIN_2 ASIN_3 ...`
    - *Intuition*: Cleora treats each line as a "context." Items appearing on the same line are behaviorally related.

### 2.2 Unsupervised Learning Engine (`run_cleora.py`)
1.  **Algorithm**: Rust-based Markov Propagation. Unlike neural networks (DeepWalk/Node2Vec), Cleora uses **recursive matrix-vector multiplication**.
2.  **Initialization**: 1024-dim vectors are generated deterministically from ASIN strings.
3.  **Markov Propagation**:
    - Performs **8 walks** (`left_markov_propagate`).
    - In each walk, an item's embedding becomes the average of the embeddings of all other items it has co-occurred with.
    - **Normalization**: L2 normalization is applied after each walk to keep the embeddings on a hypersphere (improving Cosine Similarity performance).
4.  **Persistence**: The final 1024-dim vectors are saved to `cleora_embeddings.npz`.

### 2.3 Production Retrieval (`retriever.py` & `passive_recommendation_engine.py`)
1.  **Indexing**: At API startup, Cleora embeddings are loaded into a **FAISS `IndexFlatIP`** (Inner Product / Cosine Similarity).
2.  **The Seed Mechanism**:
    - The engine identifies the user's **5 most recent interactions**.
    - Each interaction acts as a **Seed ASIN**.
    - For each seed, we query FAISS for the **Top-50 nearest neighbors** in Cleora space.
3.  **Veto & Fusion**: Behavioral candidates that fail the content sanity check (BLaIR < 0.3) are discarded to prevent "behavioral hallucinations" (books bought together by mistake).

---

## 3. The Logging Database (MongoDB)

### Why MongoDB?
- **High Write Throughput**: Perfect for "fire-and-forget" interaction logging.
- **Flexible Schema**: Allows adding metadata (device info, session context) without migrations.
- **Natural Event Storage**: Interactions are immutable events over time.

### Interaction Schema
```json
{
  "user_id": "user_123",          // UUID or Session ID
  "asin": "B000123456",           // Item interacted with
  "action": "cart",               // click | cart | purchase | skip
  "timestamp": "2026-04-06T...", 
  "source": "search",             // search | recommendation
  "is_guest": false
}
```

---

## 3. Asynchronous Pipeline (The "Queue")

Cleora re-training is a batch operation (Markov propagation on 1.7M nodes takes ~2-5 mins). We use an **Asynchronous Queue** to decouple logging from the user's response time.

1.  **Immediate Response**: `api.py` acknowledges the interaction immediately (status: 200 OK).
2.  **Background Worker**: A task is pushed to a queue (e.g., `anyio` background task or `Celery`).
3.  **Persistence**: The worker writes the event to MongoDB.
4.  **Batch Trigger**: Once every 24 hours (or every 10,000 interactions), a trigger starts the `refresh_behavioral_graph.py` process.

---

## 4. Handling Guest Users

Guest users present a "Cold Start" problem for behavioral graphs because they have no historical anchor in the Cleora space.

### Strategy: The "Seed-Neighbor" Approach
1.  **Session Tracking**: Use a frontend-generated `guest_uuid` stored in `localStorage`.
2.  **Anchor Retrieval**: When a guest clicks `Book_A`, we look up `Book_A` in the **existing** Cleora embedding space.
3.  **Behavioral Proxy**: We use `Book_A`'s neighbors as the behavioral candidates for that guest's next recommendation set.
4.  **Identity Merging**: If the user signs in:
    - Query MongoDB for all events matching `guest_uuid`.
    - Batch update those records to the new `user_id`.
    - The next Cleora run will now treat the guest history as part of the authenticated user's profile.

---

## 5. Implementation Roadmap

### Phase A: Persistence Layer (Immediate)
- Install `motor` (async MongoDB driver for Python).
- Update `api.py` to push to a `log_interaction` background task.
- Implementation of the MongoDB client in `src/utils.py`.

### Phase B: Dynamic Data Prep (Medium Term)
- Modify `src/prepare_cleora_data.py` to query MongoDB instead of downloading static Amazon CSVs.
- Combine the static "base" graph (3M rows) with the live "delta" graph (new interactions).

### Phase C: Hot-Swapping (Production)
- Add a `.reload()` method to the `Retriever` class.
- Use a File System Watcher to detect when `cleora_embeddings.npz` is updated and swap the in-memory mapping without restarting the API.

---

## 6. Current Limitations vs. Future State

| Feature | Current State | Future State (This Plan) |
| :--- | :--- | :--- |
| **Data Freshness** | Static (Amazon 2023) | Dynamic (Updated daily/weekly) |
| **New Items** | Broken (OOD items have no vec) | Content-to-Behavior projection |
| **User Scaling** | None | Real-time session adaptation |
| **Storage** | None (RAM-only profiles) | Permanent historical audit trail |

---
*Created on April 6, 2026. This document serves as the implementation guide for Phase 1.3 of the system optimization.*
