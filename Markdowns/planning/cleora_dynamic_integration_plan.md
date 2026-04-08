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

## 5. Implementation Roadmap (Dual-Speed Architecture)

To ensure the system is both responsive and capable of global learning, we implement a **Dual-Speed Behavioral Pipeline**.

### Phase A: Persistence Layer (✅ COMPLETED)
- **Infrastructure**: MongoDB + Redis Docker stack.
- **API Worker**: Async `_log_worker` in `api.py` draining the interaction queue.
- **Audit Trail**: Every click/cart/skip now stored in `nba_logs.interactions`.

### Phase B: Dual-Speed Integration (Current Focus)

#### 1. The "Fast Path": Real-Time Behavioral Projection
Instead of re-training the graph for every click, we move the user *through* the existing Cleora space.
- **Mechanism**: Update `UserProfileManager` to calculate a **1024-dim Cleora Profile**.
- **Math**: When a user clicks `Book_A` and `Book_B`, their behavioral fingerprint becomes `mean(Cleora_Vec_A, Cleora_Vec_B)`.
- **Latency**: 0ms (Vector averaging).
- **Outcome**: The "People Also Buy" tab updates immediately after a user clicks a book, reflecting their new position in the behavioral map.

#### 2. The "Slow Path": Delta-Augmented Batch Refitting
Global relationships (e.g., new books becoming "friends") are updated periodically.
- **Data Export**: A script `scripts/export_delta_hyperedges.py` queries MongoDB and formats new interactions into hyperedges.
- **Graph Augmentation**: We append these "Delta" hyperedges to the 1.7M "Base" hyperedges.
- **Up-weighting**: Multiply the "Delta" lines (e.g., repeat 100x) to ensure recent local behavior influences the global Markov propagation.
- **Refit**: Run `src/run_cleora.py` once a week or after 10,000 new interactions.

### Phase C: Production Hot-Swapping
- **Hot-Reload**: Implement a File Watcher in `retriever.py` that detects a new `cleora_embeddings.npz` and swaps the FAISS index without an API restart.
- **Session Merging**: Script to migrate `guest_uuid` interactions to registered `user_id` accounts in MongoDB.

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
