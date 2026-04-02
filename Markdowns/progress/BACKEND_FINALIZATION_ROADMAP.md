# Backend Finalization Roadmap: NBA Multimodal System

This document outlines the remaining tasks to ensure the backend operates smoothly, handles all multimodal edge cases, and provides a production-grade experience for the NBA (Next Best Action) system.

---

## 1. Detailed Folder Structure

The project follows a modular architecture separating the data layer, core ML logic, and the API serving layer.

```text
NBA_SYSTEM/
├── api.py                        # FastAPI entry point. Manages endpoints and global state.
├── PROJECT_TECHNICAL_REPORT.md   # System overview and execution flow documentation.
├── BACKEND_FINALIZATION_ROADMAP.md # This document.
│
├── src/                          # CORE ML PIPELINE
│   ├── data/                     # Data persistence layer
│   │   ├── asins.csv             # Mapping of 3.08M items (Vector ID -> ASIN)
│   │   ├── blair_index.faiss     # 12.6GB Text Index (Semantic features)
│   │   ├── clip_index.faiss      # 6.3GB Image Index (Visual features)
│   │   ├── cleora_embeddings.npz  # Behavioral manifold (375k items)
│   │   ├── item_metadata.parquet # Metadata cache (Title, Author, Image URLs)
│   │   └── hyperedges_cleora.txt # Training source for Cleora (500k users)
│   │
│   ├── config.py                 # System hyperparameters (Thresholds, K-values)
│   ├── retriever.py              # FAISS Orchestrator (Direct fetch & Similarity search)
│   ├── user_profile_manager.py   # State management (Aggregated embeddings, Temporal decay)
│   ├── active_search_engine.py   # Mode 1: Multimodal Fusion search logic
│   ├── passive_recommendation_engine.py # Mode 2: 3-Layer funnel (Cleora -> Veto -> RL)
│   ├── rl_collaborative_filter.py # Deep Q-Network (DQN) implementation
│   └── environment.py            # Simulated user reward logic
│
├── frontend/                     # REACT + VITE UI
│   ├── src/
│   │   ├── App.jsx               # Main Dashboard & Simulation UI
│   │   └── index.css             # Tailwind CSS entry
│   └── tailwind.config.js        # UI Styling configuration
│
└── venv/                         # Python 3.12 Virtual Environment
```

---

## 2. Remaining Backend Tasks

### A. Live GPU-Accelerated Encoding (Active Search)
*   **Current State**: `api.py` simulates search queries by picking random items.
*   **Required Task**: Integrate actual BLaIR and CLIP model loaders using **PyTorch CUDA**.
*   **Hardware Advantage**: Use the **NVIDIA RTX 4060 (8GB VRAM)** to run both encoders. 8GB is sufficient to hold both models in memory for sub-100ms inference.
*   **Specifics**: When a user types a query or uploads an image, the GPU will generate the embedding vector instantly.

### B. Image Upload Processing
*   **Current State**: Frontend supports drag-and-drop, but Backend doesn't yet process the `base64` bytes.
*   **Required Task**: Implement the `decode_image` helper in `api.py`.
*   **Specifics**: Convert Base64 strings into tensors and pass them to the **CLIP model on the GPU**.

### C. Metadata Coverage Expansion
*   **Current State**: `item_metadata.parquet` covers 1.7M items (56% of catalog).
*   **Required Task**: Re-run `src/build_metadata_cache.py` with a larger `sample_size` (4.4 million).
*   **Specifics**: Reach 100% metadata coverage for the 3.08M catalog.

### D. User Profile Persistence
*   **Current State**: Profiles live in RAM.
*   **Required Task**: Implement JSON or SQLite persistence for `UserBehaviorProfile`.
*   **Specifics**: Ensure RL DQN weights are saved so the system doesn't "forget" the user on restart.

---

## 3. Hardware Requirements & Optimization

The system is optimized for the following local environment:
*   **GPU**: NVIDIA RTX 4060 8GB (Primary for BLaIR/CLIP/RL-DQN).
*   **RAM**: 32GB (Required for 18GB Memory-Mapped FAISS indices).
*   **Storage**: SSD highly recommended for low-latency FAISS paging.

---

## 4. Recommended Execution Order

1.  **Immediate**: Expand the Metadata Cache to 100% coverage (4.4M samples).
2.  **Short-term**: Implement User Profile saving so history persists.
3.  **Advanced**: Switch `ActiveSearchEngine` from "Proxy Items" to "Live GPU Encoding" using the **RTX 4060**.

**Status**: Backend is 85% complete. The foundation is solid; hardware acceleration is the next major step.
