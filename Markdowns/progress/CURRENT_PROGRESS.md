# Current Progress: NBA System Implementation (DATN)

This document outlines the current state of the Next Best Action (NBA) recommendation system and details the execution flow of the primary simulation script.

## Project Status Summary
The project is currently at **Milestone 4 (Pipeline Integration)** and transitioning into **Milestone 5 (Evaluation)**.

| Milestone | Status | Description |
| :--- | :--- | :--- |
| **M1: Data Preparation** | ✅ Complete | Amazon Reviews 2023 dataset processed into `data.parquet`. |
| **M2: Cleora Embeddings** | ✅ Complete | Behavioral embeddings generated via `pycleora` in `cleora_embeddings.npz`. |
| **M3: Content Indices** | ✅ Complete | FAISS indices for BLaIR (text) and CLIP (image) built and stored. |
| **M4: Pipeline Integration** | ✅ Complete | Three-layer retrieval logic implemented in `Retriever` class. |
| **M5: Evaluation** | 🏗️ In Progress | Simulation loop in `main.py` tracks CTR using a reward environment. |

---

## Detailed Execution Flow (`src/main.py`)

When `src/main.py` is executed, the system follows a structured three-layer recommendation pipeline designed for high-precision Next Best Action (NBA) discovery.

### 1. Initialization Phase
*   **Resource Loading**: Loads the `asins.csv` mapping, `cleora_embeddings.npz`, and the pre-computed FAISS content indices (`blair_index.faiss`, `clip_index.faiss`).
*   **Index Construction**: Builds an in-memory `IndexFlatIP` for Cleora behavioral vectors to enable fast similarity search.
*   **Query Pool Selection**: Identifies a "High-Precision Query Pool" containing only items that exist in both the behavioral (Cleora) and content (BLaIR/CLIP) datasets.

### 2. The Recommendation Pipeline (per Query)
For each iteration, the system selects a random item from the query pool and processes it through three layers:

#### **Layer 1: Behavioral Candidate Generation**
*   **Logic**: Uses the **Cleora index** to find the top 50 behavioral neighbors.
*   **Goal**: Captures "wisdom of the crowd" by identifying items frequently co-interacted with by users.

#### **Layer 2: Content Verification (Scoring & Veto)**
*   **Logic**: Retrieves multimodal embeddings (BLaIR for text, CLIP for images) for all Layer 1 candidates.
*   **Similarity Scoring**: Calculates cosine similarity between the query item and candidates for both text and image modalities.
*   **Hard Veto**: Candidates are discarded if they fail to meet a **0.3 similarity threshold** in *both* modalities. This ensures that recommendations remain visually or semantically relevant.

#### **Layer 3: Decision Fusion (RRF)**
*   **Logic**: Takes the separate rankings from BLaIR and CLIP scores.
*   **Algorithm**: Applies **Reciprocal Rank Fusion (RRF)** to merge these rankings into a single, unified list.
*   **Selection**: The Top-1 item from the fused list is selected as the "Next Best Action."

### 3. Simulation & Reward Phase
*   **User Simulation**: The `environment.get_reward` function calculates a "click" probability using a sigmoid function applied to the dot product of a hidden "user preference" vector and the recommended item's CLIP embedding.
*   **Metric Logging**: The `Evaluator` tracks the success (reward = 1) or failure (reward = 0).
*   **Real-time Feedback**: Every 100 successful recommendations, the script prints the current **Click-Through Rate (CTR)**.

---

## Core Components
*   `src/retriever.py`: Orchestrates multi-index searches across behavioral and content spaces.
*   `src/environment.py`: Simulates user behavior and provides ground-truth rewards.
*   `src/evaluator.py`: Maintains performance metrics and calculates CTR stability.
*   `src/config.py`: Contains hyperparameters like `TOP_K` and threshold values.
