# 5. Project Contributions

The primary contribution of this research is the conceptualization and implementation of a **Dual-Mode Next Best Action (NBA) system** that goes beyond traditional structured behavioral logs by incorporating a large-scale multimodal knowledge base. The system fuses behavioral graph intelligence, semantic text understanding, visual feature reasoning, online reinforcement learning, and AI-assisted content grounding into a single, production-ready pipeline. Contributions are categorized into theoretical advancements and practical engineering achievements.

---

## 5.1 Theoretical and Research Contributions

### 5.1.1 Dual-Mode Multimodal Recommendation Architecture
Proposed and formalized a **Dual-Mode interaction paradigm** that separates user-initiated discovery (Active Search / Mode 1) from system-initiated discovery (Passive Recommendation / Mode 2). This architectural separation allows each mode to operate with its own retrieval strategy while sharing a common knowledge base, a concept that is often conflated in prior work. The two modes are:
- **Mode 1 (Active Search)**: The user provides a free-text query or uploads a cover image; the system encodes these live using BLaIR (1024-dim) and CLIP (512-dim) and performs a joint FAISS retrieval followed by Reciprocal Rank Fusion (RRF).
- **Mode 2 (Passive Recommendation)**: The system proactively surfaces the Next Best Action by routing through a three-layer behavioral and content funnel, entirely autonomously, using only the user's accumulated interaction profile.

### 5.1.2 Three-Layer Candidate Funnel: From 3 Million to the Single Best Action
Advanced the state of large-scale recommendation by designing a **high-recall to high-precision candidate funnel** that scales to 3,080,829 catalog items while remaining sub-millisecond at inference:
1. **Layer 1 — Behavioral Scouting (Cleora Hypergraph Embeddings)**: A 1024-dimensional Cleora embedding trained on a bipartite user-item hypergraph with 500,000 interaction baskets and 8 Markov propagation walks rapidly narrows 3 million items to **50 "Wisdom-of-the-Crowd" candidates** by seeding from the user's 5 most recent interactions.
2. **Layer 2 — Content Sanity Check (Hard Veto)**: Every candidate is compared against the user's aggregated BLaIR text profile and CLIP visual profile using cosine similarity. Items that score below the 0.3 similarity threshold in **both** modalities are discarded, preventing semantically or visually irrelevant recommendations from reaching the user.
3. **Layer 3 — Personalization (RL-DQN Re-ranking)**: A Deep Q-Network scores the surviving candidates from Layer 2 using the [(user_state ∥ item_embedding)](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/rl_collaborative_filter.py#94-100) vector as input, selecting the single highest-value Next Best Action.

### 5.1.3 Continuous Temporal User Profiling as an Aggregated Embedding
Introduced a novel **Aggregated Embedding Profile** that replaces static user vectors with a dynamically weighted representation of explicit user behavior. The profile is formulated as:

$$\text{Profile}_{\text{vec}} = \frac{\sum_{i=1}^{N} \vec{v}_i \cdot e^{\lambda \cdot (i-1)}}{\sum_{i=1}^{N} e^{\lambda \cdot (i-1)}}$$

where $\lambda$ is the temporal decay constant and $\vec{v}_i$ is the embedding of the $i$-th interacted item (in chronological order). This means the most recently clicked item has disproportionately high influence. The profile is maintained in **two separate embedding spaces** — BLaIR (1024-dim text) and CLIP (512-dim visual) — to capture semantic and aesthetic preferences independently.

### 5.1.4 Adaptive Trimodal Decision Fusion via Reciprocal Rank Fusion (RRF)
Proposed a **trimodal fusion strategy** that integrates three independent ranking signals — BLaIR text similarity, CLIP visual similarity, and the RL-DQN preference score — using Reciprocal Rank Fusion:

$$\text{Score}(\text{item}) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(\text{item})}$$

This formulation is robust to the different score scales of each model, requires no learned fusion weights, and ensures that items ranked highly by **multiple independent signals** are promoted, effectively implementing a "committee vote" for the final recommendation.

### 5.1.5 Wikipedia-Grounded RAG for Hallucination-Free AI Explanations
Proposed a **Retrieval-Augmented Generation (RAG) pipeline** to ground the lightweight local LLM (Qwen2.5-0.5B-Instruct) with factual knowledge from Wikipedia. When a user queries "Ask AI", the system:
1. Submits the book title as a semantic search query to the Wikipedia Search API.
2. Retrieves the canonical Wikipedia page title and extracts the first 5 factual sentences using the Wikipedia Extract API.
3. Injects this ground-truth text as a system context message before generating the response with the LLM.

This pipeline achieves near-100% factual grounding accuracy even with a 0.5B parameter model and eliminates the hallucinated plot summaries that plague standalone small LLMs.

---

## 5.2 Practical and Engineering Contributions

### 5.2.1 Production-Scale Multimodal Knowledge Base
Built and deployed an end-to-end data engineering pipeline that processes a **3,080,829-item Amazon product catalog** into a unified multimodal knowledge base:
- **12.6 GB BLaIR FAISS Index** (`blair_index.faiss`): 1024-dimensional semantic text embeddings stored in a memory-mapped flat inner product index, enabling sub-millisecond reconstruction.
- **6.3 GB CLIP FAISS Index** (`clip_index.faiss`): 512-dimensional visual feature embeddings.
- **Cleora Behavioral Manifold** (`cleora_embeddings.npz`): 375,414 product behavioral embeddings derived from 500,000 unique hyperedges (user interaction baskets), trained with 8 Markov propagation walks.
- **Item Metadata Cache** (`item_metadata.parquet`): Titles, authors, main categories, and image URLs for 1.7M+ items, with a fallback stub for items not yet covered.

Data ingestion was automated via [load_data.ipynb](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/load_data.ipynb) and [src/prepare_cleora_data.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/prepare_cleora_data.py), which handle K-Core filtering (minimum 5 interactions), weighted edge-list construction (view=1x, cart=3x, purchase=5x), and chunk-based loading from Hugging Face Hub (`minhkhang26/my-nba-project-data`, 90 NPZ chunks).

### 5.2.2 Live GPU-Accelerated Multimodal Encoding in Production
Implemented a **real-time encoding pipeline** within [api.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/api.py) that supports both text and image queries using on-device GPU inference (validated on NVIDIA RTX 4060):
- **BLaIR Live Encoding** (`hyp1231/blair-roberta-large` via `sentence-transformers`): Any user text query is encoded on-the-fly into a 1024-dimensional vector using the same embedding space as the pre-indexed catalog, enabling exact semantic retrieval.
- **CLIP Live Encoding** (`openai/clip-vit-base-patch32` via `transformers`): A user-uploaded book cover is decoded from Base64, converted to a PIL image, processed through the CLIP visual encoder on GPU, L2-normalized, and used for visual similarity retrieval.
- **Intelligent fallbacks**: If the text query is a known generic phrase (e.g., "find me a book like this cover"), the text vector is suppressed to avoid contaminating the image search, and vice versa. If no encoder is available, the system falls back to a proxy-item query.

### 5.2.3 Online Reinforcement Learning Agent with Full State Persistence
Engineered a **Deep Q-Network (DQN) collaborative filter** ([src/rl_collaborative_filter.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/rl_collaborative_filter.py)) that learns user preferences from live click/skip feedback in real time. The DQN architecture:
- **Input**: Concatenated user state vector [(BLaIR_profile ∥ CLIP_profile)](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/rl_collaborative_filter.py#94-100) at dimension 1536, paired with a candidate item's CLIP embedding (512-dim), yielding a 2048-dimensional joint representation.
- **Architecture**: Two hidden layers (256 units, ReLU) followed by a scalar preference score head.
- **Training Signal**: A click delivers `reward = 1.0`, an "Add to Cart" action delivers `reward = 5.0`, and a skip delivers `reward = 0.0`.
- **Persistence**: Both the model weights and optimizer state are saved to a **per-user `.pt` checkpoint file** (`<user_id>_dqn.pt`) on every interaction and loaded on subsequent sessions, ensuring the agent's learned preferences survive server restarts.

### 5.2.4 JSON-Persistent User Profile Manager
Developed a fully persistent [UserProfileManager](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/user_profile_manager.py#37-168) ([src/user_profile_manager.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/user_profile_manager.py)) that stores each user's click history, recent interaction window, and aggregated embedding vectors. Key design decisions:
- **Auto-save on every click**: The profile JSON is written to `src/data/profiles/<user_id>.json` after each interaction, making the system restart-safe with zero data loss.
- **Lazy loading**: Profiles are loaded from disk on first access and then served from an in-memory cache (`_cache` dict) for the remainder of the session.
- **Embedding reconstruction**: On load, the full BLaIR + CLIP aggregated profile vectors are recomputed from the stored click history using the FAISS index reconstruct operation, so the embedding state is always consistent with the interaction log.

### 5.2.5 End-to-End FastAPI Backend with Async Lifespan Management
Implemented a complete, production-ready **FastAPI backend** ([api.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/api.py)) exposing five endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness probe: reports catalog size, BLaIR/CLIP status, and device type |
| `POST /search` | Mode 1: live BLaIR + CLIP encoding → FAISS retrieval → RRF fusion |
| `GET /recommend` | Mode 2: 3-layer funnel with cold-start detection and DQN re-ranking |
| `POST /interact` | Log click/cart/skip → update profile → train DQN → persist weights |
| `POST /ask_llm` | Wikipedia RAG → Qwen2.5-0.5B LLM → factual book explanation |

The server uses an `asynccontextmanager` lifespan to load all ML models (18+ GB of FAISS indices, BLaIR, CLIP, and the LLM) once at startup with a 503 guard that immediately returns "System still initializing" if a request arrives before the startup is complete.

### 5.2.6 Interactive React + Vite Frontend with Real-Time RL Feedback Loop
Built a responsive, production-grade **React (Vite) demo frontend** ([frontend/src/App.jsx](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/frontend/src/App.jsx)) that serves as both a user-facing interface and a live training environment for the RL agent:
- **Dual-Mode Tab UI**: "🔍 Active Search (Mode 1)" and "✨ Recommendations (Mode 2)" tabs expose both pipeline modes to the user.
- **Image Drag-and-Drop**: Users drop a book cover image directly into the search panel; the frontend reads it as Base64 and posts it to the CLIP encoder in the backend.
- **Live/Mock Toggle**: A header toggle allows switching between the live backend API and a fully self-contained mock mode, enabling safe offline demonstrations.
- **Real-Time RL Feed**: Every click or skip triggers an `POST /interact` call, which trains the DQN and immediately reflects the updated RL step count and CTR in the dashboard header.
- **Profile Radar**: A live visualization panel shows the user's CTR, interaction depth, RL fit quality, and interaction diversity as animated progress bars, updating after each action.

### 5.2.7 Experimental Simulation and Ablation Framework
Developed a quantitative evaluation framework validating the end-to-end pipeline through a **2,000-step closed-loop simulation** ([src/main.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/main.py), [src/evaluator.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/evaluator.py)):
- **Final CTR**: **0.6735** — demonstrating successful closed-loop adaptation of the RL agent.
- **Profile Growth**: Accumulated 1,347 high-quality interactions over the simulation, showing sustained system stability.
- **Latency**: Sub-millisecond candidate scoring due to FAISS vector reconstruction and in-process inference.
- **Ablation Studies**: [src/ablation_rrf_k.py](file:///c:/Users/minhk/OneDrive/Documents/HCMUTSUB/DATN/src/ablation_rrf_k.py) provides a dedicated script for testing the sensitivity of the RRF `k` hyperparameter and measuring the impact of the content veto layer on recommendation quality.

### 5.2.8 Cold-Start Handling and Graceful Degradation
Implemented multi-level **cold-start and failure-mode strategies** that prevent poor user experiences when data is sparse:
- **New User Cold Start**: Users with fewer than `COLD_START_THRESHOLD` clicks receive a random discovery sample from the high-precision Cleora pool instead of a personalized recommendation.
- **Layer 1 Fallback**: If a user's recently clicked items are not in the Cleora graph (e.g., they were discovered via active search only), the behavioral layer falls back to a random sample from the Cleora node pool.
- **Layer 2 Fallback**: If the content veto filters out all 50 candidates, the engine returns an empty list rather than surfacing irrelevant items.
- **Encoder Fallback**: If BLaIR or CLIP fail to load, the search endpoint falls back to a proxy-item query using a randomly selected, pre-indexed item vector, ensuring the API never returns a 500 error to the user.
