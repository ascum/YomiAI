# Project Architecture & Workflow Deep Dive (Updated 19-3)

This document provides a highly detailed breakdown of the NBA (Next Best Action) Multimodal Recommendation System. It serves as an architectural blueprint, covering the specific roles of all backend files, the under-the-hood mechanics of the recommendation and RL engines, and the end-to-end multi-modal data workflow.

---

## 🏗️ 1. Backend Analysis: File Roles & Code-Level Examples

The backend is built with **FastAPI** and orchestrates multimodal retrieval, behavioral recommendation, and deep reinforcement learning.

### 📂 `api.py` (The System Orchestrator)
- **Role**: Manages API routes (`/search`, `/recommend`, `/interact`, `/ask_llm`), initializes global models (BLaIR, CLIP, local Qwen LLM), loads FAISS indices into memory map (`MMAP`), and routes data between the engines.
- **Example Usage**: When a user searches, handles encoding and triggers the search engine.
```python
# Encodes raw string using 'hyp1231/blair-roberta-large' to 1024-dim vector
text_vec = blair_model.encode(["dark fantasy magic"]).astype("float32")
# Encodes image using 'openai/clip-vit-base-patch32' to 512-dim vector
image_vec = clip_model.get_image_features(**inputs).cpu().numpy()
# Hands vectors off to the Search Engine
results = search_engine.search(user_id, text_query_vec=text_vec, image_query_vec=image_vec)
```

### 📂 `retriever.py` (The Data Access Layer)
- **Role**: Abstract wrapper around the 3 core database indices. Holds `IndexFlatIP` indices for BLaIR, CLIP, and Cleora embeddings. It is responsible for extremely fast $O(1)$ vector reconstruction and top-N inner-product searches.
- **Example Usage**: `PassiveRecommendationEngine` asks it to fetch a candidate's visual features for RL scoring.
```python
# Reconstructs the exact 512-dim visual embedding for a specific ASIN in 1 millisecond
item_visual_vec = retriever.clip_index.reconstruct(retriever.asin_to_idx["B0012345"])
```

### 📂 `active_search_engine.py` (Multimodal Fusion)
- **Role**: Implements "Mode 1". It maps a hybrid user query (text + image) into a fused ranking using Reciprocal Rank Fusion (RRF).
- **Example Usage**: 
```python
# FAISS retrieves Top-50 purely semantic matches (BLaIR space)
D_text, I_text = retriever.blair_index.search(text_vec, 50)
# FAISS retrieves Top-50 purely visual matches (CLIP space)
D_img, I_img = retriever.clip_index.search(image_vec, 50)
# Fusion: Book A is rank 2 in Text and rank 8 in Image.
# RRF Score (k=60) = (1 / (60 + 2)) + (1 / (60 + 8))
final_ranking = rrf_fusion([text_rankings, img_rankings], k=60)
```

### 📂 `user_profile_manager.py` (The Memory)
- **Role**: Tracks user interactions and dynamically updates an **Aggregated Interest Fingerprint**. It calculates an exponentially weighted moving average of the BLaIR and CLIP vectors of items the user has clicked.
- **Example Usage**:
```python
# User clicks a book. The engine fetches all past clicks.
# Applies exponential temporal decay where lambda = 0.1
weights = np.exp(0.1 * np.arange(len(clicked_blair_vecs))) 
weights /= weights.sum() # Normalize
# Produces a new 1024-dim profile vector skewed toward the *most recent* interaction
profile.text_profile = np.average(clicked_blair_vecs, axis=0, weights=weights)
```

### 📂 `rl_collaborative_filter.py` (The AI Brain)
- **Role**: A Deep Q-Network (DQN) implemented in PyTorch. It acts as a pointwise neural ranker that predicts a user's preference score for an item based on their current profile.
- **Example Usage**: During inference.
```python
# State: Concat(User Text Profile [1024], User Visual Profile [512]) -> [1, 1536]
# Action: Item Visual Features (CLIP) -> [1, 512]
# Network Feedforward: [1, 2048] -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Scalar Output
predicted_q_value = dqn_model(user_state_tensor, item_clip_tensor) 
```

### 📂 `passive_recommendation_engine.py` (The Funnel)
- **Role**: Implements "Mode 2". Executes the core 3-layer personalized recommendation pipeline.
- **Example Usage**: See Section 3 for the deep dive.

---

## 🚀 2. The Big Example: Active Search Execution Flow

**Scenario**: A user types *"Gritty detective novels"* into the search bar AND uploads an image of a moody, dark book cover. They hit "Search".

### Under the Hood: The Request Lifecycle
1. **Frontend to Backend**: `App.jsx` intercepts the upload, converts the image to a Base64 string, and sends `POST /search` with `{"query": "Gritty detective novels", "image_base64": "iVBO..."}`.
2. **Encoding (api.py)**:
    - The LLM router realizes this is a hybrid query.
    - The BLaIR encoder runs on the CPU/GPU, translating the text into a **1024-dimensional semantic manifold vector**.
    - The CLIP image processor decodes the Base64, resizes the image to 224x224, and the Vision Transformer outputs a **512-dimensional visual vector**.
3. **Retrieval (`active_search_engine.py`)**:
    - The BLaIR FAISS index is queried: *"Find me the 50 books whose content closest matches the meaning of Gritty Detective Novels."*
    - The CLIP FAISS index is queried simultaneously: *"Find me the 50 books whose covers look exactly like this moody image."*
4. **Multimodal Fusion (RRF)**:
    - The engine uses **Reciprocal Rank Fusion** ($k=60$).
    - If book X has a *great* plot (Text Rank: 1) but a very *bright* cover (Image Rank: 48), its score is $1/61 + 1/108 = 0.025$.
    - If book Y has a *decent* plot (Text Rank: 12) AND a *moody* cover (Image Rank: 3), its score is $1/72 + 1/63 = 0.029$.
    - **Result**: Book Y wins and is pushed to Rank 1.
5. **Enrichment & Response**: `api.py` fetches metadata (Title, Author, Genre) from `item_metadata.parquet` and returns the JSON payload to the React frontend, which renders the multimodal search results.

---

## 🧠 3. Under the Hood: The Recommendation Backend & RL Backbone

When the user enters the "Recommendations" tab, the system doesn't rely on simple keyword matching. It uses a **proactive 3-layer neural funnel**. Here is exactly how it works.

### Layer 1: Behavioral Scouting (Cleora)
*Goal: High Recall*
- The Engine looks at the user's **last 5 clicked items**.
- It queries the **Cleora Index** (a 1024-dim behavioral hypergraph embedding). Cleora embeddings are trained using Markov random walks, so if millions of real Amazon users frequently bought Book A alongside Book B, their vectors will be identical.
- It pulls the top nearest neighbors for those 5 seeds, returning exactly **50 candidate items**.

### Layer 2: Multimodal Content Veto
*Goal: Precision & Sanity Check*
- The engine reconstructs the raw BLaIR and CLIP features for all 50 candidates.
- It calculates the Cosine Similarity between the candidates and the user's **Aggregated Profile Fingerprint**.
- **The Veto Rule**: If an item's semantic similarity AND its visual similarity to the user's profile are BOTH `< 0.3`, it is brutally discarded. This ensures that even if an item is "popular" (Cleora), it won't be recommended if the user strictly hates that specific genre or cover aesthetic.

### Layer 3: DQN Re-Ranking (The RL Backbone)
*Goal: Personalization Optimization*
- The survivors (e.g., 30 items) are fed into the **Deep Q-Network (DQN)**.
- **State**: The user's combined 1536-dimensional BLaIR+CLIP fingerprint.
- **Action space**: The visual embeddings (512-dim) of the 30 candidate items.
- The PyTorch neural network scores each combination based on past learning. 
- A final **RRF Fusion** blends the original BLaIR score + CLIP score + the new RL DQN Score to generate the ultimate **Top-5 Next Best Actions**.

### How the RL Backbone Learns (The Feedback Loop)
When the user clicks "Interested" (or "Add to Cart"):
1. **Frontend** POSTs to `/interact` with `action: "click"`.
2. **Reward Calculation**: Click = `+1.0`, Cart = `+5.0`, Skip = `0.0`.
3. **Training Execution**: `PassiveRecommendationEngine.train_rl()` is fired.
4. **Gradient Descent**: The DQN processes its prediction vs. the actual reward. If it predicted the user wouldn't click, but they did, it calculates the **Mean Squared Error (MSE) loss**. 
5. **Backpropagation**: The Adam optimizer adjusts the linear weights (`self.fc1`, `self.fc2`). 
6. **Persistence**: The updated weights are immediately serialized to `data/profiles/{user_id}_dqn.pt`. Tomorrow, the system will use this customized, smarter network for this exact user.

---

## 💻 4. Frontend Internal Workings (`App.jsx`)

The React frontend operates as an interactive local dashboard:
- **State Hydration**: Uses the `useEffect` hook to automatically fetch personalized recommendations (`GET /recommend`) when the app boots.
- **Live vs. Mock Mode**: A boolean toggle `useMock`. If `live`, `fetch` API calls hit `localhost:8000`. If `mock`, intentional delays (`setTimeout(800)`) simulate backend latency, and static predefined objects (`MOCK_BOOKS`) are injected.
- **RL Adaptation Tracking**: The frontend maintains an `rlStep` counter. Every 3 user interactions (Clicks/Skips), it silently re-fetches the recommendations payload in the background to show the user how the RL agent is actively molding to their recent clicks.
- **Local RAG Agent (`Ask AI`)**: When a user clicks "Ask AI" on a book:
   1. Frontend sends the book details to `POST /ask_llm`.
   2. Backend searches Wikipedia for the exact book, extracts the plot summary text.
   3. Backend injects the Wikipedia context + User Prompt into the local **Qwen2.5-1.5B-Instruct** prompt template.
   4. The LLM generates a grounded 2-3 sentence markdown response.
   5. Frontend renders it in a floating glassmorphic tooltip using dangerouslySetInnerHTML equivalent.

---
*Last update: March 19, 2026. This document serves as the primary technical specification for LLM context ingestion and developer onboarding.*
