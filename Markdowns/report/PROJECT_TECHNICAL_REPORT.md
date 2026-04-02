# Technical Report: Dual-Mode Multimodal NBA System (DATN)

This report provides a comprehensive technical breakdown of the Next Best Action (NBA) recommendation system, covering data engineering, the multi-layer retrieval pipeline, and the Reinforcement Learning integration.

---

## 1. System Overview
The system is designed to solve the "Interaction-Content Gap" by combining **Behavioral Modeling** (Cleora), **Multimodal Content Understanding** (BLaIR & CLIP), and **Online Personalization** (Deep Q-Learning). It operates in two distinct modes:
*   **Active Search (Mode 1)**: User-initiated discovery via text/image queries.
*   **Passive Recommendation (Mode 2)**: Proactive system suggestions based on an evolving user profile.

---

## 2. Data Architecture & Scale
The system has been scaled to a production-grade catalog using data from **Amazon Reviews 2023**.

### A. Multimodal Knowledge Base (Content)
*   **Total Items**: 3,080,829 products.
*   **Text Encodings (BLaIR)**: 1024-dimensional semantic vectors stored in a 12.6GB FAISS index.
*   **Image Encodings (CLIP)**: 512-dimensional visual vectors stored in a 6.3GB FAISS index.
*   **Total Chunks**: 89 chunks of merged metadata and embeddings.

### B. Behavioral Manifold (Cleora)
*   **Training Data**: 500,000 unique user interaction "baskets" (hyperedges).
*   **Coverage**: 375,414 products with behavioral embeddings.
*   **Mechanism**: 8 Markov propagation walks to capture deep item-to-item correlations.
*   **Storage**: 1024-dimensional vectors stored in `cleora_embeddings.npz`.

---

## 3. The 3-Layer Funnel Architecture
The recommendation logic follows a high-recall to high-precision "funnel" to ensure efficiency at scale.

### Layer 1: Behavioral Scouting (Cleora)
*   **Input**: The user's last 5 clicked items.
*   **Process**: Performs a multi-seed similarity search in the Cleora behavioral space.
*   **Goal**: Rapidly narrow down 3 million items to **50 "Wisdom of the Crowd" candidates**.

### Layer 2: Content Sanity Check (Veto)
*   **Input**: The 50 candidates from Layer 1.
*   **Process**: Direct vector reconstruction from BLaIR and CLIP indices.
*   **Veto Logic**: Candidates are compared to the **User's Aggregated Profile**. If Cosine Similarity is **< 0.3** in both text and image modalities, the item is discarded.
*   **Goal**: Ensure recommendations match the user's specific visual and semantic taste.

### Layer 3: Personalization (RL-DQN)
*   **Input**: Surviving candidates.
*   **Model**: Deep Q-Network (DQN).
*   **Logic**: Predicts a **Preference Score** based on the current `User Profile State` and `Item Action Vector`.
*   **Goal**: Pick the single best action from the verified list.

---

## 4. Execution Flow Diagram

```mermaid
sequenceDiagram
    participant U as User (Alex)
    participant M1 as Active Search (Mode 1)
    participant M2 as Passive Rec (Mode 2)
    participant DB as Knowledge Base (3M Items)
    participant CL as Cleora (Behavioral)
    participant RL as RL-DQN Agent
    participant PR as User Profile Manager

    Note over U, PR: STEP 1: INITIALIZATION
    PR->>PR: Create Empty Profile (Alex)
    
    Note over U, PR: STEP 2: COLD START (Search Mode)
    U->>M1: Query: "Dark Fantasy" + [Image Upload]
    M1->>DB: FAISS Direct Search (BLaIR + CLIP)
    DB-->>M1: Top-10 Multimodal Results
    M1-->>U: Show Results
    U->>PR: CLICK (Item A)
    PR->>PR: Generate First Profile Fingerprint

    Note over U, PR: STEP 3: WARM START (Recommendation Mode)
    U->>M2: alex opens homepage
    M2->>CL: Layer 1: Retrieve Behavioral Neighbors of Item A
    CL-->>M2: 50 Candidates (Wisdom of Crowd)
    
    M2->>DB: Layer 2: Fetch BLaIR/CLIP vectors for 50 items
    M2->>M2: Compare to Alex's Profile (Veto < 0.3)
    
    M2->>RL: Layer 3: Score survivors via DQN
    RL-->>M2: Personalized Reranking
    
    M2->>M2: RRF Fusion (Text + Visual + RL)
    M2-->>U: Show Top-1 NBA
    
    Note over U, PR: STEP 4: CLOSED FEEDBACK LOOP
    U->>PR: Interaction (CLICK or SKIP)
    PR->>PR: Re-compute Aggregated Fingerprint (Temporal Weighting)
    PR->>RL: train_step (Update DQN Weights)
    Note right of RL: Model learns Alex's nuance live!
```

---

## 5. Key Implementation Details

### Continuous Profile Learning
The **User Behavior Profile** is not just a list of IDs. It is an **Aggregated Embedding** calculated as:
$$Profile_{vec} = \frac{\sum (Item_{vec} \cdot e^{-\lambda t})}{\sum e^{-\lambda t}}$$
Where $\lambda$ is the **Temporal Decay** constant (0.1). This ensures that Alex's most recent interests weigh more than old ones.

### Adaptive Decision Fusion (RRF)
Final scores are calculated using **Reciprocal Rank Fusion**:
$$Score(item) = \sum_{rankings} \frac{1}{k + rank_{item}}$$
This merges the independent "opinions" of the **Text Model**, **Visual Model**, and **RL Agent** into a single robust decision.

---

## 6. Validated Results
The system was validated through a 2,000-step simulation:
*   **Final CTR**: 0.6735 (Successful adaptation).
*   **Profile Growth**: Continuous enrichment from 0 to 1,347 high-quality interactions.
*   **Latency**: Sub-millisecond scoring due to FAISS vector reconstruction.

**Status**: System core is optimized, synchronized, and ready for production deployment.
