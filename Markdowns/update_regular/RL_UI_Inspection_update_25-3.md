# Project Update: Recommendation UI Split & RL Metrics Dashboard — March 25, 2026

Following the search accuracy improvements, this update focuses on the **decoupling of the recommendation engine** and the implementation of a **Real-Time RL Inspection Dashboard**. These changes provide a more transparent and personalized user experience.

---

## 🏗️ 1. Multi-Tab Recommendation Architecture

### The Problem
Previously, recommendations were returned as a single blended list. This made it difficult for users to distinguish between "Items similar to what I just saw" (retrieval) and "Items the AI thinks I will like overall" (RL-personalization).

### The Fix: Strategic Pool Separation
The recommendation engine in `src/passive_recommendation_engine.py` was refactored to return two distinct pools of data to the `/recommend` endpoint:

1.  **"People Also Buy" (Deterministic Retrieval)**:
    *   **Logic**: Uses Cleora graph-based behavioral embeddings and BLaIR/CLIP content verification.
    *   **Usecase**: High-confidence similarity to the user's *immediate* last interest.
2.  **"You Might Like" (RL-DQN Personalization)**:
    *   **Logic**: Uses the Deep Q-Network (DQN) to score a verified candidate pool based on the user's *cumulative* multi-modal state (1536-dim).
    *   **Usecase**: Long-term interest modeling and discovery.

### Frontend Integration
`App.jsx` now features a sub-tab toggle within the Recommendations section, allowing the user to switch between these two viewpoints instantly without a page reload.

---

## 📈 2. Real-Time RL Inspection Dashboard

To ensure the RL agent is "learning" correctly, we have exposed its internal mathematical performance metrics directly to the interface.

### Loss Tracking & Polling
*   **Backend**: The `RLCollaborativeFilter` now maintains a rolling window of the last 100 training losses (`self.loss_history`).
*   **API**: A new `GET /rl_metrics` endpoint streams this loss data, current buffer size, and total training steps to the frontend.
*   **Frontend**: A new **"DQN Training Loss"** widget was added to the right-side profile panel. It features:
    *   **Numeric Display**: The exact loss value from the latest PyTorch backpropagation.
    *   **Live Sparkline**: An SVG-based real-time chart tracking the convergence of the model.
    *   **Interaction Feed**: A live log of reward signals (`▲ +reward` for clicks/carting, `▼ -reward` for skipping) linked directly to the training steps.

---

## 💾 3. Standardized JSON Persistence

The user state persistence layer was overhauled for robustness and future-proofing.

*   **Schema Enforcement**: User profiles now follow a strict JSON schema in `data/profiles/{id}.json`:
    *   `history`: A detailed log of every interaction (timestamp, ASIN, action type).
    *   `state`: Stores both the current text/visual embeddings and the raw preferences.
*   **Weights Persistence**: The DQN model weights (`.pt`) and the Replay Buffer (`.pkl`) are now automatically saved per-user. The buffer persistence ensures the model can continue training on historical data immediately after a server restart.

---

## 📁 4. Files Modified in This Cycle

| File | Type | Key Change |
|---|---|---|
| `src/rl_collaborative_filter.py` | MODIFIED | Added `loss_history` tracking and `pickle` buffer persistence. |
| `src/user_profile_manager.py` | MODIFIED | Enforced strict JSON schema and state embedding persistence. |
| `src/passive_recommendation_engine.py` | MODIFIED | Refactored `recommend_for_user` to return a partitioned dictionary. |
| `api.py` | MODIFIED | Added `/rl_metrics` endpoint; updated `/recommend` and `/interact`. |
| `frontend/src/App.jsx` | MODIFIED | Implemented dual-tab UI and the real-time Loss Chart widget. |

---

## ✅ 5. Final Verification Results

A verification cycle with **35 simulated interactions** was performed:
*   **RL Steps**: Incremented correctly from 0 to 35.
*   **Buffer Size**: Corrected to 35 transitions in memory.
*   **Loss Convergence**: The `loss_history` array populated and showed a fluctuating but downward-trending curve from ~1.2 to ~0.1 during the early training phase.
*   **Serialization**: Model weights and buffer correctly saved to `data/profiles/user_demo_01_dqn.pt` and `..._buffer.pkl`.

*Report generated on March 25, 2026.* 🚀
