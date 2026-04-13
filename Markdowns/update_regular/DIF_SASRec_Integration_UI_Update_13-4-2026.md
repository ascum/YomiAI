# DIF-SASRec Integration & UI Cleanup — April 13, 2026

This document covers the changes made after completing the 100k DIF-SASRec training run,
confirming the model loads correctly on startup, and cleaning up all remaining RL/DQN
references from the frontend and API layer.

---

## Context

Following the Apr 12 PM training run (HR@10 = 0.7749, NDCG@10 = 0.5029), the pretrained
model was confirmed to load correctly on API startup via `PassiveRecommendationEngine`
→ `DIFSASRecAgent`. The "You Might Like" pipeline is now fully live.

However, the frontend still displayed RL/DQN terminology from the previous GRU-SeqDQN
architecture throughout the Profile tab and Recommendations tab. This update removes all
those references and aligns the UI with the actual running system.

---

## Changes

### `frontend/src/App.jsx`

| Location | Before | After |
| :--- | :--- | :--- |
| Header stat label | `RL Steps` | `Train Steps` |
| Recs tab subtitle | `Retrieval + RL-DQN Multi-mode` | `Retrieval + DIF-SASRec Multi-mode` |
| "You Might Like" sub-tab description | `RL-DQN Personalized` | `DIF-SASRec Personalized` |
| Profile bento card title | `RL Feed` | `Train Feed` |
| Interaction feed labels | `▲ +reward / ▼ −reward` | `▲ trained / ▼ skipped` |
| Loss card title | `DQN Loss` | `SASRec Loss` |
| Loss card subtitle | `{buffer_size}/2000 transitions` | `{step} online steps` |
| Loss empty state | `Interact to stream loss data` | `Interact to see loss converge` |
| History tab empty state | `train the DQN` | `train DIF-SASRec` |
| Mock recs layer tag | `RL-DQN` | `DIF-SASRec` |
| `rlMetrics` initial state | `{ buffer_size: 0, ... }` | `{ arch: "", ... }` |

**Why `buffer_size` had to go:** The old DQN used a replay buffer (capped at 2000
transitions) which the frontend displayed. DIF-SASRec has no replay buffer — online
training is a direct single-step gradient update on each click. The `/rl_metrics`
endpoint now returns `{loss_history, step, arch}` with no `buffer_size` field, so
displaying it would have rendered `undefined/2000 transitions`.

### `frontend/src/components/features/profile/ProfileRadar.jsx`

- Bar label `RL Fit` → `Model Fit`

### `app/api/routes/interact.py`

- Response key `rl_loss` → `sasrec_loss` to match the actual model in use.

### `app/api/routes/recommend.py`

- Docstring updated: `Cleora → Veto → RL-DQN` → `Cleora → Veto → DIF-SASRec`

### `scripts/setup_dif_sasrec.py`

- Default `--min-clicks`: 3 → **6**
- Default `--max-users`: 200,000 → **100,000**

These defaults now match the configuration of the best-performing run (Apr 12 PM,
HR@10 = 0.7749). Previously the script defaulted to 200k users / min 3 clicks, which
was the configuration that caused the Apr 12 AM regression (avg 21 clicks per user,
model underfit). Running `python scripts/setup_dif_sasrec.py` without flags now
reproduces the correct setup.

---

## How the live system works (confirmed)

**Startup:** `lifespan.py` builds `PassiveRecommendationEngine`, which initialises
`DIFSASRecAgent` and loads `data/dif_sasrec_pretrained.pt` (12.4M params) onto GPU.
The model stays resident for the process lifetime.

**`GET /recommend`:** Loads per-user personal weights if they exist →
HNSW KNN (200 candidates) → content veto → DIF-SASRec forward pass (GPU, no-grad) →
sorted top-k returned as "You Might Like".

**`POST /interact` (click/cart):** Captures click sequence before profile update →
one online `train_step` (single gradient step, no AMP, FAISS-reconstructed negatives)
→ saves per-user checkpoint to `data/profiles/<user_id>_dif_sasrec.pt`.

**Note on concurrency:** The agent model is shared in memory. `load_personal_weights`
and `save_personal_weights` swap weights in-place per request. Safe for single-user
demo; will race under concurrent multi-user load. Addressed separately.

---

## Metric explanations (documented for reference)

**Engagement Signals radar** — all computed from the local session `interactions` array:

| Bar | Formula | Meaning |
| :--- | :--- | :--- |
| CTR | `(clicks+carts) / total` | Positive engagement ratio |
| Depth | `min(total / 20, 1)` | Session engagement depth, saturates at 20 interactions |
| Model Fit | `min(clicks / 10, 1) * 0.8 + 0.1` | Proxy for positive training signal received; floor 10%, ceiling 90% |
| Diversity | `unique IDs / total` | Fraction of interactions that were unique books (not content diversity) |

**SASRec Loss sparkline** — the `loss_history` loaded from the pretrained checkpoint
contains the pretraining plateau (~5.34). The first online click appends a much lower
loss (~0.5–1.0) because: (1) online negatives are randomly sampled from 3M+ books
(easy negatives), whereas pretraining used a curated 50k pool; (2) the pretrained model
already ranks semantically similar items well; (3) the chart plots both regimes
back-to-back, making the drop look dramatic. It is not a sign of instability.

---

*Report generated April 13, 2026.*
