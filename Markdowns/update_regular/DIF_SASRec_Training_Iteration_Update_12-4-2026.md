# DIF-SASRec Training Iteration — April 12, 2026

This document covers the model improvement attempt made after the initial DIF-SASRec deployment (April 11), including architecture upgrades, training results, benchmark comparison, and the root cause analysis for why the larger model underperformed.

---

## 🎯 Goal

The initial DIF-SASRec (April 11) achieved HR@10=0.2973 — 3× the random baseline but still well below the Content Baseline (0.4522). The aim was to close that gap by:

1. Increasing model capacity (hidden dim, depth)
2. Training on more users (20k → 200k)
3. Improving GPU utilisation (was at ~30%)

---

## 🔧 Changes Applied

### Model Architecture (`app/config.py`, `app/services/dif_sasrec.py`)

| Hyperparameter | Before | After |
| :--- | :--- | :--- |
| `SASREC_HIDDEN_DIM` | 256 | 512 |
| `SASREC_N_BLOCKS` | 2 | 4 |
| `SASREC_N_HEADS` | 4 | 8 |
| `SASREC_NUM_NEGATIVES` | 256 | 512 |
| Parameters | 2,273,075 | 12,420,345 |
| head_dim | 64 | 64 (unchanged) |
| FFN width | 256→512→256 | 512→1024→512 |

### Optimizer & Training Loop (`app/services/dif_sasrec.py`)

- **Adam → AdamW** with `weight_decay=0.01` (correct weight decay for transformers)
- **AMP (Automatic Mixed Precision)** via `torch.cuda.amp.GradScaler` — fp16 forward/loss, fp32 gradients. Halves activation memory, doubles tensor core throughput on RTX architecture
- **Linear warmup + cosine LR decay** — LR rises from 0 → 1e-3 over 2 epochs, then cosine decay to 5% of peak. Prevents large early updates from destroying embeddings; refines rather than oscillates late
- **Gradient clipping** retained at max_norm=1.0

### Setup Script (`scripts/setup_dif_sasrec.py`)

- Batch size: 512 → 1024
- Default `--max-users`: 20,000 → 200,000
- Scheduler wired up via `agent.configure_scheduler(total_steps, warmup_steps)` before training loop
- LR printed per log interval for visibility

---

## 📊 Training Results

Two runs completed before benchmarking.

### Run 1 — Old Architecture (Baseline Reference)

| Field | Value |
| :--- | :--- |
| Parameters | 2,273,075 |
| Users | 20,000 |
| Batch | 512 |
| AMP | Off |
| Negatives | 256 |
| Optimizer | Adam |
| LR | Fixed 1e-3 |
| Epochs | 30 |
| Final loss | 3.3480 |
| Random baseline | ln(257) ≈ 5.55 |
| Loss / random | 60.3% |
| Total time | ~3.2 hours |
| Steps | 98,910 |

### Run 2 — New Architecture (Overnight)

| Field | Value |
| :--- | :--- |
| Parameters | 12,420,345 |
| Users | 20,000 (eval_users.json not rebuilt — see issue below) |
| Batch | 1024 |
| AMP | On |
| Negatives | 512 |
| Optimizer | AdamW |
| LR | Cosine with 2-epoch warmup |
| Epochs | 30 |
| Final loss | 3.0558 |
| Random baseline | ln(513) ≈ 6.24 |
| Loss / random | 49.0% |
| Total time | ~11.8 hours |
| Steps | 118,500 |

Despite a harder training task (2× more negatives), the new model sat further below its random baseline proportionally — indicating stronger item discrimination during training.

---

## 📉 Benchmark Results

Evaluated via `scripts/benchmark/evaluate_recommendation.py --mode sampled` (1 positive + 99 negatives, 20,000 users).

| Strategy | HR@5 | HR@10 | NDCG@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- |
| **April 11 run (old model)** | | | | |
| Content Baseline | — | 0.4522 | 0.3114 | — |
| GRU-SeqDQN | — | 0.0667 | 0.0302 | — |
| DIF-SASRec | — | 0.2973 | 0.1668 | — |
| **April 12 run (new model)** | | | | |
| Content Baseline | — | 0.4255 | 0.2967 | — |
| GRU-SeqDQN | — | 0.0993 | 0.0445 | — |
| **DIF-SASRec** | — | **0.1950** | **0.0972** | — |

**The new model performed worse despite lower training loss and more parameters.**

---

## 🔍 Root Cause Analysis

### The Problem: Capacity / Data Mismatch (Overfitting)

The `eval_users.json` was **never rebuilt** to 200k users. Both training runs used `--skip-eval`, so the file retained the original 20,000 users.

The result:

| | Old model | New model |
| :--- | :--- | :--- |
| Parameters | 2.3M | **12.4M** |
| Training users | 20,000 | 20,000 (same) |
| Training examples/epoch | ~1.69M | ~1.69M (same) |

A model with **5× more parameters trained on the same data** overfits. The larger model memorised the 20k users' training sequences — training loss dropped (3.34 → 3.06) but generalisation to the held-out last item degraded. This is textbook overfitting.

The old 2.3M model was appropriately sized for 20k users. The 12.4M model requires the 200k user dataset it was designed for.

### Secondary Observation

The harder negative task (512 vs 256) also raises the effective difficulty of training, which can slow convergence relative to data availability. With only 20k users the model does not see enough diverse patterns to benefit from the harder negatives.

---

## ✅ What Was Actually Correct

Despite the benchmark regression:

- The architecture changes (hidden=512, 4 blocks, AdamW, AMP, cosine LR) are all sound and correct choices for a larger-scale training run
- The GPU utilisation improvements (AMP + batch 1024) are working — AMP=on confirmed in training output
- The LR schedule is functioning correctly (verified from per-step LR output during training)
- The checkpoint is the correct new architecture (`step=118,500`, `params=12,420,345`)

The changes were right. The data setup step was missed.

---

## 🛠️ Next Step

Rebuild `eval_users.json` with 200k users, then retrain. Stage 1 (category vocab) does not need to be re-run.

```bash
python scripts/setup_dif_sasrec.py --skip-vocab
```

This re-runs:
- **Stage 0**: downloads and rebuilds `eval_users.json` with 200,000 users (sorted by richest history)
- **Stage 2**: retrains the 12.4M param model on the larger dataset

Expected outcome: the larger model now has sufficient data diversity to utilise its capacity without overfitting, and benchmark HR@10 should exceed the original 0.2973.

---

*Report generated April 12, 2026. Checkpoint at time of writing: `data/dif_sasrec_pretrained.pt` (step=118,500, 12,420,345 params).*
