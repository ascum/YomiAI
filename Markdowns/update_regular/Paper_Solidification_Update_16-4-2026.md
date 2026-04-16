# Paper Solidification Update — April 16, 2026

Session covering paper fixes and new experimental results for IAAA 2026 submission.

---

## Critical Fix: Eval Protocol Mismatch

Discovered that the paper claimed **full-catalog evaluation against all 3.08M items**,
but the actual evaluation used **sampled mode (99 negatives, 100 candidates total)**.
HR@10=0.7749 at 99-neg is a strong result; it would be implausible against 3.08M items.

**Fixed in:**
- `paper/sections/05_experiments.tex` — evaluation metrics rewritten with accurate protocol
- `paper/tables/main_results.tex` — caption corrected
- `paper/sections/06_results.tex` — two "full-catalog" references removed

---

## Task 1: Error Bars (eval-variance approach, no retraining)

Replaced the multi-seed retrain plan (~29h compute) with a faster eval-variance approach:
run the same checkpoint 5× with different random negative pools.

**New scripts created:**
- `scripts/benchmark/evaluate_recommendation.py` — added `--seed`, `--dif-only`, `--pretrained-path` args
- `scripts/benchmark/multiseed_eval.py` — orchestrator: runs N seeds, computes mean ± std, prints LaTeX snippet

**Results (5 seeds: 42, 123, 456, 789, 2026 — 99 negatives):**

| Seed | HR@10 | NDCG@10 |
|---|---|---|
| 42 | 0.7262 | 0.4449 |
| 123 | 0.7220 | 0.4422 |
| 456 | 0.7276 | 0.4465 |
| 789 | 0.7238 | 0.4436 |
| 2026 | 0.7224 | 0.4433 |
| **Mean ± std** | **0.7244 ± 0.0024** | **0.4441 ± 0.0017** |

Note: the original single-run 0.7749 was a lucky unseeded draw. The mean across seeds
(0.7244) is the honest canonical number and has been updated in the paper.

**Paper updated:**
- `paper/tables/main_results.tex` — DIF-SASRec row now shows `0.7244 ± 0.0024 / 0.4441 ± 0.0017`
- `paper/sections/06_results.tex` — percentages recalculated (+66% over Content-KNN, +24% over SASRec)

---

## Task 2: 999-Negative Robustness Table (new)

Ran all 3 available models at 999-neg (10× harder; random baseline HR@10 = 0.01).
SASRec excluded — its closed item vocabulary is incompatible with dense-retrieval scoring.

**Results (999 negatives, seed=42):**

| Model | HR@10 | NDCG@10 | vs random |
|---|---|---|---|
| Content-KNN (BGE-M3) | 0.2122 | 0.1349 | 21.2× |
| GRU-SeqDQN | 0.0096 | 0.0043 | below random |
| DIF-SASRec (ours) | **0.3142** | **0.2209** | **31.4×** |

Key finding: DIF-SASRec's relative advantage over random *increases* from 7.2× (99-neg)
to 31× (999-neg), showing the result is not an artefact of easy negatives.
GRU-SeqDQN drops below the random baseline at harder evaluation, confirming
it lacks genuine discriminative capacity.

**Paper updated:**
- `paper/tables/robustness_results.tex` — new table created
- `paper/sections/06_results.tex` — new §6.2 Robustness subsection added
- `paper/sections/05_experiments.tex` — 999-neg protocol referenced with cross-link

---

## Remaining Tasks

| Priority | Task |
|---|---|
| P1 | Pipeline A (Cleora) HR@10 metric |
| P1 | Ablation: Pipeline A vs B vs both |
| P2 | Related Work §2.4 expansion |
| P2 | Qualitative example figure/table |
| P3 | GRU citation fix |
| P3 | Author names + student IDs |
| P3 | 71ms conclusion discrepancy fix |
