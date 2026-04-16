# Error Bars Without Retraining — April 15, 2026

Replacing the multi-seed retrain plan (too slow at ~29h) with a faster approach
that uses the existing trained checkpoint.

---

## Why this is valid

The trained checkpoint is fixed. Running evaluation multiple times with different
random negative pools measures **evaluation variance** — how sensitive the HR@10
number is to *which* 99 distractors happen to be drawn. If variance is small,
the single-run 0.7749 is a stable point estimate. If variance is large, that's
also useful to know. Either way, a mean ± std is more honest than a bare number.

Additionally, running at `--negatives 999` (1000-item pool) provides a harder
robustness check that directly answers the reviewer question "why 99?".

---

## Execution order

### Step 1 — Harder negatives run (B)  ~30 min
```bash
python scripts/benchmark/evaluate_recommendation.py \
    --negatives 999 --dif-only --seed 42
```
Reports HR@10 / NDCG@10 at 10× harder evaluation. Shows the result holds.

### Step 2 — Eval-seed repeats (A)  ~60 min
```bash
python scripts/benchmark/multiseed_eval.py --seeds 42 123 456 789 2026
```
Runs 5 evaluations with the same checkpoint but different negative pools.
Outputs mean ± std. Saves to `evaluation/multiseed_results.json`.

### Step 3 — Paper text (C)  immediate
Update `sections/05_experiments.tex` to:
- Justify the 99-negative protocol by citation
- Note that DIF-SASRec was trained with 512 sampled negatives (harder than eval)
- Report eval-variance error bars in `tables/main_results.tex`

---

## Files changed

| File | Change |
|---|---|
| `scripts/benchmark/evaluate_recommendation.py` | `--seed`, `--dif-only`, `--pretrained-path` args |
| `scripts/benchmark/multiseed_eval.py` | New orchestrator — runs N seeds, computes mean ± std |
| `paper/sections/05_experiments.tex` | Protocol justification paragraph |
| `paper/tables/main_results.tex` | DIF-SASRec row updated with mean ± std |

---

## Status

| Task | Status |
|---|---|
| Write plan | ✅ Done |
| Modify evaluate_recommendation.py | ✅ Done |
| Create multiseed_eval.py | ✅ Done |
| Run --negatives 999 | In progress (background) |
| Run 5 eval seeds | In progress (background) |
| Update paper text | Pending results |
| Update main_results.tex | Pending results |
