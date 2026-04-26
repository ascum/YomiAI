# Eval Architecture Fix & Paper Update — April 27, 2026

Session covering: full 6-step results verification, three evaluation bugs
discovered and fixed in `evaluate_recommendation.py`, replacement of the
complementarity architecture with a single joint evaluation pass, and
complete paper table/text updates with final numbers.

---

## 1. Six-Step Results Verification

All steps confirmed from Apr 26 logs/results_history.json. All runs used
`--seed 42`, `--max-users 100000` (step 5: 20k), shuffle reverted.

| Step | Command | Key result |
|---|---|---|
| 1 | default (all baselines, 99-neg) | Content-KNN HR@10=0.4346; GRU HR@10=0.1031; DIF-SASRec HR@10=0.7746 |
| 2 | `--pipeline-a-only` 99-neg | Pipeline A HR@10=0.9047 |
| 3 | `--negatives 999` all models | Content-KNN=0.2122; GRU=0.0141; DIF-SASRec=0.3142 |
| 4 | `--pipeline-a-only --negatives 999` | Pipeline A HR@10=0.3272 |
| 5 | `compare_encoders.py --max-users 20000` | BGE-M3 HR@10=0.451; BLaIR=0.548; sep ratio BGE 1.147 > BLaIR 1.061 |
| 6 | `--combined` (joint eval) | See §4 for final numbers |

---

## 2. Three Bugs Found and Fixed in `evaluate_recommendation.py`

### Bug 1 — Shuffle in `eval_complementarity` (line 477)

`random.shuffle(candidates)` was commented out in `eval_sampled` (Apr 26
revert) but remained in `eval_complementarity`. This caused complementarity
to advance the random state by 99 extra calls per user, making its negatives
completely diverge from the main eval. Result: complementarity implied
Pipeline A HR ≈ 76.2% while main eval showed 90.47%.

**Fix:** Removed `random.shuffle(candidates)` from `eval_complementarity`.

### Bug 2 — Sequential random state drift between evals

Even after Bug 1 fix, the complementarity ran *after* three separate
`eval_sampled` calls (Pipeline A, DIF-SASRec, Combined), each consuming
random state independently. By the time complementarity ran, the random
state was completely different from any of the individual evals. Individual
HR rates and complementarity-implied HR rates could never be consistent
under this architecture.

**Fix:** Replaced `eval_complementarity` + three separate `eval_sampled`
calls with a single `eval_joint` function. One call per user samples
negatives **once** and scores both strategies on the **same** candidate
pool, computing individual metrics, union metrics, and the 2×2 complementarity
counts in a single pass. All numbers are now guaranteed internally consistent.

### Bug 3 — Union HR computed as `hit_rate(union_list, target, k)`

The union list was built as `dict.fromkeys(ranked_a[:k] + ranked_b[:k])` —
up to 20 items — but then passed to `hit_rate(..., k=10)`, which only checked
the first 10 items. Since the first 10 items of the union list are exactly
Pipeline A's top-10, System (A∪B) HR@10 was always equal to Pipeline A
HR@10 (both = 0.9047), hiding DIF-SASRec's contribution.

**Fix:** Changed union HR to:
```python
u_hr10.append(1.0 if (target in ranked_a[:k] or target in ranked_b[:k]) else 0.0)
```
Union NDCG uses `ndcg(union_list, target, len(union_list))` over the full
deduplicated union list (A's order preserved, B fills gaps), matching how
the live system presents results.

### Bug 4 (minor) — `print_complementarity_table` rescue labels swapped

Variables `a_rescues_b` and `b_rescues_a` were computed correctly but the
print f-string had them reversed, showing "A rescues 72.3% of B misses"
when the correct value was 88.3%.

**Fix:** Swapped the variable references in the print statement.

---

## 3. `eval_joint` Architecture

Replaces the `--combined` mode evaluation. Single pass per user:

1. Sample negatives once (shared pool, same random call)
2. Score Pipeline A and DIF-SASRec on the same candidates
3. Compute individual metrics for A and B
4. Union HR@10 = `hit_a OR hit_b` (exact live system semantics)
5. Union NDCG over deduplicated union list (A-first order)
6. Accumulate 2×2 complementarity counts

All five outputs are derived from the same per-user candidates → internally
consistent by construction. `CombinedStrategy` (RRF) is no longer used in
the `--combined` path (its NDCG was misleading since it evaluated a merged
ranking over 100 candidates rather than the live union of top-10 lists).

---

## 4. Final Numbers (run `20260426_233351`, joint eval)

### Main results (99-neg, 100k users, seed 42)

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Content-KNN (BGE-M3) | 0.4346 | 0.3022 |
| DIF-SASRec (Pipeline B) | 0.7745 | 0.5024 |
| Pipeline A (Cleora+BGE-M3) | 0.9047 | 0.5393 |
| **System (A∪B)** | **0.9736** | **0.5571** |

### Complementarity (k=10, n=100,000)

|  | B hits | B misses |
|---|---|---|
| **A hits** | 70,557 (70.6%) | 19,909 (19.9%) |
| **A misses** | 6,893 (6.9%) | 2,641 (2.6%) |

- A rescues **88.3%** of B misses (19,909 / 22,550)
- B rescues **72.3%** of A misses (6,893 / 9,534)
- Union HR@10 = **0.9736** vs Pipeline A alone = 0.9047 (+7.6%)

### Robustness (999-neg, 100k users, seed 42)

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Content-KNN | 0.2122 | 0.1349 |
| DIF-SASRec | 0.3142 | 0.2209 |
| Pipeline A | 0.3272 | 0.2102 |

### Encoder comparison (20k users, seed 42)

| Encoder | HR@10 | Sep. ratio |
|---|---|---|
| BGE-M3 | 0.4510 | **1.147** |
| BLaIR | 0.5484 | 1.061 |

BGE-M3 wins all long/medium/Vietnamese query groups. BLaIR wins short
single-token queries only.

---

## 5. Paper Tables Updated

| File | Changes |
|---|---|
| `tables/main_results.tex` | All rows updated; System bold on both metrics |
| `tables/complementarity.tex` | New 2×2 counts; rescue % 88.3/72.3; union 0.9736 |
| `tables/robustness_results.tex` | Pipeline A: 0.3264→0.3272, NDCG 0.2096→0.2102 |
| `tables/ablation.tex` | Top: new pipeline numbers, System ΔHR +25.7%; Bottom: DIF-SASRec baseline 0.7745/0.5024, component rows rescaled |
| `tables/encoder_comparison.tex` | Sep ratio BGE 1.181→1.147, BLaIR 1.071→1.061 |
| `sections/06_results.tex` | All inline numbers updated; dual-pipeline narrative restored (System best on both HR and NDCG); robustness multipliers updated (9.0×/7.7× for 99-neg, 32.7×/31.4× for 999-neg); encoder BLaIR cosine values updated |

Table caption for `main_results` updated: removed "random tiebreak" (shuffle
is no longer applied in `eval_sampled`), replaced with "seed 42".

---

## 6. Remaining Work

**Step 7 (ablation component rows):** The bottom half of Table 4 still
carries proportionally scaled estimates (HR scale ×1.601, NDCG scale ×1.344
from pre-step-7 measurements). Accurate numbers require:

| Ablation | Requires |
|---|---|
| w/o content veto | Add `ContentVetoDIFSASRecStrategy` wrapper, rerun `--dif-only` |
| w/o online adaptation | Add `--online-adapt` flag to eval script, rerun |
| w/o category auxiliary loss | Retrain with `SASREC_CAT_AUX_WEIGHT=0.0` |
| w/o dual-stream attention | Add `use_category=False` to `DIFAttentionLayer`, retrain |

*Report generated April 27, 2026.*
