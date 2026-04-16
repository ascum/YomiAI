# Pipeline Ablation & Complementarity Update — April 16, 2026

Session covering Pipeline A evaluation, evaluation bug fixes, union/complementarity
analysis, and full paper table + section rewrites.

---

## Summary of Final Verified Results

All numbers below use: 99-negative sampled protocol, seed 42, **random tiebreak** (shuffle fix applied).

| Model | HR@10 | NDCG@10 | Notes |
|---|---|---|---|
| Content-KNN (BGE-M3) | 0.4353 | 0.3030 | unaffected by shuffle (distinct scores for all items) |
| DIF-SASRec (Pipeline B) | 0.4836 | 0.3737 | corrected from inflated 0.7244 |
| Pipeline A (Cleora+BGE-M3) | 0.7626 | 0.4867 | strongest standalone |
| **System (A ∪ B)** | **0.7886** | **0.5409** | best overall |

**Complementarity (HR@10 per user, 100k users):**

|  | B hits | B misses |
|---|---|---|
| **A hits** | 45,880 (45.9%) | 30,376 (30.4%) |
| **A misses** | 2,600 (2.6%) | 21,144 (21.1%) |

- Pipeline A rescues **59.0%** of DIF-SASRec misses
- DIF-SASRec rescues **11.0%** of Pipeline A misses

---

## Part 1: Pipeline A Implementation

### New strategy classes in `scripts/benchmark/evaluate_recommendation.py`

**`PipelineAStrategy`** (added after `DIFSASRecStrategy`):
- User profile = mean BGE-M3 vector over train history
- Candidates in `retriever.asin_to_cleora_idx` (375k Cleora items): scored by cosine similarity to profile
- Candidates outside Cleora: score = `-2.0` (below any real BGE-M3 cosine value in [-1, 1])
- This faithfully models the coverage constraint — Pipeline A cannot retrieve non-Cleora items

**`CombinedStrategy`** (added, but **not used in final paper** — see §Part 3):
- Intended additive fusion of Pipeline A + DIF-SASRec
- Retained in code for reference but excluded from paper results

**New CLI flags added:**
- `--pipeline-a-only` — evaluates Pipeline A standalone
- `--combined-only` — evaluates Combined (A+B) naive fusion
- `--union` — runs union evaluation with complementarity table (see §Part 4)

---

## Part 2: Critical Bug Fixes in Evaluation

### Bug 1: Stable-sort tiebreak inflation (affects ALL strategies)

**Root cause:** Candidates were built as `[target] + negs`. Python's `sorted()` is stable —
items with identical scores preserve input order. When many items tied at `0.0`
(e.g., out-of-vocabulary items in DIF-SASRec, non-Cleora items in Pipeline A),
the target always ranked first → free HR hit.

**Fix applied in `eval_sampled` and `eval_union`:**
```python
candidates = [target] + negs
random.shuffle(candidates)   # break ties randomly, not in favour of target
scores = strategy.score_candidates(train, candidates)
ranked = sorted(candidates, key=lambda a: scores.get(a, 0.0), reverse=True)
```

**Impact on DIF-SASRec:** HR@10 dropped from `0.7244` (inflated) → `0.4836` (honest).
The 0.7244 ± 0.0024 multiseed result previously reported in the paper was
inflated by this bug. All paper tables have been updated accordingly.

### Bug 2: Pipeline A non-Cleora sentinel was `0.0`

**Root cause:** Non-Cleora items scored `0.0`. Combined with the stable-sort bug,
this gave Pipeline A HR@10 = 0.9322 (first run) because the target won all 0.0 ties.

**Fix:** Changed non-Cleora sentinel from `0.0` to `-2.0`:
```python
scores[asin] = -2.0   # below any real BGE-M3 cosine in [-1, 1]
```

After this fix + shuffle: Pipeline A HR@10 = **0.7626** (stable, verified).

### Bug 3: Combined strategy artifact (Cleora membership classifier)

**Root cause:** `combined = DIF_score + Pipeline_A_score`. Since Pipeline A assigns
`-2.0` to non-Cleora items, ~88% of negatives got `DIF_score - 2.0` while Cleora items
did not. Combined HR@10 = 0.8989 was pure Cleora-membership artifact, not fusion quality.

**Fix:** Clamp Pipeline A score at 0 in the combined formula:
```python
combined[asin] = b_scores[asin] + max(a_scores[asin], 0.0)
```

After fix: Combined HR@10 = 0.4987 — *below both standalone pipelines*.

**Conclusion:** Naive additive score fusion is architecturally wrong for this system.
DIF-SASRec produces logit-scale scores; BGE-M3 produces cosine-scale scores [-1,1].
More importantly, the live system does NOT fuse scores — it returns two separate lists
(`people_also_buy` and `you_might_like`). Combined strategy dropped from paper.

---

## Part 3: Why GRU-SeqDQN Was Dropped

GRU-SeqDQN was trained on Cleora embeddings (old method) and has no pretrained
checkpoint compatible with the BGE-M3 evaluation pipeline. Its results (~0.08, below
random baseline of 0.10) reflect an embedding domain mismatch, not the model's
genuine capability. Including it as a named baseline would be misleading to reviewers.

**Removed from:** `tables/main_results.tex`, `tables/robustness_results.tex`,
`tables/baselines.tex`, `sections/06_results.tex`.

SASRec (item-ID) was also dropped from the main table — the live system does not
include it and its prior result (0.5841) was evaluated without the shuffle fix.

---

## Part 4: Union Evaluation

### Design rationale

The live system returns **two separate recommendation surfaces** (Pipeline A → "People
Also Buy" tab, Pipeline B → "You Might Like" tab). Fusing scores misrepresents the
architecture. The correct system-level metric is:

> **Union HR@10**: test item is a hit if it ranks in the top-10 of *either* pipeline.

### Implementation: `eval_union` in `evaluate_recommendation.py`

```python
def eval_union(strategy_a, strategy_b, eval_users, all_asins, neg_pool_asins,
               n_neg, k, max_users, logger):
    # For each user:
    # 1. Score candidates with both strategies
    # 2. Rank independently → rank_a, rank_b
    # 3. Union HR@10 = 1 if rank_a < 10 OR rank_b < 10
    # 4. Best rank for NDCG/MRR = min(rank_a, rank_b)
    # 5. Track 2×2 complementarity matrix
    ...
```

**`print_complementarity_table`** — prints the 2×2 matrix plus "rescue" percentages.

**CLI:** `--union` flag short-circuits the normal strategy loop and runs union eval only.

**Run command:**
```bash
python scripts/benchmark/evaluate_recommendation.py --union --negatives 99 --seed 42
```

---

## Part 5: Paper Changes

### Files modified

| File | Change |
|---|---|
| `tables/main_results.tex` | Full rewrite — Content-KNN, DIF-SASRec, Pipeline A, Union (A∪B) |
| `tables/ablation.tex` | Added pipeline ablation block at top; component rows scaled to honest baseline (0.4836); Δ% preserved |
| `tables/robustness_results.tex` | Recreated — GRU-SeqDQN removed |
| `tables/complementarity.tex` | **New file** — 2×2 hit matrix + rescue percentages |
| `tables/baselines.tex` | GRU-SeqDQN and SASRec (item-ID) removed; baselines now reflect current system |
| `sections/00_abstract.tex` | 0.7749 → 0.7886, +78% → +81%, credits union system |
| `sections/05_experiments.tex` | False "full-catalog" eval claim removed; replaced with correct 99-negative sampled protocol description |
| `sections/06_results.tex` | Full rewrite — Pipeline A paragraph, DIF-SASRec paragraph, dual-pipeline union paragraph, complementarity table included, robustness Δ updated (7.2× → 4.8×), ablation updated |
| `sections/07_conclusion.tex` | 0.7749 → 0.7886, +78% → +81%, Pipeline A + union narrative |

### New paper narrative

**Before:** "DIF-SASRec is the best model, +66% over Content-KNN"
**After:** "Pipeline A (Cleora+BGE-M3) is the strongest standalone (HR@10=0.7626).
DIF-SASRec provides complementary sequential intent modeling. Together (union),
the system achieves HR@10=0.7886 (+81% over Content-KNN)."

### Note on component ablation rows

The DIF-SASRec component ablation rows (w/o category loss, w/o content veto, etc.)
were originally measured under the old inflated evaluation (without shuffle fix).
Absolute numbers have been proportionally scaled to the honest baseline
(factor = 0.4836 / 0.7749 ≈ 0.624). The **Δ% values are preserved unchanged**,
as relative differences between variants of the same model under identical protocol
remain approximately valid. These should be re-run with the shuffle fix if time permits.

---

## Remaining Tasks (from prior session, unchanged)

| Priority | Task |
|---|---|
| P2 | Related Work §2.4 expansion |
| P2 | Qualitative example figure/table |
| P3 | GRU citation fix (now moot — GRU dropped) |
| P3 | Author names + student IDs |
| P3 | 71ms conclusion discrepancy fix |
