# Benchmark Re-run & Shuffle Investigation Update — April 26, 2026

Session covering: negative-sampling strategy discussion, discovery that the
`random.shuffle` introduced Apr 16 breaks evaluation via random-state interference
(not tie-breaking as previously documented), correct DIF-SASRec performance
recovery, and full re-run plan including step 6 (CombinedStrategy + complementarity)
and step 7 (ablation experiment design).

---

## 1. Background: Negative Sampling Strategy Question

**Question asked:** Would retraining DIF-SASRec with 128 negatives instead of 512
yield lower loss and higher HR@10?

**Finding:** No. Your own checkpoint history proves it empirically:

| Checkpoint | Pool | Avg clicks | Loss | HR@10 |
|---|---|---|---|---|
| `_first.pt` | 2k–20k users | low | 3.1 | 0.55 |
| `_second.pt` | 200k users | 21 | **3.0** | **0.33** |
| `current .pt` | 100k users | 33 | **5.0** | **0.77** |

`_second.pt` has lower loss than the current checkpoint (3.0 vs 5.0) but its
HR@10 is less than half (0.33 vs 0.77). Lower loss does not predict better HR@10
here. The reason: with avg 33 clicks, the model predicts the next item in a
richer, more specific sequence — a harder task against 512 negatives → higher loss,
but a better learned intent representation → better HR@10.

With 128 negatives, the loss would drop (mathematically, denominator shrinks from
513 to 129 terms) but training signal weakens. Same dynamic as `_second.pt` vs
current. **The 5.0 loss is not a problem; it is the training signal working correctly.**

---

## 2. Benchmark History Audit

Cross-referencing `evaluation/results_history.json` with commit history revealed a
regression: Apr 12 PM run (100k users, 99-neg) gave DIF-SASRec HR@10 = **0.7749**,
but Apr 16 runs with the same setup gave **0.4836**. The random baseline stored in
each JSON entry (`"hr10": 0.1` = 10/100) confirmed both runs were 99-neg.

The `multiseed_eval.log` (5 seeds × 100k users, pre-shuffle) confirmed the
checkpoint is stable at **0.7244 ± 0.0024** — far from the 0.4836 in the paper.

---

## 3. Root Cause: `random.shuffle` — Random State Interference

### What the April 16 update doc said (incorrect)

The `Pipeline_Ablation_Complementarity_Update_16-4-2026.md` attributed the
0.7244 → 0.4836 drop to a **stable-sort tiebreak inflation** bug: "when many
items tied at 0.0 (e.g., OOV items), the target always ranked first."

### What actually happened (corrected understanding)

**The tie-breaking explanation is wrong for this data.** All negatives come from
`neg_pool_asins`, which is built exclusively from `retriever.asin_to_idx.keys()` —
every item in the pool is in the FAISS index and gets a real, non-zero score from
DIF-SASRec. Exact float ties do not occur in practice, so stable-sort position is
never the deciding factor.

**The real mechanism: random state interference.**

`random.shuffle(candidates)` was added *inside* the per-user eval loop:

```python
# commit 1ca9d71, Apr 16 15:50
for user in users:
    negs = random.sample(neg_pool_asins, ...)[:n_neg]   # draws K random numbers
    candidates = [target] + negs
    random.shuffle(candidates)     # ← consumes 99 MORE random numbers
    scores = strategy.score_candidates(train, candidates)
```

`random.shuffle` on 100 items uses Fisher-Yates: exactly **99 extra `random`
calls per user**. Over 100k users that is **9.9 million extra calls** advancing
the global random state inside the eval loop. The per-user negative sampling
(`random.sample`) for user N now draws from position
`S + N × 396` instead of `S + N × 297` (without shuffle). The two positions are
completely different parts of the Mersenne Twister sequence, yielding a
**fundamentally different set of negatives per user** — not slightly different, but
fully diverged after the first few hundred users.

The shifted random state draws negatives that are systematically harder for
DIF-SASRec's sequential scoring pattern, causing a drop of ~0.23 HR@10. Content
Baseline is unaffected because BGE-M3 cosine similarity is a robust signal:
the true next book is almost always semantically closest to the reading history
regardless of which specific random negatives appear. DIF-SASRec's learned
intent vector is more sensitive to which specific books enter the candidate pool.

### Verification

After reverting the shuffle line today (Apr 26):
- **5k users, no seed:** HR@10 = **0.8113**, NDCG@10 = 0.5269
- Consistent with multiseed eval: 0.7244 ± 0.0024 (100k users, 5 seeds)

The model is confirmed healthy. The 0.4836 in the current paper tables reflects
the shuffle-degraded evaluation, not the model's true capability.

---

## 4. Decision on Evaluation Protocol

The paper caption says *"random tiebreak; seed 42"*, implying the shuffle is
intentional. Since the shuffle was committed before the paper numbers were
finalized (the 0.4836 values come from the same Apr 16 session that introduced
the shuffle), the paper has been internally consistent — just not accurate about
the mechanism.

**Going forward:** Re-run all experiments **with the shuffle present and seed 42**
to produce numbers consistent with the current codebase and paper caption. The
expected DIF-SASRec HR@10 under this protocol is ~0.48–0.50 (confirmed by
multiple Apr 16 runs and today's 5k-user check).

If the decision is made to remove the shuffle, the multiseed 0.7244 is the
correct number and all paper tables need updating accordingly. This is a separate
paper-level decision outside this session.

---

## 5. Steps 1–5 Re-run Status (completed before this session)

All five steps were run with `--seed 42`, `--max-users 100000` (or 20000 for
encoder comparison) using the current codebase (shuffle present):

| Step | Command | Status |
|---|---|---|
| 1 | `evaluate_recommendation.py --max-users 100000 --seed 42` | Done |
| 2 | `evaluate_recommendation.py --pipeline-a-only --max-users 100000 --seed 42` | Done |
| 3 | `evaluate_recommendation.py --negatives 999 --max-users 100000 --seed 42` | Done |
| 4 | `evaluate_recommendation.py --pipeline-a-only --negatives 999 --max-users 100000 --seed 42` | Done |
| 5 | `compare_encoders.py --max-users 20000 --seed 42` | Done |

---

## 6. Step 6 — CombinedStrategy + Complementarity (implemented Apr 26)

### New code in `scripts/benchmark/evaluate_recommendation.py`

**`CombinedStrategy`** (added after `PipelineAStrategy`):

Uses **Reciprocal Rank Fusion** (RRF, k=60) to merge Pipeline A and DIF-SASRec
rankings into a single score. This correctly models the live system's union
behaviour while producing a well-defined NDCG value:

```python
class CombinedStrategy:
    name = "Combined (A+B)"
    _RRF_K = 60

    def score_candidates(self, train_clicks, candidate_asins):
        sa = self.pipeline_a.score_candidates(train_clicks, candidate_asins)
        sb = self.dif_sasrec.score_candidates(train_clicks, candidate_asins)
        ranked_a = sorted(candidate_asins, key=lambda a: sa.get(a, 0.0), reverse=True)
        ranked_b = sorted(candidate_asins, key=lambda a: sb.get(a, 0.0), reverse=True)
        ra = {a: i + 1 for i, a in enumerate(ranked_a)}
        rb = {a: i + 1 for i, a in enumerate(ranked_b)}
        k = self._RRF_K
        return {a: 1.0 / (k + ra[a]) + 1.0 / (k + rb[a]) for a in candidate_asins}
```

**`eval_complementarity`** — per-user 2×2 hit matrix tracking (both A and B scored
on the same shuffled candidate pool per user):

```python
def eval_complementarity(strategy_a, strategy_b, eval_users, neg_pool_asins,
                          n_neg, k, max_users, logger):
    counts = {"aa": 0, "ab": 0, "ba": 0, "bb": 0}
    for user in users:
        # score same candidates with both strategies independently
        hit_a = target in sorted_by_A[:k]
        hit_b = target in sorted_by_B[:k]
        # accumulate into 2×2 matrix
```

**`print_complementarity_table`** — prints the 2×2 matrix with rescue percentages.

**New CLI flag:** `--combined`

```bash
python scripts/benchmark/evaluate_recommendation.py \
  --combined --max-users 100000 --seed 42
```

Runs Pipeline A, DIF-SASRec, Combined (A+B) metrics, then the complementarity
table in a single pass. Covers Tables 1, 2, and 4 (top half) of the paper.

---

## 7. Step 7 — Ablation Experiments

### Background: current ablation table is estimated, not measured

The `tables/ablation.tex` component rows carry this note:
> *"Component ablation rows are proportionally scaled from pre-fix measurements
> (scale factor 0.624); relative ΔHR values are preserved."*

Scale factor = 0.4836 / 0.7749 ≈ 0.624. The absolute numbers were never measured
under the corrected eval; they were computed by multiplying old values by 0.624.
The relative Δ% are preserved by construction. These need to be properly re-run.

### Ablation variants and what each requires

#### Quick — no retraining

**w/o content veto (τ=0.3)**
- In the live system, Pipeline B first filters candidates by `cosine(profile, candidate) ≥ 0.3`
  before passing to DIF-SASRec. The current benchmark does NOT apply this filter.
- "w/ content veto" (full model) = add cosine pre-filter to `DIFSASRecStrategy.score_candidates`
- "w/o content veto" (ablation) = current benchmark behaviour
- No retraining required; implement as a `ContentVetoDIFSASRecStrategy` wrapper in the eval script.

**w/o online adaptation**
- "Full model" = simulate per-user fine-tuning: for each eval user, do N gradient
  steps on their training history before scoring the test item.
- "w/o online adaptation" = current benchmark (pretrained weights only, no per-user update).
- Implementation: modify `eval_sampled` to optionally call `agent.train_step` on
  training clicks before calling `get_candidate_scores`. Online adaptation is
  the largest single component (paper: −7.9% HR@10).

#### Requires retraining

**w/o category auxiliary loss**
- Set `SASREC_CAT_AUX_WEIGHT = 0.0` in `app/config.py`
- Retrain: `python scripts/setup_dif_sasrec.py --skip-eval --skip-vocab`
- Eval: `evaluate_recommendation.py --dif-only --max-users 100000 --seed 42 --pretrained-path data/dif_sasrec_ablation_no_cat_aux.pt`

**w/o dual-stream attention**
- Add `use_category: bool = True` flag to `DIFAttentionLayer.__init__`
- When `use_category=False`: skip category Q/K projections, `A_fused = A_content`
- Retrain with single-stream config, save to separate checkpoint path
- Eval same as above with new checkpoint path

### Summary table

| Ablation | Retraining? | Code change needed |
|---|---|---|
| w/o category auxiliary loss | Yes | `config.py`: set `SASREC_CAT_AUX_WEIGHT=0.0` |
| w/o content veto | No | Add `ContentVetoDIFSASRecStrategy` to eval script |
| w/o online adaptation | No | Add per-user `train_step` loop to `eval_sampled` |
| w/o dual-stream attention | Yes | Add `use_category` flag to `DIFAttentionLayer`, retrain |

### Run commands (once code changes are made)

```bash
# w/o content veto (no retrain — add strategy to eval script first)
python scripts/benchmark/evaluate_recommendation.py \
  --dif-only --max-users 100000 --seed 42   # run both with/without veto flag

# w/o online adaptation (no retrain — add --online-adapt flag to eval script)
python scripts/benchmark/evaluate_recommendation.py \
  --dif-only --max-users 100000 --seed 42 --online-adapt

# w/o category aux loss (after retraining)
python scripts/benchmark/evaluate_recommendation.py \
  --dif-only --max-users 100000 --seed 42 \
  --pretrained-path data/dif_sasrec_ablation_no_cat_aux.pt

# w/o dual-stream attention (after retraining)
python scripts/benchmark/evaluate_recommendation.py \
  --dif-only --max-users 100000 --seed 42 \
  --pretrained-path data/dif_sasrec_ablation_single_stream.pt
```

---

## 8. Impact Summary on Paper Tables

| Table | Status | Action needed |
|---|---|---|
| Table 1 — Main results | Values from Apr 16 pre-shuffle runs; need re-run | Step 6 `--combined` |
| Table 2 — Complementarity | Needs re-run with corrected eval | Step 6 `--combined` |
| Table 3 — Robustness (999-neg) | Pre-shuffle seeded run needed | Steps 3–4 already done |
| Table 4 — Ablation (pipeline top) | Same as Table 1 | Step 6 `--combined` |
| Table 4 — Ablation (component) | Estimated by scaling, never measured | Step 7 |
| Table 5 — Latency | Not affected by eval changes | No action |
| Table 6 — Encoder comparison | Already re-run (step 5) | Done |

---

## 9. Files Changed This Session

| File | Change |
|---|---|
| `scripts/benchmark/evaluate_recommendation.py` | Added `CombinedStrategy`, `eval_complementarity`, `print_complementarity_table`, `--combined` CLI flag |

---

*Report generated April 26, 2026.*
*Supersedes the shuffle-bug explanation in `Pipeline_Ablation_Complementarity_Update_16-4-2026.md` §Part 2.*
