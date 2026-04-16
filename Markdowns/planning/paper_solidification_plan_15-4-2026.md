# Paper Solidification Plan — April 15, 2026

Prepared ahead of IAAA 2026 submission.
Based on full review of `paper/` and all progress updates in `Markdowns/update_regular/`.

---

## High-Priority (must do before submission)

### 1. Multi-seed DIF-SASRec runs for error bars
HR@10=0.7749 is currently a single-run number. Reviewers will question reproducibility.

**Steps:**
1. Re-run DIF-SASRec training with 3–5 different random seeds (same hyperparameters)
2. Report **mean ± std** in `tables/main_results.tex`, e.g. `0.7749 ± 0.003`

**Effort:** ~1 day compute (can run in background)

---

### 2. Pipeline A (Cleora) evaluation metric
The paper is titled "Dual-Mode Multimodal" but only has metrics for Pipeline B (DIF-SASRec sequential). Pipeline A ("People Also Buy") has no quantitative evaluation.

**Steps:**
1. For users in the test split, check whether Cleora's top-10 "People Also Buy" candidates contain the held-out item
2. Report **HR@10 / Recall@10 for Pipeline A alone**
3. Add a row or column to `tables/main_results.tex` (or a new table)

**Effort:** 1–2 hours coding + run

---

### 3. Ablation: Pipeline A vs Pipeline B vs combined
The current ablation (`tables/ablation.tex`) only ablates components *within* DIF-SASRec. It does not show the value of combining both pipelines.

**Steps:**
1. Evaluate Pipeline A only (Cleora recommendations for the test split)
2. Evaluate Pipeline B only (DIF-SASRec recommendations)
3. Evaluate combined (current full system)
4. Add 3 rows to the ablation table comparing the pipelines

This directly supports the "dual complementary pipelines" claim in Section 1.

**Effort:** ~2 hours

---

## Medium Priority

### 4. Expand Related Work §2.4 (Hybrid Systems)
**File:** `paper/sections/02_related_work.tex:27–31`

Currently one sentence. Add 2–3 citations on multi-modal recommenders (e.g., VBPR, MMSSL, or two-tower models) and explain how they differ from the dual-mode active/passive framing used here.

**Effort:** 1 hour

---

### 5. Add qualitative example (figure or mini-table)
A single worked example makes the system tangible for reviewers:
- Show a user's last 5 interactions
- Show what Pipeline A returns (People Also Buy)
- Show what Pipeline B returns (You Might Like)
- Show what the held-out item actually was

Can be a small table in the Results section or a compact figure.

**Effort:** 1–2 hours

---

## Low-Risk Cleanup

### 6. Fix GRU-SeqDQN citation placeholder
**File:** `paper/tables/baselines.tex:4`

`% TODO: replace with correct GRU-SeqDQN citation` is still open.

### 7. Fill in author names and student IDs
**File:** `paper/main.tex:32–39`

`[Author Name]`, `[Co-Author Name]`, `[student-id]` are all still placeholders.

### 8. Fix latency discrepancy in conclusion
**File:** `paper/sections/07_conclusion.tex:13`

The conclusion states "71ms" but the latency table (`tables/latency.tex`) shows warm P50 = 17ms internal / 34ms E2E. Reconcile with the actual measured number.

---

## Priority Summary

| Priority | Task | Effort | Status |
|---|---|---|---|
| P0 | Architecture diagram | 2–4h | ✅ Done |
| P1 | Multi-seed / eval-variance error bars | ~1.5h compute | ✅ Done (16-4-2026) |
| P1 | 999-neg robustness table | ~35min compute | ✅ Done (16-4-2026) |
| P1 | Eval protocol fix (sampled vs full-catalog) | immediate | ✅ Done (15-4-2026) |
| P1 | Pipeline A (Cleora) metric | 1–2h | Pending |
| P1 | Ablation: Pipeline A vs B vs both | ~2h | Pending |
| P2 | Related Work §2.4 expansion | 1h | Pending |
| P2 | Qualitative example figure/table | 1–2h | Pending |
| P3 | GRU citation fix | 30min | Pending |
| P3 | Author names + student IDs | 15min | Pending |
| P3 | 71ms conclusion discrepancy | 15min | Pending |

---

## Notes

- **BERT4Rec + MostPop baselines** — dropped. Not pursuing additional baselines.
- The **LLM assistant ("Ask AI")** is a product feature, not a research contribution. Do not add metrics for it.
- The latency and concurrency sections are already thorough — no additional benchmarks needed there.
- The **HR@10=0.7749 against 3.08M items** is the paper's strongest claim and will receive the most scrutiny. Make sure the eval protocol section clearly states that the held-out item is ranked against all 3.08M catalog items with no subsampling.
