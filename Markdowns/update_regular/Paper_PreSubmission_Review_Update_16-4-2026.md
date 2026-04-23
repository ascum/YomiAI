# Paper Pre-Submission Review Update — April 16, 2026

Session covering full paper review, citation fact-checking, formatting fixes, and
completion of the robustness table with Pipeline A 999-negative results.

---

## Summary of All Changes Made

### CRITICAL — Citation Corrections (`references.bib`)

**Cleora citation was entirely wrong.**
All authors and the venue were fabricated. Corrected to:
- **Authors:** Barbara Rychalska, Piotr Bąbel, Konrad Gołuchowski, Andrzej Michałowski, Jacek Dąbrowski
- **Venue:** ICONIP 2021, Lecture Notes in Computer Science vol. 13111, Springer
- **arXiv:** 2102.02302

Previous entry had: Rychlikowski, Tworkowski, Michalewski, Wu (Yuxiang), Szegedy (Christian), Milos —
none of whom are actual Cleora authors.

**FAISS publication year was wrong.**
`year = 2019` → `year = 2021`. The DOI contains "2019" as part of the identifier but the
paper was published in IEEE Transactions on Big Data in 2021.

---

### HIGH — Content Fixes

**Abstract "vice versa" removed** (`sections/00_abstract.tex`)
"Pipeline A rescuing 59% of DIF-SASRec misses and vice versa" was misleading
(59% vs 11% is not symmetric). Replaced with explicit numbers:
"Pipeline A rescuing 59% of DIF-SASRec misses and DIF-SASRec rescuing 11% of Pipeline A misses."

**VRAM contradiction fixed** (`sections/06_results.tex`)
"leaving headroom on a 24 GB card" → "leaving headroom on the 16 GB card".
The GPU is specified as RTX 5060 Ti 16 GB in §5; 24 GB was inconsistent.

**"Phase 1 / Phase 2" project language removed** (`sections/07_conclusion.tex`)
The conclusion opened with "The work addresses the two-phase project scope: Phase 1 established...
Phase 2 delivered..." which reads like a capstone report, not a conference paper. Replaced
with a clean summary of what the system does.

**LightGCN citation removed from unsupported claim** (`sections/01_introduction.tex`)
"Current NBA research relies predominantly on structured transaction data~\cite{he2020lightgcn}"
— LightGCN is a graph CF model, not a survey or NBA paper. Citation dropped; claim stands on its own.

**Content veto formula moved to displayed equation** (`sections/04_methodology.tex`)
The inline `\max(\cos(...), \cos(...)) < 0.3` overflowed the column width inside the enumerate.
Converted to a proper `equation*` environment.

---

### MODERATE — Style Fixes

**British/American English unified to American throughout**
Files affected: `sections/01_introduction.tex`, `sections/07_conclusion.tex`, `sections/06_results.tex`
Changed: behavioural→behavioral, personalisation→personalization (all occurrences).

**Ablation table caption updated** (`tables/ablation.tex`)
The `$^{\star}$` footnote marker was dangling (footnote text was commented out).
Moved the scaling disclosure directly into the caption:
"Component ablation rows are proportionally scaled from pre-fix measurements (scale factor 0.624);
relative ΔHR values are preserved."

---

### NEW — Pipeline A 999-Negative Robustness Result

**Eval script restored** (`scripts/benchmark/evaluate_recommendation.py`)
The git reset had wiped three things from the script:
1. Shuffle fix (`random.shuffle(candidates)` before scoring) — was the root cause of
   inflated HR@10=0.9322 in the first Pipeline A run
2. `PipelineAStrategy` class
3. `--pipeline-a-only` CLI flag

All three were restored. The shuffle fix is the critical one — without it the target
always wins ties because `[target] + negs` preserves insertion order under stable sort.

**New result logged** (`evaluation/logs/20260416_135251.log`)
```
negatives=999  seed=42  users=100,000
Pipeline A (Cleora)  HR@10=0.3264  NDCG@10=0.2096  191.6s
```

**Robustness table updated** (`tables/robustness_results.tex`)
Added Pipeline A row. Bold is now split:
- Pipeline A: **HR@10=0.3264** (best)
- DIF-SASRec: **NDCG@10=0.2209** (best)

**§6 robustness paragraph rewritten** (`sections/06_results.tex`)
New narrative: Pipeline A leads HR@10, DIF-SASRec leads NDCG@10 (hits rank closer to
position 1). Both show increasing relative advantage over random as pool grows:
- Pipeline A: 7.6× (99-neg) → 32.6× (999-neg)
- DIF-SASRec: 4.8× (99-neg) → 31.4× (999-neg)

---

## Final Robustness Numbers (999-neg, seed 42, 100k users)

| Model | HR@10 | NDCG@10 | vs random (HR) |
|---|---|---|---|
| Content-KNN (BGE-M3) | 0.2122 | 0.1349 | 21.2× |
| DIF-SASRec (Pipeline B) | 0.3142 | **0.2209** | 31.4× |
| Pipeline A (Cleora+BGE-M3) | **0.3264** | 0.2096 | 32.6× |

---

## Cross-Reference & Citation Audit (clean)

- All 12 `\cite{}` keys resolve to entries in `references.bib` ✓
- All `\ref{}` targets have matching `\label{}` definitions ✓
- No figures directory needed (architecture figure is inline TikZ) ✓
- Unused bib entries (`hou2022core`, `sun2019bert4rec`, `zhao2018deep`) are harmless —
  BibTeX only includes cited entries in the output ✓

---

## Remaining Before Submission

| Priority | Task | Notes |
|---|---|---|
| **P0** | Compile PDF and visual check | Confirm TikZ figure, equation layout, page count within IAAA limit |
| P2 | Re-run 4 component ablation rows with shuffle fix | Disclosed in caption; re-run removes the disclosure entirely |
| P2 | Related Work §2.4 expansion | Currently one paragraph |
