# CLIP Upgrade Decision — March 31, 2026

## Current model
- **Name**: `openai/clip-vit-base-patch32`
- **Embedding dim**: 512
- **Device**: see startup log `CLIP model loaded: openai/clip-vit-base-patch32 │ dim=512 │ device=…`
- **Mean visual similarity score across test covers**: *run `scripts/audit_clip_quality.py` to populate*

---

## Quality audit findings

> Fill in after running `python scripts/audit_clip_quality.py`

```
Test image: Dark/noir cover
  Mean similarity score (top-5): X.XXXX
  Top-5 results: ...

Test image: Bright children's book
  Mean similarity score (top-5): X.XXXX
  ...
```

**Coherence check** (fill manually):
- [ ] Noir/dark covers → dark, moody results?
- [ ] Children's → colorful, illustrated results?
- [ ] Sci-fi → space/futuristic results?
- [ ] Romance → warm, illustrated results?
- [ ] History → academic/textbook results?

Overall: **X/5 test images returned visually coherent top-5 results**

---

## Upgrade candidate benchmarks

| Model | Dim | Flickr30k R@1 | MS-COCO R@1 | Relative gain vs ViT-B/32 |
|---|---|---|---|---|
| `clip-vit-base-patch32` *(current)* | 512 | ~68% | ~58% | baseline |
| `clip-vit-large-patch14` | 768 | ~76% | ~65% | ~+11% |
| `CLIP-ViT-H-14-laion2B` (OpenCLIP) | 1024 | ~84% | ~73% | ~+22% |

*Benchmarks sourced from the LAION-5B paper and OpenCLIP leaderboard.*

---

## Upgrade cost (if proceeding)

1. **Re-embed ~3M book covers** → estimated time: 10–20 hours on single GPU
2. **Rebuild FAISS CLIP index** from scratch (90 NPZ chunks → new dim)
3. **Update DQN input dim**: currently `1024 (BLaIR) + 512 (CLIP) = 1536`
   - ViT-L upgrade: `1024 + 768 = 1792` → must retrain DQN from scratch
   - ViT-H upgrade: `1024 + 1024 = 2048` → same caveat
4. **Update all 90 NPZ chunks** to new dimension

---

## Decision

**GO / NO-GO** ← *fill in after running the audit*

**Heuristic**: 
- If current model returns visually coherent top-5 for ≥ 4/5 test images AND mean similarity ≥ 0.20 → **NO-GO** (document and move on)
- If mean similarity < 0.15 or ≥ 2 test images return clearly random results → **GO** (schedule re-embed pass as a separate task — do not block Week 1 work)

**Reason**: *(1–2 sentences — key factors: quality gap from audit, compute cost, DQN retraining cost, thesis timeline)*

---

*Last updated: March 31, 2026. Run `scripts/audit_clip_quality.py` to fill in the quality metrics.*
