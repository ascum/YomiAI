# Encoder Comparison Benchmark: BLaIR (Legacy) vs BGE-M3

## Background

The system was originally built with BLaIR embeddings. After migrating to BGE-M3, both
flat indices remain on disk (`blair_index_flat_legacy.faiss`, `bge_index_flat.faiss`).
This benchmark quantifies whether the migration improved retrieval quality, using the same
evaluation protocol as `evaluate_recommendation.py` so numbers are directly comparable
in the paper's results table.

No live BLaIR model is needed — all item vectors are pre-computed in the flat indices.

---

## Data Requirements

| File | Role |
|---|---|
| `data/bge_index_flat.faiss` | BGE-M3 pre-computed item vectors |
| `data/bge_index_hnsw.faiss` | BGE-M3 HNSW graph (Step 5b only) |
| `data/blair_index_flat_legacy.faiss` | BLaIR pre-computed item vectors |
| `data/asins.csv` | Shared ASIN↔FAISS-slot mapping |
| `evaluation/eval_users.json` | Ground truth train/test click sequences |
| `data/item_metadata.parquet` | `categories` field for genre-overlap analysis |

---

## Evaluation Methodology

### Sampled evaluation (Step 3) — primary quality metric

Exact same protocol as `evaluate_recommendation.py --mode sampled`:

- For each user: form a profile vector = mean of train-click embeddings
- Rank the 1 real test item against 99 random negatives
- Metrics: HR@5, HR@10, NDCG@10, MRR@10
- Same random seed → same negative pool for both models (fair comparison)

No index search happens here. `.reconstruct()` from the flat index + dot products only.

### Score distribution analysis (Step 4)

Sample 5,000 ASINs with metadata. For each model:
- **Intra-genre cosine sim**: mean over 300 random same-genre pairs
- **Inter-genre cosine sim**: mean over 300 random cross-genre pairs
- **Separation ratio**: intra/inter — higher means tighter genre clusters

### Top-K overlap — flat search for both (Step 5a)

For N random query ASINs, search both flat indices (exact, brute-force):
- BGE: `bge_flat.search(q, k+1)`
- BLaIR: `blair_flat.search(q, k+1)`

Flat for both = fair embedding quality comparison. Default N=200 (flat search is
slow at 3M vectors; ~50–200ms per query on CPU).

Metrics: mean Jaccard@10 between result sets, genre-precision@10 per model.

### HNSW recall validation — BGE only (Step 5b)

Compare BGE flat (exact) vs BGE HNSW (approximate) on the same queries:
- Recall@10 = |flat ∩ hnsw| / k
- Latency: flat p50/p95 vs HNSW p50/p95
- Speedup: flat_p50 / hnsw_p50

This validates that the HNSW production index isn't losing meaningful recall.
BLaIR has no HNSW index — this section is BGE-only by design.

### Optional: BGE-M3 encode latency (Step 6)

Enabled with `--benchmark-latency`. Loads the live BGE-M3 model and times
encoding of 60 standard queries. Reports p50, p95, p99 ms.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Item-as-query (profile vector) | BLaIR model not available; also matches live system behavior |
| Same neg pool across models | Eliminates sampling variance from the comparison |
| Flat search for 5a (not HNSW vs flat) | Compares embedding spaces, not search algorithms |
| Step 5b separate, BGE-only | HNSW recall is an index quality question, not an encoder quality question |
| Same protocol as `evaluate_recommendation.py` | Numbers slot into paper table without methodology footnotes |

---

## CLI

```
python scripts/benchmark/compare_encoders.py

Options:
  --negatives 99          # sampled eval negatives (default: 99)
  --max-users 5000        # cap users for fast iteration (default: all)
  --k 10                  # rank cutoff (default: 10)
  --seed 42               # reproducibility (default: 42)
  --sample-n 5000         # ASINs for distribution analysis (default: 5000)
  --overlap-queries 200   # queries for flat-search overlap (default: 200)
  --benchmark-latency     # also load BGE-M3 and time query encoding
  --no-save               # skip writing JSON output
```

---

## Outputs

- **Console**: results table (mirrors `evaluate_recommendation.py` style) + LaTeX rows
- **JSON**: `evaluation/encoder_comparison_<run_id>.json`

### Console table (example shape)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ENCODER COMPARISON   negatives=99   k=10                                │
└──────────────────────────────────────────────────────────────────────────┘

  Model                HR@5      HR@10    NDCG@10    MRR@10    Users    Time
  ────────────────────────────────────────────────────────────────────────
  BGE-M3 (current)   0.XXXX    0.XXXX    0.XXXX    0.XXXX    4,821    12.3s  ◄
  BLaIR (legacy)     0.XXXX    0.XXXX    0.XXXX    0.XXXX    4,821    11.9s
  ────────────────────────────────────────────────────────────────────────
  Random baseline    0.0500    0.1000    0.0141    0.0500

  Score separation (intra-genre / inter-genre cosine):
    BGE-M3:  0.XXX / 0.XXX  →  ratio=X.XX
    BLaIR:   0.XXX / 0.XXX  →  ratio=X.XX

  Top-10 Jaccard overlap (BGE flat vs BLaIR flat):  0.XX
  HNSW recall@10 vs exact:  0.XXXX  |  HNSW speedup: X.Xx
```

### LaTeX rows

```latex
BGE-M3 (ours)  & \textbf{0.XXXX} & \textbf{0.XXXX} & \textbf{0.XXXX} & \textbf{0.XXXX} \\
BLaIR (legacy) & 0.XXXX          & 0.XXXX          & 0.XXXX          & 0.XXXX          \\
```
