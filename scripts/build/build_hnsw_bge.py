"""
build_hnsw_bge.py — Rebuild BGE HNSW index from existing chunk files.

Fixes the zero-vector graph pollution problem in the original HNSW by building
with only the 1.73M non-zero (metadata) vectors instead of all 3M.

Uses IndexIDMap2(IndexHNSWFlat) so the search returns original FAISS slot IDs
and the existing asins[i] mapping in faiss_repo.py continues to work unchanged.

Peak RAM: ~14-15 GB (vs ~32 GB for full 3M build)
Build time: ~5-10 min (vs ~75 min for full 3M build)

Usage:
    python scripts/build/build_hnsw_bge.py
    python scripts/build/build_hnsw_bge.py --m 16          # smaller graph, less RAM
    python scripts/build/build_hnsw_bge.py --ef-search 64  # faster query, less accurate
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import faiss
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from app.config import settings

DATA_DIR       = settings.DATA_DIR
CHUNK_PATTERN  = os.path.join(DATA_DIR, "bge_embeddings_chunk_*.npz")
OUT_HNSW       = os.path.join(DATA_DIR, "blair_index_bge_hnsw.faiss")

EMBED_DIM      = 1024
HNSW_M         = 32
EF_CONSTRUCTION = 200
EF_SEARCH      = 128
CHUNK_SIZE     = 200_000


# ── Step 1: Load non-zero vectors from chunks ─────────────────────────────────

def load_nonzero_from_chunks() -> tuple[np.ndarray, np.ndarray]:
    """
    Scan all chunk files, keep only rows with norm > 0, L2-normalize them,
    and return (vecs, ids) where ids are the original global slot indices.

    Memory: proportional to non-zero rows only (~7 GB for 1.73M × 1024 fp32).
    """
    chunk_files = sorted(glob.glob(CHUNK_PATTERN))
    if not chunk_files:
        print(f"ERROR: No chunk files found matching {CHUNK_PATTERN}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files.")

    vec_parts = []
    id_parts  = []
    total_rows = 0
    nonzero_rows = 0

    for f in tqdm(chunk_files, desc="Loading chunks"):
        data      = np.load(f)
        vecs      = data["embeddings"].astype(np.float32)   # (N, 1024)
        start_idx = int(data["start_idx"])

        norms = np.linalg.norm(vecs, axis=1)                # (N,)
        mask  = norms > 1e-6

        total_rows   += len(vecs)
        nonzero_rows += mask.sum()

        if not mask.any():
            continue

        valid_vecs  = vecs[mask]
        valid_norms = norms[mask, None]
        valid_vecs  = valid_vecs / valid_norms              # L2-normalize

        local_indices  = np.where(mask)[0].astype(np.int64)
        global_indices = local_indices + start_idx

        vec_parts.append(valid_vecs)
        id_parts.append(global_indices)

    print(f"\nTotal slots : {total_rows:,}")
    print(f"Non-zero    : {nonzero_rows:,}  ({nonzero_rows/total_rows*100:.1f}%)")
    print(f"Zero (skip) : {total_rows - nonzero_rows:,}")

    all_vecs = np.vstack(vec_parts)    # (nonzero_rows, 1024)
    all_ids  = np.concatenate(id_parts)  # (nonzero_rows,)
    return all_vecs, all_ids


# ── Step 2: Build HNSW ────────────────────────────────────────────────────────

def build_hnsw(all_vecs: np.ndarray, all_ids: np.ndarray,
               m: int, ef_construction: int, ef_search: int) -> faiss.Index:
    """
    Build IndexIDMap2(IndexHNSWFlat) with METRIC_INNER_PRODUCT.

    IndexIDMap2 maps external IDs (original FAISS slot indices) to internal
    HNSW positions, so .search() returns slot IDs directly.
    """
    n = len(all_vecs)
    print(f"\nBuilding IndexHNSWFlat (M={m}, efConstruction={ef_construction}) "
          f"over {n:,} vectors ...")
    print("NOTE: HNSW build is CPU-only. Be patient.")

    hnsw_base = faiss.IndexHNSWFlat(EMBED_DIM, m, faiss.METRIC_INNER_PRODUCT)
    hnsw_base.hnsw.efConstruction = ef_construction

    hnsw_index = faiss.IndexIDMap2(hnsw_base)

    for start in tqdm(range(0, n, CHUNK_SIZE), desc="Adding to HNSW"):
        end = min(start + CHUNK_SIZE, n)
        hnsw_index.add_with_ids(
            all_vecs[start:end],
            all_ids[start:end],
        )

    hnsw_base.hnsw.efSearch = ef_search
    print(f"HNSW index built: {hnsw_index.ntotal:,} vectors")
    return hnsw_index


# ── Step 3: Verify ────────────────────────────────────────────────────────────

def verify(hnsw_index: faiss.Index, all_vecs: np.ndarray, all_ids: np.ndarray):
    """
    Spot-check: search with 5 known vectors, confirm scores are realistic
    and returned IDs are in the non-zero set (i.e. have metadata).
    """
    print("\nVerifying index ...")
    id_set = set(all_ids.tolist())

    sample_indices = [0, len(all_vecs)//4, len(all_vecs)//2,
                      3*len(all_vecs)//4, len(all_vecs)-1]
    failures = 0
    for local_i in sample_indices:
        q   = all_vecs[local_i].reshape(1, -1)
        D, I = hnsw_index.search(q, 10)
        top_id    = int(I[0][0])
        top_score = float(D[0][0])
        in_set    = all(int(i) in id_set for i in I[0] if i != -1)
        status    = "OK  " if in_set and top_score > 0.5 else "FAIL"
        print(f"  [{status}] query_slot={int(all_ids[local_i]):,}  "
              f"top_result={top_id:,}  score={top_score:.4f}  "
              f"all_results_in_nonzero_set={in_set}")
        if status == "FAIL":
            failures += 1

    if failures:
        print(f"\nWARNING: {failures} verification checks failed.")
    else:
        print("All spot-checks passed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",             type=int, default=HNSW_M,
                        help=f"HNSW M parameter (default: {HNSW_M})")
    parser.add_argument("--ef-construction", type=int, default=EF_CONSTRUCTION,
                        help=f"efConstruction (default: {EF_CONSTRUCTION})")
    parser.add_argument("--ef-search",     type=int, default=EF_SEARCH,
                        help=f"efSearch at query time (default: {EF_SEARCH})")
    args = parser.parse_args()

    print("=" * 60)
    print("  BGE HNSW Index Builder (non-zero vectors only)")
    print(f"  Output : {OUT_HNSW}")
    print(f"  M={args.m}  efConstruction={args.ef_construction}  efSearch={args.ef_search}")
    print("=" * 60)

    t0 = time.perf_counter()

    # 1. Load
    all_vecs, all_ids = load_nonzero_from_chunks()

    # 2. Build
    hnsw_index = build_hnsw(all_vecs, all_ids, args.m, args.ef_construction, args.ef_search)

    # 3. Verify
    verify(hnsw_index, all_vecs, all_ids)

    # 4. Save
    print(f"\nSaving → {OUT_HNSW}")
    faiss.write_index(hnsw_index, OUT_HNSW)
    size_gb = os.path.getsize(OUT_HNSW) / 1024**3
    print(f"File size: {size_gb:.2f} GB")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print("=" * 60)
    print("Next steps:")
    print("  1. Update faiss_repo.py to load blair_index_bge_hnsw.faiss first")
    print("  2. python scripts/audit/check_alignment.py")
    print("  3. Restart server")
    print("=" * 60)


if __name__ == "__main__":
    main()
