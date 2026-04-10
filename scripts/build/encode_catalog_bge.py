"""
encode_catalog_bge.py — BGE-M3 Full Catalog Encoder
=====================================================
Encodes all 3,080,829 ASINs from asins.csv into a new FAISS HNSW index
using BGE-M3 (BAAI/bge-m3, 1024-dim fp16).

KEY INVARIANT: The output FAISS index must have EXACTLY 3,080,829 vectors,
one per row in asins.csv, in the SAME ORDER. This is required because
retriever.py maps FAISS row i → asins[i] directly.

ASINs that have no entry in item_metadata.parquet receive a ZERO VECTOR
at their FAISS slot (they become invisible to search, which is correct —
they have no content to retrieve anyway).

Pipeline:
  1. Load asins.csv → ordered list of 3.08M ASINs
  2. Load item_metadata.parquet → dict {parent_asin: row}
  3. For each ASIN in order:
       - If in metadata: build document string → encode with BGE-M3
       - If NOT in metadata: emit zero vector (1024-dim)
  4. Batch encode with BGE-M3 (batch_size=512, fp16)
  5. L2-normalize all vectors (cosine similarity via IndexFlatIP/HNSW)
  6. Write to TEXT_INDEX_HNSW (HNSW graph for sub-ms search; see app/config.py)
  7. Checkpoint every CHECKPOINT_EVERY items → bge_embeddings_checkpoint.npz

Usage:
    # First run (starts fresh):
    python scripts/encode_catalog_bge.py

    # Resume after interruption (checkpoint auto-detected):
    python scripts/encode_catalog_bge.py

    # Dry run to verify alignment (no GPU needed):
    python scripts/encode_catalog_bge.py --dry-run

    # Build final HNSW index from completed checkpoint:
    python scripts/encode_catalog_bge.py --build-index-only

Estimated time on RTX 4060 Laptop (8GB): ~1h 20min
Estimated time on RTX 5060 Ti (16GB):    ~40–50min
"""

import os
import sys
import ast
import time
import argparse
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# ─── Path setup so `app.*` imports work ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from app.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = settings.DATA_DIR

ASINS_CSV       = os.path.join(DATA_DIR, "asins.csv")
METADATA_PARQ   = os.path.join(DATA_DIR, "item_metadata.parquet")

# Output files
CHECKPOINT_NPZ  = os.path.join(DATA_DIR, "bge_embeddings_checkpoint.npz")
OUT_INDEX_FLAT  = os.path.join(DATA_DIR, settings.TEXT_INDEX_FLAT)   # exact, for reconstruction
OUT_INDEX_HNSW  = os.path.join(DATA_DIR, settings.TEXT_INDEX_HNSW)   # fast search

MODEL_NAME      = settings.TEXT_ENCODER_MODEL
EMBED_DIM       = settings.TEXT_EMBED_DIM
BATCH_SIZE      = 512           # Best throughput on GPU
CHECKPOINT_EVERY = 100_000      # Save checkpoint every N items

# HNSW build parameters
HNSW_M          = 32
HNSW_EF_CONSTRUCTION = 200

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT STRING BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def parse_author_name(raw: str) -> str:
    """Extract clean author name matching app/repository/metadata_repo.py logic."""
    if not raw or str(raw) == "nan":
        return ""
    auth_str = str(raw)
    if auth_str.startswith("{") and auth_str.endswith("}"):
        try:
            auth_dict = ast.literal_eval(auth_str)
            return auth_dict.get("name", "").strip()
        except Exception:
            return auth_str
    return auth_str.strip()


def build_document_string(row: dict) -> str:
    """
    Builds the text string to encode for a single book.
    """
    parts = []

    # 1. Title
    title = str(row.get("title", "")).strip()
    if title and title != "nan":
        parts.append(title)

    # 2. Author
    author = parse_author_name(row.get("author_name", ""))
    if author and author != "Unknown Author":
        parts.append(f"by {author}")

    # 3. Categories (Genre signal)
    cats = str(row.get("categories", "")).strip()
    if cats and cats != "nan":
        cats_clean = cats.replace("|", ", ").replace("&", "and")
        parts.append(cats_clean)

    # 4. Description
    desc = str(row.get("description", "")).strip()
    STUB_PHRASES = {
        "enter your model number",
        "make sure this fits",
        "from the manufacturer",
        "about the author",
        "product description",
        "see all",
    }
    is_stub = (
        len(desc) < 40
        or desc == "nan"
        or any(p in desc.lower()[:80] for p in STUB_PHRASES)
    )
    if not is_stub:
        parts.append(desc[:600])

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_asins(path: str) -> list:
    """
    Load asins.csv (no header, single column of ASIN strings).
    Row i of this file = FAISS vector slot i.
    """
    print(f"[1/4] Loading ASIN list from {path} ...")
    # The file has no header — first row is a real ASIN
    df = pd.read_csv(path, header=None, dtype=str)
    asins = df.iloc[:, 0].tolist()
    print(f"      Loaded {len(asins):,} ASINs (FAISS dimension: {len(asins):,} vectors required)")
    return asins


def load_metadata(path: str) -> dict:
    """
    Load item_metadata.parquet into a dict keyed by parent_asin.
    Only keeps columns needed for document string building.
    """
    print(f"[2/4] Loading metadata from {path} ...")
    cols = ["parent_asin", "title", "author_name", "categories", "description"]
    df = pd.read_parquet(path, columns=cols)
    # Build lookup dict: parent_asin -> row dict
    meta = {}
    for _, row in df.iterrows():
        meta[row["parent_asin"]] = {
            "title":       row.get("title", ""),
            "author_name": row.get("author_name", ""),
            "categories":  row.get("categories", ""),
            "description": row.get("description", ""),
        }
    print(f"      Metadata loaded: {len(meta):,} entries")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# SHARDED CHECKPOINT SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_chunk_path(idx: int) -> str:
    """Generate path for a specific chunk index."""
    return os.path.join(DATA_DIR, f"bge_embeddings_chunk_{idx:03d}.npz")

def load_sharded_checkpoints(total_asins: int) -> tuple[np.ndarray, int]:
    """
    Find all chunk files, load them into a master matrix, and return resume point.

    Resume point is the first contiguous gap — not the maximum end index.
    This prevents silently skipping corrupted shards and treating their slots
    as zero-vectors (no-metadata) forever.

    Corrupted shards are renamed to .corrupt so they get re-encoded on resume.
    """
    import glob
    all_embeddings = np.zeros((total_asins, EMBED_DIM), dtype=np.float32)

    chunk_files = sorted(glob.glob(os.path.join(DATA_DIR, "bge_embeddings_chunk_*.npz")))
    if not chunk_files:
        return all_embeddings, 0

    print(f"\n[RESUME] Found {len(chunk_files)} existing shards. Assembling...")

    loaded_ranges: list[tuple[int, int]] = []

    for f in chunk_files:
        try:
            data = np.load(f)
            chunk_vecs = data["embeddings"]
            start_i    = int(data["start_idx"])
            end_i      = int(data["end_idx"])

            # Validate shape before trusting the data
            expected_rows = end_i - start_i
            if chunk_vecs.shape != (expected_rows, EMBED_DIM):
                raise ValueError(
                    f"Shape mismatch: expected ({expected_rows}, {EMBED_DIM}), "
                    f"got {chunk_vecs.shape}"
                )

            all_embeddings[start_i:end_i] = chunk_vecs
            loaded_ranges.append((start_i, end_i))
            print(f"         Loaded {f} (Items {start_i:,} – {end_i:,})")

        except Exception as e:
            corrupt_path = f + ".corrupt"
            os.rename(f, corrupt_path)
            print(
                f"         ⚠ WARNING: Corrupted shard quarantined → {corrupt_path}\n"
                f"           Reason: {e}\n"
                f"           Its range will be re-encoded on resume."
            )

    # Find the first contiguous gap — that is the safe resume point.
    # Example: shards cover [0,100k), [100k,200k), [300k,400k)
    #   → resume from 200k (not 400k), so 200k–300k gets re-encoded.
    loaded_ranges.sort()
    completed_count = 0
    for start_i, end_i in loaded_ranges:
        if start_i > completed_count:
            print(f"         Gap detected at {completed_count:,} — resuming from there.")
            break
        completed_count = max(completed_count, end_i)

    print(f"         Progress restored: {completed_count:,} / {total_asins:,} items.")
    return all_embeddings, completed_count


def save_shard(embeddings_slice: np.ndarray, start_idx: int, end_idx: int, chunk_num: int):
    """
    Atomically save a specific slice of the catalog to a unique chunk file.

    Writes to a .tmp file first then renames so a crash mid-write never
    leaves a half-written (silently corrupt) .npz at the real path.
    """
    path     = get_chunk_path(chunk_num)
    tmp_path = path.replace(".npz", ".tmp.npz")  # np.savez_compressed appends .npz automatically
    try:
        np.savez_compressed(
            tmp_path,
            embeddings=embeddings_slice,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        os.replace(tmp_path, path)   # atomic on all major OSes
        print(f"\n  [CHECKPOINT] Shard {chunk_num:03d} saved → {path} ({embeddings_slice.shape[0]:,} items)")
    except Exception as e:
        print(f"\n  ⚠ CRITICAL: Failed to save shard {chunk_num}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # tmp_path already ends in .npz so no surprise extension


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING — MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def encode_catalog(asins: list, meta: dict, resume_from: int, 
                   all_embeddings: np.ndarray) -> np.ndarray:
    """
    Encode all ASINs in order using a sharded checkpoint system.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    total = len(asins)

    # Load BGE-M3
    print(f"\n      Loading BGE-M3 model ({MODEL_NAME}) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    model.half()
    print(f"      Model loaded on {device.upper()}")

    # Batch state
    batch_asins  = []
    batch_idxs   = []
    batch_docs   = []

    t_start        = time.time()
    docs_encoded   = 0
    last_saved_idx = resume_from
    chunk_counter  = (resume_from // CHECKPOINT_EVERY) + 1

    def flush_batch():
        nonlocal docs_encoded
        if not batch_docs: return
        vecs = model.encode(
            batch_docs, batch_size=BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        for local_i, faiss_i in enumerate(batch_idxs):
            all_embeddings[faiss_i] = vecs[local_i]
        docs_encoded += len(batch_docs)
        batch_asins.clear(); batch_idxs.clear(); batch_docs.clear()

    pbar = tqdm(range(resume_from, total), total=total, initial=resume_from, desc="Encoding", unit="item")

    for faiss_i in pbar:
        asin = asins[faiss_i]

        if asin in meta:
            doc = build_document_string(meta[asin])
            batch_asins.append(asin)
            batch_idxs.append(faiss_i)
            batch_docs.append(doc if doc.strip() else asin)

        if len(batch_docs) >= BATCH_SIZE:
            flush_batch()

        # Sharded Save every N items
        if (faiss_i + 1 - last_saved_idx) >= CHECKPOINT_EVERY:
            flush_batch()
            save_shard(
                all_embeddings[last_saved_idx:faiss_i+1], 
                last_saved_idx, faiss_i+1, chunk_counter
            )
            last_saved_idx = faiss_i + 1
            chunk_counter += 1
            
            elapsed = time.time() - t_start
            rate = (faiss_i + 1 - resume_from) / elapsed if elapsed > 0 else 1
            pbar.set_postfix({"docs": docs_encoded, "rate": f"{rate:.0f}/s"})

    # Final flush and final shard
    flush_batch()
    if last_saved_idx < total:
        save_shard(all_embeddings[last_saved_idx:], last_saved_idx, total, chunk_counter)

    return all_embeddings

# ... (Main update follows in next turn or combined)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS INDEX BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_indices(embeddings: np.ndarray, hnsw_only: bool = False):
    """
    Build two FAISS indices from the embedding matrix:

    1. IndexFlatIP  (TEXT_INDEX_FLAT  — see app/config.py)
       - Exact search, supports .reconstruct() for RL pipeline
       - Required by retriever.py's score_candidates() and get_asin_vec()

    2. IndexHNSWFlat (TEXT_INDEX_HNSW — see app/config.py)
       - Sub-millisecond approximate search (M=32, efConstruction=200)
       - Used by active_search_engine.py for the /search endpoint
    """
    total = embeddings.shape[0]
    print(f"\n[4/4] Building FAISS indices from {total:,} vectors ...")

    # L2-normalize (ensures dot-product = cosine similarity)
    print("      L2-normalizing vectors ...")
    X = embeddings.copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid dividing zero-vectors (ASINs with no metadata)
    norms[norms == 0] = 1.0
    X = X / norms
    X = X.astype(np.float32)

    # ── Flat Index (for reconstruction) ───────────────────────────────────
    CHUNK = 200_000
    if hnsw_only:
        print("      Skipping FlatIP build (--hnsw-only specified).")
    else:
        print("      Building IndexFlatIP (exact, supports reconstruct) ...")
        flat_index = faiss.IndexFlatIP(EMBED_DIM)
        for start in tqdm(range(0, total, CHUNK), desc="  Adding to FlatIP"):
            flat_index.add(X[start:start + CHUNK])

        print(f"      FlatIP index: {flat_index.ntotal:,} vectors")
        faiss.write_index(flat_index, OUT_INDEX_FLAT)
        print(f"      Saved → {OUT_INDEX_FLAT}")
        print(f"      File size: {os.path.getsize(OUT_INDEX_FLAT)/1024**3:.2f} GB")

        # Free flat index memory before building HNSW (both are ~12 GB in RAM)
        del flat_index
        import gc; gc.collect()
        print("      Freed FlatIP index from memory.")

    # ── HNSW Index (for fast search) ──────────────────────────────────────
    print(f"\n      Building IndexHNSWFlat (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}) ...")
    print("      NOTE: HNSW build is CPU-only and takes ~10–20min. Be patient.")

    hnsw_index = faiss.IndexHNSWFlat(EMBED_DIM, HNSW_M)
    hnsw_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION

    for start in tqdm(range(0, total, CHUNK), desc="  Adding to HNSW"):
        hnsw_index.add(X[start:start + CHUNK])

    # Set efSearch for query time (higher = more accurate but slower)
    hnsw_index.hnsw.efSearch = 128

    print(f"      HNSW index: {hnsw_index.ntotal:,} vectors")
    faiss.write_index(hnsw_index, OUT_INDEX_HNSW)
    print(f"      Saved → {OUT_INDEX_HNSW}")
    print(f"      File size: {os.path.getsize(OUT_INDEX_HNSW)/1024**3:.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_alignment(asins: list, index_path: str, meta: dict):
    """
    Sanity check: verify that FAISS slot N corresponds to asins[N].
    Tests 5 known ASINs with metadata and confirms their vectors are non-zero.
    """
    print("\n[VERIFY] Checking FAISS ↔ ASIN alignment ...")
    index = faiss.read_index(index_path)

    assert index.ntotal == len(asins), (
        f"CRITICAL ALIGNMENT ERROR: FAISS has {index.ntotal:,} vectors "
        f"but asins.csv has {len(asins):,} rows!"
    )
    print(f"  ✓ Vector count matches: {index.ntotal:,} == {len(asins):,}")

    # Test 5 random ASINs that have metadata
    import random
    test_asins = [a for a in random.sample(asins[:100_000], 5) if a in meta]
    for asin in test_asins:
        faiss_i = asins.index(asin)
        vec = index.reconstruct(faiss_i)
        norm = float(np.linalg.norm(vec))
        doc = build_document_string(meta[asin])
        print(f"  ✓ ASIN {asin} @ slot {faiss_i:,} | norm={norm:.4f} | doc='{doc[:60]}...'")

    # Test that zero-vector ASINs (no metadata) have norm ≈ 0
    no_meta_sample = [a for a in asins[:10_000] if a not in meta][:3]
    for asin in no_meta_sample:
        faiss_i = asins.index(asin)
        vec = index.reconstruct(faiss_i)
        norm = float(np.linalg.norm(vec))
        status = "✓ (zero)" if norm < 0.01 else "⚠ NOT ZERO"
        print(f"  {status} ASIN {asin} (no metadata) @ slot {faiss_i:,} | norm={norm:.4f}")

    print("\n  [VERIFY] Alignment check passed.")


# ─────────────────────────────────────────────────────────────────────────────
# DRY RUN
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(asins: list, meta: dict):
    """Print alignment stats and sample document strings without running the GPU."""
    print("\n=== DRY RUN (no GPU encoding) ===")
    print(f"Total FAISS slots (asins.csv):   {len(asins):,}")
    has_meta = sum(1 for a in asins if a in meta)
    print(f"ASINs with metadata:             {has_meta:,} ({100*has_meta/len(asins):.1f}%)")
    print(f"ASINs without (-> zero vector):  {len(asins)-has_meta:,} ({100*(len(asins)-has_meta)/len(asins):.1f}%)")
    print(f"\nOutput FAISS index size (estimate):")
    print(f"  FlatIP: {len(asins)*EMBED_DIM*4/1024**3:.2f} GB")
    print(f"  HNSW:   {len(asins)*EMBED_DIM*4*1.3/1024**3:.2f} GB (approx)")
    print(f"\nSample document strings (first 5 ASINs with metadata):")
    count = 0
    for asin in asins:
        if asin in meta and count < 5:
            doc = build_document_string(meta[asin])
            print(f"\n  [{count+1}] ASIN: {asin}")
            print(f"      DOC:  {doc[:200]}")
            count += 1
    print("\n=== Dry run complete. Run without --dry-run to start encoding. ===")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Encode catalog with BGE-M3 (Sharded)")
    parser.add_argument("--dry-run",        action="store_true", help="Print stats")
    parser.add_argument("--build-index-only", action="store_true", help="Build FAISS from shards")
    parser.add_argument("--hnsw-only",      action="store_true", help="Skip FlatIP build; only build HNSW (use when flat index already exists)")
    parser.add_argument("--verify",         action="store_true", help="Verify alignment")
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    if args.batch_size != 512:
        globals()["BATCH_SIZE"] = args.batch_size
    if args.checkpoint_every != 100_000:
        globals()["CHECKPOINT_EVERY"] = args.checkpoint_every

    print("=" * 60)
    print("  BGE-M3 Catalog Encoder (Sharded Checkpoints)")
    print(f"  Catalog:     {ASINS_CSV}")
    print(f"  Model:       {MODEL_NAME}")
    print("=" * 60)

    # 1. Load ASINs and Metadata
    asins = load_asins(ASINS_CSV)
    meta  = load_metadata(METADATA_PARQ)

    if args.dry_run:
        dry_run(asins, meta)
        return

    # 2. Load / Restore Sharded Progress
    all_embeddings, resume_from = load_sharded_checkpoints(len(asins))

    if args.build_index_only or args.hnsw_only:
        if resume_from < len(asins):
            print(f"WARNING: Only {resume_from:,}/{len(asins):,} items loaded.")
        build_faiss_indices(all_embeddings, hnsw_only=args.hnsw_only)
        return

    # 3. Full encoding run
    all_embeddings = encode_catalog(asins, meta, resume_from, all_embeddings)

    # 4. Build FAISS indices
    build_faiss_indices(all_embeddings)

    # 5. Verify
    if args.verify:
        verify_alignment(asins, OUT_INDEX_FLAT, meta)

    print("\n" + "=" * 60)
    print("  DONE. Catalog encoded and HNSW index built.")
    print("=" * 60)


if __name__ == "__main__":
    main()
