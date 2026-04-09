"""
scripts/audit/check_alignment.py — Cross-index alignment audit.

Checks that all data artefacts (Tantivy, FAISS flat, metadata parquet, asins.csv)
are consistent with each other before the server is started.

Usage:
    python scripts/audit/check_alignment.py
    python scripts/audit/check_alignment.py --sample 5
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(_ROOT, "data")

PASS = "OK  "
FAIL = "FAIL"
WARN = "WARN"

errors = []
warnings = []

def ok(msg):   print(f"  {PASS} {msg}")
def fail(msg): print(f"  {FAIL} {msg}"); errors.append(msg)
def warn(msg): print(f"  {WARN} {msg}"); warnings.append(msg)


# ── 1. Metadata parquet ───────────────────────────────────────────────────────

def check_metadata():
    print("\n[1] item_metadata.parquet")
    path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if not os.path.exists(path):
        fail(f"Not found: {path}")
        return None

    df = pd.read_parquet(path)
    if "parent_asin" in df.columns:
        df = df.set_index("parent_asin")
    df.index = df.index.map(str)

    ok(f"{len(df):,} items indexed by parent_asin")

    for col in ["title", "author_name", "categories"]:
        missing = df[col].isna().sum() if col in df.columns else len(df)
        pct = missing / len(df) * 100
        if pct > 50:
            warn(f"Column '{col}': {missing:,} nulls ({pct:.1f}%)")
        else:
            ok(f"Column '{col}': {missing:,} nulls ({pct:.1f}%)")

    return df


# ── 2. asins.csv ──────────────────────────────────────────────────────────────

def check_asins(meta_df):
    print("\n[2] asins.csv")
    path = os.path.join(DATA_DIR, "asins.csv")
    if not os.path.exists(path):
        fail(f"Not found: {path}")
        return None

    asins = pd.read_csv(path, header=None)[0].astype(str).tolist()
    ok(f"{len(asins):,} rows")

    if meta_df is not None:
        overlap = len(set(asins) & set(meta_df.index))
        pct = overlap / len(meta_df) * 100
        if overlap == len(meta_df):
            ok(f"All {overlap:,} metadata ASINs present in asins.csv")
        else:
            fail(f"Only {overlap:,}/{len(meta_df):,} metadata ASINs in asins.csv ({pct:.1f}%)")

    return asins


# ── 3. FAISS flat index ───────────────────────────────────────────────────────

def check_faiss_flat(asins, meta_df, n_sample):
    print("\n[3] blair_index_bge_flat.faiss")
    try:
        import faiss
    except ImportError:
        warn("faiss not installed — skipping FAISS checks")
        return

    path = os.path.join(DATA_DIR, "blair_index_bge_flat.faiss")
    if not os.path.exists(path):
        fail(f"Not found: {path}")
        return

    idx = faiss.read_index(path, faiss.IO_FLAG_MMAP)
    ok(f"ntotal={idx.ntotal:,}  metric={'IP' if idx.metric_type == 0 else 'L2'}")

    if asins and idx.ntotal != len(asins):
        fail(f"Vector count mismatch: FAISS={idx.ntotal:,}  asins.csv={len(asins):,}")
    elif asins:
        ok(f"Vector count matches asins.csv ({idx.ntotal:,})")

    if meta_df is None or asins is None:
        return

    # Sample n_sample metadata ASINs and verify their vectors are non-zero
    meta_asins = list(meta_df.index)
    sample = meta_asins[:n_sample]
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    zero_count = 0
    for asin in sample:
        if asin not in asin_to_idx:
            fail(f"ASIN {asin} in metadata but missing from asins.csv")
            continue
        slot = asin_to_idx[asin]
        vec = idx.reconstruct(slot)
        norm = float(np.linalg.norm(vec))
        if norm < 0.01:
            zero_count += 1
    if zero_count:
        fail(f"{zero_count}/{n_sample} sampled metadata ASINs have zero vectors in flat index")
    else:
        ok(f"{n_sample} sampled metadata ASINs all have non-zero vectors")

    # Check a few non-metadata ASINs ARE zero
    non_meta = [a for a in asins[:50_000] if a not in meta_df.index][:n_sample]
    non_zero = 0
    for asin in non_meta:
        slot = asin_to_idx[asin]
        vec = idx.reconstruct(slot)
        if float(np.linalg.norm(vec)) > 0.01:
            non_zero += 1
    if non_zero:
        warn(f"{non_zero}/{len(non_meta)} no-metadata ASINs have non-zero vectors (unexpected)")
    else:
        ok(f"{len(non_meta)} no-metadata ASINs correctly have zero vectors")


# ── 4. FAISS HNSW index ───────────────────────────────────────────────────────

def check_faiss_hnsw(asins, meta_df, n_sample):
    print("\n[4] blair_index_bge_hnsw.faiss")
    try:
        import faiss
    except ImportError:
        warn("faiss not installed — skipping HNSW checks")
        return

    path = os.path.join(DATA_DIR, "blair_index_bge_hnsw.faiss")
    if not os.path.exists(path):
        warn(f"Not found (flat index will be used as fallback): {path}")
        return

    idx = faiss.read_index(path, faiss.IO_FLAG_MMAP)
    ok(f"ntotal={idx.ntotal:,}  metric={'IP' if idx.metric_type == 0 else 'L2'}")

    # Should contain only metadata vectors (~1.73M), not all 3M
    if meta_df is not None:
        if idx.ntotal == len(meta_df):
            ok(f"Vector count matches metadata count ({idx.ntotal:,}) — zero vectors correctly excluded")
        elif asins and idx.ntotal == len(asins):
            fail(f"ntotal={idx.ntotal:,} equals asins.csv length — zero vectors were NOT excluded, "
                 f"index has graph pollution (expected ~{len(meta_df):,})")
        else:
            warn(f"ntotal={idx.ntotal:,} — expected ~{len(meta_df):,} (metadata) or {len(asins) if asins else '?'} (all)")

    if meta_df is None or asins is None:
        return

    asin_to_idx = {a: i for i, a in enumerate(asins)}
    flat_path = os.path.join(DATA_DIR, "blair_index_bge_flat.faiss")
    if not os.path.exists(flat_path):
        warn("Flat index not found — skipping HNSW search quality check")
        return

    flat = faiss.read_index(flat_path, faiss.IO_FLAG_MMAP)
    meta_asins = list(meta_df.index)
    sample = meta_asins[:n_sample]

    # Search with known vectors, check results are in metadata and scores are realistic
    bad_results = 0
    zero_vector_results = 0
    for asin in sample:
        if asin not in asin_to_idx:
            continue
        slot = asin_to_idx[asin]
        q = flat.reconstruct(slot).reshape(1, -1)
        D, I = idx.search(q, 10)
        for k in range(len(I[0])):
            i = int(I[0][k])
            if i == -1:
                continue
            result_asin = asins[i]
            if result_asin not in meta_df.index:
                bad_results += 1
                break
        top_score = float(D[0][0])
        if top_score < 0.3:
            zero_vector_results += 1

    if bad_results:
        fail(f"{bad_results}/{n_sample} queries returned results not in metadata — graph pollution likely")
    else:
        ok(f"{n_sample} queries all returned metadata-only results")

    if zero_vector_results:
        fail(f"{zero_vector_results}/{n_sample} queries had top score < 0.3 — possible zero-vector returns")
    else:
        ok(f"All top scores >= 0.3 — no zero-vector pollution detected")


# ── 5. Tantivy index ──────────────────────────────────────────────────────────

def check_tantivy(meta_df, n_sample):
    print("\n[5] tantivy_index")
    try:
        import tantivy
    except ImportError:
        warn("tantivy not installed — skipping Tantivy checks")
        return

    path = os.path.join(DATA_DIR, "tantivy_index")
    if not os.path.exists(path):
        fail(f"Not found: {path}")
        return

    try:
        index    = tantivy.Index.open(path)
        searcher = index.searcher()
    except Exception as e:
        fail(f"Failed to open: {e}")
        return

    num_docs = searcher.num_docs
    ok(f"{num_docs:,} documents")

    if meta_df is not None:
        if num_docs != len(meta_df):
            warn(f"Doc count mismatch: Tantivy={num_docs:,}  metadata={len(meta_df):,}")
        else:
            ok(f"Doc count matches metadata ({num_docs:,})")

    # Schema check
    try:
        index.parse_query("test", ["title", "author", "genres"])
        ok("Schema has all required fields: title, author, genres")
    except Exception as e:
        fail(f"Schema missing field: {e}")
        return

    # Functional search test
    try:
        qp = index.parse_query("adventure", ["title"])
        results = searcher.search(qp, 5)
        if results.hits:
            ok(f"Search works ({len(results.hits)} hits for 'adventure')")
        else:
            warn("Search returned 0 hits for 'adventure' — index may be empty")
    except Exception as e:
        fail(f"Search failed: {e}")
        return

    # ASIN alignment: sample Tantivy docs and verify ASINs are in metadata
    if meta_df is None:
        return

    try:
        qp       = index.parse_query("the", ["title"])
        results  = searcher.search(qp, n_sample * 2)
        bad      = 0
        checked  = 0
        for _, addr in results.hits[:n_sample]:
            doc      = searcher.doc(addr)
            asin_str = str(doc["asin"][0])
            if asin_str not in meta_df.index:
                bad += 1
                if bad == 1:
                    fail(f"ASIN alignment broken — '{asin_str}' from Tantivy not in metadata "
                         f"(looks like integer index: {asin_str.isdigit()})")
            checked += 1
        if bad == 0 and checked > 0:
            ok(f"{checked} sampled Tantivy ASINs all present in metadata")
        elif bad > 0:
            fail(f"{bad}/{checked} Tantivy ASINs not found in metadata — rebuild index")
    except Exception as e:
        warn(f"ASIN alignment check failed: {e}")


# ── 5. Summary ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of items to spot-check per test (default: 10)")
    args = parser.parse_args()

    print("=" * 55)
    print("  Data Alignment Audit")
    print(f"  DATA_DIR: {DATA_DIR}")
    print("=" * 55)

    meta_df = check_metadata()
    asins   = check_asins(meta_df)
    check_faiss_flat(asins, meta_df, args.sample)
    check_faiss_hnsw(asins, meta_df, args.sample)
    check_tantivy(meta_df, args.sample)

    print("\n" + "=" * 55)
    if errors:
        print(f"  RESULT: {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors:
            print(f"    {FAIL} {e}")
        sys.exit(1)
    elif warnings:
        print(f"  RESULT: OK with {len(warnings)} warning(s)")
        for w in warnings:
            print(f"    {WARN} {w}")
    else:
        print("  RESULT: All checks passed")
    print("=" * 55)


if __name__ == "__main__":
    main()
