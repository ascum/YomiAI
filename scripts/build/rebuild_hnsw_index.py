import faiss
import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

def rebuild_index(name, dim):
    flat_path = os.path.join(DATA_DIR, f"{name}_index.faiss")
    hnsw_path = os.path.join(DATA_DIR, f"{name}_index_hnsw.faiss")
    
    if not os.path.exists(flat_path):
        print(f"Skipping {name}: {flat_path} not found.")
        return

    print(f"\n--- Rebuilding {name} Index ({dim} dimensions) ---")
    
    # 1. Load the flat index
    print(f"Loading existing flat index: {flat_path}")
    t0 = time.perf_counter()
    index_flat = faiss.read_index(flat_path)
    n_total = index_flat.ntotal
    print(f"Loaded {n_total:,} vectors in {time.perf_counter()-t0:.2f}s")

    # 2. Extract vectors from the flat index
    print("Extracting vectors...")
    t1 = time.perf_counter()
    # FAISS allows reconstructing vectors from Flat indices
    vectors = index_flat.reconstruct_n(0, n_total)
    print(f"Extracted in {time.perf_counter()-t1:.2f}s")

    # 3. Create HNSW index
    # M=32 is a good balance between speed and accuracy for 1M+ vectors
    print(f"Building HNSW index (M=32, Metric=InnerProduct)...")
    t2 = time.perf_counter()
    index_hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    
    # HNSW doesn't require training, just adding
    index_hnsw.add(vectors)
    print(f"Built HNSW in {time.perf_counter()-t2:.2f}s")

    # 4. Save
    print(f"Saving to {hnsw_path}...")
    faiss.write_index(index_hnsw, hnsw_path)
    print("Done.")

if __name__ == "__main__":
    # BLaIR is 1024-dim
    rebuild_index("blair", 1024)
    
    # CLIP is 512-dim
    rebuild_index("clip", 512)
