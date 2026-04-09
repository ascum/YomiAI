import tantivy
import pandas as pd
import os
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(_ROOT, "data", "tantivy_index")
META_PATH  = os.path.join(_ROOT, "data", "item_metadata.parquet")

def investigate(query_str):
    print(f"=== Deep Investigation: '{query_str}' ===\n")

    # 1. Check Metadata
    print("1. Loading Metadata...")
    df = pd.read_parquet(META_PATH)
    # Check current index state
    print(f"   - Index type: {df.index.dtype}")
    print(f"   - Sample ID from index: '{df.index[0]}' (Type: {type(df.index[0])})")
    
    # Simulate the api.py fix
    print("   - Applying api.py string-force fix...")
    if "parent_asin" in df.columns:
        df["parent_asin"] = df["parent_asin"].astype(str)
        df.set_index("parent_asin", inplace=True)
    else:
        df.index = df.index.map(str)
    print(f"   - New index type: {df.index.dtype}")
    print(f"   - Sample ID after fix: '{df.index[0]}' (Type: {type(df.index[0])})")

    # 2. Check Tantivy
    print("\n2. Opening Tantivy Index...")
    index = tantivy.Index.open(INDEX_PATH)
    searcher = index.searcher()
    
    # Simulate the query parsing
    print(f"   - Parsing query: '{query_str}'")
    parser = index.parse_query(query_str, ["title", "author"])
    results = searcher.search(parser, 10)
    
    if not results.hits:
        print("   - FAILED: Tantivy returned 0 hits for this query.")
        return

    print(f"   - Found {len(results.hits)} raw hits.")

    # 3. Trace the Filtering
    print("\n3. Tracing the Filtering Loop:")
    for score, addr in results.hits:
        doc = searcher.doc(addr)
        asin_from_rust = str(doc["asin"][0])
        
        in_metadata = asin_from_rust in df.index
        
        print(f"   - Hit ASIN: '{asin_from_rust}' | Score: {score:.2f} | In Metadata? {in_metadata}")
        
        if not in_metadata:
            # If not found, let's see why. Maybe a space?
            # Check for partial matches in index
            exists_stripped = asin_from_rust.strip() in df.index
            exists_int = False
            if asin_from_rust.isdigit():
                exists_int = int(asin_from_rust) in df.index
            
            print(f"     -> Debug: StripMatch={exists_stripped}, IntMatch={exists_int}")

if __name__ == "__main__":
    investigate("jojo bizarre")
