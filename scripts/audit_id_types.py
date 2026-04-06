import pandas as pd
import numpy as np
import os

DATA_DIR = "src/data"

def audit():
    print("=== ASIN Data Type Audit ===\n")

    # 1. Check Metadata Parquet
    meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if os.path.exists(meta_path):
        df = pd.read_parquet(meta_path)
        print(f"Metadata Parquet ({len(df):,} rows):")
        print(f"  - Index Name: {df.index.name}")
        print(f"  - Index Type: {df.index.dtype}")
        sample_id = df.index[0]
        print(f"  - Sample ID: {sample_id} (Type: {type(sample_id)})")
        
        # Check if 'parent_asin' is a column instead
        if 'parent_asin' in df.columns:
            print(f"  - 'parent_asin' Column Type: {df['parent_asin'].dtype}")
    
    # 2. Check asins.csv (FAISS mapping)
    csv_path = os.path.join(DATA_DIR, "asins.csv")
    if os.path.exists(csv_path):
        # Read without header to see raw behavior
        csv_df = pd.read_csv(csv_path, header=None)
        print(f"\nFAISS asins.csv ({len(csv_df):,} rows):")
        print(f"  - Column 0 Type: {csv_df[0].dtype}")
        sample_csv = csv_df[0].iloc[0]
        print(f"  - Sample ID: {sample_csv} (Type: {type(sample_csv)})")

    # 3. Check Cleora (if applicable)
    cleora_path = os.path.join(DATA_DIR, "cleora_embeddings.npz")
    if os.path.exists(cleora_path):
        data = np.load(cleora_path)
        if 'asins' in data:
            print(f"\nCleora NPZ:")
            print(f"  - ASINs Array Type: {data['asins'].dtype}")
            sample_cl = data['asins'][0]
            print(f"  - Sample ID: {sample_cl} (Type: {type(sample_cl)})")

if __name__ == "__main__":
    audit()
