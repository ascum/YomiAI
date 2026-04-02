import pandas as pd
from huggingface_hub import hf_hub_download
import os

def prepare_cleora_data(user_limit=500000, k_core=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Load valid ASINs from our content index
    print("Loading valid ASINs from content index...")
    asins_path = os.path.join(data_dir, "asins.csv")
    if not os.path.exists(asins_path):
        print("Error: asins.csv not found. Run src/build_large_index.py first.")
        return
    valid_asins = set(pd.read_csv(asins_path, header=None)[0].astype(str).tolist())
    print(f"Tracking {len(valid_asins)} valid products.")

    # 2. Download the pre-processed CSV benchmark data
    print("Downloading processed interaction data from McAuley-Lab/Amazon-Reviews-2023...")
    repo_id = "McAuley-Lab/Amazon-Reviews-2023"
    # This file contains user_id, parent_asin, rating, timestamp
    filename = "benchmark/5core/timestamp/Books.train.csv"
    
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        print(f"Reading {filename}...")
        # Columns in this CSV are typically: user_id,parent_asin,rating,timestamp
        df = pd.read_csv(local_path)
        
        # Rename columns to match our expected format if necessary
        # Usually it's parent_asin for Amazon 2023
        if 'parent_asin' not in df.columns and 'asin' in df.columns:
            df.rename(columns={'asin': 'parent_asin'}, inplace=True)
            
        print(f"Read {len(df)} total interactions.")
        
        # Filter by valid asins
        df = df[df['parent_asin'].isin(valid_asins)]
        print(f"Filtered to {len(df)} interactions matching our catalog.")

        # 3. K-Core Filtering
        print(f"Applying {k_core}-core filtering...")
        while True:
            user_counts = df['user_id'].value_counts()
            item_counts = df['parent_asin'].value_counts()
            
            mask = df['user_id'].isin(user_counts[user_counts >= k_core].index) & \
                   df['parent_asin'].isin(item_counts[item_counts >= k_core].index)
            
            new_df = df[mask]
            if len(new_df) == len(df):
                break
            df = new_df
            print(f"Refined to {len(df)} interactions...")

        # Limit to target user count
        unique_users = df['user_id'].unique()
        if len(unique_users) > user_limit:
            print(f"Limiting to {user_limit} unique users...")
            keep_users = unique_users[:user_limit]
            df = df[df['user_id'].isin(keep_users)]

        # 4. Generate hyperedge format
        print("Generating hyperedge format (one line per user)...")
        hyperedges = df.groupby('user_id')['parent_asin'].apply(lambda x: ' '.join(list(set(x)))).reset_index()
        hyperedge_path = os.path.join(data_dir, 'hyperedges_cleora.txt')
        with open(hyperedge_path, 'w', encoding='utf-8') as f:
            for edge in hyperedges['parent_asin']:
                f.write(edge + "\n")
                
        print(f"Saved {len(hyperedges)} hyperedges (users) to {hyperedge_path}")
        print(f"Behavioral items covered: {df['parent_asin'].nunique()}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prepare_cleora_data(user_limit=500000)
