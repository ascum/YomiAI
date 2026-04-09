import os
import pandas as pd
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

def extract_author_name(x):
    if isinstance(x, dict) and 'name' in x:
        return x['name']
    elif isinstance(x, str):
        return x
    return "Unknown Author"

def extract_image_url(x):
    if isinstance(x, dict) and 'large' in x:
        urls = x['large']
        if isinstance(urls, (list, np.ndarray)) and len(urls) > 0:
            return urls[0]
    return None

def build_metadata_cache(sample_size=1000000):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    asins_path = os.path.join(data_dir, "asins.csv")
    dataset_path = os.path.join(data_dir, "Books_meta_extracted", "Books_meta")
    output_path = os.path.join(data_dir, "item_metadata.parquet")

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print("Loading catalog ASINs...")
    valid_asins = set(pd.read_csv(asins_path, header=None)[0].astype(str).tolist())
    print(f"Catalog size: {len(valid_asins)}")

    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)
    
    print(f"Dataset loaded. Total records: {len(ds)}")
    print(f"Taking a sample of {sample_size} records...")
    
    metadata_list = []
    # Using a subset or iterating with a progress bar
    # For safety, let's process in chunks of 100k
    chunk_size = 100000
    for start in tqdm(range(0, min(len(ds), sample_size), chunk_size)):
        end = min(start + chunk_size, len(ds))
        subset = ds.select(range(start, end))
        
        # Convert to pandas
        df = subset.to_pandas()
        
        # Filter by our catalog
        mask = df['parent_asin'].isin(valid_asins)
        df_filtered = df[mask].copy()
        
        if not df_filtered.empty:
            # Extract nested fields
            df_filtered['author_name'] = df_filtered['author'].apply(extract_author_name)
            df_filtered['image_url'] = df_filtered['images'].apply(extract_image_url)

            # Extract description: join the list of strings, cap at 1000 chars
            def extract_description(x):
                if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
                    text = ' '.join(str(v) for v in x if v and str(v).strip())
                    return text[:1000] if text else ''
                return ''
            df_filtered['description'] = df_filtered['description'].apply(extract_description)

            # Select relevant columns
            cols = ['parent_asin', 'title', 'author_name', 'main_category', 'image_url', 'description']
            existing = [c for c in cols if c in df_filtered.columns]
            metadata_list.append(df_filtered[existing])

    if metadata_list:
        print("\nConsolidating metadata...")
        final_df = pd.concat(metadata_list, ignore_index=True)
        final_df.drop_duplicates(subset=['parent_asin'], inplace=True)
        print(f"Final metadata size: {len(final_df)} unique items.")
        
        # Save to parquet
        final_df.to_parquet(output_path, index=False)
        print(f"Metadata cache saved to {output_path}")
    else:
        print("\nNo matching metadata found in the sample!")

if __name__ == "__main__":
    # Increased sample size to 5 million to cover all 4.4M raw records (matching the 3.1M FAISS index)
    build_metadata_cache(sample_size=5000000)
