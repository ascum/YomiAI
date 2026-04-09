import os
import requests
import pandas as pd
import random
from urllib.parse import urlparse

def download_sample_covers(num_samples=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parquet_path = os.path.join(base_dir, "data", "item_metadata.parquet")
    output_dir = os.path.join(base_dir, "sample_covers")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading metadata from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Filter to ensure we have valid URLs
    valid_df = df[df['image_url'].notna() & (df['image_url'] != "")]
    
    if valid_df.empty:
        print("No valid image URLs found!")
        return
        
    print(f"Found {len(valid_df)} books with images. Sampling {num_samples}...")
    
    # Sample random rows
    samples = valid_df.sample(n=min(num_samples, len(valid_df)))
    
    downloaded = 0
    for _, row in samples.iterrows():
        url = row['image_url']
        asin = row['parent_asin']
        title = str(row['title']).replace("/", "_").replace("\\", "_").replace(":", "_")[:30].strip()
        
        # Determine extension from URL
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1]
        if not ext:
            ext = ".jpg" # fallback
            
        filename = f"{asin}_{title}{ext}"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Downloading [{asin}] {title}...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                print(f"  -> Saved to {filename}")
            else:
                print(f"  -> Failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    print(f"\nDone! Successfully downloaded {downloaded}/{num_samples} images to the 'sample_covers' folder.")

if __name__ == "__main__":
    download_sample_covers()
