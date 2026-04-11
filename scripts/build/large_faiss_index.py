from huggingface_hub import login, hf_hub_download
import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

def build_index(num_chunks=90):
    # Use your provided token for authentication
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_TOKEN not found in environment variables.")
    login(token=hf_token)
    
    repo_id = "minhkhang26/my-nba-project-data"
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    asins = []
    # DIMENSIONS
    clip_dim = 512
    blair_dim = 1024
    
    # We will use IndexFlatIP for exact search and reconstruction support
    clip_index = faiss.IndexFlatIP(clip_dim)
    blair_index = faiss.IndexFlatIP(blair_dim)

    print(f"Building high-precision index from ALL {num_chunks} chunks...")
    print("Warning: This will use ~15-20GB of RAM and significant disk space.")

    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        filename = f"meta_img_merged_chunks/meta_img_merged_chunk_{i:03d}.npz"
        
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            data = np.load(local_path, allow_pickle=True)
            
            # Extract and convert
            chunk_asins = list(data["asin"])
            chunk_clip = data["emb_clip_img"].astype("float32")
            chunk_blair = data["emb_blair"].astype("float32")
            
            # Normalize for Cosine Similarity
            faiss.normalize_L2(chunk_clip)
            faiss.normalize_L2(chunk_blair)
            
            # Add to indices
            clip_index.add(chunk_clip)
            blair_index.add(chunk_blair)
            
            # Store ASINs
            asins.extend(chunk_asins)
            
        except Exception as e:
            print(f"\nError processing chunk {i}: {e}")
            continue

    # Save Results
    print(f"\nSaving final indices for {len(asins)} items...")
    faiss.write_index(clip_index, os.path.join(data_dir, "clip_index.faiss"))
    faiss.write_index(blair_index, os.path.join(data_dir, "blair_index.faiss"))
    pd.Series(asins).to_csv(os.path.join(data_dir, "asins.csv"), index=False, header=False)
    
    print("Full-scale index build complete! Files saved in data/")

if __name__ == "__main__":
    build_index(num_chunks=90)
