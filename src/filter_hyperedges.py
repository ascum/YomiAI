import os
import pandas as pd
from tqdm import tqdm

def filter_hyperedges():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Load the 3,080,829 valid ASINs
    print("Loading valid ASINs from updated content index...")
    asins_path = os.path.join(data_dir, "asins.csv")
    valid_asins = set(pd.read_csv(asins_path, header=None)[0].astype(str).tolist())
    print(f"Syncing with {len(valid_asins)} multimodal items.")

    # 2. Filter hyperedges
    input_path = os.path.join(data_dir, "hyperedges_cleora.txt")
    output_path = os.path.join(data_dir, "hyperedges_cleora_filtered.txt")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Filtering hyperedges to match content index...")
    valid_hyperedges_count = 0
    total_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Filtering lines"):
            total_lines += 1
            asins = line.strip().split()
            # Keep only ASINs that exist in our new 3M index
            filtered_asins = [a for a in asins if a in valid_asins]
            
            # Only save the line if it still has at least 2 items (for behavioral relations)
            # or 1 item if you want to keep single-item users
            if len(filtered_asins) >= 1:
                fout.write(" ".join(filtered_asins) + "\n")
                valid_hyperedges_count += 1

    print(f"\nFiltering Complete:")
    print(f"Original Hyperedges: {total_lines}")
    print(f"Filtered Hyperedges: {valid_hyperedges_count}")
    
    # Replace the old file
    os.replace(output_path, input_path)
    print(f"Updated {input_path} successfully.")

if __name__ == "__main__":
    filter_hyperedges()
