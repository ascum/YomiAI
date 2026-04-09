import os
import shutil
import tantivy
import pandas as pd
import time
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
INDEX_PATH = os.path.join(DATA_DIR, "tantivy_index")

def build_index():
    # 1. Setup Schema: ASIN (ID), Title, and Author
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("asin", stored=True)
    schema_builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("author", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()

    # 2. Prepare Directory
    if os.path.exists(INDEX_PATH):
        print(f"Removing existing index at {INDEX_PATH}...")
        shutil.rmtree(INDEX_PATH)
    os.makedirs(INDEX_PATH)

    # 3. Initialize Index
    index = tantivy.Index(schema, path=INDEX_PATH)
    # Using 1GB heap for fast indexing
    writer = index.writer(heap_size=1024*1024*1024)

    # 4. Load Data
    meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return

    print(f"Loading metadata from {meta_path}...")
    df = pd.read_parquet(meta_path, columns=["parent_asin", "title", "author_name"])
    
    # 5. Index Documents
    print(f"Indexing {len(df):,} documents into Rust engine...")
    t0 = time.perf_counter()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        asin = row['parent_asin']
        if not asin: continue
        writer.add_document(tantivy.Document(
            asin=str(asin),
            title=str(row['title'] or ""),
            author=str(row['author_name'] or "")
        ))

    print("Committing changes...")
    writer.commit()
    # Wait for background merger
    writer.wait_merging_threads()
    
    duration = time.perf_counter() - t0
    print(f"\n✅ Success! Tantivy index built in {duration:.2f}s")
    print(f"Location: {INDEX_PATH}")

if __name__ == "__main__":
    build_index()
