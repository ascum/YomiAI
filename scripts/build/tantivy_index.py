import os
import shutil
import tantivy
import pandas as pd
import time
from tqdm import tqdm

# Calculate DATA_DIR relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "tantivy_index")

def build_index():
    # 1. Setup Schema: ASIN, Title, Author, and Genres
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("asin", stored=True)
    schema_builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("author", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("genres", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()

    # 2. Prepare Directory
    if os.path.exists(INDEX_PATH):
        print(f"Removing existing index at {INDEX_PATH}...")
        shutil.rmtree(INDEX_PATH)
    os.makedirs(INDEX_PATH)

    # 3. Initialize Index
    index = tantivy.Index(schema, path=INDEX_PATH)
    writer = index.writer(heap_size=1024*1024*1024)

    # 4. Load Data
    meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return

    print(f"Loading metadata from {meta_path}...")
    # Load 'categories' column which contains the rich genre data
    df = pd.read_parquet(meta_path, columns=["parent_asin", "title", "author_name", "categories"])
    df = df.set_index("parent_asin")
    
    # 5. Index Documents
    print(f"Indexing {len(df):,} documents into Rust engine...")
    t0 = time.perf_counter()
    
    for asin, row in tqdm(df.iterrows(), total=len(df)):
        if not asin: continue
        
        # Clean categories (replace pipe separators with spaces for the search engine)
        raw_cats = str(row.get("categories", ""))
        clean_genres = raw_cats.replace("|", " ").replace("&", "and") if raw_cats != "nan" else ""

        writer.add_document(tantivy.Document(
            asin=str(asin),
            title=str(row['title'] or ""),
            author=str(row['author_name'] or ""),
            genres=clean_genres
        ))

    print("Committing changes...")
    writer.commit()
    writer.wait_merging_threads()
    
    duration = time.perf_counter() - t0
    print(f"\n✅ Success! Tantivy index built in {duration:.2f}s")
    print(f"Location: {INDEX_PATH}")

if __name__ == "__main__":
    build_index()
