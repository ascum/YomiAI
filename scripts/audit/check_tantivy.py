import tantivy
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(_ROOT, "data", "tantivy_index")

def check():
    if not os.path.exists(INDEX_PATH):
        print(f"Error: {INDEX_PATH} not found.")
        return

    index = tantivy.Index.open(INDEX_PATH)
    searcher = index.searcher()
    
    num_docs = index.searcher().num_docs
    print(f"Total documents in index: {num_docs:,}")
    
    # Try a simple search
    query_str = "jojo adventure"
    print(f"\nTesting search for: '{query_str}'")
    parser = index.parse_query(query_str, ["title"])
    results = searcher.search(parser, 5)
    
    print(f"Found {len(results.hits)} hits:")
    for score, addr in results.hits:
        doc = searcher.doc(addr)
        print(f"- [{score:.2f}] {doc['asin'][0]} | {doc['title'][0]}")

if __name__ == "__main__":
    check()
