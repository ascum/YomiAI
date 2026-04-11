import time
import logging
import sys
import os

# Add the project root to sys.path to import app modules
sys.path.append(os.getcwd())

from app.services import llm as llm_service

logging.basicConfig(level=logging.INFO)

def benchmark_llm(title, author, user_prompt):
    print(f"Benchmarking LLM for: {title} by {author}")
    
    start_total = time.time()
    
    # 1. Loading
    start_load = time.time()
    loaded = llm_service.ensure_loaded()
    end_load = time.time()
    print(f"Loading/Ensuring loaded: {end_load - start_load:.4f}s")
    
    if not loaded:
        print("Failed to load LLM")
        return

    # 2. Wikipedia fetch
    start_wiki = time.time()
    wiki_summary = llm_service.fetch_wikipedia_summary(title)
    end_wiki = time.time()
    print(f"Wikipedia fetch: {end_wiki - start_wiki:.4f}s")
    
    # 3. Generation
    start_gen = time.time()
    # We'll call a modified version or just the generate function
    # To be precise, let's look at what generate does
    answer = llm_service.generate(title, author, user_prompt)
    end_gen = time.time()
    print(f"LLM Generation: {end_gen - start_gen:.4f}s")
    
    end_total = time.time()
    print(f"Total time: {end_total - start_total:.4f}s")
    print(f"Response: {answer[:100]}...")

if __name__ == "__main__":
    test_books = [
        ("The Great Gatsby", "F. Scott Fitzgerald", ""),
        ("Clean Code", "Robert C. Martin", ""),
        ("DoesNotExistBook123", "Unknown", "")
    ]
    
    for title, author, prompt in test_books:
        benchmark_llm(title, author, prompt)
        print("-" * 30)
