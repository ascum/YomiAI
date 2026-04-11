import requests
import time
import json

def test_stream(title, author):
    url = "http://127.0.0.1:8000/ask_llm_stream"
    payload = {
        "item_id": "test",
        "title": title,
        "author": author,
        "user_prompt": "Tell me about this book."
    }
    
    print(f"\nTesting stream for: {title}")
    start_time = time.perf_counter()
    first_token_time = None
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=30) as r:
            if r.status_code != 200:
                print(f"Error: {r.status_code}")
                return
            
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        print(f"TTFT (Time to First Token): {first_token_time - start_time:.4f}s")
                    
                    text = chunk.decode("utf-8")
                    print(text, end="", flush=True)
            
            end_time = time.perf_counter()
            print(f"\n\nTotal time: {end_time - start_time:.4f}s")
    except Exception as e:
        print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    test_stream("1984", "George Orwell")
