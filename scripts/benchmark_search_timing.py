import requests
import time
import json
import numpy as np

API_URL = "http://127.0.0.1:8000/search"

TEST_QUERIES = [
    "detective mystery novels",
    "tiểu thuyết trinh thám",  # Vietnamese
    "fantasy with magic systems",
    "lịch sử thế giới",        # Vietnamese
    "self-help for productivity",
    "cookbook for beginners",
    "science fiction space opera",
    "biography of famous scientists",
    "horror stories for kids",
    "romance novels set in Paris"
]

import requests
import time
import json
import numpy as np
import sys
import os
from datetime import datetime

API_URL = "http://127.0.0.1:8000/search"

# ... (TEST_QUERIES remains same) ...
TEST_QUERIES = [
    "detective mystery novels",
    "tiểu thuyết trinh thám",
    "fantasy with magic systems",
    "lịch sử thế giới",
    "self-help for productivity",
    "cookbook for beginners",
    "science fiction space opera",
    "biography of famous scientists",
    "horror stories for kids",
    "romance novels set in Paris"
]

def format_human_time(ms):
    # ... (remains same) ...
    seconds = int(ms // 1000)
    remainder_ms = int(ms % 1000)
    minutes = int(seconds // 60)
    remainder_seconds = int(seconds % 60)
    parts = []
    if minutes > 0: parts.append(f"{minutes}m")
    if remainder_seconds > 0 or minutes > 0: parts.append(f"{remainder_seconds}s")
    parts.append(f"{remainder_ms}ms")
    return " ".join(parts)

DESCRIPTIONS = {
    "translate_ms":           "NLLB Translation (VI -> EN)",
    "encode_blair_ms":        "BLaIR Text Encoding (GPU)",
    "encode_clip_ms":         "CLIP Image Encoding (GPU)",
    "bm25_ms":                "BM25 Keyword Search",
    "blair_search_ms":        "FAISS Text Search",
    "clip_search_ms":         "FAISS Image Search",
    "rrf_ms":                 "Reciprocal Rank Fusion",
    "meta_filter_ms":         "Parquet ASIN Filter",
    "reranker_ms":            "BGE Cross-Encoder Rerank",
    "search_engine_total_ms": "Total Search Engine Time",
    "metadata_hydration_ms":  "Metadata Parquet Lookup",
    "total_ms":               "Total API Internal Time",
    "e2e_wall_clock_ms":      "Full E2E Wall Clock Cycle"
}

def run_benchmark(stage_name, num_runs=3):
    print(f"Starting Benchmark Stage: [{stage_name}]")
    print(f"({len(TEST_QUERIES)} queries, {num_runs} runs each)\n")
    
    all_timings = []
    
    for query in TEST_QUERIES:
        print(f"Testing query: '{query}'")
        for i in range(num_runs):
            try:
                t_start = time.perf_counter()
                response = requests.post(f"{API_URL}?debug=true", json={"query": query})
                t_end = time.perf_counter()
                
                if response.status_code == 200:
                    data = response.json()
                    if "_debug_timings" in data:
                        timings = data["_debug_timings"]
                        timings["e2e_wall_clock_ms"] = round((t_end - t_start) * 1000, 2)
                        all_timings.append(timings)
                    else:
                        print(f"  Run {i+1}: No debug timings in response")
                else:
                    print(f"  Run {i+1}: Failed with status {response.status_code}")
            except Exception as e:
                print(f"  Run {i+1}: Error: {e}")
            
            time.sleep(0.2)

    if not all_timings:
        print("No timings collected. Is the API running?")
        return

    # Aggregate results
    keys = all_timings[0].keys()
    stats = {}
    for key in keys:
        values = [t[key] for t in all_timings if key in t]
        stats[key] = {
            "avg": round(float(np.mean(values)), 2),
            "p95": round(float(np.percentile(values, 95)), 2)
        }

    # Print Table
    print("\n" + "="*110)
    print(f"{'Stage / Component':<30} | {'Description':<35} | {'Avg (ms)':<12} | {'Human Readable':<15}")
    print("-" * 110)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg'], reverse=True)
    for stage, s in sorted_stats:
        desc = DESCRIPTIONS.get(stage, "N/A")
        readable = format_human_time(s['avg'])
        print(f"{stage:<30} | {desc:<35} | {s['avg']:<12} | {readable:<15}")
    print("="*110)

    # Save unique file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "metadata": {
            "stage": stage_name,
            "timestamp": timestamp,
            "queries": len(TEST_QUERIES),
            "runs_per_query": num_runs
        },
        "results": stats
    }
    
    filename = f"profiling/benchmark_{stage_name}_{timestamp}.json"
    os.makedirs("profiling", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {filename}")

if __name__ == "__main__":
    # To run: python scripts/benchmark_search_timing.py "stage_1_threading"
    stage = sys.argv[1] if len(sys.argv) > 1 else "unnamed_stage"
    run_benchmark(stage)
