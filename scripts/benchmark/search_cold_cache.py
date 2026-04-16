"""
scripts/benchmark/search_cold_cache.py — True cold-cache latency benchmark.

Why this exists:
    search_timing.py uses runs_per_query=3 with the same 10 queries each run.
    Runs 2 & 3 hit the in-process LRU caches (translation: 2048 entries,
    encoding: 4096 entries), so median/P50 reflects warm-cache state, not the
    realistic latency a new user with a new query experiences.

This script:
    - Uses a large pool of unique queries (never repeated between calls)
    - Runs each query exactly ONCE so every observation is a cache miss
    - Saves results with stage label "cold_cache" for comparison

Usage:
    python scripts/benchmark/search_cold_cache.py
    python scripts/benchmark/search_cold_cache.py --queries 30
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

API_URL  = "http://127.0.0.1:8000/search"
ROOT     = Path(__file__).resolve().parent.parent.parent
PROF_DIR = ROOT / "profiling"
RUNS_DIR = PROF_DIR / "runs"
HISTORY  = PROF_DIR / "history.jsonl"

# Large pool of unique queries — mix of EN and non-EN, varied lengths & topics
UNIQUE_QUERY_POOL = [
    # English
    ("mysteries set in ancient rome",           "en"),
    ("artificial intelligence for beginners",   "en"),
    ("wilderness survival handbook",            "en"),
    ("japanese poetry haiku anthology",         "en"),
    ("financial planning for millennials",      "en"),
    ("short stories magical realism",           "en"),
    ("world war two personal memoirs",          "en"),
    ("plant based cooking vegan",               "en"),
    ("architecture history gothic cathedrals",  "en"),
    ("graphic novel superhero origin",          "en"),
    ("philosophy of mind consciousness",        "en"),
    ("travel guide southeast asia backpacker",  "en"),
    ("chess strategy grandmaster games",        "en"),
    ("climate change solutions environment",    "en"),
    ("bedtime stories toddlers illustrated",    "en"),
    ("startup business lean methodology",       "en"),
    ("poetry collections modern american",      "en"),
    ("history of mathematics great theorems",   "en"),
    ("psychological thriller unreliable narrator", "en"),
    ("yoga meditation mindfulness practice",    "en"),
    # Vietnamese
    ("văn học thiếu nhi việt nam",              "vi"),
    ("truyện tranh manga nhật bản",             "vi"),
    ("lịch sử chiến tranh thế giới thứ hai",    "vi"),
    ("kỹ năng lãnh đạo quản lý",                "vi"),
    ("tiểu thuyết lãng mạn hiện đại",           "vi"),
    # French
    ("roman policier français classique",       "fr"),
    ("cuisine méditerranéenne recettes",        "fr"),
    # Spanish
    ("novela histórica edad media",             "es"),
    ("ciencia ficción latinoamericana",         "es"),
    # Chinese
    ("中国古典文学四大名著",                          "zh"),
    ("现代商业管理书籍",                              "zh"),
    # Japanese
    ("日本推理小説ミステリー",                          "ja"),
    # Korean
    ("한국 현대 소설 문학",                            "ko"),
    # German
    ("deutsche literatur klassiker goethe",     "de"),
    # Portuguese
    ("literatura brasileira contemporânea",     "pt"),
]


def sys_info() -> dict:
    info = {"platform": platform.system(), "python": platform.python_version()}
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["cuda"] = "unknown"
    return info


def next_run_number() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RUNS_DIR.glob("run_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem
    m = re.match(r"run_(\d+)", last)
    return int(m.group(1)) + 1 if m else 1


def run_benchmark(n_queries: int) -> list[dict]:
    pool = UNIQUE_QUERY_POOL[:n_queries]
    results = []

    print(f"\n  Cold-cache benchmark — {len(pool)} unique queries, 1 run each")
    print(f"  Every observation is a guaranteed LRU cache miss.\n")
    print(f"  {'#':>3}  {'Lang':>4}  {'Query':<45}  {'total_ms':>9}  {'e2e_ms':>9}")
    print("  " + "─" * 80)

    for i, (query, lang) in enumerate(pool, 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(f"{API_URL}?debug=true", json={"query": query}, timeout=15)
            t1 = time.perf_counter()
            e2e = round((t1 - t0) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                t = data.get("_debug_timings", {})
                t["e2e_wall_clock_ms"] = e2e
                t["_query"] = query
                t["_lang"] = lang
                results.append(t)
                total = t.get("total_ms", "?")
                print(f"  {i:>3}  [{lang:>2}]  {query[:45]:<45}  {total:>9}  {e2e:>9}")
            else:
                print(f"  {i:>3}  [{lang:>2}]  {query[:45]:<45}  HTTP {resp.status_code}")

        except requests.exceptions.ConnectionError:
            print("  Connection refused — is the server running on :8000?")
            sys.exit(1)
        except Exception as e:
            print(f"  {i:>3}  [{lang:>2}]  error: {e}")

        time.sleep(0.1)   # small gap to avoid thermal throttle distortion

    return results


def compute_stats(timings: list[dict]) -> dict:
    keys = [k for k in timings[0] if not k.startswith("_")]
    stats = {}
    for key in keys:
        values = [t[key] for t in timings if key in t and isinstance(t[key], (int, float))]
        if not values:
            continue
        stats[key] = {
            "avg": round(float(np.mean(values)),         2),
            "med": round(float(np.median(values)),       2),
            "p95": round(float(np.percentile(values, 95)), 2),
            "p99": round(float(np.percentile(values, 99)), 2),
            "min": round(float(np.min(values)),          2),
            "max": round(float(np.max(values)),          2),
            "std": round(float(np.std(values)),          2),
            "n":   len(values),
        }
    return stats


LABELS = {
    "translate_ms":           "NLLB Translation",
    "encode_text_ms":         "Text Encoding (BGE-M3)",
    "encode_clip_ms":         "CLIP Image Encoding",
    "tantivy_ms":             "Tantivy Keyword Search",
    "text_search_ms":         "FAISS Text Search",
    "clip_search_ms":         "FAISS Image Search",
    "rrf_ms":                 "Reciprocal Rank Fusion",
    "meta_filter_ms":         "Parquet ASIN Filter",
    "reranker_ms":            "Cross-Encoder Rerank",
    "metadata_hydration_ms":  "Metadata Lookup",
    "total_search_engine_ms": "Search Engine Total",
    "total_ms":               "Total API Internal",
    "e2e_wall_clock_ms":      "E2E Wall Clock",
}

PRINT_ORDER = [
    "translate_ms", "encode_text_ms", "encode_clip_ms",
    "tantivy_ms", "text_search_ms", "clip_search_ms",
    "rrf_ms", "meta_filter_ms", "reranker_ms",
    "metadata_hydration_ms",
    "total_search_engine_ms", "total_ms", "e2e_wall_clock_ms",
]


def print_summary(stats: dict):
    print()
    print("  COLD-CACHE RESULTS (all unique queries, 1 run each = true cache miss)")
    print(f"  {'Component':<32} {'Avg':>7} {'Med(P50)':>9} {'P95':>7} {'P99':>7} {'Max':>7}")
    print("  " + "─" * 75)
    for key in PRINT_ORDER:
        if key not in stats:
            continue
        s = stats[key]
        label = LABELS.get(key, key)
        sep = "  " + "─"*75 if key in ("metadata_hydration_ms",) else ""
        print(f"  {label:<32} {s['avg']:>7.1f} {s['med']:>9.1f} {s['p95']:>7.1f}"
              f" {s['p99']:>7.1f} {s['max']:>7.1f}")
        if sep:
            print(sep)

    # Per-language breakdown
    print()
    print("  Per-language P50 (total_ms):")
    lang_groups: dict[str, list] = {}
    for key_unused in []:  # populated below
        pass

    return stats


def print_lang_breakdown(timings: list[dict], stats: dict):
    groups: dict[str, list] = {}
    for t in timings:
        lang = t.get("_lang", "?")
        groups.setdefault(lang, []).append(t.get("total_ms", 0))

    print("  Per-language breakdown (total_ms):")
    for lang in sorted(groups):
        vals = groups[lang]
        med = float(np.median(vals))
        avg = float(np.mean(vals))
        p95 = float(np.percentile(vals, 95)) if len(vals) >= 3 else vals[-1]
        n   = len(vals)
        print(f"    [{lang:>2}]  n={n:<3}  avg={avg:6.1f}ms  P50={med:6.1f}ms  P95={p95:6.1f}ms")


def save_run(stats: dict, timings: list[dict], n_queries: int) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    num  = next_run_number()
    path = RUNS_DIR / f"run_{num:03d}_cold_cache.json"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "run_id":    f"run_{num:03d}",
        "stage":     "cold_cache",
        "note":      "Unique queries, 1 run each — true LRU cache miss measurement",
        "timestamp": ts,
        "system":    sys_info(),
        "config":    {"queries": n_queries, "runs_per_query": 1, "api_url": API_URL},
        "stats":     stats,
        "per_query": [
            {"query": t["_query"], "lang": t["_lang"], "total_ms": t.get("total_ms")}
            for t in timings
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "run_id":    payload["run_id"],
        "stage":     "cold_cache",
        "timestamp": ts,
        "file":      path.name,
        "key_metrics": {k: stats[k]["avg"] for k in
                        ["translate_ms","encode_text_ms","total_ms","e2e_wall_clock_ms"]
                        if k in stats},
    }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Cold-cache latency benchmark")
    parser.add_argument("--queries", type=int, default=len(UNIQUE_QUERY_POOL),
                        help=f"Number of unique queries to run (max {len(UNIQUE_QUERY_POOL)})")
    args = parser.parse_args()

    n = min(args.queries, len(UNIQUE_QUERY_POOL))
    timings = run_benchmark(n)

    if not timings:
        print("No data collected.")
        return

    stats = compute_stats(timings)
    print_summary(stats)
    print_lang_breakdown(timings, stats)

    saved = save_run(stats, timings, n)
    print(f"\n  Saved → {saved.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
