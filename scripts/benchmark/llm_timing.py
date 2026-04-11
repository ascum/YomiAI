"""
scripts/benchmark/llm_timing.py — LLM assistant latency benchmark.

Usage:
    python scripts/benchmark/llm_timing.py                        # unnamed run
    python scripts/benchmark/llm_timing.py "optimized_greedy"      # named run
    python scripts/benchmark/llm_timing.py --compare              # diff last 2 runs
    python scripts/benchmark/llm_timing.py --runs 3               # 3 runs per book
    python scripts/benchmark/llm_timing.py --list                 # show run history
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

# ── Config ────────────────────────────────────────────────────────────────────

API_URL  = "http://127.0.0.1:8000/ask_llm"
ROOT     = Path(__file__).resolve().parent.parent.parent
PROF_DIR = ROOT / "profiling"
RUNS_DIR = PROF_DIR / "runs_llm"
HISTORY  = PROF_DIR / "history_llm.jsonl"

TEST_BOOKS = [
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("Clean Code", "Robert C. Martin"),
    ("1984", "George Orwell"),
    ("The Hobbit", "J.R.R. Tolkien"),
    ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("The Catcher in the Rye", "J.D. Salinger"),
    ("Brave New World", "Aldous Huxley"),
    ("Fahrenheit 451", "Ray Bradbury"),
    ("Dune", "Frank Herbert"),
]

LABELS = {
    "wiki_fetch_ms":     "Wikipedia Fetch",
    "llm_gen_ms":       "LLM Generation",
    "total_ms":         "Total API Internal",
    "e2e_wall_clock_ms": "E2E Wall Clock",
}

BUDGET = {
    "wiki_fetch_ms":     1000,
    "llm_gen_ms":       10000,
    "total_ms":         11000,
    "e2e_wall_clock_ms": 12000,
}

ORDERED_KEYS = ["wiki_fetch_ms", "llm_gen_ms"]
SUMMARY_KEYS = ["total_ms", "e2e_wall_clock_ms"]

# ── Formatting helpers ────────────────────────────────────────────────────────

def ms(v) -> str:
    return f"{v:7.1f}"

def delta_str(new_v, old_v) -> str:
    d = new_v - old_v
    sign = "+" if d >= 0 else ""
    pct  = (d / old_v * 100) if old_v else 0
    arrow = "▲" if d > 10 else ("▼" if d < -10 else "─")
    return f"{sign}{d:+6.1f}ms ({pct:+.0f}%) {arrow}"

def budget_flag(key, avg) -> str:
    limit = BUDGET.get(key)
    if limit is None:
        return ""
    return "✓" if avg <= limit else f"⚠ >{limit}"

def bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)

def next_run_number() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RUNS_DIR.glob("run_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem
    m = re.match(r"run_(\d+)", last)
    return int(m.group(1)) + 1 if m else 1

def sys_info() -> dict:
    info = {"platform": platform.system(), "python": platform.python_version()}
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["cuda"] = "unknown"
    return info

# ── Collection ────────────────────────────────────────────────────────────────

def collect(num_runs: int, warmup: bool) -> tuple[list, list]:
    all_timings = []
    per_book    = []

    total_calls = len(TEST_BOOKS) * (num_runs + (1 if warmup else 0))
    done = 0

    for title, author in TEST_BOOKS:
        book_runs = []
        total_iters = num_runs + (1 if warmup else 0)

        for i in range(total_iters):
            is_warmup = warmup and i == 0
            label = "warmup" if is_warmup else f"run {i if warmup else i+1}/{num_runs}"
            done += 1
            pct_done = done / total_calls * 100
            print(f"  [{pct_done:5.1f}%] {label:8s}  {title[:45]}", end="  ", flush=True)

            try:
                t_start = time.perf_counter()
                response = requests.post(
                    f"{API_URL}?debug=true",
                    json={
                        "item_id": "bench",
                        "title": title,
                        "author": author,
                        "user_prompt": "Tell me about this book."
                    },
                    timeout=30
                )
                t_end = time.perf_counter()

                if response.status_code == 200:
                    data = response.json()
                    if "_debug_timings" in data:
                        t = data["_debug_timings"].copy()
                        t["e2e_wall_clock_ms"] = round((t_end - t_start) * 1000, 2)
                        if not is_warmup:
                            all_timings.append(t)
                            book_runs.append(t)
                        print(f"→ {t.get('total_ms', '?')}ms total")
                    else:
                        print("→ no debug timings")
                else:
                    print(f"→ HTTP {response.status_code}")
            except Exception as e:
                print(f"→ error: {e}")

            # Sleep slightly to avoid slamming the Wikipedia API
            time.sleep(0.1)

        per_book.append({"title": title, "author": author, "runs": book_runs})

    return all_timings, per_book

# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(timings: list) -> dict:
    if not timings:
        return {}
    keys = [k for k in timings[0] if not k.startswith("_")]
    stats = {}
    for key in keys:
        values = [t[key] for t in timings if key in t]
        if not values:
            continue
        stats[key] = {
            "avg":  round(float(np.mean(values)),       2),
            "med":  round(float(np.median(values)),     2),
            "p95":  round(float(np.percentile(values, 95)), 2),
            "max":  round(float(np.max(values)),        2),
            "std":  round(float(np.std(values)),        2),
            "n":    len(values),
        }
    return stats

# ── Display ───────────────────────────────────────────────────────────────────

W = 100

def print_header(stage: str, n_books: int, num_runs: int, timestamp: str, run_label: str):
    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = f"  LLM TIMING BENCHMARK  —  {run_label}"
    print(f"║{title:<{W-2}}║")
    sub = f"  Stage: {stage}   │   {timestamp}   │   {n_books} books × {num_runs} runs"
    print(f"║{sub:<{W-2}}║")
    print("╚" + "═" * (W - 2) + "╝")

def print_stats_table(stats: dict, compare: dict = None):
    C = [28, 8, 8, 8, 8, 20, 8]
    head = (f"  {'Component':<{C[0]}} {'Avg':>{C[1]}} {'Med':>{C[2]}} {'P95':>{C[3]}} "
            f"{'Max':>{C[4]}}  {'% of total':<{C[5]}} {'Budget':<{C[6]}}")
    if compare:
        head += f"  {'vs prev (avg)':>20}"
    print()
    print(head)
    print("  " + "─" * (W - 2))

    total_avg = stats.get("total_ms", {}).get("avg") or 1.0

    def print_row(key, separator=False):
        if key not in stats:
            return
        s    = stats[key]
        label = LABELS.get(key, key)
        frac  = s["avg"] / total_avg
        bflag = budget_flag(key, s["avg"])
        row = (f"  {label:<{C[0]}} {ms(s['avg']):>{C[1]}} {ms(s['med']):>{C[2]}} "
               f"{ms(s['p95']):>{C[3]}} {ms(s['max']):>{C[4]}}  "
               f"{bar(frac):<{C[5]}} {bflag:<{C[6]}}")
        if compare and key in compare:
            row += f"  {delta_str(s['avg'], compare[key]['avg']):>20}"
        print(row)
        if separator:
            print("  " + "─" * (W - 2))

    for key in ORDERED_KEYS:
        print_row(key)

    print("  " + "─" * (W - 2))
    for key in SUMMARY_KEYS:
        print_row(key, separator=(key == "total_ms"))

def save_run(stage: str, stats: dict, per_book: list, num_runs: int, timestamp: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    num = next_run_number()
    slug = re.sub(r"[^\w]", "_", stage)[:40]
    filename = RUNS_DIR / f"run_{num:03d}_{slug}.json"

    payload = {
        "run_id":    f"run_{num:03d}",
        "stage":     stage,
        "timestamp": timestamp,
        "system":    sys_info(),
        "stats":     stats,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = {
        "run_id":    payload["run_id"],
        "stage":     stage,
        "timestamp": timestamp,
        "file":      str(filename.name),
        "key_metrics": {
            k: stats[k]["avg"] for k in SUMMARY_KEYS if k in stats
        },
    }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    return filename

def list_runs():
    if not HISTORY.exists():
        print("No LLM benchmark history found.")
        return
    print()
    print(f"  {'Run ID':<10} {'Timestamp':<18} {'Stage':<35} {'total':>10} {'e2e':>10}")
    print("  " + "─" * 100)
    with open(HISTORY, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = r.get("key_metrics", {})
            print(f"  {r['run_id']:<10} {r['timestamp']:<18} {r['stage']:<35} "
                  f"{m.get('total_ms', '─'):>10.1f} "
                  f"{m.get('e2e_wall_clock_ms', '─'):>10.1f}")

def main():
    parser = argparse.ArgumentParser(description="LLM pipeline latency benchmark")
    parser.add_argument("stage",           nargs="?", default="unnamed",
                        help="Descriptive stage name")
    parser.add_argument("--runs",          type=int, default=2,
                        help="Runs per book (default: 2)")
    parser.add_argument("--warmup",        action="store_true",
                        help="Discard first run")
    parser.add_argument("--compare",       nargs="?", const="prev",
                        help="Compare against previous run")
    parser.add_argument("--list",          action="store_true",
                        help="List runs")
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if args.compare is not None:
        candidates = sorted(RUNS_DIR.glob("run_*.json"))
        if len(candidates) < 2:
            print("Need at least 2 saved runs to compare.")
            return
        run_b_path = candidates[-1]
        run_a_path = candidates[-2] if args.compare == "prev" else next(p for p in candidates if args.compare in p.name)
        
        with open(run_a_path, encoding="utf-8") as f: data_a = json.load(f)
        with open(run_b_path, encoding="utf-8") as f: data_b = json.load(f)
        
        print(f"\nComparing {run_a_path.stem} vs {run_b_path.stem}")
        print_stats_table(data_b["stats"], compare=data_a["stats"])
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_timings, per_book = collect(args.runs, args.warmup)
    
    if not all_timings: return
    
    stats = compute_stats(all_timings)
    print_header(args.stage, len(TEST_BOOKS), args.runs, timestamp, f"run_{next_run_number():03d}")
    print_stats_table(stats)
    
    save_run(args.stage, stats, per_book, args.runs, timestamp)

if __name__ == "__main__":
    main()
