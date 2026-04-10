"""
scripts/benchmark/search_timing.py — Search pipeline latency benchmark.

Usage:
    python scripts/benchmark/search_timing.py                        # unnamed run
    python scripts/benchmark/search_timing.py "post_nllb_600m"      # named run
    python scripts/benchmark/search_timing.py --compare              # diff last 2 runs
    python scripts/benchmark/search_timing.py --compare run_004      # diff latest vs named
    python scripts/benchmark/search_timing.py --runs 5               # 5 runs per query
    python scripts/benchmark/search_timing.py --warmup               # discard first run
    python scripts/benchmark/search_timing.py --list                 # show run history
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

API_URL  = "http://127.0.0.1:8000/search"
ROOT     = Path(__file__).resolve().parent.parent.parent
PROF_DIR = ROOT / "profiling"
RUNS_DIR = PROF_DIR / "runs"
HISTORY  = PROF_DIR / "history.jsonl"   # append-only, one line per run

# Queries labelled by language so we can split EN vs non-EN stats
TEST_QUERIES = [
    ("detective mystery novels",        "en"),
    ("tiểu thuyết trinh thám",          "vi"),
    ("fantasy with magic systems",      "en"),
    ("lịch sử thế giới",                "vi"),
    ("self-help for productivity",      "en"),
    ("cookbook for beginners",          "en"),
    ("science fiction space opera",     "en"),
    ("biography of famous scientists",  "en"),
    ("horror stories for kids",         "en"),
    ("romance novels set in Paris",     "en"),
]

# Human-readable labels for each timing key returned by the API
LABELS = {
    "translate_ms":           "NLLB Translation",
    "encode_text_ms":         "Text Encoding (BGE-M3)",
    "encode_clip_ms":         "CLIP Image Encoding",
    "tantivy_ms":             "Tantivy Keyword Search",
    "text_search_ms":         "FAISS Text Search",
    "clip_search_ms":         "FAISS Image Search",
    "rrf_ms":                 "Reciprocal Rank Fusion",
    "meta_filter_ms":         "Parquet ASIN Filter",
    "reranker_ms":            "BGE Cross-Encoder Rerank",
    "total_search_engine_ms": "Search Engine Total",
    "metadata_hydration_ms":  "Metadata Parquet Lookup",
    "total_ms":               "Total API Internal",
    "e2e_wall_clock_ms":      "E2E Wall Clock",
}

# Budget thresholds (ms) — flagged if avg exceeds
BUDGET = {
    "translate_ms":           100,
    "encode_text_ms":          50,
    "encode_clip_ms":          80,
    "tantivy_ms":              20,
    "text_search_ms":          10,
    "rrf_ms":                  10,
    "meta_filter_ms":          10,
    "metadata_hydration_ms":   20,
    "total_ms":               300,
    "e2e_wall_clock_ms":      350,
}

# Print order (summary keys last)
ORDERED_KEYS = [
    "translate_ms", "encode_text_ms", "encode_clip_ms",
    "tantivy_ms", "text_search_ms", "clip_search_ms",
    "rrf_ms", "meta_filter_ms", "reranker_ms",
    "metadata_hydration_ms",
]
SUMMARY_KEYS = ["total_search_engine_ms", "total_ms", "e2e_wall_clock_ms"]


# ── Formatting helpers ────────────────────────────────────────────────────────

def ms(v) -> str:
    """Right-aligned ms value, 2 decimal places."""
    return f"{v:7.1f}"

def delta_str(new_v, old_v) -> str:
    d = new_v - old_v
    sign = "+" if d >= 0 else ""
    pct  = (d / old_v * 100) if old_v else 0
    arrow = "▲" if d > 1 else ("▼" if d < -1 else "─")
    return f"{sign}{d:+6.1f}ms ({pct:+.0f}%) {arrow}"

def budget_flag(key, avg) -> str:
    limit = BUDGET.get(key)
    if limit is None:
        return ""
    return "✓" if avg <= limit else f"⚠ >{limit}"

def bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)

def run_id_from_path(path: Path) -> str:
    return path.stem   # e.g. "run_007_post_nllb_600m"

def next_run_number() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RUNS_DIR.glob("run_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem  # "run_007_..."
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
    """
    Run all test queries and return (all_timings, per_query_timings).
    per_query_timings: list of {query, lang, runs: [timing_dict]}
    """
    all_timings     = []
    per_query       = []

    total_calls = len(TEST_QUERIES) * (num_runs + (1 if warmup else 0))
    done = 0

    for query, lang in TEST_QUERIES:
        query_runs = []
        total_iters = num_runs + (1 if warmup else 0)

        for i in range(total_iters):
            is_warmup = warmup and i == 0
            label = "warmup" if is_warmup else f"run {i if warmup else i+1}/{num_runs}"
            done += 1
            pct_done = done / total_calls * 100
            print(f"  [{pct_done:5.1f}%] {label:8s}  {query[:45]}", end="  ", flush=True)

            try:
                t_start = time.perf_counter()
                response = requests.post(f"{API_URL}?debug=true", json={"query": query},
                                         timeout=10)
                t_end = time.perf_counter()

                if response.status_code == 200:
                    data = response.json()
                    if "_debug_timings" in data:
                        t = data["_debug_timings"].copy()
                        t["e2e_wall_clock_ms"] = round((t_end - t_start) * 1000, 2)
                        t["_n_results"] = len(data.get("results", []))
                        if not is_warmup:
                            all_timings.append(t)
                            query_runs.append(t)
                        print(f"→ {t.get('total_ms', '?')}ms total  ({t['_n_results']} results)")
                    else:
                        print("→ no debug timings")
                else:
                    print(f"→ HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("→ connection refused — is the server running?")
                return [], []
            except Exception as e:
                print(f"→ error: {e}")

            time.sleep(0.2)

        per_query.append({"query": query, "lang": lang, "runs": query_runs})

    return all_timings, per_query


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
            "p99":  round(float(np.percentile(values, 99)), 2),
            "min":  round(float(np.min(values)),        2),
            "max":  round(float(np.max(values)),        2),
            "std":  round(float(np.std(values)),        2),
            "n":    len(values),
        }
    return stats

def per_lang_stats(per_query: list) -> dict:
    """Group timings by language, return {lang: stats_dict}."""
    groups: dict[str, list] = {}
    for pq in per_query:
        lang = pq["lang"]
        groups.setdefault(lang, []).extend(pq["runs"])
    return {lang: compute_stats(runs) for lang, runs in groups.items()}

def slowest_queries(per_query: list, n: int = 5) -> list:
    """Return top-N slowest queries by avg total_ms."""
    rows = []
    for pq in per_query:
        if not pq["runs"]:
            continue
        avg = float(np.mean([r.get("total_ms", 0) for r in pq["runs"]]))
        rows.append((avg, pq["query"], pq["lang"]))
    return sorted(rows, reverse=True)[:n]


# ── Display ───────────────────────────────────────────────────────────────────

W = 100   # total line width

def print_header(stage: str, n_queries: int, num_runs: int, timestamp: str, run_label: str):
    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = f"  SEARCH TIMING BENCHMARK  —  {run_label}"
    print(f"║{title:<{W-2}}║")
    sub = f"  Stage: {stage}   │   {timestamp}   │   {n_queries} queries × {num_runs} runs = {n_queries*num_runs} samples"
    print(f"║{sub:<{W-2}}║")
    print("╚" + "═" * (W - 2) + "╝")

def print_stats_table(stats: dict, compare: dict = None):
    # Column widths
    C = [28, 8, 8, 8, 8, 8, 20, 8]   # label, avg, med, p95, p99, max, bar, budget
    head = (f"  {'Component':<{C[0]}} {'Avg':>{C[1]}} {'Med':>{C[2]}} {'P95':>{C[3]}} "
            f"{'P99':>{C[4]}} {'Max':>{C[5]}}  {'% of total':<{C[6]}} {'Budget':<{C[7]}}")
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
               f"{ms(s['p95']):>{C[3]}} {ms(s['p99']):>{C[4]}} {ms(s['max']):>{C[5]}}  "
               f"{bar(frac):<{C[6]}} {bflag:<{C[7]}}")
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

def print_language_breakdown(lang_stats: dict):
    print()
    print("  ── By Query Language " + "─" * 60)
    for lang, stats in sorted(lang_stats.items()):
        if "total_ms" not in stats:
            continue
        s   = stats["total_ms"]
        t   = stats.get("translate_ms", {})
        row = (f"  [{lang.upper()}]  avg {s['avg']:.0f}ms  "
               f"med {s['med']:.0f}ms  p95 {s['p95']:.0f}ms  "
               f"max {s['max']:.0f}ms")
        if t:
            row += f"   (translation avg {t['avg']:.0f}ms)"
        print(row)

def print_slowest(slowest: list):
    print()
    print("  ── Slowest Queries " + "─" * 62)
    for rank, (avg, query, lang) in enumerate(slowest, 1):
        print(f"  {rank}.  {avg:6.0f}ms  [{lang}]  \"{query}\"")

def print_variability(stats: dict):
    """Flag keys with high coefficient of variation (std/avg > 0.3)."""
    noisy = []
    for key, s in stats.items():
        if key in SUMMARY_KEYS or key == "e2e_wall_clock_ms":
            continue
        if s["avg"] > 0.5 and s["std"] / s["avg"] > 0.30:
            noisy.append((s["std"] / s["avg"], key, s))
    if not noisy:
        return
    print()
    print("  ── High Variability (std/avg > 30%) " + "─" * 45)
    for cv, key, s in sorted(noisy, reverse=True):
        label = LABELS.get(key, key)
        print(f"  {label:<30}  avg {s['avg']:6.1f}ms  std {s['std']:6.1f}ms  "
              f"cv {cv:.0%}  range [{s['min']:.1f} – {s['max']:.1f}]")

def print_comparison_header(run_a: str, run_b: str):
    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = f"  COMPARISON  {run_a}  →  {run_b}"
    print(f"║{title:<{W-2}}║")
    print("╚" + "═" * (W - 2) + "╝")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_run(stage: str, stats: dict, per_query: list, num_runs: int,
             timestamp: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    num = next_run_number()
    slug = re.sub(r"[^\w]", "_", stage)[:40]
    filename = RUNS_DIR / f"run_{num:03d}_{slug}.json"

    payload = {
        "run_id":    f"run_{num:03d}",
        "stage":     stage,
        "timestamp": timestamp,
        "system":    sys_info(),
        "config": {
            "queries":      len(TEST_QUERIES),
            "runs_per_query": num_runs,
            "api_url":      API_URL,
        },
        "stats": stats,
        "per_query": [
            {
                "query": pq["query"],
                "lang":  pq["lang"],
                "avg_total_ms": round(float(np.mean([r.get("total_ms", 0) for r in pq["runs"]])), 2)
                                if pq["runs"] else None,
            }
            for pq in per_query
        ],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Append compact summary to history.jsonl
    summary = {
        "run_id":    payload["run_id"],
        "stage":     stage,
        "timestamp": timestamp,
        "file":      str(filename.name),
        "key_metrics": {
            k: stats[k]["avg"]
            for k in ["translate_ms", "encode_text_ms", "total_ms", "e2e_wall_clock_ms"]
            if k in stats
        },
    }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    return filename

def load_run(identifier: str) -> tuple[dict, str]:
    """Load a run by run_id (e.g. 'run_004') or partial stage name."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(RUNS_DIR.glob("*.json"))
    if not candidates:
        return {}, ""

    # Exact run_id match first
    for p in candidates:
        if p.stem.startswith(identifier):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("stats", {}), p.stem

    # Fallback: most-recent
    with open(candidates[-1], encoding="utf-8") as f:
        d = json.load(f)
    return d.get("stats", {}), candidates[-1].stem

def list_runs():
    if not HISTORY.exists():
        print("No benchmark history found.")
        return
    print()
    print(f"  {'Run ID':<10} {'Timestamp':<18} {'Stage':<35} "
          f"{'translate':>10} {'total':>8} {'e2e':>8}")
    print("  " + "─" * 100)
    with open(HISTORY, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = r.get("key_metrics", {})
            print(f"  {r['run_id']:<10} {r['timestamp']:<18} {r['stage']:<35} "
                  f"{m.get('translate_ms', '─'):>10} "
                  f"{m.get('total_ms', '─'):>8} "
                  f"{m.get('e2e_wall_clock_ms', '─'):>8}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Search pipeline latency benchmark")
    parser.add_argument("stage",           nargs="?", default="unnamed",
                        help="Descriptive stage name (saved in log)")
    parser.add_argument("--runs",          type=int, default=3,
                        help="Runs per query (default: 3)")
    parser.add_argument("--warmup",        action="store_true",
                        help="Run one discarded warmup pass before recording")
    parser.add_argument("--compare",       nargs="?", const="prev", metavar="RUN_ID",
                        help="Compare latest run against a previous one "
                             "(default: second-to-last)")
    parser.add_argument("--list",          action="store_true",
                        help="List all recorded runs and exit")
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if args.compare is not None:
        # Load last two runs and diff them
        candidates = sorted(RUNS_DIR.glob("run_*.json")) if RUNS_DIR.exists() else []
        if len(candidates) < 2:
            print("Need at least 2 saved runs to compare. Run the benchmark first.")
            return

        run_b_path = candidates[-1]
        if args.compare == "prev":
            run_a_path = candidates[-2]
        else:
            matches = [p for p in candidates if args.compare in p.stem]
            run_a_path = matches[-1] if matches else candidates[-2]

        with open(run_a_path, encoding="utf-8") as f:
            data_a = json.load(f)
        with open(run_b_path, encoding="utf-8") as f:
            data_b = json.load(f)

        stats_a, stats_b = data_a["stats"], data_b["stats"]
        print_comparison_header(run_a_path.stem, run_b_path.stem)

        # Print table B with A as comparison baseline
        print_stats_table(stats_b, compare=stats_a)

        # Summary delta
        print()
        print("  ── Summary Delta " + "─" * 64)
        for key in ["translate_ms", "encode_text_ms", "total_ms", "e2e_wall_clock_ms"]:
            if key in stats_a and key in stats_b:
                label = LABELS.get(key, key)
                print(f"  {label:<30}  "
                      f"{stats_a[key]['avg']:7.1f}ms  →  {stats_b[key]['avg']:7.1f}ms  "
                      f"  {delta_str(stats_b[key]['avg'], stats_a[key]['avg'])}")
        print()
        return

    # ── Normal benchmark run ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nBenchmark stage : {args.stage}")
    print(f"Queries         : {len(TEST_QUERIES)}")
    print(f"Runs per query  : {args.runs}" + (" + 1 warmup (discarded)" if args.warmup else ""))
    print(f"Warmup pass     : {'yes' if args.warmup else 'no'}")
    print()

    all_timings, per_query = collect(args.runs, args.warmup)

    if not all_timings:
        print("\nNo data collected — aborting.")
        return

    stats      = compute_stats(all_timings)
    lang_stats = per_lang_stats(per_query)
    slowest    = slowest_queries(per_query)

    # ── Output ────────────────────────────────────────────────────────────────
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_header(args.stage, len(TEST_QUERIES), args.runs, timestamp,
                 f"run_{next_run_number():03d}")
    print_stats_table(stats)
    print_language_breakdown(lang_stats)
    print_slowest(slowest)
    print_variability(stats)

    # ── Save ──────────────────────────────────────────────────────────────────
    saved_path = save_run(args.stage, stats, per_query, args.runs, ts_file)
    print()
    print(f"  Saved  →  {saved_path.relative_to(ROOT)}")
    print(f"  History→  {HISTORY.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
