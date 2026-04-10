"""
scripts/benchmark_search.py
============================
Task 1.1 — End-to-end pipeline latency benchmark.

Sends 5 representative queries to a running FastAPI server on localhost:8000
(must have been started with: uvicorn api:app --host 0.0.0.0 --port 8000)
with the ?debug=true flag to collect per-stage timing breakdowns.

Usage:
    python scripts/benchmark_search.py
    python scripts/benchmark_search.py --host localhost --port 8000 --runs 3

Output:
    - Formatted timing table printed to stdout
    - Results saved to profiling/benchmark_results_<date>.txt
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

# ─── Ensure running from project root ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Test query definitions ───────────────────────────────────────────────────
def _make_gray_image_b64(size: int = 224) -> str:
    """Generate a minimal gray 224x224 JPEG encoded as base64 (fallback placeholder)."""
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (size, size), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Emit a 1-pixel JPEG as a minimal placeholder (still valid base64)
        TINY_JPEG = (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
            "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
            "AAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/"
            "aAAwDAQACEQMRAD8AJQAB/9k="
        )
        return TINY_JPEG


# Build test image once
_GRAY_IMAGE_B64 = _make_gray_image_b64()

# Check for real book cover JPGs in sample_covers/
_SAMPLE_COVERS_DIR = ROOT / "sample_covers"
_real_image_b64 = None
if _SAMPLE_COVERS_DIR.exists():
    for jpg in sorted(_SAMPLE_COVERS_DIR.glob("*.jpg"))[:1]:
        try:
            _real_image_b64 = base64.b64encode(jpg.read_bytes()).decode()
            print(f"[benchmark] Using real cover image: {jpg.name}")
            break
        except Exception:
            pass

_IMAGE_B64 = _real_image_b64 or _GRAY_IMAGE_B64

TEST_CASES = [
    {
        "label": "Short query (keyword scan path)",
        "payload": {"query": "jojo's bizarre adventure", "top_k": 10},
        "image": False,
    },
    {
        "label": "Long query (pure dense retrieval)",
        "payload": {"query": "gritty detective novels set in noir cities", "top_k": 10},
        "image": False,
    },
    {
        "label": "Vietnamese query (VI→EN translation path)",
        "payload": {"query": "tiểu thuyết trinh thám", "top_k": 10},
        "image": False,
    },
    {
        "label": "Image-only query",
        "payload": {"query": "", "image_base64": _IMAGE_B64, "top_k": 10},
        "image": True,
    },
    {
        "label": "Hybrid text + image",
        "payload": {"query": "dark fantasy", "image_base64": _IMAGE_B64, "top_k": 10},
        "image": True,
    },
    {
        "label": "Nonsense query (threshold test)",
        "payload": {"query": "xzqwerty blorp nonsense", "top_k": 10},
        "image": False,
    },
]

# ─── HTTP helper ─────────────────────────────────────────────────────────────

def post_search(host: str, port: int, payload: dict) -> tuple[dict, float]:
    """
    POST /search?debug=true and return (response_dict, wall_clock_ms).
    Raises on HTTP errors.
    """
    url = f"http://{host}:{port}/search?debug=true"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    wall_ms = (time.perf_counter() - t0) * 1000
    return data, wall_ms


# ─── Report helpers ──────────────────────────────────────────────────────────

STAGE_KEYS = [
    ("encode_text_ms",        "Text encode"),
    ("encode_clip_ms",        "CLIP encode"),
    ("faiss_rrf_rerank_ms",   "FAISS+RRF+Rerank"),
    ("metadata_hydration_ms", "Metadata hydration"),
    ("total_ms",              "TOTAL"),
]

BUDGET_MS = {
    "encode_text_ms":        100,
    "encode_clip_ms":        80,
    "faiss_rrf_rerank_ms":   600,   # FAISS (<50ms) + Reranker (<400ms) combined
    "metadata_hydration_ms": 20,
    "total_ms":              1000,
}


def format_table(label: str, timings: dict, wall_ms: float, n_results: int) -> str:
    lines = []
    lines.append(f"\n{'─'*62}")
    lines.append(f"  {label}")
    lines.append(f"{'─'*62}")
    lines.append(f"  {'Stage':<28} {'ms':>8}  {'Budget':>8}  {'Status':>8}")
    lines.append(f"  {'-'*58}")
    for key, display in STAGE_KEYS:
        val = timings.get(key, 0.0)
        budget = BUDGET_MS.get(key, 9999)
        status = "✅" if val <= budget else "🔴 OVER"
        lines.append(f"  {display:<28} {val:>8.1f}  {budget:>8}  {status}")
    lines.append(f"  {'─'*58}")
    lines.append(f"  Wall-clock (client round-trip):  {wall_ms:>8.1f} ms")
    lines.append(f"  Results returned:                {n_results:>8}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark NBA /search latency")
    parser.add_argument("--host",  default="localhost")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--runs",  type=int, default=1,
                        help="Number of runs per query (averages timings)")
    args = parser.parse_args()

    # Verify server is up
    try:
        health_url = f"http://{args.host}:{args.port}/health"
        with urllib.request.urlopen(health_url, timeout=10) as r:
            health = json.loads(r.read())
        if health.get("status") != "ready":
            print(f"[WARNING] Server reports status: {health.get('status')} — models may still be loading.")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {args.host}:{args.port}: {e}")
        print("  Start the server with:  uvicorn api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    date_str   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir    = ROOT / "profiling"
    out_dir.mkdir(exist_ok=True)
    out_file   = out_dir / f"benchmark_results_{date_str}.txt"

    all_totals = []
    report_lines = [
        f"NBA Benchmark Results",
        f"Run date : {datetime.now().isoformat()}",
        f"Server   : {args.host}:{args.port}",
        f"Runs/query: {args.runs}",
        f"{'='*62}",
    ]

    bottleneck_stages = []

    for tc in TEST_CASES:
        label   = tc["label"]
        payload = tc["payload"]
        accumulated = {k: [] for k, _ in STAGE_KEYS}
        wall_times  = []
        n_results   = 0

        for run in range(args.runs):
            try:
                resp, wall_ms = post_search(args.host, args.port, payload)
            except Exception as e:
                print(f"[ERROR] Query '{label}' failed: {e}")
                break

            timings   = resp.get("_debug_timings", {})
            n_results = resp.get("total", 0)
            wall_times.append(wall_ms)
            for key, _ in STAGE_KEYS:
                accumulated[key].append(timings.get(key, 0.0))

        if not wall_times:
            continue

        avg_timings = {k: sum(v) / len(v) for k, v in accumulated.items()}
        avg_wall    = sum(wall_times) / len(wall_times)
        all_totals.append(avg_timings.get("total_ms", 0.0))

        table = format_table(label, avg_timings, avg_wall, n_results)
        print(table)
        report_lines.append(table)

        # Track bottlenecks
        for key, display in STAGE_KEYS:
            if key == "total_ms":
                continue
            if avg_timings.get(key, 0.0) > BUDGET_MS[key]:
                bottleneck_stages.append((display, avg_timings[key], BUDGET_MS[key]))

    # Summary
    summary_lines = [
        f"\n{'='*62}",
        f"SUMMARY",
        f"  Queries benchmarked    : {len(all_totals)}",
        f"  Mean total latency     : {sum(all_totals)/max(1,len(all_totals)):.1f} ms",
        f"  Max total latency      : {max(all_totals, default=0):.1f} ms",
        f"  Under 1s budget (<1000ms): {'YES ✅' if max(all_totals, default=9999) < 1000 else 'NO 🔴'}",
    ]
    if bottleneck_stages:
        summary_lines.append(f"\n  BOTTLENECKS (exceeded budget):")
        for stage, val, budget in bottleneck_stages:
            summary_lines.append(f"    🔴 {stage:<28} {val:.1f}ms  (budget={budget}ms)")
    else:
        summary_lines.append("  All stages within budget ✅")

    summary_str = "\n".join(summary_lines)
    print(summary_str)
    report_lines.append(summary_str)

    # Write file
    full_report = "\n".join(report_lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n[benchmark] Results saved to: {out_file}")


if __name__ == "__main__":
    main()
