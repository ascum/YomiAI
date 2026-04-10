"""
scripts/benchmark/language_detection.py
========================================
Benchmark lingua vs langdetect for language detection.

Measures:
  - Accuracy on a labelled test set (VI, EN, FR, DE, ZH, KO, JA, mixed)
  - Latency: cold start (first call) and warm throughput (N calls / second)
  - Memory footprint (RSS before vs after model load)

Usage:
    python scripts/benchmark/language_detection.py
    python scripts/benchmark/language_detection.py --runs 200 --no-langdetect
"""
import argparse
import os
import sys
import time
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

# ── Test corpus ───────────────────────────────────────────────────────────────

TEST_CASES = [
    # (text, expected_iso2)
    # Vietnamese — the primary non-English language this system handles
    ("sách khoa học viễn tưởng hay nhất",           "vi"),
    ("tiểu thuyết tình cảm lãng mạn",               "vi"),
    ("lịch sử Việt Nam",                             "vi"),
    ("truyện tranh thiếu nhi",                       "vi"),
    ("sách học lập trình Python",                    "vi"),
    ("tâm lý học hành vi",                           "vi"),
    # Short Vietnamese (hard case)
    ("sách hay",                                     "vi"),
    ("tiểu thuyết",                                  "vi"),
    # English
    ("science fiction books",                        "en"),
    ("best fantasy novels 2024",                     "en"),
    ("machine learning for beginners",               "en"),
    ("history of ancient Rome",                      "en"),
    ("children picture books",                       "en"),
    # Short English (hard case)
    ("mystery",                                      "en"),
    ("books",                                        "en"),
    # French
    ("les meilleurs romans policiers",               "fr"),
    ("littérature française classique",              "fr"),
    # German
    ("Wissenschaft und Technologie Bücher",          "de"),
    ("deutsche Literatur Klassiker",                 "de"),
    # Chinese (Simplified)
    ("科幻小说推荐",                                   "zh"),
    ("中国历史书籍",                                   "zh"),
    # Korean
    ("한국 소설 추천",                                  "ko"),
    # Japanese
    ("日本の歴史の本",                                  "ja"),
    # Mixed / ambiguous (no hard expected — just check it doesn't crash)
    ("Harry Potter sách",                            None),
    ("machine learning sách tốt nhất",              None),
]

WARM_QUERIES = [tc[0] for tc in TEST_CASES] * 4   # repeat for throughput test


# ── Helpers ───────────────────────────────────────────────────────────────────

def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return float("nan")


def pct(correct, total) -> str:
    return f"{correct}/{total}  ({100 * correct / total:.1f}%)" if total else "n/a"


def print_header(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


# ── Lingua benchmark ──────────────────────────────────────────────────────────

def bench_lingua(runs: int):
    print_header("LINGUA")
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        print("  [SKIP] lingua not installed  (pip install lingua-language-detector)")
        return None

    mem_before = rss_mb()
    t_load = time.perf_counter()
    detector = LanguageDetectorBuilder.from_all_languages().build()
    cold_ms = (time.perf_counter() - t_load) * 1000
    mem_after = rss_mb()

    print(f"  Cold load time : {cold_ms:.0f} ms")
    print(f"  Memory delta   : {mem_after - mem_before:+.0f} MB  (RSS {mem_after:.0f} MB)")

    def detect(text: str) -> str:
        lang = detector.detect_language_of(text)
        if lang is None:
            return "unknown"
        return lang.iso_code_639_1.name.lower()

    # Accuracy
    correct, total_labelled = 0, 0
    print(f"\n  {'Text':<42} {'Expected':>8}  {'Got':>8}  {'OK?'}")
    print(f"  {'-'*68}")
    for text, expected in TEST_CASES:
        got = detect(text)
        if expected is not None:
            total_labelled += 1
            ok = got == expected
            if ok:
                correct += 1
            flag = "✓" if ok else "✗"
            short = text[:40]
            print(f"  {short:<42} {expected:>8}  {got:>8}  {flag}")
        else:
            short = text[:40]
            print(f"  {short:<42} {'(any)':>8}  {got:>8}  -")

    print(f"\n  Accuracy : {pct(correct, total_labelled)}")

    # Throughput
    warm_queries = (WARM_QUERIES * ((runs // len(WARM_QUERIES)) + 1))[:runs]
    t0 = time.perf_counter()
    for q in warm_queries:
        detect(q)
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / len(warm_queries) * 1000
    throughput   = len(warm_queries) / elapsed

    print(f"  Throughput : {throughput:.0f} calls/s  ({per_call_ms:.2f} ms/call)  "
          f"[{len(warm_queries)} calls]")

    return {"lib": "lingua", "accuracy": correct / total_labelled if total_labelled else 0,
            "per_call_ms": per_call_ms, "throughput": throughput, "load_ms": cold_ms,
            "mem_delta_mb": mem_after - mem_before}


# ── langdetect benchmark ───────────────────────────────────────────────────────

def bench_langdetect(runs: int):
    print_header("LANGDETECT")
    try:
        from langdetect import detect as ld_detect, DetectorFactory, LangDetectException
        DetectorFactory.seed = 42          # make langdetect deterministic
    except ImportError:
        print("  [SKIP] langdetect not installed  (pip install langdetect)")
        return None

    mem_before = rss_mb()
    t_load = time.perf_counter()
    # langdetect is lazy — force it to load by running one call
    try:
        ld_detect("warmup")
    except Exception:
        pass
    cold_ms = (time.perf_counter() - t_load) * 1000
    mem_after = rss_mb()

    print(f"  Cold load time : {cold_ms:.0f} ms")
    print(f"  Memory delta   : {mem_after - mem_before:+.0f} MB  (RSS {mem_after:.0f} MB)")

    def detect(text: str) -> str:
        try:
            return ld_detect(text)
        except Exception:
            return "unknown"

    # Accuracy
    correct, total_labelled = 0, 0
    print(f"\n  {'Text':<42} {'Expected':>8}  {'Got':>8}  {'OK?'}")
    print(f"  {'-'*68}")
    for text, expected in TEST_CASES:
        got = detect(text)
        if expected is not None:
            total_labelled += 1
            ok = got == expected
            if ok:
                correct += 1
            flag = "✓" if ok else "✗"
            short = text[:40]
            print(f"  {short:<42} {expected:>8}  {got:>8}  {flag}")
        else:
            short = text[:40]
            print(f"  {short:<42} {'(any)':>8}  {got:>8}  -")

    print(f"\n  Accuracy : {pct(correct, total_labelled)}")

    # Throughput
    warm_queries = (WARM_QUERIES * ((runs // len(WARM_QUERIES)) + 1))[:runs]
    t0 = time.perf_counter()
    for q in warm_queries:
        detect(q)
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / len(warm_queries) * 1000
    throughput   = len(warm_queries) / elapsed

    print(f"  Throughput : {throughput:.0f} calls/s  ({per_call_ms:.2f} ms/call)  "
          f"[{len(warm_queries)} calls]")

    return {"lib": "langdetect", "accuracy": correct / total_labelled if total_labelled else 0,
            "per_call_ms": per_call_ms, "throughput": throughput, "load_ms": cold_ms,
            "mem_delta_mb": mem_after - mem_before}


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list):
    results = [r for r in results if r is not None]
    if len(results) < 2:
        return

    print_header("SUMMARY")
    fmt = f"  {'Metric':<22}"
    for r in results:
        fmt += f"  {r['lib']:>14}"
    print(fmt)
    print(f"  {'-'*60}")

    metrics = [
        ("Accuracy",      lambda r: f"{r['accuracy']*100:.1f}%"),
        ("Load time",     lambda r: f"{r['load_ms']:.0f} ms"),
        ("Per-call",      lambda r: f"{r['per_call_ms']:.2f} ms"),
        ("Throughput",    lambda r: f"{r['throughput']:.0f} /s"),
        ("Memory delta",  lambda r: f"{r['mem_delta_mb']:+.0f} MB"),
    ]
    for label, fn in metrics:
        row = f"  {label:<22}"
        for r in results:
            row += f"  {fn(r):>14}"
        print(row)

    print()
    winner_acc  = max(results, key=lambda r: r["accuracy"])
    winner_spd  = max(results, key=lambda r: r["throughput"])
    print(f"  Accuracy winner  : {winner_acc['lib']}")
    print(f"  Speed winner     : {winner_spd['lib']}")

    # Current choice in translation.py
    print(f"\n  Current choice   : lingua  (app/infrastructure/translation.py)")
    if winner_acc["lib"] != "lingua" or winner_spd["lib"] != "lingua":
        other = [r for r in results if r["lib"] != "lingua"][0]
        print(f"  Suggestion       : consider switching to {other['lib']} if its "
              f"accuracy is acceptable and speed matters more")
    else:
        print(f"  Verdict          : lingua is the better choice — no change needed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",          type=int, default=100,
                        help="Number of warm calls for throughput test (default: 100)")
    parser.add_argument("--no-lingua",     action="store_true")
    parser.add_argument("--no-langdetect", action="store_true")
    args = parser.parse_args()

    print(f"Language Detection Benchmark")
    print(f"Warm runs per library : {args.runs}")

    results = []
    if not args.no_lingua:
        results.append(bench_lingua(args.runs))
    if not args.no_langdetect:
        results.append(bench_langdetect(args.runs))

    print_summary(results)


if __name__ == "__main__":
    main()
