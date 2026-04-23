"""
scripts/benchmark/compare_encoders.py — BLaIR (legacy) vs BGE-M3 encoder comparison.

Compares embedding quality using the same sampled-eval protocol as
evaluate_recommendation.py so numbers are directly comparable in the paper.

Steps:
  1. Load BGE flat, BGE HNSW, BLaIR HNSW (legacy)
  2. Build dual embedding cache (BGE flat + BLaIR HNSW reconstruct)
  3. Sampled evaluation: HR@5/10, NDCG@10, MRR@10  (shared negatives per user)
  4. Score distribution: intra- vs inter-genre cosine sim, separation ratio
  5a. Top-K overlap: HNSW vs HNSW (same algorithm, fair embedding comparison)
  5b. HNSW recall: BGE flat vs BGE HNSW (validates production index quality)
  6. BGE-M3 live query encode latency (skip with --no-latency)
  7. Query retrieval quality: load both live encoders, encode text queries
     (short / medium / long / Vietnamese), search respective HNSW indices,
     measure genre-precision@K per group. This is the real retrieval test.

Usage:
    python scripts/benchmark/compare_encoders.py
    python scripts/benchmark/compare_encoders.py --max-users 2000 --seed 42
    python scripts/benchmark/compare_encoders.py --blair-model hyp1231/blair-roberta-large

Output:
    evaluation/encoder_comparison_<run_id>.json
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import faiss
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.config import settings

DATA_DIR   = settings.DATA_DIR
EVAL_PATH  = os.path.join(ROOT, "evaluation", "eval_users.json")
OUTPUT_DIR = os.path.join(ROOT, "evaluation")


# ── Query test suite ───────────────────────────────────────────────────────────
# (query_text, expected_genre_keywords, group)
# Groups: short (1-3 words), medium (4-8 words), long (15+ words / review-like),
#         vi_short, vi_long — tests cross-lingual ability (BGE-M3 only).
QUERY_SUITE = [
    # Short
    ("mystery",                          ["mystery", "detective", "crime", "thriller"],     "short"),
    ("fantasy magic",                    ["fantasy", "magic", "wizard"],                    "short"),
    ("self help",                        ["self-help", "self help", "success", "habits"],   "short"),
    ("science fiction",                  ["science fiction", "sci-fi", "space"],            "short"),
    ("war history",                      ["history", "war", "military"],                    "short"),
    ("horror",                           ["horror", "supernatural", "gothic"],              "short"),
    # Medium
    ("detective mystery crime thriller investigation",
     ["mystery", "detective", "crime", "thriller"],                                         "medium"),
    ("fantasy magic sword quest adventure",
     ["fantasy", "magic", "adventure"],                                                     "medium"),
    ("self improvement habits success personal growth",
     ["self-help", "self help", "success"],                                                 "medium"),
    ("science fiction space travel alien civilization",
     ["science fiction", "sci-fi", "space"],                                                "medium"),
    ("historical war military conflict soldiers battle",
     ["history", "war", "military"],                                                        "medium"),
    ("horror supernatural haunted ghost paranormal",
     ["horror", "supernatural", "paranormal"],                                              "medium"),
    # Long / review-like (BLaIR training distribution)
    (
        "A brilliant detective investigates a series of mysterious murders in a dark city"
        " full of corruption and betrayal",
        ["mystery", "detective", "crime", "thriller"],
        "long",
    ),
    (
        "An epic tale of a young orphan who discovers magical powers and must defeat an"
        " ancient evil threatening the entire realm",
        ["fantasy", "magic", "adventure"],
        "long",
    ),
    (
        "Practical advice on building lasting habits and achieving personal success through"
        " discipline mindset shifts and daily routines",
        ["self-help", "self help", "success", "habits"],
        "long",
    ),
    (
        "Humanity ventures beyond the solar system and makes first contact with an alien"
        " civilization with a radically different concept of time and consciousness",
        ["science fiction", "sci-fi", "space"],
        "long",
    ),
    # Vietnamese short — BGE-M3 multilingual; BLaIR is English-only
    ("tiểu thuyết trinh thám",           ["mystery", "detective", "crime", "thriller"],    "vi_short"),
    ("phép thuật kỳ ảo",                 ["fantasy", "magic"],                             "vi_short"),
    ("sách phát triển bản thân",         ["self-help", "self help", "success"],            "vi_short"),
    ("khoa học viễn tưởng",              ["science fiction", "sci-fi"],                    "vi_short"),
    # Vietnamese long
    (
        "Câu chuyện trinh thám hấp dẫn về thám tử tài ba điều tra vụ án giết người bí ẩn"
        " trong thành phố đầy tham nhũng và phản bội",
        ["mystery", "detective", "crime", "thriller"],
        "vi_long",
    ),
    (
        "Cuốn sách hướng dẫn xây dựng thói quen tốt và phát triển bản thân để đạt thành"
        " công trong cuộc sống và sự nghiệp",
        ["self-help", "self help", "success", "habits"],
        "vi_long",
    ),
]


# ── Tee logger (writes all print output to file + console) ────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

class _Tee:
    """Write to console with ANSI colours; strip ANSI before writing to log file."""
    def __init__(self, console, logfile):
        self._console = console
        self._logfile = logfile

    def write(self, data):
        self._console.write(data)
        self._logfile.write(_ANSI_RE.sub('', data))

    def flush(self):
        self._console.flush()
        self._logfile.flush()

    def isatty(self):
        return self._console.isatty()


# ── ANSI helpers ───────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()

def _c(code, text): return f"\033[{code}m{text}\033[0m" if _IS_TTY else text
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def hit_rate(ranked, target, k):
    return 1.0 if target in ranked[:k] else 0.0

def ndcg(ranked, target, k):
    for i, a in enumerate(ranked[:k]):
        if a == target:
            return 1.0 / math.log2(i + 2)
    return 0.0

def mrr(ranked, target, k):
    for i, a in enumerate(ranked[:k]):
        if a == target:
            return 1.0 / (i + 1)
    return 0.0


# ── Step 1: Load indices ───────────────────────────────────────────────────────

def load_indices():
    bge_flat_path  = os.path.join(DATA_DIR, settings.TEXT_INDEX_FLAT)
    bge_hnsw_path  = os.path.join(DATA_DIR, settings.TEXT_INDEX_HNSW)
    blair_hnsw_path = os.path.join(DATA_DIR, settings.TEXT_INDEX_HNSW_LEGACY)

    missing = [p for p in [bge_flat_path, blair_hnsw_path] if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing index files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print(bold("Loading indices (MMAP) ..."))
    t0 = time.time()

    bge_flat   = faiss.read_index(bge_flat_path,   faiss.IO_FLAG_MMAP)
    blair_hnsw = faiss.read_index(blair_hnsw_path, faiss.IO_FLAG_MMAP)
    bge_hnsw   = (faiss.read_index(bge_hnsw_path, faiss.IO_FLAG_MMAP)
                  if os.path.exists(bge_hnsw_path) else None)

    print(f"  BGE   flat : {bge_flat.ntotal:,} vectors  (dim={bge_flat.d})")
    print(f"  BLaIR HNSW : {blair_hnsw.ntotal:,} vectors  (dim={blair_hnsw.d})")
    if bge_hnsw:
        print(f"  BGE   HNSW : {bge_hnsw.ntotal:,} vectors  (dim={bge_hnsw.d})")
    else:
        print(f"  BGE   HNSW : {yellow('not found')} — step 5b will be skipped")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if bge_flat.ntotal != blair_hnsw.ntotal:
        print(yellow(f"  WARNING: BGE ({bge_flat.ntotal:,}) and BLaIR ({blair_hnsw.ntotal:,})"
                     " have different vector counts — evaluation uses shared asins.csv range."))

    return bge_flat, bge_hnsw, blair_hnsw


# ── Step 2: Build dual embedding cache ────────────────────────────────────────

def build_dual_cache(bge_flat, blair_hnsw, asins, asin_to_idx, eval_users, n_neg):
    print(bold("\nBuilding dual embedding cache ..."))

    n_safe = min(bge_flat.ntotal, blair_hnsw.ntotal)

    eval_set = set()
    for u in eval_users:
        eval_set.update(u.get("train_clicks", []))
        eval_set.update(u.get("test_clicks",  []))
    eval_set = {a for a in eval_set if a in asin_to_idx and asin_to_idx[a] < n_safe}

    pool = [a for a in asins if asin_to_idx.get(a, n_safe) < n_safe]
    NEG_POOL       = min(200_000, len(pool))
    neg_pool_raw   = random.sample(pool, NEG_POOL)
    to_load        = eval_set | set(neg_pool_raw)

    bge_cache   = {}
    blair_cache = {}
    n           = len(to_load)
    t0          = time.time()

    for i, asin in enumerate(to_load):
        idx = asin_to_idx[asin]
        bge_cache[asin]   = bge_flat.reconstruct(idx).astype("float32")
        blair_cache[asin] = blair_hnsw.reconstruct(idx).astype("float32")

        if (i + 1) % 10_000 == 0 or i + 1 == n:
            pct     = (i + 1) / n
            filled  = int(30 * pct)
            bar     = green("█" * filled) + dim("░" * (30 - filled))
            elapsed = time.time() - t0
            speed   = (i + 1) / elapsed if elapsed > 0 else 1
            eta     = (n - i - 1) / speed
            print(f"\r  [{bar}] {pct*100:5.1f}%  {i+1:>7,}/{n:,}  "
                  f"{speed:,.0f}/s  ETA {eta:.0f}s  ", end="", flush=True)

    neg_pool = [a for a in neg_pool_raw if a in bge_cache and a in blair_cache]
    size_mb  = n * 1024 * 4 * 2 / 1e6
    print(f"\r  {green('Done')}  {n:,} ASINs × 2 models  (~{size_mb:.0f} MB)  "
          f"neg pool: {len(neg_pool):,}  in {time.time()-t0:.1f}s          ")

    return bge_cache, blair_cache, neg_pool


# ── Step 3: Sampled evaluation ─────────────────────────────────────────────────

def _score_candidates(cache, train_clicks, candidates):
    vecs = [cache[a] for a in train_clicks if a in cache]
    if not vecs:
        return {a: 0.0 for a in candidates}
    profile = np.mean(vecs, axis=0).astype("float32")
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm
    return {a: float(profile @ cache[a]) for a in candidates if a in cache}


def pregenerate_negatives(eval_users, neg_pool, n_neg, max_users):
    """
    Draw per-user negative sets ONCE before any model evaluation.
    Both models receive the exact same candidates per user — the only
    variable between them is the scoring function (their embedding space).
    """
    users    = eval_users[:max_users] if max_users else eval_users
    per_user = []
    for user in users:
        train  = user.get("train_clicks", [])
        test   = user.get("test_clicks",  [])
        if not train or not test:
            per_user.append(None)
            continue
        seen = set(train) | set(test)
        negs = [a for a in random.sample(neg_pool, min(n_neg * 3, len(neg_pool)))
                if a not in seen][:n_neg]
        per_user.append(negs if len(negs) >= n_neg // 2 else None)
    return per_user


def run_sampled_eval(cache, eval_users, per_user_negs, k, max_users):
    hr5, hr10, ndcg_vals, mrr_vals = [], [], [], []
    t0    = time.time()
    users = eval_users[:max_users] if max_users else eval_users

    for user, negs in zip(users, per_user_negs):
        if negs is None:
            continue
        train  = user.get("train_clicks", [])
        test   = user.get("test_clicks",  [])
        target = test[0]
        if target not in cache:
            continue

        candidates = [target] + negs
        scores  = _score_candidates(cache, train, candidates)
        ranked  = sorted(candidates, key=lambda a: scores.get(a, 0.0), reverse=True)

        hr5.append(hit_rate(ranked, target, 5))
        hr10.append(hit_rate(ranked, target, 10))
        ndcg_vals.append(ndcg(ranked, target, k))
        mrr_vals.append(mrr(ranked, target, k))

    n = len(hr5) or 1
    return (
        sum(hr5)       / n,
        sum(hr10)      / n,
        sum(ndcg_vals) / n,
        sum(mrr_vals)  / n,
        len(hr5),
        time.time() - t0,
    )


# ── Step 4: Score distribution analysis ───────────────────────────────────────

def analyze_score_distribution(bge_cache, blair_cache, meta_df, n_sample=5_000):
    print(bold("\nAnalyzing score distributions ..."))

    common = [a for a in bge_cache if a in blair_cache]
    if len(common) > n_sample:
        common = random.sample(common, n_sample)

    genre_map = {}
    for asin in common:
        genre_map[asin] = "Unknown"
        if meta_df is not None and asin in meta_df.index:
            cats = str(meta_df.loc[asin].get("categories", ""))
            if cats and cats != "nan":
                parts = [p.strip() for p in cats.split("|")]
                genre_map[asin] = parts[1] if len(parts) > 1 else parts[0]

    groups = defaultdict(list)
    for asin, genre in genre_map.items():
        groups[genre].append(asin)
    groups = {g: v for g, v in groups.items() if len(v) >= 10}
    all_genres = list(groups.keys())

    if len(all_genres) < 2:
        print(yellow("  Not enough genre diversity for distribution analysis — skipping"))
        return {}

    N_PAIRS  = 300
    results  = {}

    for model_name, cache in [("BGE-M3", bge_cache), ("BLaIR", blair_cache)]:
        intra, inter = [], []

        for _ in range(N_PAIRS):
            g = random.choice(all_genres)
            if len(groups[g]) < 2:
                continue
            a1, a2 = random.sample(groups[g], 2)
            if a1 in cache and a2 in cache:
                intra.append(float(cache[a1] @ cache[a2]))

        for _ in range(N_PAIRS):
            g1, g2 = random.sample(all_genres, 2)
            a1 = random.choice(groups[g1])
            a2 = random.choice(groups[g2])
            if a1 in cache and a2 in cache:
                inter.append(float(cache[a1] @ cache[a2]))

        mean_intra = float(np.mean(intra)) if intra else 0.0
        mean_inter = float(np.mean(inter)) if inter else 0.0
        ratio      = mean_intra / mean_inter if mean_inter > 0 else 0.0
        results[model_name] = {
            "mean_intra_sim":   round(mean_intra, 4),
            "mean_inter_sim":   round(mean_inter, 4),
            "separation_ratio": round(ratio,      4),
        }
        print(f"  {model_name:<12}  intra={mean_intra:.4f}  "
              f"inter={mean_inter:.4f}  ratio={ratio:.3f}")

    return results


# ── Step 5a: Top-K overlap — HNSW vs HNSW ────────────────────────────────────

def analyze_topk_overlap(bge_hnsw, blair_hnsw, asin_to_idx, asins,
                         n_queries=500, k=10):
    n_safe  = min(bge_hnsw.ntotal, blair_hnsw.ntotal)
    pool    = [a for a in asins if asin_to_idx.get(a, n_safe) < n_safe]
    queries = random.sample(pool, min(n_queries, len(pool)))

    print(bold(f"\nTop-K overlap — HNSW vs HNSW (k={k}, n={len(queries)}) ..."))

    jaccard_list = []
    t0 = time.time()

    for q_asin in queries:
        q_idx = asin_to_idx[q_asin]

        bge_q   = bge_hnsw.reconstruct(q_idx).reshape(1, -1).astype("float32")
        blair_q = blair_hnsw.reconstruct(q_idx).reshape(1, -1).astype("float32")

        _, I_bge   = bge_hnsw.search(bge_q,    k + 1)
        _, I_blair = blair_hnsw.search(blair_q, k + 1)

        top_bge   = {asins[i] for i in I_bge[0]   if 0 <= i < len(asins) and asins[i] != q_asin}
        top_blair = {asins[i] for i in I_blair[0]  if 0 <= i < len(asins) and asins[i] != q_asin}
        top_bge   = set(list(top_bge)[:k])
        top_blair = set(list(top_blair)[:k])

        if top_bge and top_blair:
            jaccard_list.append(len(top_bge & top_blair) / len(top_bge | top_blair))

    mean_j = float(np.mean(jaccard_list)) if jaccard_list else 0.0
    label  = ("high divergence" if mean_j < 0.2
               else "moderate overlap" if mean_j < 0.5
               else "high overlap")
    print(f"  Mean Jaccard@{k}: {mean_j:.4f}  ({label})  in {time.time()-t0:.1f}s")

    return {"mean_jaccard": round(mean_j, 4), "k": k, "n_queries": len(jaccard_list)}


# ── Step 5b: HNSW recall — BGE only ───────────────────────────────────────────

def analyze_hnsw_recall(bge_flat, bge_hnsw, asin_to_idx, asins,
                        n_queries=500, k=10):
    if bge_hnsw is None:
        print(bold("\nHNSW recall: skipped (bge_index_hnsw.faiss not found)"))
        return None

    print(bold(f"\nHNSW recall validation — BGE only (k={k}, n={n_queries}) ..."))

    n_safe  = bge_flat.ntotal
    pool    = [a for a in asins if asin_to_idx.get(a, n_safe) < n_safe]
    queries = random.sample(pool, min(n_queries, len(pool)))

    recall_list, flat_ms, hnsw_ms = [], [], []

    for q_asin in queries:
        q_idx = asin_to_idx[q_asin]
        vec   = bge_flat.reconstruct(q_idx).reshape(1, -1).astype("float32")

        t0 = time.perf_counter()
        _, I_flat = bge_flat.search(vec, k + 1)
        flat_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _, I_hnsw = bge_hnsw.search(vec, k + 1)
        hnsw_ms.append((time.perf_counter() - t0) * 1000)

        top_flat = {asins[i] for i in I_flat[0] if 0 <= i < len(asins) and asins[i] != q_asin}
        top_hnsw = {asins[i] for i in I_hnsw[0] if 0 <= i < len(asins) and asins[i] != q_asin}
        top_flat = set(list(top_flat)[:k])
        top_hnsw = set(list(top_hnsw)[:k])

        if top_flat and top_hnsw:
            recall_list.append(len(top_flat & top_hnsw) / k)

    mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
    f50, f95    = float(np.median(flat_ms)),        float(np.percentile(flat_ms, 95))
    h50, h95    = float(np.median(hnsw_ms)),        float(np.percentile(hnsw_ms, 95))
    speedup     = f50 / h50 if h50 > 0 else 0.0

    quality = "good" if mean_recall > 0.95 else "acceptable" if mean_recall > 0.85 else "degraded"
    print(f"  HNSW recall@{k}: {mean_recall:.4f}  ({quality})")
    print(f"  Flat  p50={f50:.2f}ms  p95={f95:.2f}ms")
    print(f"  HNSW  p50={h50:.2f}ms  p95={h95:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x  (median)")

    return {
        "mean_recall_at_k": round(mean_recall, 4),
        "k":                k,
        "flat_p50_ms":      round(f50, 2),
        "flat_p95_ms":      round(f95, 2),
        "hnsw_p50_ms":      round(h50, 2),
        "hnsw_p95_ms":      round(h95, 2),
        "speedup_median":   round(speedup, 2),
    }


# ── Step 6: Optional BGE-M3 encode latency ────────────────────────────────────

def benchmark_encode_latency():
    print(bold("\nBenchmarking BGE-M3 encode latency ..."))
    import torch
    from sentence_transformers import SentenceTransformer

    QUERIES = [
        "detective mystery novels",
        "self-help personal development books",
        "magic fantasy",
        "world war history",
        "children's books",
        "dark fantasy sword sorcery",
        "science fiction space opera",
        "romance novel historical",
        "horror supernatural thriller",
        "biography memoir autobiography",
        "cooking recipes italian",
        "business leadership management",
        "philosophy ethics morality",
        "tiểu thuyết trinh thám",
        "sách phát triển bản thân",
    ] * 4  # 60 queries

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SentenceTransformer(settings.TEXT_ENCODER_MODEL, device=str(device),
                                 trust_remote_code=True)
    model.max_seq_length = 64
    if device.type == "cuda":
        model.half()

    model.encode(["warmup"], normalize_embeddings=True)

    times = []
    for q in QUERIES:
        t0 = time.perf_counter()
        model.encode([q], normalize_embeddings=True)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.median(times))
    p95 = float(np.percentile(times, 95))
    p99 = float(np.percentile(times, 99))
    print(f"  n={len(times)}  p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  device={device}")

    return {"n": len(times), "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2), "p99_ms": round(p99, 2), "device": str(device)}


# ── Step 7: Query retrieval quality ───────────────────────────────────────────

def _mean_pool_normalize(last_hidden_state, attention_mask):
    """Mean-pool token embeddings and L2-normalise (HF transformers path)."""
    import torch.nn.functional as F
    mask   = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2, dim=1)


def load_query_encoders(blair_model_id: str):
    """
    Load BGE-M3 (SentenceTransformer) and BLaIR (ST if possible, else HF transformers).
    Returns (bge_pack, blair_pack); blair_pack is None if model cannot be loaded.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    print(bold("\nLoading live query encoders ..."))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # BGE-M3
    print(f"  BGE-M3 ({settings.TEXT_ENCODER_MODEL}) ...", end="", flush=True)
    t0     = time.time()
    bge_st = SentenceTransformer(settings.TEXT_ENCODER_MODEL, device=device,
                                 trust_remote_code=True)
    bge_st.max_seq_length = 512
    if device == "cuda":
        bge_st.half()
    bge_pack = {"model": bge_st, "type": "st", "name": "BGE-M3"}
    print(f" {green('OK')} ({time.time()-t0:.1f}s)")

    # BLaIR — try SentenceTransformer first, fall back to HF transformers + mean pool
    blair_pack = None
    print(f"  BLaIR  ({blair_model_id}) ...", end="", flush=True)
    t0 = time.time()
    try:
        blair_st = SentenceTransformer(blair_model_id, device=device)
        blair_pack = {"model": blair_st, "type": "st", "name": "BLaIR"}
        print(f" {green('OK via SentenceTransformer')} ({time.time()-t0:.1f}s)")
    except Exception as e_st:
        print(f"\n    ST failed ({e_st}), trying HF transformers ...", end="", flush=True)
        try:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained(blair_model_id)
            mdl = AutoModel.from_pretrained(blair_model_id)
            mdl.to(device).eval()
            if device == "cuda":
                mdl.half()
            blair_pack = {"model": mdl, "tokenizer": tok, "type": "hf",
                          "name": "BLaIR", "device": device}
            print(f" {green('OK via HF transformers')} ({time.time()-t0:.1f}s)")
        except Exception as e_hf:
            print(f" {yellow(f'FAILED: {e_hf}')}")
            print(yellow("  BLaIR encoder unavailable — query test will be BGE-only"))

    return bge_pack, blair_pack


def encode_query(text: str, pack) -> np.ndarray:
    """Encode one query string → L2-normalised float32 vector."""
    if pack["type"] == "st":
        vec = pack["model"].encode([text], normalize_embeddings=True,
                                   show_progress_bar=False)
        return np.array(vec[0], dtype="float32")
    # HF transformers path (BLaIR fallback)
    import torch
    tok, mdl, device = pack["tokenizer"], pack["model"], pack["device"]
    inputs = tok([text], padding=True, truncation=True,
                 max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs)
    vec = _mean_pool_normalize(out.last_hidden_state, inputs["attention_mask"])
    return vec.squeeze(0).cpu().float().numpy()


def _genre_precision(result_asins, meta_df, expected_genres, k=10):
    """Fraction of top-K results whose metadata contains any expected genre keyword."""
    if meta_df is None or meta_df.empty:
        return None
    hits = 0
    for asin in result_asins[:k]:
        if asin not in meta_df.index:
            continue
        row  = meta_df.loc[asin]
        text = (str(row.get("categories",    "")) + " " +
                str(row.get("main_category", ""))).lower()
        if any(g.lower() in text for g in expected_genres):
            hits += 1
    return hits / k


def run_query_test(bge_pack, blair_pack, bge_hnsw, blair_hnsw,
                   asins, meta_df, k=10):
    """
    Encode every QUERY_SUITE entry with both live models, search the respective
    HNSW indices, compute genre-precision@K, print a grouped table.
    """
    print(bold(f"\nQuery retrieval quality — live encoder test (k={k}) ..."))
    if blair_pack is None:
        print(yellow("  BLaIR encoder unavailable — BGE-M3 only"))

    GROUPS  = ["short", "medium", "long", "vi_short", "vi_long"]
    records = []

    for query_text, expected_genres, group in QUERY_SUITE:
        rec = {"query": query_text, "group": group,
               "expected_genres": expected_genres,
               "bge_gp": None, "blair_gp": None}

        try:
            qv       = encode_query(query_text, bge_pack).reshape(1, -1)
            _, I_bge = bge_hnsw.search(qv, k + 1)
            hits     = [asins[i] for i in I_bge[0] if 0 <= i < len(asins)][:k]
            rec["bge_gp"] = _genre_precision(hits, meta_df, expected_genres, k)
        except Exception as exc:
            print(yellow(f"  BGE error on '{query_text[:40]}': {exc}"))

        if blair_pack is not None:
            try:
                qv         = encode_query(query_text, blair_pack).reshape(1, -1)
                _, I_blair = blair_hnsw.search(qv, k + 1)
                hits       = [asins[i] for i in I_blair[0] if 0 <= i < len(asins)][:k]
                rec["blair_gp"] = _genre_precision(hits, meta_df, expected_genres, k)
            except Exception as exc:
                print(yellow(f"  BLaIR error on '{query_text[:40]}': {exc}"))

        records.append(rec)

    # ── Per-query table ────────────────────────────────────────────────────────
    print()
    print(bold(f"  {'Query':<52}  {'Grp':<8}  {'BGE':>6}  {'BLaIR':>6}  {'Δ':>6}"))
    print("  " + "─" * 84)

    for rec in records:
        q_s = (rec["query"][:49] + "...") if len(rec["query"]) > 52 else rec["query"]
        b_s = f"{rec['bge_gp']:.2f}"   if rec["bge_gp"]   is not None else "  N/A"
        r_s = f"{rec['blair_gp']:.2f}" if rec["blair_gp"] is not None else "  N/A"

        if rec["bge_gp"] is not None and rec["blair_gp"] is not None:
            d     = rec["bge_gp"] - rec["blair_gp"]
            d_fmt = (green(f"{d:+.2f}") if d > 0.05
                     else yellow(f"{d:+.2f}") if d < -0.05
                     else dim(f"{d:+.2f}"))
        else:
            d_fmt = dim("   N/A")

        b_fmt = (green(b_s) if rec["bge_gp"]   is not None and rec["bge_gp"]   >= 0.5
                 else b_s)
        r_fmt = (green(r_s) if rec["blair_gp"] is not None and rec["blair_gp"] >= 0.5
                 else r_s)

        print(f"  {q_s:<52}  {dim(rec['group']):<8}  {b_fmt:>6}  {r_fmt:>6}  {d_fmt:>6}")

    print("  " + "─" * 84)

    # ── Per-group summary ──────────────────────────────────────────────────────
    print()
    print(bold(f"  {'Group':<10}  {'N':>3}  {'BGE avg':>9}  {'BLaIR avg':>10}  {'Winner':>9}"))
    print("  " + "─" * 48)

    group_summary = {}
    for grp in GROUPS:
        rows = [r for r in records if r["group"] == grp]
        if not rows:
            continue
        bge_vals   = [r["bge_gp"]   for r in rows if r["bge_gp"]   is not None]
        blair_vals = [r["blair_gp"] for r in rows if r["blair_gp"] is not None]
        bge_avg    = float(np.mean(bge_vals))   if bge_vals   else None
        blair_avg  = float(np.mean(blair_vals)) if blair_vals else None

        b_s = f"{bge_avg:.3f}"   if bge_avg   is not None else "  N/A"
        r_s = f"{blair_avg:.3f}" if blair_avg is not None else "  N/A"

        if bge_avg is not None and blair_avg is not None:
            winner = green("BGE  ✓") if bge_avg > blair_avg else yellow("BLaIR ✓")
        elif bge_avg is not None:
            winner = "BGE only"
        else:
            winner = dim("—")

        print(f"  {grp:<10}  {len(rows):>3}  {b_s:>9}  {r_s:>10}  {winner:>9}")
        group_summary[grp] = {
            "n_queries":    len(rows),
            "bge_avg_gp":   round(bge_avg,   4) if bge_avg   is not None else None,
            "blair_avg_gp": round(blair_avg, 4) if blair_avg is not None else None,
        }

    print()
    return {"per_query": records, "per_group": group_summary}


# ── Report ─────────────────────────────────────────────────────────────────────

def print_results_table(results_map, n_neg, k):
    baseline_hr10   = 10 / (n_neg + 1)
    baseline_ndcg10 = sum(1 / math.log2(i + 2) for i in range(k)) / (n_neg + 1)
    best_hr10       = max(r["hr10"] for r in results_map.values())

    w = 72
    print()
    print(bold("┌" + "─" * (w - 2) + "┐"))
    print(bold(f"│  {cyan(f'ENCODER COMPARISON   negatives={n_neg}   k={k}'):<{w-4}}  │"))
    print(bold("└" + "─" * (w - 2) + "┘"))
    print()
    print(bold(f"  {'Model':<22}  {'HR@5':>8}  {'HR@10':>8}  {'NDCG@10':>9}  "
               f"{'MRR@10':>8}  {'Users':>7}  {'Time':>7}"))
    print("  " + "─" * 68)

    for name, r in results_map.items():
        marker = " ◄" if r["hr10"] == best_hr10 else "  "

        def cell(v, base):
            s = f"{v:.4f}"
            if v >= base * 1.5: return green(s)
            if v >= base:       return yellow(s)
            return s

        print(f"  {bold(name):<22}  "
              f"{cell(r['hr5'],    baseline_hr10):>8}  "
              f"{cell(r['hr10'],   baseline_hr10):>8}  "
              f"{cell(r['ndcg10'], baseline_ndcg10):>9}  "
              f"{cell(r['mrr10'],  baseline_hr10):>8}  "
              f"{r['users']:>7,}  "
              f"{r['time_s']:>5.1f}s{marker}")

    print("  " + "─" * 68)
    print(f"  {dim('Random baseline'):<22}  "
          f"{dim(f'{baseline_hr10:.4f}'):>8}  "
          f"{dim(f'{baseline_hr10:.4f}'):>8}  "
          f"{dim(f'{baseline_ndcg10:.4f}'):>9}  "
          f"{dim(f'{baseline_hr10:.4f}'):>8}")
    print()


def print_latex(results_map):
    best_hr10 = max(r["hr10"] for r in results_map.values())
    print(bold("  LaTeX rows:"))
    for name, r in results_map.items():
        is_best = r["hr10"] == best_hr10
        def fmt(v):
            s = f"{v:.4f}"
            return f"\\textbf{{{s}}}" if is_best else s
        print(f"  {name:<24} & {fmt(r['hr5'])} & {fmt(r['hr10'])} "
              f"& {fmt(r['ndcg10'])} & {fmt(r['mrr10'])} \\\\")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BLaIR vs BGE-M3 encoder comparison")
    p.add_argument("--negatives",         type=int,  default=99)
    p.add_argument("--max-users",         type=int,  default=None)
    p.add_argument("--k",                 type=int,  default=10)
    p.add_argument("--seed",              type=int,  default=42)
    p.add_argument("--sample-n",          type=int,  default=5_000,
                   help="ASINs for distribution analysis (default: 5000)")
    p.add_argument("--overlap-queries",   type=int,  default=500,
                   help="Queries for HNSW overlap step (default: 500)")
    p.add_argument("--hnsw-queries",      type=int,  default=500,
                   help="Queries for HNSW recall step (default: 500, HNSW is fast)")
    p.add_argument("--no-latency", action="store_true",
                   help="Skip BGE-M3 encode latency benchmark")
    p.add_argument("--blair-model", default="hyp1231/blair-roberta-large",
                   help="HuggingFace model ID for BLaIR encoder (default: hyp1231/blair-roberta-large)")
    p.add_argument("--no-query-test", action="store_true",
                   help="Skip live query retrieval quality test (step 7)")
    p.add_argument("--no-save",           action="store_true")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_global = time.time()

    # ── Tee stdout → log file ─────────────────────────────────────────────────
    log_dir  = os.path.join(ROOT, "evaluation", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"encoder_comparison_{run_id}.log")
    _log_fh  = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)  # console gets ANSI, file gets plain text

    print(bold(f"\n{'═'*72}"))
    print(bold(f"  BLaIR vs BGE-M3 Encoder Comparison   run={cyan(run_id)}"))
    print(bold(f"  negatives={args.negatives}   k={args.k}   seed={args.seed}"))
    print(bold(f"{'═'*72}"))

    # ── 1. Load indices ───────────────────────────────────────────────────────
    bge_flat, bge_hnsw, blair_hnsw = load_indices()

    # ── Load ASIN list + eval users ───────────────────────────────────────────
    print(bold("\nLoading ASINs and eval users ..."))
    asins_df    = pd.read_csv(os.path.join(DATA_DIR, "asins.csv"),
                              header=None, dtype=str)
    asins       = asins_df.iloc[:, 0].tolist()
    asin_to_idx = {a: i for i, a in enumerate(asins)}

    if not os.path.exists(EVAL_PATH):
        print(f"ERROR: {EVAL_PATH} not found. Run scripts/setup_dif_sasrec.py first.")
        sys.exit(1)
    with open(EVAL_PATH) as f:
        eval_users = json.load(f)
    print(f"  {len(asins):,} ASINs  |  {len(eval_users):,} eval users")

    # ── 2. Build dual cache ───────────────────────────────────────────────────
    bge_cache, blair_cache, neg_pool = build_dual_cache(
        bge_flat, blair_hnsw, asins, asin_to_idx, eval_users, args.negatives
    )

    # ── 3. Sampled evaluation ─────────────────────────────────────────────────
    print(bold("\nPre-generating negatives (shared across both models) ..."))
    per_user_negs = pregenerate_negatives(
        eval_users, neg_pool, n_neg=args.negatives, max_users=args.max_users
    )
    valid = sum(1 for x in per_user_negs if x is not None)
    print(f"  {valid:,} users with valid candidate sets")

    print(bold("\nRunning sampled evaluation ..."))
    results_map = {}

    for model_name, cache in [("BGE-M3 (current)", bge_cache),
                               ("BLaIR (legacy)",   blair_cache)]:
        print(f"  {model_name} ...", end="", flush=True)
        hr5, hr10, nd, mr, n, t = run_sampled_eval(
            cache, eval_users, per_user_negs,
            k=args.k, max_users=args.max_users,
        )
        print(f"\r  {bold(model_name):<22}  {green(f'HR@10={hr10:.4f}')}  "
              f"NDCG@10={nd:.4f}  {t:.1f}s")
        results_map[model_name] = {
            "hr5": hr5, "hr10": hr10, "ndcg10": nd, "mrr10": mr,
            "users": n, "time_s": round(t, 1),
        }

    print_results_table(results_map, args.negatives, args.k)
    print_latex(results_map)

    # ── 4. Score distributions ────────────────────────────────────────────────
    meta_df   = None
    meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if os.path.exists(meta_path):
        try:
            meta_df = pd.read_parquet(
                meta_path, columns=["parent_asin", "categories", "main_category"])
        except Exception:
            meta_df = pd.read_parquet(meta_path, columns=["parent_asin", "categories"])
        if "parent_asin" in meta_df.columns:
            meta_df.set_index("parent_asin", inplace=True)
    else:
        print(yellow("  item_metadata.parquet not found — skipping distribution analysis"))

    dist_results = (analyze_score_distribution(bge_cache, blair_cache,
                                               meta_df, n_sample=args.sample_n)
                    if meta_df is not None else {})

    # ── 5a. Top-K overlap — flat vs flat ─────────────────────────────────────
    overlap_results = analyze_topk_overlap(
        bge_hnsw, blair_hnsw, asin_to_idx, asins,
        n_queries=args.overlap_queries, k=args.k,
    )

    # ── 5b. HNSW recall — BGE only ───────────────────────────────────────────
    hnsw_results = analyze_hnsw_recall(
        bge_flat, bge_hnsw, asin_to_idx, asins,
        n_queries=args.hnsw_queries, k=args.k,
    )

    # ── 6. Optional latency ───────────────────────────────────────────────────
    latency_results = benchmark_encode_latency() if not args.no_latency else None

    # ── 7. Query retrieval quality ────────────────────────────────────────────
    query_results = None
    if not args.no_query_test:
        if bge_hnsw is not None:
            bge_enc, blair_enc = load_query_encoders(args.blair_model)
            query_results = run_query_test(
                bge_enc, blair_enc, bge_hnsw, blair_hnsw,
                asins, meta_df, k=args.k,
            )
        else:
            print(yellow("\nStep 7 skipped: BGE HNSW index not found"))

    # ── Summary printout ──────────────────────────────────────────────────────
    if dist_results:
        print(bold("\nScore separation (intra-genre / inter-genre cosine):"))
        for model_name, d in dist_results.items():
            print(f"  {model_name:<14}  {d['mean_intra_sim']:.4f} / "
                  f"{d['mean_inter_sim']:.4f}  →  ratio={d['separation_ratio']:.3f}")

    if overlap_results:
        print(bold(f"\nTop-10 Jaccard overlap (BGE HNSW vs BLaIR HNSW): "
                   f"{overlap_results['mean_jaccard']:.4f}"))

    if hnsw_results:
        print(bold(f"\nHNSW recall@{args.k} vs exact: "
                   f"{hnsw_results['mean_recall_at_k']:.4f}  |  "
                   f"speedup: {hnsw_results['speedup_median']:.1f}x"))

    # ── Save ──────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_global
    summary = {
        "run_id":            run_id,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "negatives":         args.negatives,
        "k":                 args.k,
        "seed":              args.seed,
        "total_elapsed_s":   round(total_elapsed, 1),
        "eval_results":      {
            name: {kk: round(v, 4) if isinstance(v, float) else v
                   for kk, v in r.items()}
            for name, r in results_map.items()
        },
        "score_distributions": dist_results,
        "topk_overlap":        overlap_results,
        "hnsw_recall":         hnsw_results,
        "encode_latency":      latency_results,
        "query_test":          query_results,
    }

    if not args.no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"encoder_comparison_{run_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  {green('Saved')} → {out_path}")

    print(bold(f"\n{'═'*72}"))
    print(bold(f"  Done in {total_elapsed:.1f}s"))
    print(bold(f"  Log  → {log_path}"))
    print(bold(f"{'═'*72}\n"))

    sys.stdout = sys.__stdout__
    _log_fh.close()


if __name__ == "__main__":
    main()
