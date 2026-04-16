"""
scripts/benchmark/multiseed_eval.py — Eval-variance error bars for DIF-SASRec.

Runs evaluate_recommendation.py N times with different random seeds against the
same trained checkpoint. Each seed draws a different random negative pool,
so the spread measures evaluation variance (sensitivity to which 99 distractors
are drawn).

This is a fast alternative to multi-seed retraining (~1h vs ~29h).

Usage:
    python scripts/benchmark/multiseed_eval.py
    python scripts/benchmark/multiseed_eval.py --seeds 42 123 456 789 2026
    python scripts/benchmark/multiseed_eval.py --negatives 999

Output:
    evaluation/multiseed_results.json   — per-seed results + mean ± std
    Prints a LaTeX-ready snippet for tables/main_results.tex
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.services.category_encoder import CategoryEncoder
from app.services.dif_sasrec import DIFSASRecAgent

DATA_DIR        = settings.DATA_DIR
EVAL_PATH       = os.path.join(ROOT, "evaluation", "eval_users.json")
OUTPUT_PATH     = os.path.join(ROOT, "evaluation", "multiseed_results.json")
DEFAULT_SEEDS   = [42, 123, 456, 789, 2026]


# ── Metric helpers ────────────────────────────────────────────────────────────

def hit_rate(ranked, target, k):
    return 1.0 if target in ranked[:k] else 0.0

def ndcg(ranked, target, k):
    for i, a in enumerate(ranked[:k]):
        if a == target:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ── Single-seed evaluation ────────────────────────────────────────────────────

def run_one_seed(seed, eval_users, retriever, cat_encoder, emb_cache,
                 all_asins, pretrained_path, n_neg, k, max_users):
    """
    Evaluate DIF-SASRec for one random seed.
    Returns dict with hr10, ndcg10, n_users, elapsed_s.
    """
    random.seed(seed)
    np.random.seed(seed)

    from app.services.dif_sasrec import DIFSASRecAgent
    agent = DIFSASRecAgent(
        retriever, cat_encoder,
        pretrained_path if os.path.exists(pretrained_path) else None,
    )

    # Build negative pool with this seed
    NEG_POOL       = min(200_000, len(all_asins))
    neg_pool_asins = random.sample(all_asins, NEG_POOL)
    neg_pool_asins = [a for a in neg_pool_asins if a in emb_cache]

    users     = eval_users[:max_users] if max_users else eval_users
    all_set   = set(all_asins)
    hr10_list = []
    nd10_list = []
    skipped   = 0
    t0        = time.time()

    for user in users:
        train  = user.get("train_clicks", [])
        test   = user.get("test_clicks",  [])
        if not train or not test:
            skipped += 1
            continue
        target = test[0]
        if target not in all_set:
            skipped += 1
            continue

        seen = set(train) | set(test)
        negs = [a for a in random.sample(neg_pool_asins,
                                         min(n_neg * 3, len(neg_pool_asins)))
                if a not in seen][:n_neg]
        if len(negs) < n_neg // 2:
            skipped += 1
            continue

        candidates = [target] + negs
        cat_ids    = cat_encoder.encode_sequence(train)
        scores     = agent.get_candidate_scores(train, cat_ids, candidates)
        ranked     = sorted(candidates, key=lambda a: scores.get(a, 0.0), reverse=True)

        hr10_list.append(hit_rate(ranked, target, k))
        nd10_list.append(ndcg(ranked, target, k))

    n       = len(hr10_list) or 1
    elapsed = time.time() - t0
    return {
        "seed":      seed,
        "hr10":      sum(hr10_list) / n,
        "ndcg10":    sum(nd10_list) / n,
        "n_users":   len(hr10_list),
        "skipped":   skipped,
        "elapsed_s": round(elapsed, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Eval-variance error bars for DIF-SASRec")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help=f"Random seeds to evaluate (default: {DEFAULT_SEEDS})")
    p.add_argument("--negatives", type=int, default=99,
                   help="Number of random negatives per user (default 99)")
    p.add_argument("--k", type=int, default=10,
                   help="Cutoff rank for metrics (default 10)")
    p.add_argument("--max-users", type=int, default=None,
                   help="Cap number of eval users (default: all)")
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to DIF-SASRec checkpoint (default: data/dif_sasrec_pretrained.pt)")
    return p.parse_args()


def main():
    args            = parse_args()
    pretrained_path = args.pretrained_path or os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")

    print(f"\n{'═'*70}")
    print(f"  DIF-SASRec Eval-Variance Run")
    print(f"  seeds={args.seeds}  negatives={args.negatives}  k={args.k}")
    print(f"  checkpoint: {pretrained_path}")
    print(f"{'═'*70}\n")

    # ── Load shared resources once ────────────────────────────────────────────
    print("Loading FAISS indices ...")
    cleora_data = np.load(os.path.join(DATA_DIR, "cleora_embeddings.npz"))
    retriever   = Retriever(DATA_DIR, cleora_data)
    print(f"  {len(retriever.asins):,} ASINs\n")

    cat_encoder    = CategoryEncoder()
    cat_vocab_path = os.path.join(DATA_DIR, "category_vocab.json")
    if os.path.exists(cat_vocab_path):
        cat_encoder.load(cat_vocab_path)
    else:
        cat_encoder.build_from_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))

    with open(EVAL_PATH) as f:
        eval_users = json.load(f)
    print(f"Eval users: {len(eval_users):,}\n")

    all_asins = list(retriever.asin_to_idx.keys())

    # ── Pre-cache embeddings once (shared across all seeds) ───────────────────
    print("Pre-loading embeddings into RAM ...")
    eval_set = set()
    for u in eval_users:
        eval_set.update(u.get("train_clicks", []))
        eval_set.update(u.get("test_clicks",  []))
    NEG_POOL        = min(200_000, len(all_asins))
    neg_sample      = random.sample(all_asins, NEG_POOL)
    to_load         = eval_set | set(neg_sample)
    to_load         = {a for a in to_load if a in retriever.asin_to_idx}

    t0        = time.time()
    emb_cache = {}
    for asin in to_load:
        emb_cache[asin] = retriever.text_flat.reconstruct(retriever.asin_to_idx[asin])
    print(f"  {len(emb_cache):,} ASINs cached in {time.time()-t0:.1f}s\n")

    # ── Run each seed ─────────────────────────────────────────────────────────
    per_seed = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"[{i}/{len(args.seeds)}] seed={seed} ...", end="", flush=True)
        result = run_one_seed(
            seed, eval_users, retriever, cat_encoder, emb_cache,
            all_asins, pretrained_path,
            n_neg=args.negatives, k=args.k, max_users=args.max_users,
        )
        per_seed.append(result)
        print(f"  HR@10={result['hr10']:.4f}  NDCG@10={result['ndcg10']:.4f}"
              f"  ({result['elapsed_s']:.0f}s)")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    hr10s   = [r["hr10"]   for r in per_seed]
    ndcg10s = [r["ndcg10"] for r in per_seed]

    mean_hr10   = float(np.mean(hr10s))
    std_hr10    = float(np.std(hr10s, ddof=1))
    mean_ndcg10 = float(np.mean(ndcg10s))
    std_ndcg10  = float(np.std(ndcg10s, ddof=1))

    summary = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "negatives":    args.negatives,
        "k":            args.k,
        "n_seeds":      len(args.seeds),
        "seeds":        args.seeds,
        "per_seed":     per_seed,
        "mean_hr10":    round(mean_hr10,   4),
        "std_hr10":     round(std_hr10,    4),
        "mean_ndcg10":  round(mean_ndcg10, 4),
        "std_ndcg10":   round(std_ndcg10,  4),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Results across {len(args.seeds)} seeds  (negatives={args.negatives})")
    print(f"{'─'*70}")
    print(f"  HR@10  : {mean_hr10:.4f} ± {std_hr10:.4f}")
    print(f"  NDCG@10: {mean_ndcg10:.4f} ± {std_ndcg10:.4f}")
    print(f"\n  Saved → {OUTPUT_PATH}")
    print(f"{'═'*70}")

    print(f"\n  LaTeX row for tables/main_results.tex:")
    print(f"  DIF-SASRec (ours) & "
          f"\\textbf{{{mean_hr10:.4f} $\\pm$ {std_hr10:.4f}}} & "
          f"\\textbf{{{mean_ndcg10:.4f} $\\pm$ {std_ndcg10:.4f}}} \\\\")
    print()


if __name__ == "__main__":
    main()
