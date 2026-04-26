"""
scripts/benchmark/evaluate_recommendation.py — Offline recommendation evaluation.

Two evaluation modes:

  --mode sampled  (DEFAULT, academic standard)
      For each user: rank the 1 real test item against N random negatives.
      N=99  -> rank among 100  items. Random baseline HR@10 = 10/100  = 0.100
      N=999 -> rank among 1000 items. Random baseline HR@10 = 10/1000 = 0.010
      This is the protocol used in SASRec / BERT4Rec / DIF-SASRec papers.

  --mode full
      Fetch HNSW candidates from the full 3M catalog and check if the test item
      appears in the top-K. This is "full-ranking" — numbers will be very low
      (0.001–0.010 even for good models against a 3M catalog).

Usage:
    python scripts/benchmark/evaluate_recommendation.py
    python scripts/benchmark/evaluate_recommendation.py --mode full
    python scripts/benchmark/evaluate_recommendation.py --negatives 999 --max-users 5000
"""
import argparse
import json
import logging
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

DATA_DIR     = settings.DATA_DIR
EVAL_PATH    = os.path.join(ROOT, "evaluation", "eval_users.json")
HISTORY_PATH = os.path.join(ROOT, "evaluation", "results_history.json")
LOG_DIR      = os.path.join(ROOT, "evaluation", "logs")

# ─── ANSI colours (auto-disabled when not a TTY) ─────────────────────────────
_IS_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)


# ─── Logger setup ─────────────────────────────────────────────────────────────

def setup_logger(run_id: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")

    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — plain text, full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # Console handler — INFO only, no timestamps (we format our own output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)   # suppress routine INFO from console (we print directly)
    logger.addHandler(ch)

    return logger, log_path


# ─── Strategy implementations ─────────────────────────────────────────────────

class ContentBaseline:
    """BGE-M3 profile vector -> HNSW KNN. No trained model."""
    name = "Content Baseline"

    def __init__(self, retriever, emb_cache: dict = None):
        self.retriever = retriever
        self.emb_cache = emb_cache or {}

    def _vec(self, asin):
        if asin in self.emb_cache:
            return self.emb_cache[asin]
        if asin in self.retriever.asin_to_idx:
            return self.retriever.text_flat.reconstruct(self.retriever.asin_to_idx[asin])
        return None

    def score_candidates(self, train_clicks: list, candidate_asins: list) -> dict:
        vecs = [v for a in train_clicks if (v := self._vec(a)) is not None]
        if not vecs:
            return {a: 0.0 for a in candidate_asins}
        profile = np.mean(vecs, axis=0)
        scores = {}
        for asin in candidate_asins:
            v = self._vec(asin)
            scores[asin] = float(profile @ v) if v is not None else 0.0
        return scores

    def recommend_full(self, train_clicks: list, k: int, exclude: set) -> list:
        vecs = [v for a in train_clicks if (v := self._vec(a)) is not None]
        if not vecs:
            return []
        profile = np.mean(vecs, axis=0)
        candidates = self.retriever.get_content_candidates(
            profile, top_n=settings.PERSONAL_CANDIDATES, exclude_asins=exclude
        )
        scores = self.score_candidates(train_clicks, candidates)
        return [a for a, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]


class GRUSeqDQNStrategy:
    """Existing GRU-Sequential DQN."""
    name = "GRU-SeqDQN"

    def __init__(self, retriever, emb_cache: dict = None):
        from app.services.rl_filter import RLSequentialFilter
        self.agent     = RLSequentialFilter(retriever)
        self.retriever = retriever
        self.emb_cache = emb_cache or {}
        if hasattr(self.agent, "set_embedding_cache"):
            self.agent.set_embedding_cache(self.emb_cache)

    def _vec(self, asin):
        if asin in self.emb_cache:
            return self.emb_cache[asin]
        if asin in self.retriever.asin_to_idx:
            return self.retriever.text_flat.reconstruct(self.retriever.asin_to_idx[asin])
        return None

    def score_candidates(self, train_clicks: list, candidate_asins: list) -> dict:
        return self.agent.get_candidate_scores(train_clicks, candidate_asins)

    def recommend_full(self, train_clicks: list, k: int, exclude: set) -> list:
        vecs = [v for a in train_clicks if (v := self._vec(a)) is not None]
        if not vecs:
            return []
        profile = np.mean(vecs, axis=0)
        candidates = self.retriever.get_content_candidates(
            profile, top_n=settings.PERSONAL_CANDIDATES, exclude_asins=exclude
        )
        scores = self.score_candidates(train_clicks, candidates)
        return [a for a, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]


class DIFSASRecStrategy:
    """DIF-SASRec personal recommendation."""
    name = "DIF-SASRec"

    def __init__(self, retriever, category_encoder, emb_cache: dict = None,
                 pretrained_path=None):
        from app.services.dif_sasrec import DIFSASRecAgent
        self.agent       = DIFSASRecAgent(retriever, category_encoder, pretrained_path)
        self.retriever   = retriever
        self.cat_encoder = category_encoder
        self.emb_cache   = emb_cache or {}
        if self.emb_cache:
            self.agent.set_embedding_cache(self.emb_cache)

    def _vec(self, asin):
        if asin in self.emb_cache:
            return self.emb_cache[asin]
        if asin in self.retriever.asin_to_idx:
            return self.retriever.text_flat.reconstruct(self.retriever.asin_to_idx[asin])
        return None

    def score_candidates(self, train_clicks: list, candidate_asins: list) -> dict:
        cat_ids = self.cat_encoder.encode_sequence(train_clicks)
        return self.agent.get_candidate_scores(train_clicks, cat_ids, candidate_asins)

    def recommend_full(self, train_clicks: list, k: int, exclude: set) -> list:
        vecs = [v for a in train_clicks if (v := self._vec(a)) is not None]
        if not vecs:
            return []
        profile = np.mean(vecs, axis=0)
        candidates = self.retriever.get_content_candidates(
            profile, top_n=settings.PERSONAL_CANDIDATES, exclude_asins=exclude
        )
        scores = self.score_candidates(train_clicks, candidates)
        return [a for a, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]


class PipelineAStrategy:
    """
    Pipeline A: Cleora behavioral graph + BGE-M3 profile similarity ranking.

    Scoring rule:
      - Candidate in Cleora index  → cosine(profile, candidate_bge_vec)
      - Candidate outside Cleora   → -2.0  (below any real cosine in [-1, 1])

    This faithfully models the coverage constraint: Pipeline A cannot
    retrieve non-Cleora items in the live system.
    """
    name = "Pipeline A (Cleora)"

    def __init__(self, retriever, emb_cache: dict = None):
        self.retriever = retriever
        self.emb_cache = emb_cache or {}

    def _vec(self, asin):
        if asin in self.emb_cache:
            return self.emb_cache[asin]
        if asin in self.retriever.asin_to_idx:
            return self.retriever.text_flat.reconstruct(self.retriever.asin_to_idx[asin])
        return None

    def score_candidates(self, train_clicks: list, candidate_asins: list) -> dict:
        vecs = [v for a in train_clicks if (v := self._vec(a)) is not None]
        if not vecs:
            return {a: -2.0 for a in candidate_asins}
        profile = np.mean(vecs, axis=0)
        scores = {}
        for asin in candidate_asins:
            if asin not in self.retriever.asin_to_cleora_idx:
                scores[asin] = -2.0
            else:
                v = self._vec(asin)
                scores[asin] = float(profile @ v) if v is not None else -2.0
        return scores


class CombinedStrategy:
    """
    Union of Pipeline A (Cleora+BGE-M3) and DIF-SASRec via RRF fusion.

    HR@K equals P(hit_A or hit_B) — the target appears in either pipeline's
    top-K.  RRF preserves this property while producing a single merged
    ranking for NDCG.  Standard k=60 constant.
    """
    name = "Combined (A+B)"
    _RRF_K = 60

    def __init__(self, pipeline_a: PipelineAStrategy, dif_sasrec: DIFSASRecStrategy):
        self.pipeline_a = pipeline_a
        self.dif_sasrec = dif_sasrec

    def score_candidates(self, train_clicks: list, candidate_asins: list) -> dict:
        sa = self.pipeline_a.score_candidates(train_clicks, candidate_asins)
        sb = self.dif_sasrec.score_candidates(train_clicks, candidate_asins)

        ranked_a = sorted(candidate_asins, key=lambda a: sa.get(a, 0.0), reverse=True)
        ranked_b = sorted(candidate_asins, key=lambda a: sb.get(a, 0.0), reverse=True)
        ra = {a: i + 1 for i, a in enumerate(ranked_a)}
        rb = {a: i + 1 for i, a in enumerate(ranked_b)}

        k = self._RRF_K
        return {a: 1.0 / (k + ra[a]) + 1.0 / (k + rb[a]) for a in candidate_asins}

    def recommend_full(self, train_clicks: list, k: int, exclude: set) -> list:
        top_a = self.pipeline_a.recommend_full(train_clicks, k, exclude)
        top_b = self.dif_sasrec.recommend_full(train_clicks, k, exclude)
        union = list(dict.fromkeys(top_a + top_b))
        scores = self.score_candidates(train_clicks, union)
        return sorted(union, key=lambda a: scores.get(a, 0.0), reverse=True)[:k]


# ─── Metric helpers ───────────────────────────────────────────────────────────

def hit_rate(ranked: list, target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0

def ndcg(ranked: list, target: str, k: int) -> float:
    for i, a in enumerate(ranked[:k]):
        if a == target:
            return 1.0 / math.log2(i + 2)
    return 0.0

def mrr(ranked: list, target: str, k: int) -> float:
    for i, a in enumerate(ranked[:k]):
        if a == target:
            return 1.0 / (i + 1)
    return 0.0


# ─── Embedding cache ──────────────────────────────────────────────────────────

def build_eval_cache(retriever, eval_users, all_asins, n_neg, logger):
    """
    Pre-load into RAM every embedding needed for evaluation:
      - All train + test ASINs from eval_users
      - A shared negative pool of min(200_000, len(all_asins)) random ASINs

    Returns (emb_cache, neg_pool_asins).
    """
    print(bold("\nPre-loading embeddings into RAM ..."))
    logger.info("Building embedding cache for evaluation")
    t0 = time.time()

    eval_set = set()
    for u in eval_users:
        eval_set.update(u.get("train_clicks", []))
        eval_set.update(u.get("test_clicks",  []))
    eval_set = {a for a in eval_set if a in retriever.asin_to_idx}
    logger.info(f"Eval sequences cover {len(eval_set):,} unique ASINs")

    NEG_POOL       = min(200_000, len(all_asins))
    neg_pool_asins = random.sample(all_asins, NEG_POOL)
    neg_pool_set   = {a for a in neg_pool_asins if a in retriever.asin_to_idx}

    to_load    = eval_set | neg_pool_set
    emb_cache  = {}
    n          = len(to_load)
    bar_width  = 30

    for i, asin in enumerate(to_load):
        idx = retriever.asin_to_idx[asin]
        emb_cache[asin] = retriever.text_flat.reconstruct(idx)

        if (i + 1) % 10_000 == 0 or i + 1 == n:
            pct      = (i + 1) / n
            filled   = int(bar_width * pct)
            bar      = green("█" * filled) + dim("░" * (bar_width - filled))
            elapsed  = time.time() - t0
            speed    = (i + 1) / elapsed
            eta      = (n - i - 1) / speed if speed > 0 else 0
            print(f"\r  [{bar}] {pct*100:5.1f}%  "
                  f"{i+1:>7,}/{n:,}  "
                  f"{speed:,.0f} ASINs/s  "
                  f"ETA {eta:4.0f}s  ", end="", flush=True)

    neg_pool_asins = [a for a in neg_pool_asins if a in emb_cache]
    elapsed = time.time() - t0
    size_mb = len(emb_cache) * 1024 * 4 / 1e6
    print(f"\r  {green('Done')}  {len(emb_cache):,} ASINs cached  "
          f"({size_mb:.0f} MB)  neg pool: {len(neg_pool_asins):,}  "
          f"in {elapsed:.1f}s          ")
    logger.info(f"Cache ready: {len(emb_cache):,} ASINs ({size_mb:.0f} MB) in {elapsed:.1f}s")
    return emb_cache, neg_pool_asins


# ─── Evaluation modes ─────────────────────────────────────────────────────────

def eval_sampled(strategy, eval_users, all_asins, neg_pool_asins,
                 n_neg, k, max_users, logger):
    """
    Sampled evaluation (academic standard).
    Rank test item among itself + n_neg random negatives drawn from neg_pool_asins.
    """
    hr5, hr10, ndcg10, mrr10 = [], [], [], []
    t0          = time.time()
    all_asins_s = set(all_asins)
    users       = eval_users[:max_users] if max_users else eval_users
    n_users     = len(users)
    skipped     = 0
    LOG_EVERY   = max(1, n_users // 20)   # log progress ~20 times

    logger.info(f"[{strategy.name}] sampled eval  n_neg={n_neg}  users={n_users:,}")

    for i, user in enumerate(users):
        train  = user.get("train_clicks", [])
        test   = user.get("test_clicks",  [])
        if not train or not test:
            skipped += 1
            continue
        target = test[0]
        if target not in all_asins_s:
            skipped += 1
            continue

        seen = set(train) | set(test)
        negs = [a for a in random.sample(neg_pool_asins, min(n_neg * 3, len(neg_pool_asins)))
                if a not in seen][:n_neg]
        if len(negs) < n_neg // 2:
            skipped += 1
            continue

        candidates = [target] + negs
        # random.shuffle(candidates)          # break ties randomly, not in favour of target
        scores     = strategy.score_candidates(train, candidates)
        ranked     = sorted(candidates, key=lambda a: scores.get(a, 0.0), reverse=True)

        hr5.append(hit_rate(ranked, target, 5))
        hr10.append(hit_rate(ranked, target, 10))
        ndcg10.append(ndcg(ranked, target, 10))
        mrr10.append(mrr(ranked, target, 10))

        if (i + 1) % LOG_EVERY == 0:
            done = len(hr5)
            avg_hr10 = sum(hr10) / done if done else 0
            elapsed  = time.time() - t0
            logger.info(f"  [{strategy.name}] {i+1:>6,}/{n_users:,}  "
                        f"HR@10={avg_hr10:.4f}  {elapsed:.0f}s")

    n = len(hr5) or 1
    elapsed = time.time() - t0
    logger.info(f"[{strategy.name}] done  n={len(hr5):,}  skipped={skipped}  "
                f"HR@10={sum(hr10)/n:.4f}  NDCG@10={sum(ndcg10)/n:.4f}  {elapsed:.1f}s")
    return (
        sum(hr5)    / n,
        sum(hr10)   / n,
        sum(ndcg10) / n,
        sum(mrr10)  / n,
        len(hr5),
        elapsed,
    )


def eval_full(strategy, eval_users, k, max_users, logger):
    """Full-ranking evaluation against retrieved candidates from the 3M catalog."""
    hr5, hr10, ndcg10, mrr10 = [], [], [], []
    t0    = time.time()
    users = eval_users[:max_users] if max_users else eval_users
    logger.info(f"[{strategy.name}] full eval  k={k}  users={len(users):,}")

    for user in users:
        train   = user.get("train_clicks", [])
        test    = user.get("test_clicks",  [])
        if not train or not test:
            continue
        target  = test[0]
        exclude = set(train) | set(test)
        ranked  = strategy.recommend_full(train, k=max(k, 10), exclude=exclude)

        hr5.append(hit_rate(ranked, target, 5))
        hr10.append(hit_rate(ranked, target, 10))
        ndcg10.append(ndcg(ranked, target, 10))
        mrr10.append(mrr(ranked, target, 10))

    n       = len(hr5) or 1
    elapsed = time.time() - t0
    logger.info(f"[{strategy.name}] done  n={len(hr5):,}  HR@10={sum(hr10)/n:.4f}  {elapsed:.1f}s")
    return (
        sum(hr5)    / n,
        sum(hr10)   / n,
        sum(ndcg10) / n,
        sum(mrr10)  / n,
        len(hr5),
        elapsed,
    )



# ─── Complementarity ──────────────────────────────────────────────────────────

def eval_complementarity(strategy_a, strategy_b, eval_users, neg_pool_asins,
                          n_neg, k, max_users, logger):
    """
    Per-user 2×2 hit matrix: did Pipeline A hit? did Pipeline B hit?

    Both strategies score the same shuffled candidate pool so the results
    are directly comparable and the combined OR hit is consistent with the
    main eval_sampled numbers.

    Returns (counts_dict, n_evaluated, elapsed_s) where counts_dict has
    keys 'aa' (both hit), 'ab' (A only), 'ba' (B only), 'bb' (neither).
    """
    counts = {"aa": 0, "ab": 0, "ba": 0, "bb": 0}
    users  = eval_users[:max_users] if max_users else eval_users
    t0     = time.time()
    logger.info(f"Complementarity  n_users={len(users):,}  k={k}")

    for user in users:
        train  = user.get("train_clicks", [])
        test   = user.get("test_clicks",  [])
        if not train or not test:
            continue
        target = test[0]
        seen   = set(train) | set(test)
        negs   = [a for a in random.sample(neg_pool_asins,
                  min(n_neg * 3, len(neg_pool_asins))) if a not in seen][:n_neg]
        if len(negs) < n_neg // 2:
            continue

        candidates = [target] + negs
        # shuffle removed — was causing random-state interference analogous
        # to the main eval_sampled shuffle that was reverted on 2026-04-26
        sa = strategy_a.score_candidates(train, candidates)
        sb = strategy_b.score_candidates(train, candidates)

        ranked_a = sorted(candidates, key=lambda a: sa.get(a, 0.0), reverse=True)
        ranked_b = sorted(candidates, key=lambda a: sb.get(a, 0.0), reverse=True)

        hit_a = target in ranked_a[:k]
        hit_b = target in ranked_b[:k]

        if   hit_a and     hit_b: counts["aa"] += 1
        elif hit_a and not hit_b: counts["ab"] += 1
        elif not hit_a and hit_b: counts["ba"] += 1
        else:                     counts["bb"] += 1

    elapsed = time.time() - t0
    n = sum(counts.values()) or 1
    logger.info(
        f"Complementarity done  n={n:,}  {elapsed:.1f}s  "
        f"A∩B={counts['aa']:,}  A-only={counts['ab']:,}  "
        f"B-only={counts['ba']:,}  neither={counts['bb']:,}"
    )
    return counts, n, elapsed


def print_complementarity_table(counts: dict, n: int, k: int):
    aa, ab, ba, bb = counts["aa"], counts["ab"], counts["ba"], counts["bb"]
    combined_hr = (aa + ab + ba) / n

    print()
    print(bold(f"  Pipeline complementarity  (k={k}, n={n:,})"))
    print()
    print(f"  {'':30s}  {'B hits':>14}  {'B misses':>14}")
    print("  " + "─" * 64)
    print(f"  {'A hits':30s}  "
          f"{green(f'{aa:,}  ({aa/n*100:.1f}%)'):>23}  "
          f"{f'{ab:,}  ({ab/n*100:.1f}%)':>14}")
    print(f"  {'A misses':30s}  "
          f"{f'{ba:,}  ({ba/n*100:.1f}%)':>14}  "
          f"{dim(f'{bb:,}  ({bb/n*100:.1f}%)'):>23}")
    print("  " + "─" * 64)
    b_rescues_a = ba / (ba + bb) if (ba + bb) else 0.0
    a_rescues_b = ab / (ab + bb) if (ab + bb) else 0.0
    print(f"  A rescues {b_rescues_a*100:.1f}% of B misses  |  "
          f"B rescues {a_rescues_b*100:.1f}% of A misses")
    print(f"  Combined HR@{k} = {green(f'{combined_hr:.4f}')}")
    print()


# ─── History ──────────────────────────────────────────────────────────────────

def load_history() -> list:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(record: dict):
    history = load_history()
    history.append(record)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def build_record(run_id, args, results_map, baseline_hr10, baseline_ndcg10,
                 n_eval_users, total_elapsed):
    """Build the history dict for one evaluation run."""
    return {
        "run_id":          run_id,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "mode":            args.mode,
        "negatives":       args.negatives if args.mode == "sampled" else None,
        "k":               args.k,
        "max_users":       args.max_users,
        "n_eval_users":    n_eval_users,
        "total_elapsed_s": round(total_elapsed, 1),
        "random_baseline": {
            "hr10":    round(baseline_hr10,    4),
            "ndcg10":  round(baseline_ndcg10,  4),
        } if args.mode == "sampled" else None,
        "results": {
            name: {
                "hr5":    round(r["hr5"],    4),
                "hr10":   round(r["hr10"],   4),
                "ndcg10": round(r["ndcg10"], 4),
                "mrr10":  round(r["mrr10"],  4),
                "users":  r["users"],
                "time_s": round(r["time_s"], 1),
            }
            for name, r in results_map.items()
        },
    }


# ─── Pretty printing ──────────────────────────────────────────────────────────

def print_header(mode, negatives, k, n_users):
    width = 76
    print()
    print(bold("┌" + "─" * (width - 2) + "┐"))
    if mode == "sampled":
        title = f"SAMPLED EVALUATION   negatives={negatives}   pool={negatives+1}   k={k}"
    else:
        title = f"FULL-RANKING EVALUATION   k={k}   (numbers will be low vs 3M catalog)"
    print(bold(f"│  {cyan(title):<{width - 4}}  │"))
    print(bold(f"│  {dim(f'Evaluating {n_users:,} users'):<{width - 4}}  │"))
    print(bold("└" + "─" * (width - 2) + "┘"))


def _metric_cell(val, baseline=None):
    """Colour a metric value relative to the random baseline."""
    s = f"{val:.4f}"
    if baseline is None:
        return s
    if val >= baseline * 1.5:
        return green(s)
    if val >= baseline * 1.0:
        return yellow(s)
    return s


def print_results_table(results_map, mode, negatives, k):
    baseline_hr10   = 10 / (negatives + 1) if mode == "sampled" else None
    baseline_ndcg10 = (sum(1 / math.log2(i + 2) for i in range(10)) / (negatives + 1)
                       if mode == "sampled" else None)

    # Column widths
    w_name = max(len(n) for n in results_map) + 2
    print()
    print(bold(f"  {'Strategy':<{w_name}}  {'HR@5':>8}  {'HR@10':>8}  "
               f"{'NDCG@10':>9}  {'MRR@10':>8}  {'Users':>7}  {'Time':>7}"))
    print("  " + "─" * (w_name + 60))

    best_hr10 = max(r["hr10"] for r in results_map.values())

    for name, r in results_map.items():
        marker = " *" if r["hr10"] == best_hr10 else "  "
        hr5_s   = _metric_cell(r["hr5"],    baseline_hr10)
        hr10_s  = _metric_cell(r["hr10"],   baseline_hr10)
        ndcg_s  = _metric_cell(r["ndcg10"], baseline_ndcg10)
        mrr_s   = _metric_cell(r["mrr10"],  baseline_hr10)
        time_s  = f"{r['time_s']:>5.1f}s"
        print(f"  {bold(name):<{w_name + (9 if _IS_TTY else 0)}}"
              f"  {hr5_s:>8}  {hr10_s:>8}  {ndcg_s:>9}  {mrr_s:>8}"
              f"  {r['users']:>7,}  {time_s}{marker}")

    if mode == "sampled":
        print("  " + "─" * (w_name + 60))
        b_hr10  = f"{baseline_hr10:.4f}"
        b_ndcg  = f"{baseline_ndcg10:.4f}"
        print(f"  {dim('Random baseline'):<{w_name}}  "
              f"{dim(b_hr10):>8}  {dim(b_hr10):>8}  {dim(b_ndcg):>9}  "
              f"{dim(b_hr10):>8}")

    print()


def print_history_trend(history, mode):
    """Show the last 5 runs of the same mode for trend tracking."""
    same = [r for r in history if r.get("mode") == mode]
    if len(same) < 2:
        return

    recent = same[-5:]
    print(bold(f"  Run history ({mode} mode, last {len(recent)} runs):"))
    print(f"  {'Timestamp':<22}  {'Strategy':<22}  {'HR@10':>8}  {'NDCG@10':>9}")
    print("  " + "─" * 68)

    for rec in recent:
        ts  = rec["timestamp"]
        for name, m in rec["results"].items():
            print(f"  {dim(ts):<22}  {name:<22}  {m['hr10']:>8.4f}  {m['ndcg10']:>9.4f}")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Offline recommendation evaluation")
    p.add_argument("--mode", choices=["sampled", "full"], default="sampled",
                   help="sampled = rank vs N negatives (default, academic standard); "
                        "full = rank in full HNSW retrieval pool")
    p.add_argument("--negatives", type=int, default=99,
                   help="Number of random negatives per user in sampled mode (default 99)")
    p.add_argument("--k", type=int, default=10,
                   help="Cutoff rank for metrics (default 10)")
    p.add_argument("--max-users", type=int, default=None,
                   help="Limit eval users for speed (default: all 20k)")
    p.add_argument("--no-history", action="store_true",
                   help="Skip saving result to evaluation/results_history.json")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for negative pool sampling (default: unseeded)")
    p.add_argument("--dif-only", action="store_true",
                   help="Only evaluate DIF-SASRec — skip Content-KNN and GRU-SeqDQN")
    p.add_argument("--pipeline-a-only", action="store_true",
                   help="Only evaluate Pipeline A (Cleora+BGE-M3) — no model inference needed")
    p.add_argument("--combined", action="store_true",
                   help="Run Pipeline A + DIF-SASRec + Combined (A+B) and print complementarity table")
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to DIF-SASRec checkpoint (default: data/dif_sasrec_pretrained.pt)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_path = setup_logger(run_id)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(bold(f"\n{'═' * 76}"))
    print(bold(f"  Recommendation Evaluation   run_id={cyan(run_id)}"))
    print(bold(f"{'═' * 76}"))
    print(f"  Log: {dim(log_path)}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    logger.info(f"Run start  mode={args.mode}  negatives={args.negatives}  k={args.k}  seed={args.seed}")

    # ── Load FAISS + encoders ─────────────────────────────────────────────────
    print(f"\n{bold('Loading FAISS indices ...')}")
    t_load = time.time()
    cleora_data = np.load(os.path.join(DATA_DIR, "cleora_embeddings.npz"))
    retriever   = Retriever(DATA_DIR, cleora_data)
    print(f"  Retriever ready — {len(retriever.asins):,} ASINs  "
          f"({time.time()-t_load:.1f}s)")
    logger.info(f"Retriever loaded: {len(retriever.asins):,} ASINs")

    cat_encoder    = CategoryEncoder()
    cat_vocab_path = os.path.join(DATA_DIR, "category_vocab.json")
    if os.path.exists(cat_vocab_path):
        cat_encoder.load(cat_vocab_path)
    else:
        cat_encoder.build_from_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))

    pretrained_path = args.pretrained_path or os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")
    if os.path.exists(pretrained_path):
        print(f"  DIF-SASRec checkpoint: {green('found')} ({pretrained_path})")
    else:
        print(f"  DIF-SASRec checkpoint: {yellow('not found')} — random weights")

    if not os.path.exists(EVAL_PATH):
        print(f"\n  ERROR: {EVAL_PATH} not found. Run setup_dif_sasrec.py first.")
        sys.exit(1)

    with open(EVAL_PATH) as f:
        eval_users = json.load(f)
    n_eval = min(len(eval_users), args.max_users or len(eval_users))
    logger.info(f"Eval users loaded: {len(eval_users):,} total  using {n_eval:,}")

    all_asins = list(retriever.asin_to_idx.keys())

    # ── Pre-cache embeddings ──────────────────────────────────────────────────
    emb_cache, neg_pool_asins = build_eval_cache(
        retriever, eval_users, all_asins, args.negatives, logger
    )

    # ── Build strategies ─────────────────────────────────────────────────────
    if args.pipeline_a_only:
        strategies = [PipelineAStrategy(retriever, emb_cache)]
    elif args.dif_only:
        dif_strategy = DIFSASRecStrategy(
            retriever, cat_encoder, emb_cache,
            pretrained_path=pretrained_path if os.path.exists(pretrained_path) else None,
        )
        strategies = [dif_strategy]
    elif args.combined:
        pipeline_a   = PipelineAStrategy(retriever, emb_cache)
        dif_strategy = DIFSASRecStrategy(
            retriever, cat_encoder, emb_cache,
            pretrained_path=pretrained_path if os.path.exists(pretrained_path) else None,
        )
        combined     = CombinedStrategy(pipeline_a, dif_strategy)
        strategies   = [pipeline_a, dif_strategy, combined]
    else:
        dif_strategy = DIFSASRecStrategy(
            retriever, cat_encoder, emb_cache,
            pretrained_path=pretrained_path if os.path.exists(pretrained_path) else None,
        )
        strategies = [
            ContentBaseline(retriever, emb_cache),
            GRUSeqDQNStrategy(retriever, emb_cache),
            dif_strategy,
        ]

    # ── Print run header ─────────────────────────────────────────────────────
    print_header(args.mode, args.negatives, args.k, n_eval)

    t_eval_start = time.time()
    results_map  = {}

    if args.mode == "sampled":
        baseline_hr10   = 10 / (args.negatives + 1)
        baseline_ndcg10 = sum(1 / math.log2(i + 2) for i in range(10)) / (args.negatives + 1)

        for s in strategies:
            print(f"  {bold(s.name)} ...", end="", flush=True)
            r = eval_sampled(s, eval_users, all_asins, neg_pool_asins,
                             n_neg=args.negatives, k=args.k,
                             max_users=args.max_users, logger=logger)
            hr5, hr10, nd, mr, n, t = r
            print(f"\r  {bold(s.name):<22}  done  {green(f'HR@10={hr10:.4f}')}"
                  f"  NDCG@10={nd:.4f}  {t:.1f}s")
            results_map[s.name] = {
                "hr5": hr5, "hr10": hr10, "ndcg10": nd, "mrr10": mr,
                "users": n, "time_s": t,
            }

        print_results_table(results_map, args.mode, args.negatives, args.k)

        # ── Complementarity (only when --combined and sampled mode) ──────────
        if args.combined and args.mode == "sampled":
            print(bold("\n  Computing complementarity table ..."))
            counts, n_comp, t_comp = eval_complementarity(
                pipeline_a, dif_strategy, eval_users, neg_pool_asins,
                n_neg=args.negatives, k=args.k,
                max_users=args.max_users, logger=logger,
            )
            print_complementarity_table(counts, n_comp, args.k)
            logger.info(f"Complementarity done in {t_comp:.1f}s")

    else:
        baseline_hr10   = None
        baseline_ndcg10 = None

        for s in strategies:
            print(f"  {bold(s.name)} ...", end="", flush=True)
            r = eval_full(s, eval_users, k=args.k,
                          max_users=args.max_users, logger=logger)
            hr5, hr10, nd, mr, n, t = r
            print(f"\r  {bold(s.name):<22}  done  {green(f'HR@10={hr10:.4f}')}"
                  f"  NDCG@10={nd:.4f}  {t:.1f}s")
            results_map[s.name] = {
                "hr5": hr5, "hr10": hr10, "ndcg10": nd, "mrr10": mr,
                "users": n, "time_s": t,
            }

        print_results_table(results_map, args.mode, args.negatives, args.k)

    total_elapsed = time.time() - t_eval_start
    logger.info(f"All strategies done in {total_elapsed:.1f}s")

    # ── Save history ─────────────────────────────────────────────────────────
    if not args.no_history:
        record = build_record(
            run_id, args, results_map,
            baseline_hr10   or 0,
            baseline_ndcg10 or 0,
            n_eval, total_elapsed,
        )
        save_history(record)
        print(f"  Result saved to {green(HISTORY_PATH)}")
        logger.info(f"Result saved to {HISTORY_PATH}")

    # ── Trend view ────────────────────────────────────────────────────────────
    history = load_history()
    print_history_trend(history, args.mode)

    print(bold(f"{'═' * 76}"))
    print(bold(f"  Total time: {total_elapsed:.1f}s   Log: {dim(log_path)}"))
    print(bold(f"{'═' * 76}\n"))
    logger.info(f"Run complete  total={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
