"""
scripts/data/build_eval_users.py — Build evaluation/eval_users.json

Downloads the Amazon Reviews 2023 Books 5-core benchmark from HuggingFace
(same dataset used to build Cleora), filters to ASINs in our FAISS index,
and produces a leave-one-out train/test split.

Output format (evaluation/eval_users.json):
[
  {
    "user_id":     "AGGXXX...",
    "train_clicks": ["B001", "B002", "B003", "B004"],  # chronological, all but last
    "test_clicks":  ["B005"]                            # last interaction only
  },
  ...
]

Usage:
    python scripts/data/build_eval_users.py
    python scripts/data/build_eval_users.py --min-interactions 5 --max-users 20000
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd
from huggingface_hub import hf_hub_download

from app.config import settings

DATA_DIR  = settings.DATA_DIR
EVAL_PATH = os.path.join(ROOT, "evaluation", "eval_users.json")
REPO_ID   = "McAuley-Lab/Amazon-Reviews-2023"
FILENAME  = "benchmark/5core/timestamp/Books.train.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Build eval_users.json from Amazon Reviews 2023")
    p.add_argument("--min-interactions", type=int, default=5,
                   help="Minimum interactions per user to be included (default 5)")
    p.add_argument("--max-users", type=int, default=20000,
                   help="Cap on number of users in output (default 20000)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load valid ASINs from our FAISS index
    asins_path = os.path.join(DATA_DIR, "asins.csv")
    print(f"Loading valid ASINs from {asins_path} ...")
    valid_asins = set(pd.read_csv(asins_path, header=None)[0].astype(str).tolist())
    print(f"  {len(valid_asins):,} ASINs in our index")

    # 2. Download / load Amazon Reviews 2023 Books 5-core
    print(f"\nDownloading {FILENAME} from HuggingFace ...")
    print("  (Uses cache if already downloaded — safe to re-run)")
    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"  Cached at: {local_path}")

    df = pd.read_csv(local_path)
    print(f"  {len(df):,} raw interactions loaded")

    # Normalise column names
    if "parent_asin" not in df.columns and "asin" in df.columns:
        df.rename(columns={"asin": "parent_asin"}, inplace=True)

    # 3. Filter to ASINs in our index
    df = df[df["parent_asin"].isin(valid_asins)].copy()
    print(f"  {len(df):,} interactions after filtering to indexed ASINs")

    if len(df) == 0:
        print("ERROR: No interactions remain after filtering. "
              "Check that asins.csv matches the Amazon Books catalog.")
        sys.exit(1)

    # 4. Sort chronologically and deduplicate (keep first occurrence per user+item)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first", inplace=True)

    # 5. Keep only users with enough interactions
    user_counts = df.groupby("user_id")["parent_asin"].count()
    eligible    = user_counts[user_counts >= args.min_interactions].index
    df          = df[df["user_id"].isin(eligible)]
    print(f"  {df['user_id'].nunique():,} users with ≥{args.min_interactions} interactions")

    # 6. Leave-one-out split: last item is test, rest is train
    print(f"\nBuilding leave-one-out train/test split ...")
    eval_users = []
    for user_id, group in df.groupby("user_id"):
        items = group["parent_asin"].tolist()  # already sorted by timestamp
        if len(items) < 2:
            continue
        eval_users.append({
            "user_id":      user_id,
            "train_clicks": items[:-1],
            "test_clicks":  [items[-1]],
        })

    # 7. Cap at max_users (take the users with the most interactions for richer data)
    if len(eval_users) > args.max_users:
        eval_users.sort(key=lambda u: len(u["train_clicks"]), reverse=True)
        eval_users = eval_users[: args.max_users]
        print(f"  Capped to {args.max_users:,} users (kept those with most clicks)")

    print(f"  Final: {len(eval_users):,} eval users")
    avg_train = sum(len(u["train_clicks"]) for u in eval_users) / max(len(eval_users), 1)
    print(f"  Avg train_clicks per user: {avg_train:.1f}")

    # 8. Save
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_users, f, ensure_ascii=False)
    print(f"\nSaved → {EVAL_PATH}  ({os.path.getsize(EVAL_PATH)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
