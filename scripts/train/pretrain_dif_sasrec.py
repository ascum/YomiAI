"""
scripts/train/pretrain_dif_sasrec.py — Offline pre-training for DIF-SASRec.

Trains a global DIF-SASRec checkpoint using existing user interaction histories
from evaluation/eval_users.json. This checkpoint is loaded for ALL new users as
their starting point; per-user weights diverge from it via online fine-tuning.

Usage:
    python scripts/train/pretrain_dif_sasrec.py [--epochs N] [--device cuda]

Output:
    data/dif_sasrec_pretrained.pt   — model checkpoint
    data/category_vocab.json        — category vocabulary (built if not present)
"""
import argparse
import json
import os
import sys
import time

# Add project root so we can import app.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.services.category_encoder import CategoryEncoder
from app.services.dif_sasrec import DIFSASRecAgent

DATA_DIR        = settings.DATA_DIR
EVAL_PATH       = os.path.join(ROOT, "evaluation", "eval_users.json")
PRETRAINED_PATH = os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")
CAT_VOCAB_PATH  = os.path.join(DATA_DIR, "category_vocab.json")


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train DIF-SASRec on eval_users.json")
    p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    p.add_argument("--min-clicks", type=int, default=3,
                   help="Minimum train_clicks for a user to be included")
    return p.parse_args()


def main():
    args   = parse_args()
    t_start = time.time()

    # 1. Load Retriever (needs Cleora for the eval data, but DIF model itself is Cleora-free)
    print("Loading FAISS indices and Cleora embeddings ...")
    cleora_path = os.path.join(DATA_DIR, "cleora_embeddings.npz")
    cleora_data = np.load(cleora_path)
    retriever   = Retriever(DATA_DIR, cleora_data)
    print(f"  Retriever ready — {len(retriever.asins):,} ASINs in index")

    # 2. Load / build CategoryEncoder
    cat_encoder = CategoryEncoder()
    if os.path.exists(CAT_VOCAB_PATH):
        cat_encoder.load(CAT_VOCAB_PATH)
    else:
        print("Building category vocabulary from item_metadata.parquet ...")
        cat_encoder.build_from_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))
        cat_encoder.save(CAT_VOCAB_PATH)

    # 3. Build DIFSASRecAgent (fresh — no pretrained path yet)
    agent = DIFSASRecAgent(retriever, cat_encoder)
    param_count = sum(p.numel() for p in agent.model.parameters())
    print(f"Model parameters: {param_count:,}")

    # 4. Load eval users
    if not os.path.exists(EVAL_PATH):
        print(f"ERROR: eval file not found at {EVAL_PATH}")
        sys.exit(1)
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_users = json.load(f)

    eligible = [u for u in eval_users if len(u.get("train_clicks", [])) >= args.min_clicks]
    print(f"Eval users: {len(eval_users):,} total, {len(eligible):,} with ≥{args.min_clicks} clicks")

    all_asins = list(retriever.asin_to_idx.keys())

    # 5. Training loop
    print(f"\nStarting training for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        t_epoch     = time.time()
        total_loss  = 0.0
        n_steps     = 0
        n_skipped   = 0

        for user in eligible:
            train_clicks = user.get("train_clicks", [])

            for t in range(2, len(train_clicks)):
                input_seq  = train_clicks[:t]
                target     = train_clicks[t]
                target_cat = cat_encoder.get_category_id(target)

                loss = agent.train_step(input_seq, target, target_cat, all_asins)
                if loss is not None:
                    total_loss += loss
                    n_steps    += 1
                else:
                    n_skipped  += 1

        avg   = total_loss / max(n_steps, 1)
        elapsed = time.time() - t_epoch
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"steps={n_steps:,}  skipped={n_skipped}  "
              f"avg_loss={avg:.4f}  "
              f"epoch_time={elapsed:.0f}s  "
              f"total_elapsed={time.time()-t_start:.0f}s")

    # 6. Save checkpoint
    agent.save(PRETRAINED_PATH)
    print(f"\nPre-trained checkpoint saved → {PRETRAINED_PATH}")
    print(f"Total training time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
