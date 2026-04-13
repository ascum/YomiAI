"""
scripts/setup_dif_sasrec.py — One-time setup for the DIF-SASRec personal pipeline.

Runs three stages in order:
  Stage 0: Build evaluation/eval_users.json from Amazon Reviews 2023
  Stage 1: Build category vocabulary from item_metadata.parquet
  Stage 2: Pre-train the DIF-SASRec model on eval_users.json

After this script completes, just run the API normally:
    uvicorn app.main:app --host 127.0.0.1 --port 8000

The API will auto-load both artifacts (category_vocab.json + dif_sasrec_pretrained.pt).

Usage:
    python scripts/setup_dif_sasrec.py               # full run, 30 epochs
    python scripts/setup_dif_sasrec.py --epochs 5    # quick smoke test (~10 min total)
    python scripts/setup_dif_sasrec.py --skip-eval   # skip Stage 0 if eval_users.json exists
    python scripts/setup_dif_sasrec.py --skip-vocab  # skip Stage 1 if category_vocab.json exists
    python scripts/setup_dif_sasrec.py --skip-eval --skip-vocab  # only run training
"""
import argparse
import json
import os
import sys
import time

# ── Project root on path so we can import app.* ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.services.category_encoder import CategoryEncoder
from app.services.dif_sasrec import DIFSASRecAgent

DATA_DIR        = settings.DATA_DIR
CAT_VOCAB_PATH  = os.path.join(DATA_DIR, "category_vocab.json")
PRETRAINED_PATH = os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")
EVAL_PATH       = os.path.join(ROOT, "evaluation", "eval_users.json")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Full DIF-SASRec setup: eval data -> category vocab -> pre-training"
    )
    p.add_argument("--epochs",     type=int, default=30,
                   help="Training epochs (default 30, use 5 for a quick test)")
    p.add_argument("--min-clicks", type=int, default=6,
                   help="Min train_clicks for a user to be included in training (default 6)")
    p.add_argument("--max-users",  type=int, default=100000,
                   help="Max users in eval_users.json (default 100000)")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Training batch size (default 2048; use 1024 if VRAM is tight)")
    p.add_argument("--skip-eval",  action="store_true",
                   help="Skip Stage 0 if evaluation/eval_users.json already exists")
    p.add_argument("--skip-vocab", action="store_true",
                   help="Skip Stage 1 if data/category_vocab.json already exists")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip Stage 2 (only build eval data + vocab)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def stage0_build_eval_users(skip: bool, max_users: int):
    """Download Amazon Reviews 2023 and build leave-one-out eval split."""
    print("\n" + "=" * 60)
    print("STAGE 0 — Build eval_users.json")
    print("=" * 60)

    if skip and os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            n = len(json.load(f))
        print(f"Skipped — loaded existing file ({n:,} users)")
        return

    import pandas as pd
    from huggingface_hub import hf_hub_download

    REPO_ID  = "McAuley-Lab/Amazon-Reviews-2023"
    FILENAME = "benchmark/5core/timestamp/Books.train.csv"

    # Load valid ASINs from our index
    asins_path = os.path.join(DATA_DIR, "asins.csv")
    print(f"Loading valid ASINs from {asins_path} ...")
    valid_asins = set(pd.read_csv(asins_path, header=None)[0].astype(str).tolist())
    print(f"  {len(valid_asins):,} ASINs in our FAISS index")

    # Download the dataset (cached by HuggingFace — safe to re-run)
    print(f"\nFetching {FILENAME} from HuggingFace ...")
    print("  (Uses local HF cache if already downloaded)")
    t0         = time.time()
    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    df         = pd.read_csv(local_path)
    print(f"  {len(df):,} raw interactions  ({time.time()-t0:.0f}s)")

    # Normalise column name
    if "parent_asin" not in df.columns and "asin" in df.columns:
        df.rename(columns={"asin": "parent_asin"}, inplace=True)

    # Filter to ASINs in our catalog, sort by time, deduplicate
    df = df[df["parent_asin"].isin(valid_asins)].copy()
    print(f"  {len(df):,} interactions match our catalog")

    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first", inplace=True)

    # Keep users with ≥ 2 interactions (need at least 1 train + 1 test)
    counts   = df.groupby("user_id")["parent_asin"].count()
    eligible = counts[counts >= 2].index
    df       = df[df["user_id"].isin(eligible)]
    print(f"  {df['user_id'].nunique():,} users with ≥2 interactions")

    # Leave-one-out: last item = test, rest = train
    eval_users = []
    for user_id, group in df.groupby("user_id"):
        items = group["parent_asin"].tolist()
        eval_users.append({
            "user_id":      user_id,
            "train_clicks": items[:-1],
            "test_clicks":  [items[-1]],
        })

    # Cap at max_users, prefer users with richer histories
    if len(eval_users) > max_users:
        eval_users.sort(key=lambda u: len(u["train_clicks"]), reverse=True)
        eval_users = eval_users[:max_users]

    avg_train = sum(len(u["train_clicks"]) for u in eval_users) / max(len(eval_users), 1)
    print(f"\n  Final: {len(eval_users):,} users  |  avg train_clicks: {avg_train:.1f}")

    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_users, f, ensure_ascii=False)

    size_mb = os.path.getsize(EVAL_PATH) / 1e6
    print(f"Stage 0 complete -> {EVAL_PATH}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
def stage1_build_vocab(skip: bool) -> CategoryEncoder:
    print("\n" + "=" * 60)
    print("STAGE 1 — Category Vocabulary")
    print("=" * 60)

    cat_encoder = CategoryEncoder()

    if skip and os.path.exists(CAT_VOCAB_PATH):
        cat_encoder.load(CAT_VOCAB_PATH)
        print(f"Skipped — loaded existing vocab ({cat_encoder.num_categories} categories)")
        return cat_encoder

    meta_path = os.path.join(DATA_DIR, "item_metadata.parquet")
    if not os.path.exists(meta_path):
        print(f"ERROR: item_metadata.parquet not found at {meta_path}")
        sys.exit(1)

    t0 = time.time()
    cat_encoder.build_from_parquet(meta_path)
    cat_encoder.save(CAT_VOCAB_PATH)
    print(f"Stage 1 complete in {time.time()-t0:.0f}s -> {CAT_VOCAB_PATH}")
    return cat_encoder


# ─────────────────────────────────────────────────────────────────────────────
def stage2_pretrain(cat_encoder: CategoryEncoder, epochs: int, min_clicks: int,
                    batch_size: int = 2048):
    print("\n" + "=" * 60)
    print("STAGE 2 — DIF-SASRec Pre-training")
    print("=" * 60)

    # Load FAISS + Cleora (Cleora only used to build the Retriever object;
    # DIF-SASRec itself never reads Cleora embeddings)
    print("Loading FAISS indices and Cleora embeddings ...")
    cleora_path = os.path.join(DATA_DIR, "cleora_embeddings.npz")
    if not os.path.exists(cleora_path):
        print(f"ERROR: cleora_embeddings.npz not found at {cleora_path}")
        sys.exit(1)

    cleora_data = np.load(cleora_path)
    retriever   = Retriever(DATA_DIR, cleora_data)
    print(f"  Retriever ready — {len(retriever.asins):,} ASINs in FAISS index")

    # Load eval users
    if not os.path.exists(EVAL_PATH):
        print(f"ERROR: eval_users.json not found at {EVAL_PATH}")
        print("       Run Stage 0 first (remove --skip-eval flag).")
        sys.exit(1)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_users = json.load(f)

    eligible = [u for u in eval_users
                if len(u.get("train_clicks", [])) >= min_clicks]
    total_steps_per_epoch = sum(
        max(0, len(u["train_clicks"]) - 2) for u in eligible
    )
    print(f"  Users: {len(eval_users):,} total, {len(eligible):,} eligible  "
          f"|  ~{total_steps_per_epoch:,} steps/epoch")

    # ── Pre-load embeddings into RAM ─────────────────────────────────────────
    # The bottleneck in naive training is FAISS mmap reads (disk I/O) for every
    # sequence item and every negative sample. Pre-loading unique ASINs into RAM
    # eliminates this — reconstruction happens once, training runs from memory.
    print("\n  Pre-loading embeddings into RAM ...")
    t_cache = time.time()

    # Collect all unique ASINs that appear in training sequences
    unique_asins = set()
    for user in eligible:
        unique_asins.update(user["train_clicks"])
    unique_asins = [a for a in unique_asins if a in retriever.asin_to_idx]

    # Reconstruct from FAISS flat index into a dict (one-time cost)
    emb_cache: dict = {}
    n = len(unique_asins)
    for i, asin in enumerate(unique_asins):
        idx = retriever.asin_to_idx[asin]
        emb_cache[asin] = retriever.text_flat.reconstruct(idx)
        if (i + 1) % 50_000 == 0 or i + 1 == n:
            pct = (i + 1) / n * 100
            print(f"    {i+1:>7,}/{n:,}  ({pct:.0f}%)  "
                  f"{time.time()-t_cache:.0f}s elapsed", end="\r")

    print(f"\n  Cached {len(emb_cache):,} ASINs  "
          f"({len(emb_cache)*1024*4/1e6:.0f} MB)  "
          f"in {time.time()-t_cache:.0f}s")

    # Negative pool is rebuilt each epoch inside the training loop to expose the
    # model to fresh hard negatives — reusing the same 50k across all epochs
    # lets the model memorise which negatives are "easy".
    NEG_POOL_SIZE   = min(50_000, len(emb_cache))
    import random
    _all_cache_asins = list(emb_cache.keys())
    print(f"  Negative pool: {NEG_POOL_SIZE:,} ASINs resampled each epoch  "
          f"(pool source: {len(_all_cache_asins):,} cached ASINs)")

    # Fresh agent with the embedding cache injected
    agent = DIFSASRecAgent(retriever, cat_encoder)
    agent.set_embedding_cache(emb_cache)
    param_count = sum(p.numel() for p in agent.model.parameters())
    print(f"  Model parameters: {param_count:,}  |  device: {agent.device}")

    # ── Build flat list of all (input_seq, target, cat_id) examples ──────────
    # Doing this once lets us shuffle across all users each epoch, which gives
    # better gradient diversity than processing user-by-user in fixed order.
    print("\n  Building training examples list ...")
    all_examples = []
    for user in eligible:
        tc = user.get("train_clicks", [])
        for t in range(2, len(tc)):
            all_examples.append((
                tc[:t],
                tc[t],
                cat_encoder.get_category_id(tc[t]),
            ))
    print(f"  {len(all_examples):,} total (seq, target, cat) examples")

    # ── Batch size — controlled by --batch-size CLI arg (default 2048) ─────────
    # 2048 targets ~8-10 GB VRAM with hidden_dim=512 + AMP (fp16 halves activation mem)
    # Drop to 1024 if you get OOM errors on the first epoch
    BATCH_SIZE = batch_size
    n_batches  = (len(all_examples) + BATCH_SIZE - 1) // BATCH_SIZE
    LOG_EVERY  = max(1, n_batches // 10)   # print ~10 lines per epoch

    # ── LR scheduler: linear warmup → cosine decay ───────────────────────────
    total_steps  = n_batches * epochs
    warmup_steps = n_batches * settings.SASREC_WARMUP_EPOCHS
    agent.configure_scheduler(total_steps, warmup_steps)

    t_start = time.time()
    print(f"\n  Training {epochs} epochs  |  batch={BATCH_SIZE}  "
          f"batches/epoch={n_batches:,}  AMP={'on' if agent._amp_enabled else 'off'}\n")
    print(f"  {'Epoch':>5}  {'Batch':>8}  {'Avg Loss':>10}  {'LR':>10}  {'Elapsed':>9}")
    print("  " + "-" * 50)

    for epoch in range(1, epochs + 1):
        # Shuffle every epoch for better gradient diversity
        import random as _random
        _random.shuffle(all_examples)

        # Resample the negative pool each epoch — prevents the model from
        # memorising which specific negatives are easy across epochs
        neg_pool_asins = random.sample(_all_cache_asins, NEG_POOL_SIZE)
        neg_pool_vecs  = np.array([emb_cache[a] for a in neg_pool_asins],
                                  dtype=np.float32)

        total_loss = 0.0
        n_batches_done = 0
        t_epoch = time.time()

        for b_start in range(0, len(all_examples), BATCH_SIZE):
            batch = all_examples[b_start : b_start + BATCH_SIZE]
            seqs     = [e[0] for e in batch]
            targets  = [e[1] for e in batch]
            cat_ids  = [e[2] for e in batch]

            loss = agent.train_step_batch(seqs, targets, cat_ids, neg_pool_vecs)
            if loss is not None:
                total_loss     += loss
                n_batches_done += 1

            if n_batches_done > 0 and n_batches_done % LOG_EVERY == 0:
                avg_so_far = total_loss / n_batches_done
                done_pct   = b_start / len(all_examples) * 100
                cur_lr     = agent.optimizer.param_groups[0]["lr"]
                print(f"  {epoch:>5}  {n_batches_done:>5,}/{n_batches:,} "
                      f"({done_pct:4.0f}%)  {avg_so_far:>10.4f}  "
                      f"{cur_lr:>10.2e}  {time.time()-t_epoch:>7.0f}s")

        avg     = total_loss / max(n_batches_done, 1)
        cur_lr  = agent.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch:>2} done  avg_loss={avg:.4f}  lr={cur_lr:.2e}  "
              f"epoch_time={time.time()-t_epoch:.0f}s  total={elapsed:.0f}s")

        # Save a checkpoint every 5 epochs so training can be safely interrupted.
        # The final save below overwrites this with the completed model.
        if epoch % 5 == 0:
            ckpt_path = PRETRAINED_PATH.replace(".pt", f"_epoch{epoch}.pt")
            agent.save(ckpt_path)
            print(f"  [checkpoint] saved -> {ckpt_path}")

    agent.save(PRETRAINED_PATH)
    total_time = time.time() - t_start
    print(f"\nStage 2 complete in {total_time:.0f}s -> {PRETRAINED_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\nDIF-SASRec Setup")
    print(f"  Data dir:     {DATA_DIR}")
    print(f"  Eval users:   {EVAL_PATH}")
    print(f"  Vocab output: {CAT_VOCAB_PATH}")
    print(f"  Model output: {PRETRAINED_PATH}")
    print(f"  Epochs:       {args.epochs}")

    t_total = time.time()

    stage0_build_eval_users(skip=args.skip_eval, max_users=args.max_users)
    cat_encoder = stage1_build_vocab(skip=args.skip_vocab)

    if not args.skip_train:
        stage2_pretrain(cat_encoder, epochs=args.epochs, min_clicks=args.min_clicks,
                        batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print(f"Setup complete in {time.time()-t_total:.0f}s")
    print("=" * 60)
    print("\nStart the API:")
    print("  uvicorn app.main:app --host 127.0.0.1 --port 8000\n")


if __name__ == "__main__":
    main()
