"""
scripts/audit_clip_quality.py
==============================
Task 1.4 — CLIP model quality audit.

1. Loads 5 test cover images (real samples from sample_covers/ OR generated placeholders)
2. Encodes each with the current CLIP model (openai/clip-vit-base-patch32)
3. Searches the FAISS CLIP index for top-5 visual nearest neighbors
4. Looks up titles and main_category from item_metadata.parquet
5. Prints results + mean similarity scores
6. Saves output to profiling/clip_audit_<date>.txt

Usage:
    python scripts/audit_clip_quality.py
    python scripts/audit_clip_quality.py --top-k 5
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from config import CLIP_DIM


def load_clip(device):
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    print(f"[clip-audit] Loading {model_name} on {device}…")
    model     = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    # Task 1.4: confirm model configuration
    print(
        f"[clip-audit] Model: {model_name} | embedding dim: {CLIP_DIM} | device: {device}"
    )
    return model, processor, model_name


def encode_image(image, model, processor, device) -> np.ndarray:
    from PIL import Image
    if isinstance(image, (str, Path)):
        image = Image.open(str(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        if not isinstance(feat, torch.Tensor):
            feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat[1]
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()


def make_placeholder_image(color: tuple, size: int = 224):
    """Generate a solid-color PIL image as a test placeholder."""
    from PIL import Image
    return Image.new("RGB", (size, size), color=color)


def get_test_images():
    """
    Return a list of (label, image_or_path) for testing.
    Prefers real JPGs from sample_covers/ if available.
    Falls back to generated solid-color placeholders.
    """
    covers_dir = ROOT / "sample_covers"
    test_covers_dir = ROOT / "evaluation" / "test_covers"
    test_covers_dir.mkdir(parents=True, exist_ok=True)

    real_images = []
    for d in [covers_dir, test_covers_dir]:
        if d.exists():
            for jpg in sorted(d.glob("*.jpg"))[:5]:
                real_images.append((jpg.stem, jpg))

    if len(real_images) >= 3:
        print(f"[clip-audit] Using {len(real_images)} real book cover images from disk.")
        return real_images[:5]

    # Fall back to solid-color placeholders representing different aesthetic moods
    print("[clip-audit] Using solid-color placeholder images (no real covers found).")
    placeholder_specs = [
        ("Dark/noir cover (dark gray)",   (40, 40, 40)),
        ("Bright children's book (yellow)", (255, 220, 50)),
        ("Science fiction (deep blue)",   (20, 20, 120)),
        ("Romance (warm pink)",           (240, 130, 160)),
        ("History / textbook (beige)",    (210, 195, 170)),
    ]
    images = []
    for label, color in placeholder_specs:
        img = make_placeholder_image(color)
        # Save the placeholder so it can be inspected
        out_path = test_covers_dir / f"{label.split('(')[0].strip().replace('/', '_').replace(' ', '_')}.jpg"
        img.save(str(out_path))
        images.append((label, img))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load resources
    from retriever import Retriever
    DATA_DIR   = ROOT / "src" / "data"
    cleora_data  = np.load(str(DATA_DIR / "cleora_embeddings.npz"))
    retriever    = Retriever(str(DATA_DIR), cleora_data)

    meta_path    = DATA_DIR / "item_metadata.parquet"
    metadata_df  = pd.read_parquet(str(meta_path))
    metadata_df.set_index("parent_asin", inplace=True)
    print(f"[clip-audit] Metadata rows: {len(metadata_df):,}")

    clip_model, clip_processor, model_name = load_clip(device)

    test_images = get_test_images()
    date_str    = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir     = ROOT / "profiling"
    out_dir.mkdir(exist_ok=True)
    out_file    = Path(args.output) if args.output else out_dir / f"clip_audit_{date_str}.txt"

    report_lines = [
        f"CLIP Quality Audit",
        f"Date        : {datetime.now().isoformat()}",
        f"Model       : {model_name}",
        f"Embedding dim: {CLIP_DIM}",
        f"Device      : {device}",
        f"Top-K       : {args.top_k}",
        "=" * 64,
    ]

    all_mean_scores = []

    for label, image_or_path in test_images:
        print(f"\n[clip-audit] Processing: {label}")
        vec = encode_image(image_or_path, clip_model, clip_processor, device)

        # FAISS CLIP search
        D, I = retriever.clip_index.search(vec.reshape(1, -1), args.top_k)
        results = [
            (retriever.asins[idx], float(D[0][r]))
            for r, idx in enumerate(I[0]) if idx != -1
        ]

        scores = [s for _, s in results]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        all_mean_scores.append(mean_score)

        block = [
            f"\n  Test image: {label}",
            f"  Mean similarity score (top-{args.top_k}): {mean_score:.4f}",
            f"  {'ASIN':<15} {'Score':>8}  {'Title':<50}  {'Category'}",
            f"  {'-'*100}",
        ]
        for asin, score in results:
            if asin in metadata_df.index:
                row      = metadata_df.loc[asin]
                title    = str(row.get("title", "") or "")[:48]
                category = str(row.get("main_category", "") or "")[:20]
            else:
                title    = "(no metadata)"
                category = ""
            block.append(f"  {asin:<15} {score:>8.4f}  {title:<50}  {category}")

        block_str = "\n".join(block)
        print(block_str)
        report_lines.append(block_str)

    # Overall assessment
    overall_mean = sum(all_mean_scores) / len(all_mean_scores) if all_mean_scores else 0.0
    verdict = "GOOD (NO-GO on upgrade)" if overall_mean >= 0.20 else "POOR (consider upgrade)"
    summary = [
        "\n" + "=" * 64,
        f"OVERALL MEAN SIMILARITY: {overall_mean:.4f}",
        f"VERDICT: {verdict}",
        "  Heuristic: mean >= 0.20 AND visually coherent top-5 → NO-GO on upgrade.",
        "  If mean < 0.15 or results look random → GO (schedule re-embed pass).",
    ]
    summary_str = "\n".join(summary)
    print(summary_str)
    report_lines.append(summary_str)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[clip-audit] Saved to: {out_file}")


if __name__ == "__main__":
    main()
