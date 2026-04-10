"""
scripts/audit/vietnamese_search.py
===================================
Diagnose text encoder quality on Vietnamese vs. English queries.

For each VI/EN query pair:
  1. Encode both with the text encoder (BGE-M3, via app.core.models)
  2. Search the FAISS text index, retrieve top-10
  3. Look up titles and main_category from item_metadata.parquet
  4. Report: top-10 titles, mean score, genre hit rate

Also runs a second pass with VI→EN translation applied to measure the fix.

Usage:
    python scripts/audit/vietnamese_search.py
    python scripts/audit/vietnamese_search.py --top-k 10 --no-fix
"""

import argparse
import os
import sys

# Force UTF-8 encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.core import models as model_loader


def load_resources():
    """Load text encoder, FAISS index and metadata parquet."""
    DATA_DIR = Path(settings.DATA_DIR)
    cleora_path = DATA_DIR / "cleora_embeddings.npz"

    print("[audit] Loading FAISS indices…")
    cleora_data = np.load(str(cleora_path))
    retriever   = Retriever(str(DATA_DIR), cleora_data)

    print("[audit] Loading item metadata…")
    meta_path   = DATA_DIR / "item_metadata.parquet"
    metadata_df = pd.read_parquet(str(meta_path))
    metadata_df.set_index("parent_asin", inplace=True)
    print(f"[audit] Metadata loaded: {len(metadata_df):,} items")

    print(f"[audit] Loading text encoder ({settings.TEXT_ENCODER_MODEL})…")
    import torch
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model  = model_loader.load_text_encoder(device)
    print(f"[audit] Text encoder ready on {device}")

    return retriever, metadata_df, text_model


def encode(text_model, text: str) -> np.ndarray:
    return text_model.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")


def search_top_k(retriever, vec: np.ndarray, top_k: int):
    D, I = retriever.text_index.search(vec.reshape(1, -1), top_k)
    return [
        (retriever.asins[i], float(D[0][idx]))
        for idx, i in enumerate(I[0]) if i != -1
    ]


def genre_hit_rate(results, metadata_df, expected_genres):
    """Fraction of top-K results whose title/category contains any expected genre keyword."""
    hits = 0
    for asin, _ in results:
        if asin not in metadata_df.index:
            continue
        row    = metadata_df.loc[asin]
        text   = " ".join([
            str(row.get("title", "") or ""),
            str(row.get("main_category", "") or ""),
        ]).lower()
        if any(g.lower() in text for g in expected_genres):
            hits += 1
    return hits, len(results)


def audit_pair(text_model, retriever, metadata_df, vi_q, en_q, expected_genres, top_k: int, apply_fix: bool):
    """Run a single VI vs EN comparison, optionally with translation fix."""
    # Raw VI (no translation)
    vi_vec    = encode(text_model, vi_q)
    vi_results = search_top_k(retriever, vi_vec, top_k)
    vi_scores  = [s for _, s in vi_results]
    vi_mean    = sum(vi_scores) / len(vi_scores) if vi_scores else 0.0
    vi_hits, vi_total = genre_hit_rate(vi_results, metadata_df, expected_genres)

    # English equivalent (no translation)
    en_vec    = encode(text_model, en_q)
    en_results = search_top_k(retriever, en_vec, top_k)
    en_scores  = [s for _, s in en_results]
    en_mean    = sum(en_scores) / len(en_scores) if en_scores else 0.0
    en_hits, en_total = genre_hit_rate(en_results, metadata_df, expected_genres)

    # Translation-fixed VI
    fix_mean  = None
    fix_hits  = None
    fix_total = None
    if apply_fix:
        try:
            from app.infrastructure.translation import translate_vi_to_en
            translated = translate_vi_to_en(vi_q)
            print(f"    [fix] '{vi_q}' → '{translated}'")
            fix_vec     = encode(text_model, translated)
            fix_results = search_top_k(retriever, fix_vec, top_k)
            fix_scores  = [s for _, s in fix_results]
            fix_mean    = sum(fix_scores) / len(fix_scores) if fix_scores else 0.0
            fix_hits, fix_total = genre_hit_rate(fix_results, metadata_df, expected_genres)
        except Exception as e:
            print(f"    [fix] Translation unavailable: {e}")

    # Sample top-5 VI titles
    vi_titles = []
    for asin, score in vi_results[:5]:
        title = str(metadata_df.loc[asin, "title"]) if asin in metadata_df.index else asin
        vi_titles.append(f"      • {title[:60]}  ({score:.4f})")

    return {
        "vi_q": vi_q, "en_q": en_q,
        "vi_mean": vi_mean, "en_mean": en_mean,
        "vi_hits": vi_hits, "vi_total": vi_total,
        "en_hits": en_hits, "en_total": en_total,
        "fix_mean": fix_mean, "fix_hits": fix_hits, "fix_total": fix_total,
        "vi_titles": vi_titles,
        "delta": en_mean - vi_mean,
    }


def format_result(r) -> str:
    lines = [
        f"\n  Query pair: '{r['vi_q']}'  vs  '{r['en_q']}'",
        f"  {'':4} VI (raw) mean score : {r['vi_mean']:.4f}",
        f"  {'':4} EN (raw) mean score : {r['en_mean']:.4f}",
        f"  {'':4} Score delta (EN-VI) : {r['delta']:+.4f}",
        f"  {'':4} VI genre hit rate   : {r['vi_hits']}/{r['vi_total']}",
        f"  {'':4} EN genre hit rate   : {r['en_hits']}/{r['en_total']}",
    ]
    if r["fix_mean"] is not None:
        lines += [
            f"  {'':4} VI+fix mean score   : {r['fix_mean']:.4f}  (Δ from raw VI: {r['fix_mean']-r['vi_mean']:+.4f})",
            f"  {'':4} VI+fix genre hits   : {r['fix_hits']}/{r['fix_total']}",
        ]
    lines.append("  Top-5 VI raw results:")
    lines += r["vi_titles"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k",   type=int, default=10)
    parser.add_argument("--no-fix",  action="store_true",
                        help="Skip the translation-fix pass")
    parser.add_argument("--output",  default="")
    args = parser.parse_args()

    retriever, metadata_df, text_model = load_resources()

    from evaluation.vi_test_queries import VI_TEST_QUERIES, EN_EQUIVALENT_QUERIES

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir  = ROOT / "profiling"
    out_dir.mkdir(exist_ok=True)
    suffix   = "" if not args.no_fix else "_no_fix"
    out_file = Path(args.output) if args.output else out_dir / f"vi_search_audit{suffix}_{date_str}.txt"

    apply_fix = not args.no_fix
    report    = [
        "Vietnamese Search Audit",
        f"Date      : {datetime.now().isoformat()}",
        f"Encoder   : {settings.TEXT_ENCODER_MODEL}",
        f"Top-K     : {args.top_k}",
        f"Fix pass  : {'enabled' if apply_fix else 'disabled'}",
        "=" * 64,
    ]

    vi_degraded = 0
    for vi_entry, en_q in zip(VI_TEST_QUERIES, EN_EQUIVALENT_QUERIES):
        print(f"\n[audit] Running: {vi_entry['query']}")
        r = audit_pair(
            text_model, retriever, metadata_df,
            vi_entry["query"], en_q, vi_entry["expected_genres"],
            args.top_k, apply_fix,
        )
        text = format_result(r)
        print(text)
        report.append(text)

        if r["vi_mean"] < 0.30 or r["vi_hits"] < 3:
            vi_degraded += 1

    conclusion = [
        "\n" + "=" * 64,
        "CONCLUSION",
        f"  Queries with clear VI degradation (score<0.30 OR hits<3/10): {vi_degraded}/{len(VI_TEST_QUERIES)}",
    ]
    if vi_degraded >= 3:
        conclusion.append(
            "  VERDICT: Encoder degrades significantly on Vietnamese — apply translation fix."
        )
    elif vi_degraded >= 1:
        conclusion.append(
            "  VERDICT: Mild degradation — apply language-adaptive threshold or translation fix."
        )
    else:
        conclusion.append(
            "  VERDICT: Vietnamese performance acceptable — no fix required."
        )
    conc_str = "\n".join(conclusion)
    print(conc_str)
    report.append(conc_str)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n[audit] Saved to: {out_file}")


if __name__ == "__main__":
    main()
