"""
test_search_fusion.py — Unit tests for the BM25 confidence-weighted adaptive search engine.

Run from the project root:
    python src/test_search_fusion.py

Does NOT require FAISS indices, GPU, or any ML models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import pandas as pd

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ─── Stub retriever ───────────────────────────────────────────────────────────
class _StubRetriever:
    """Minimal retriever that returns random vectors and knows a fixed set of ASINs."""
    def __init__(self, asins):
        self.asins       = asins
        self.asin_to_idx = {a: i for i, a in enumerate(asins)}

    def get_asin_vec(self, asin):
        idx = self.asin_to_idx.get(asin)
        if idx is None:
            return None
        rng = np.random.default_rng(idx)
        return rng.standard_normal(1024).astype(np.float32), rng.standard_normal(512).astype(np.float32)


# ─── Stub profile manager ─────────────────────────────────────────────────────
class _StubProfileManager:
    def log_search(self, *args, **kwargs):
        pass


# ─── Synthetic metadata ───────────────────────────────────────────────────────
FAKE_ASINS = ["A001", "A002", "A003", "A004", "A005"]
FAKE_META  = pd.DataFrame({
    "parent_asin": FAKE_ASINS,
    "title": [
        "JoJo's Bizarre Adventure: Phantom Blood",
        "A Court of Thorns and Roses",
        "The Name of the Wind",
        "Dune",
        "Brandon Sanderson's Mistborn",
    ],
    "author_name": [
        "Hirohiko Araki",
        "Sarah J. Maas",
        "Patrick Rothfuss",
        "Frank Herbert",
        "Brandon Sanderson",
    ],
    "main_category": ["Books"] * 5,
    "image_url":     [None] * 5,
    "description":   [""] * 5,
}).set_index("parent_asin")

retriever    = _StubRetriever(FAKE_ASINS)
profile_mgr  = _StubProfileManager()


# ─── Import engine ────────────────────────────────────────────────────────────
section("Setup: import ActiveSearchEngine")
try:
    from active_search_engine import ActiveSearchEngine
    print(f"{PASS}  ActiveSearchEngine imported OK")
except ImportError as e:
    print(f"{FAIL}  Import failed: {e}")
    sys.exit(1)

engine = ActiveSearchEngine(retriever, profile_mgr, reranker=None, metadata_df=FAKE_META)


# ─── 1. BM25 index was built ──────────────────────────────────────────────────
section("1. BM25 index construction")
assert engine.bm25_index is not None, f"{FAIL}  BM25 index should have been built"
assert len(engine.bm25_asins) == len(FAKE_ASINS), (
    f"{FAIL}  Expected {len(FAKE_ASINS)} indexed docs, got {len(engine.bm25_asins)}")
print(f"{PASS}  BM25 index built with {len(engine.bm25_asins)} documents")


# ─── 2. Tokenizer correctness ─────────────────────────────────────────────────
section("2. _tokenize()")
cases = [
    ("Jojo's Bizarre Adventure",         ["jojos", "bizarre", "adventure"]),
    ("A Court of Thorns & Roses",         ["a", "court", "of", "thorns", "roses"]),
    ("Spider-Man: No Way Home",           ["spiderman", "no", "way", "home"]),
    ("",                                  []),
    ("   ",                               []),
    ("Dune",                              ["dune"]),
]
for text, expected in cases:
    result = engine._tokenize(text)
    assert result == expected, f"{FAIL}  _tokenize({text!r}) → {result}, expected {expected}"
    print(f"{PASS}  _tokenize({text!r})")


# ─── 3. BM25 search — exact title hit (short query) ──────────────────────────
section("3. _bm25_search() — exact title query")
hits, conf = engine._bm25_search("jojo's bizarre adventure")
assert len(hits) > 0, f"{FAIL}  No hits for exact title query"
top_asin = hits[0][0]
assert top_asin == "A001", f"{FAIL}  Expected A001 (JoJo) as top hit, got {top_asin}"
assert conf >= 0.5, f"{FAIL}  Confidence should be high for exact match, got {conf:.3f}"
print(f"{PASS}  'jojo's bizarre adventure' → top={top_asin}, conf={conf:.3f}")

# Long title (>4 words) — old heuristic would have blocked this
hits2, conf2 = engine._bm25_search("a court of thorns and roses")
assert len(hits2) > 0, f"{FAIL}  No hits for long exact title"
assert hits2[0][0] == "A002", f"{FAIL}  Expected A002 (ACOTAR), got {hits2[0][0]}"
print(f"{PASS}  'a court of thorns and roses' → top={hits2[0][0]}, conf={conf2:.3f}  (old heuristic would have missed this)")


# ─── 4. BM25 search — author search ──────────────────────────────────────────
section("4. _bm25_search() — author name query")
hits3, conf3 = engine._bm25_search("brandon sanderson")
assert len(hits3) > 0, f"{FAIL}  No hits for author query"
assert hits3[0][0] == "A005", f"{FAIL}  Expected A005 (Sanderson), got {hits3[0][0]}"
print(f"{PASS}  'brandon sanderson' → top={hits3[0][0]}, conf={conf3:.3f}")


# ─── 5. BM25 search — conceptual query returns low confidence ────────────────
section("5. _bm25_search() — conceptual / semantic query")
hits4, conf4 = engine._bm25_search("dark gritty detective noir magic")
# Concept query — BM25 should return nothing meaningful OR very low confidence
print(f"      hits={len(hits4)}, conf={conf4:.3f}")
if len(hits4) > 0:
    assert conf4 < 0.5, (
        f"{FAIL}  Concept query confidence should be low, got {conf4:.3f}")
    print(f"{PASS}  Concept query → low confidence ({conf4:.3f}) — BLaIR will dominate")
else:
    print(f"{PASS}  Concept query → no BM25 hits — BLaIR will dominate")


# ─── 6. _adaptive_rrf weight scaling ─────────────────────────────────────────
section("6. _adaptive_rrf() — channel weight scaling")
from config import RRF_K

# Two channels with same items but different weights
# Channel A: weight 2.0,  Channel B: weight 0.5
# A001 is rank-1 in both, A002 rank-1 only in B
chan_a = [("blair", [("A001", 0.9), ("A003", 0.7)], 2.0)]
chan_b = [("clip",  [("A002", 0.8), ("A001", 0.6)], 0.5)]
result = engine._adaptive_rrf(chan_a + chan_b)

result_asins = [a for a, _ in result]
# A001 should be first (weight-2.0 channel scores it rank-1)
assert result_asins[0] == "A001", f"{FAIL}  Expected A001 first, got {result_asins[0]}"
print(f"{PASS}  Heavy-weight channel (w=2.0) correctly dominates — A001 is rank-1")

# A002 should appear (it was rank-1 in the 0.5 channel)
assert "A002" in result_asins, f"{FAIL}  A002 missing from result"
print(f"{PASS}  Light-weight channel (w=0.5) still contributes — A002 in results")


# ─── 7. Graceful fallback when BM25 index is None ────────────────────────────
section("7. Graceful fallback — BM25 disabled")
engine_no_bm25 = ActiveSearchEngine(
    retriever, profile_mgr, reranker=None, metadata_df=None)
hits5, conf5 = engine_no_bm25._bm25_search("jojo")
assert hits5 == [], f"{FAIL}  Should return empty hits when metadata_df is None"
assert conf5 == 0.0, f"{FAIL}  Should return 0.0 confidence when metadata_df is None"
print(f"{PASS}  BM25 disabled → _bm25_search returns ([], 0.0) without crashing")


# ─── 8. data_dict keys are present in adaptive_rrf output ───────────────────
section("8. data_dict schema in _adaptive_rrf output")
chan = [("blair", [("A001", 0.9), ("A002", 0.7)], 1.0)]
r = engine._adaptive_rrf(chan)
assert len(r) == 2
asin0, d0 = r[0]
for key in ("score", "text_sim", "img_sim", "bm25_score"):
    assert key in d0, f"{FAIL}  Missing key '{key}' in data_dict"
print(f"{PASS}  data_dict contains keys: score, text_sim, img_sim, bm25_score")


# ─── Summary ──────────────────────────────────────────────────────────────────
section("All search fusion tests passed 🎉")
