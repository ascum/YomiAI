# Project Update — March 25, 2026

This document captures all functional changes made to the NBA Multimodal Recommendation System between **March 19** and **March 25, 2026**. It should be read alongside `Project_update_19-3.md` and `item_metadata_update_19-3.md`.

---

## Summary of Changes

| Area | Change |
|---|---|
| Search — `active_search_engine.py` | Added smart short/long text routing with keyword pre-scan |
| Search — `active_search_engine.py` | Removed RRF when only one retrieval modality is active |
| Search — `active_search_engine.py` | Applied metadata filter to active search results |
| Backend — `api.py` | Updated reward signal for `"cart"` action to `+5.0` |

---

## 🔍 1. Smarter Text Search: Short vs. Long Query Routing

### The Problem (Pre-March 22)
The previous pipeline **always** ran both BLaIR semantic search and directly returned dense vector matches. This caused a frustrating bug: searching for a proper noun like `"jojo's bizarre adventure"` would **not** surface the actual title at the top spot, because BLaIR encodes the *semantics* of words — and "Jojo" has no inherent semantic meaning. The semantic space naturally gravitated toward unrelated books with similar-sounding conceptual themes.

### The Fix: `_keyword_scan()` in `active_search_engine.py`

A new **keyword pre-scan layer (Step 0)** was introduced with the following architecture:

```
Query ("jojo's bizarre adventure")
│
├─ Step 0: _keyword_scan()  ← NEW
│     • Only runs when query has ≤ 4 words   (short / proper-noun style)
│     • Performs case-insensitive substring match over title + author columns
│     • Returns up to 30 (ASIN, 1/rank_score) pairs where the title
│       or author_name contains the query string
│     • Only keeps ASINs that exist in the FAISS index
│
├─ Step 1: BLaIR semantic search (Top-50 dense text matches)
│
├─ Step 2: CLIP visual search   (Top-50 visual matches)
│
└─ Step 3: Fusion + BGE Reranker
```

**Long queries (> 4 words)** such as `"gritty detective novels set in noir cities"` skip the keyword pre-scan entirely. A long query is a conceptual request — dense retrieval handles it far better than substring matching.

### Injection Mechanic
Keyword hits are **injected at the front** of the final ranking list, *before* the dense results. This guarantees exact-match items always surface first while the semantic results still fill the remaining slots:

```python
# active_search_engine.py — Step 3 / Injection
kw_ranked   = [(asin, {score=1/(i+1), ...}) for i, asin in keyword_hits]
remaining   = [(a, d) for a, d in dense_ranking if a not in keyword_asin_set]
final_ranking = kw_ranked + remaining  ← exact hits always lead
```

### Observable Result
- Typing `"jojo's bizarre adventure"` → the Jojo volume appears **first** instead of unrelated books.
- Typing `"dark fantasy magic systems"` → unchanged, still pure dense retrieval.

---

## ⚡ 2. RRF Bypass for Single-Modality Queries

### The Problem
**Reciprocal Rank Fusion (RRF)** has a real computational and conceptual cost when it serves no purpose. Prior to this change, the fusion function was called even when only one retrieval channel was active (e.g., a text-only query with no image). Running RRF over a single list adds no fusion benefit — it was dead code on every single-modal request.

### The Fix

The `search()` function in `active_search_engine.py` now contains a **conditional branch** in Step 3:

```python
# Step 3: Fusion — only RRF when both modalities are active
if blair_results and clip_results:
    # ✅ Both text + image: real multimodal fusion — RRF makes sense
    dense_ranking = self.rrf_fusion([('blair', blair_results), ('clip', clip_results)])

elif blair_results:
    # ✅ Text only: sort BLaIR by raw similarity score — no fusion needed
    dense_ranking = [(asin, {'score': score, 'text_sim': score, 'img_sim': 0.0})
                     for asin, score in sorted(blair_results, key=lambda x: x[1], reverse=True)]

elif clip_results:
    # ✅ Image only: sort CLIP by raw similarity score — no fusion needed
    dense_ranking = [(asin, {'score': score, 'text_sim': 0.0, 'img_sim': score})
                     for asin, score in sorted(clip_results, key=lambda x: x[1], reverse=True)]

elif keyword_hits:
    # ✅ Keyword-only fallback (encoder unavailable)
    dense_ranking = [(asin, {...}) for asin, score in keyword_hits]
```

**RRF is now only invoked when a true text+image hybrid query is submitted.** All other paths bypass it entirely for a cleaner, faster pipeline.

> **Note:** RRF is still present and unchanged in `passive_recommendation_engine.py` for the 3-layer recommendation funnel (BLaIR + CLIP + RL-DQN fusion), where it genuinely fuses three separate signals.

---

## 🛡️ 3. Metadata Filter Applied to Active Search Results

### The Problem
The FAISS BLaIR and CLIP indices contain **~3M items** but the `item_metadata.parquet` database only covers **~1.73M items**. This gap meant some returned ASINs had no title, author, or cover art — they would surface as generic stub entries ("Book B001XXX").

This filter was already applied in the `/recommend` endpoint (passive recommendations) but was **missing** from the active search path.

### The Fix: Explicit ASIN Filter in `active_search_engine.py`

A metadata guard is applied after the dense fusion step, before keyword injection:

```python
# ── Metadata filter ─────────────────────────────────────────────────────────
if self.metadata_df is not None:
    dense_ranking = [
        (asin, data) for asin, data in dense_ranking
        if asin in self.metadata_df.index   ← drops ghost items with no metadata
    ]

# ── Inject keyword hits (also filtered) at the front ───────────────────────
kw_ranked = [
    (asin, {...}) for i, (asin, _) in enumerate(keyword_hits)
    if self.metadata_df is None or asin in self.metadata_df.index
]
```

Ghost items are now silently dropped from both keyword and dense results before they can reach the API response.

---

## 💰 4. Updated Reward Signal for "Add to Cart"

### Change
The `/interact` endpoint in `api.py` was updated to define a dedicated `"cart"` action with a higher reward than a plain click:

| Action | Previous Reward | New Reward |
|--------|----------------|------------|
| `"click"` | `+1.0` | `+1.0` (unchanged) |
| `"cart"` | `+1.0` (same as click) | **`+5.0`** |
| `"skip"` | `0.0` | `0.0` (unchanged) |

```python
# api.py — /interact endpoint
if req.action == "cart":
    reward = 5.0                           # ← NEW: strongest positive signal
    profile_manager.log_click(req.user_id, req.item_id, source="web_ui")
elif req.action == "click":
    reward = 1.0
    profile_manager.log_click(req.user_id, req.item_id, source="web_ui")
else:  # skip
    reward = 0.0
    ...
```

### Rationale
"Add to Cart" is a strong **purchase intent** signal. By scaling the reward to `+5.0`, a single cart action has the same gradient weight as five independent clicks. This biases the DQN to learn that items similar to what the user actually wants to buy should be ranked higher in future recommendations — not just items they casually browse.

---

## 📁 5. Files Changed in This Update Cycle

| File | Type | Key Change |
|---|---|---|
| `src/active_search_engine.py` | MODIFIED | `_keyword_scan()` method added; RRF bypass; metadata filter |
| `api.py` | MODIFIED | Cart action reward changed to `+5.0`; dedicated `"cart"` branch in `/interact` |

No schema changes to `item_metadata.parquet` were made in this cycle. The BGE Reranker, Cleora pipeline, and all model files remain unchanged from March 19.

---

## ✅ 6. Verification Notes

- **"jojo's bizarre adventure" test**: Confirmed the correct Jojo volume appears as the top result. Previously this query returned semantically unrelated books.
- **Single-modality path**: Verified that a text-only query no longer calls `rrf_fusion()` — the code path goes directly to the `elif blair_results:` branch.
- **Cart reward**: Verified the `/interact` endpoint returns `{"reward": 5.0}` for `"cart"` actions.
- **Ghost item filter**: Verified that all ASINs in the search response are present in `metadata_df.index`.

---

*Last update: March 25, 2026. Next update cycle will cover any further changes to the RL training loop or frontend UI.*
