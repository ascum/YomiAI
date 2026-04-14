# DIF-SASRec Implementation Plan (Revised — April 11, 2026)

> **For Claude Code Agent**: Execute these steps IN ORDER. Each step has exact file paths and the current code state. Do not skip steps.

---

## What Changed Since the Original Plan

The project underwent a **full architectural refactor** between the original plan and now. Here is a complete mapping of the breaking changes:

### 1. Directory Structure (CRITICAL)

| Original Plan Reference | Actual Current Path |
|:---|:---|
| `src/config.py` | `app/config.py` |
| `src/retriever.py` | `app/repository/faiss_repo.py` |
| `src/user_profile_manager.py` | `app/repository/profile_repo.py` |
| `src/passive_recommendation_engine.py` | `app/services/passive_recommend.py` |
| `src/rl_collaborative_filter.py` | `app/services/rl_filter.py` |
| `src/sequential_dqn.py` | `app/services/sequential_dqn.py` |
| `api.py` (monolith) | Split into `app/core/lifespan.py`, `app/api/routes/recommend.py`, `app/api/routes/interact.py` |
| `scripts/evaluate_recommendation.py` | Does not exist — create as `scripts/benchmark/evaluate_recommendation.py` |

### 2. Variable Naming (CRITICAL)

All `blair_*` names were renamed to `text_*` (Translation_Pipeline_Update_10-4-2026):

| Old name (plan uses this) | Current name |
|:---|:---|
| `self.blair_index` | **`self.text_index`** (HNSW) + **`self.text_flat`** (flat/reconstruct) |
| `BLAIR_DIM = 1024` | **`settings.TEXT_EMBED_DIM`** (= 1024) |
| `blair_index_bge_hnsw.faiss` | **`bge_index_hnsw.faiss`** (`settings.TEXT_INDEX_HNSW`) |
| `blair_index.faiss` | **`bge_index_flat.faiss`** (`settings.TEXT_INDEX_FLAT`) |
| `retriever.blair_index.reconstruct(idx)` | **`retriever.text_flat.reconstruct(idx)`** |

### 3. Config is a Frozen Dataclass

`app/config.py` is NOT a flat file of constants. It is a **`@dataclass(frozen=True) class Settings`**. New hyperparameters must be added as fields **inside** the dataclass, then accessed via `settings.FIELD_NAME`.

### 4. UserProfileManager is Fully Async + MongoDB-backed

`get_profile()`, `get_click_sequence()`, `log_click()`, `save_profile()` are all **`async`**. Any new method that calls these must also be `async`. No local JSON files — profiles live in MongoDB via `app.infrastructure.database.db`.

### 5. HNSW Auto-Detection Already Done

The original plan's Step 4a (add HNSW auto-detection) is **already implemented** in `app/repository/faiss_repo.py::_load_text_index()`. The retriever loads HNSW as `self.text_index` automatically if `bge_index_hnsw.faiss` exists (it does). Step 4 is now **only** adding `get_content_candidates()`.

### 6. API Uses Dependency Injection

No more `_state` dict. Routes receive an `AppContainer` instance via `Depends(require_ready)`. New services added in lifespan must be wired into `AppContainer` and accessed via `container.X`.

### 7. FAISS Files on Disk

`data/bge_index_hnsw.faiss` and `data/bge_index_flat.faiss` both **exist**. No legacy fallback needed for new code.

---

## Codebase State (READ FIRST)

```
Project root:   /DATN (1)/
App code:       app/
  config.py                    — Settings dataclass (single source of truth)
  core/
    container.py               — AppContainer dependency injection holder
    lifespan.py                — FastAPI startup (replaces api.py startup block)
  repository/
    faiss_repo.py              — Retriever: text_index(HNSW) + text_flat(flat)
    profile_repo.py            — UserProfileManager (async, MongoDB-backed)
  services/
    passive_recommend.py       — PassiveRecommendationEngine (dual-tab funnel)
    rl_filter.py               — RLSequentialFilter (GRU-SeqDQN — KEEP, do not delete)
    sequential_dqn.py          — SequentialDQN model (KEEP, do not delete)
  api/routes/
    recommend.py               — GET /recommend, GET /rl_metrics
    interact.py                — POST /interact (RL training hook)
Data:
  data/bge_index_hnsw.faiss   — BGE-M3 HNSW (fast ANN, 3M vectors, primary search)
  data/bge_index_flat.faiss   — BGE-M3 Flat (exact, used for .reconstruct())
  data/item_metadata.parquet  — 1.7M items with 'parent_asin' and 'categories' columns
  data/cleora_embeddings.npz  — Cleora behavioral embeddings
  data/asins.csv              — ASIN list (index = faiss row)
Evaluation:
  evaluation/eval_users.json  — (same as before)
```

**Key class names / attributes in current code:**
- `Retriever.text_index` — HNSW index (search)
- `Retriever.text_flat` — Flat index (`.reconstruct(idx)` → 1024-dim BGE-M3 vec)
- `Retriever.asin_to_idx` — `{asin: faiss_row_int}` mapping
- `Retriever.asins` — list indexed by FAISS row
- `settings.TEXT_EMBED_DIM` = 1024 (BGE-M3 dim)

---

## Step 1: Create `app/services/category_encoder.py` (NEW FILE)

**Purpose**: Build a category vocabulary from `item_metadata.parquet` and provide ASIN → category_id lookups.

**Create file**: `app/services/category_encoder.py`

```python
"""
app/services/category_encoder.py — Category Vocabulary for DIF-SASRec

Parses the 'categories' field from item_metadata.parquet into a vocabulary
mapping. Uses LEAF categories (last pipe-separated segment) for sharpest
genre signal.

Category format in parquet: "Books|Literature & Fiction|Action & Adventure"
Leaf category extracted:    "Action & Adventure"
"""
import json
import os
import pandas as pd


class CategoryEncoder:
    """
    Manages category vocabulary for the DIF-SASRec personal recommendation model.

    Special tokens:
        PAD_ID = 0  (padding for sequence alignment)
        UNK_ID = 1  (unknown / missing category)
    """

    PAD_ID = 0
    UNK_ID = 1

    def __init__(self):
        self.vocab = {}           # {category_string: int_id}
        self.id_to_cat = {}       # {int_id: category_string}
        self.asin_to_cat_id = {}  # {asin: int_id}
        self.num_categories = 2   # starts with PAD + UNK

    # ─── Build from metadata ──────────────────────────────────────────────────

    def build_from_parquet(self, metadata_path: str):
        """
        Build category vocabulary from item_metadata.parquet.

        Reads 'parent_asin' and 'categories' columns.
        Extracts the leaf (last segment) of each pipe-separated category string.
        Assigns integer IDs starting from 2 (0=PAD, 1=UNK).
        """
        print(f"[CategoryEncoder] Loading metadata from {metadata_path} ...")
        df = pd.read_parquet(metadata_path, columns=["parent_asin", "categories"])

        cat_set = set()
        asin_cats = {}

        for _, row in df.iterrows():
            asin = str(row["parent_asin"])
            raw = str(row.get("categories", ""))
            leaf = self._parse_leaf_category(raw)
            if leaf:
                cat_set.add(leaf)
                asin_cats[asin] = leaf

        sorted_cats = sorted(cat_set)
        self.vocab = {cat: idx + 2 for idx, cat in enumerate(sorted_cats)}
        self.id_to_cat = {0: "PAD", 1: "UNK"}
        self.id_to_cat.update({idx + 2: cat for idx, cat in enumerate(sorted_cats)})
        self.num_categories = len(self.vocab) + 2

        self.asin_to_cat_id = {}
        for asin, cat in asin_cats.items():
            self.asin_to_cat_id[asin] = self.vocab.get(cat, self.UNK_ID)

        print(f"[CategoryEncoder] Vocabulary built: {self.num_categories} categories "
              f"({len(self.vocab)} unique + PAD + UNK)")
        print(f"[CategoryEncoder] ASINs with categories: {len(self.asin_to_cat_id):,}")

        from collections import Counter
        freq = Counter(asin_cats.values())
        print("[CategoryEncoder] Top 10 categories:")
        for cat, count in freq.most_common(10):
            print(f"    {cat}: {count:,}")

    # ─── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_leaf_category(raw: str) -> str:
        """
        Extract the leaf category from a pipe-separated string.

        Examples:
            "Books|Literature & Fiction|Action & Adventure" → "Action & Adventure"
            "Books|Science Fiction" → "Science Fiction"
            "Books" → "Books"
            "" or "nan" → ""
        """
        if not raw or raw == "nan" or raw.strip() == "":
            return ""
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        return parts[-1] if parts else ""

    # ─── Lookups ──────────────────────────────────────────────────────────────

    def get_category_id(self, asin: str) -> int:
        """Return the category ID for an ASIN. Returns UNK_ID if not found."""
        return self.asin_to_cat_id.get(asin, self.UNK_ID)

    def get_category_name(self, cat_id: int) -> str:
        """Reverse lookup: int_id → category string."""
        return self.id_to_cat.get(cat_id, "UNK")

    def encode_sequence(self, asin_sequence: list) -> list:
        """Convert a list of ASINs into a list of category IDs."""
        return [self.get_category_id(asin) for asin in asin_sequence]

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save vocabulary to JSON."""
        payload = {
            "vocab": self.vocab,
            "asin_to_cat_id": self.asin_to_cat_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CategoryEncoder] Saved to {path}")

    def load(self, path: str):
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.vocab = payload["vocab"]
        self.asin_to_cat_id = payload.get("asin_to_cat_id", {})
        self.id_to_cat = {0: "PAD", 1: "UNK"}
        self.id_to_cat.update({v: k for k, v in self.vocab.items()})
        self.num_categories = len(self.vocab) + 2
        print(f"[CategoryEncoder] Loaded {self.num_categories} categories from {path}")
```

---

## Step 2: Modify `app/config.py`

**Add DIF-SASRec hyperparameters inside the `Settings` dataclass** (after the `NEG_SAMPLE_SIZE` field, before the closing of the class).

The config is a **frozen dataclass** — new fields go inside the class body, not appended after it.

Add these fields to the `Settings` dataclass:

```python
    # ── DIF-SASRec (Personal Pipeline — "You Might Like") ────────────────────
    # Replaces GRU-SeqDQN for the personal "You Might Like" tab.
    # Uses decoupled category attention to model individual user taste evolution.
    # Candidates come from HNSW BGE-M3 index — zero Cleora dependency.
    SASREC_HIDDEN_DIM:     int   = 256    # model internal dimension (projects 1024→256)
    SASREC_N_BLOCKS:       int   = 2      # DIF-attention transformer layers
    SASREC_N_HEADS:        int   = 4      # attention heads (head_dim = 256/4 = 64)
    SASREC_DROPOUT:        float = 0.2    # dropout in attention and FFN
    SASREC_LR:             float = 1e-3   # Adam learning rate
    SASREC_ALPHA_INIT:     float = 0.7    # initial content vs category attention balance
    SASREC_CAT_AUX_WEIGHT: float = 0.1   # category prediction auxiliary loss weight
    SASREC_NUM_NEGATIVES:  int   = 256    # sampled softmax negatives per step
    PERSONAL_CANDIDATES:   int   = 200    # HNSW KNN retrieval count (no Cleora)
```

---

## Step 3: Create `app/services/dif_sasrec.py` (NEW FILE)

**Purpose**: The core DIF-SASRec model and its training/inference agent.

**CRITICAL naming**: Use `retriever.text_flat.reconstruct(idx)` (NOT `blair_index.reconstruct()`). Use `settings.TEXT_EMBED_DIM` (NOT `BLAIR_DIM`). Import with `from app.config import settings`.

**Create file**: `app/services/dif_sasrec.py`

### 3.1 `ContentProjector(nn.Module)`

```
Input:  [batch, seq_len, 1024]  ← BGE-M3 embeddings from text_flat.reconstruct()
Output: [batch, seq_len, 256]   ← projected to model dimension

Layers: Linear(1024, 256) → LayerNorm(256) → Dropout(0.2)
```

### 3.2 `DIFAttentionLayer(nn.Module)`

```
Content stream:  Q, K, V from content embeddings (3 × Linear(256→256))
Category stream: Q, K from category embeddings  (2 × Linear(256→256)) — NO V!

Fusion: A_fused = α·softmax(Q_c·K_c^T/√64) + (1-α)·softmax(Q_k·K_k^T/√64)
Output: A_fused · V_content  (values ALWAYS from content stream)

α: learnable scalar, initialized so sigmoid(α_logit) ≈ SASREC_ALPHA_INIT (0.7)
Multi-head: SASREC_N_HEADS=4 heads, head_dim = 64
Causal mask: lower-triangular (prevent attending to future positions)
```

### 3.3 `DIFSASRecBlock(nn.Module)`

```
Pre-norm architecture:
  content → LayerNorm → DIFAttention(content, category) → + residual
  → LayerNorm → FFN(256→512→256, GELU) → + residual
```

### 3.4 `DIFSASRecModel(nn.Module)`

```
__init__(num_categories, hidden_dim=256, n_blocks=2, max_len=50):
    self.content_proj   = ContentProjector(1024, 256)
    self.category_emb   = nn.Embedding(num_categories, 256, padding_idx=0)
    self.position_emb   = nn.Embedding(50, 256)
    self.blocks         = nn.ModuleList([DIFSASRecBlock(256) × n_blocks])
    self.final_norm     = nn.LayerNorm(256)
    self.candidate_proj = ContentProjector(1024, 256)
    self.category_head  = nn.Linear(256, num_categories)

forward(bge_seqs, cat_ids, lengths):
    Input:  bge_seqs [B, T, 1024], cat_ids [B, T], lengths [B]
    Output: hidden [B, T, 256], intent [B, 256], cat_logits [B, T, num_cats]

score_candidates(intent, candidate_bge):
    Input:  intent [B, 256], candidate_bge [N, 1024]
    Output: scores [B, N]
```

### 3.5 `DIFSASRecAgent`

**This is the interface class used by `passive_recommend.py` and routes.**

```python
__init__(self, retriever, category_encoder, pretrained_path=None):
    from app.config import settings
    # Creates DIFSASRecModel on GPU
    # Loads pretrained checkpoint if provided

_build_tensors(click_seq_asins, cat_ids=None):
    # Reconstruct BGE-M3 embeddings from FAISS using:
    #   idx = self.retriever.asin_to_idx[asin]
    #   vec = self.retriever.text_flat.reconstruct(idx)  ← NOT blair_index!
    # Returns (bge_tensor [1,T,1024], cat_tensor [1,T], lengths [1])

get_intent_vector(click_seq_asins, cat_ids=None) → np.ndarray[256]:
    # Reconstructs BGE-M3 embeddings, runs forward, returns last-position hidden

get_candidate_scores(click_seq_asins, cat_ids, candidate_asins) → dict{asin: float}:
    # Scores a list of candidate ASINs
    # Drop-in replacement for RLSequentialFilter.get_candidate_scores()

train_step(click_seq_asins, target_asin, target_cat_id, neg_pool_asins) → float:
    # Sampled Softmax loss (positive = target, negatives = random from neg_pool)
    # Category CE auxiliary loss (predict category from intent vector)
    # Returns total loss

save(path) / load(path):
    # Saves/loads model state dict + optimizer + step counter
    # Checkpoint key: "arch": "dif_sasrec_v1"
```

**Full file skeleton to implement:**

```python
"""
app/services/dif_sasrec.py — DIF-SASRec model and online training agent.

Architecture reference: "DIF-SR: Decoupled Interest-aware Feature for Sequential Recommendation"
Variant implemented here: category-decoupled attention for book genre modeling.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from app.config import settings

TEXT_EMBED_DIM = settings.TEXT_EMBED_DIM   # 1024 — BGE-M3
HIDDEN_DIM     = settings.SASREC_HIDDEN_DIM
N_BLOCKS       = settings.SASREC_N_BLOCKS
N_HEADS        = settings.SASREC_N_HEADS
DROPOUT        = settings.SASREC_DROPOUT
LR             = settings.SASREC_LR
ALPHA_INIT     = settings.SASREC_ALPHA_INIT
CAT_AUX_WEIGHT = settings.SASREC_CAT_AUX_WEIGHT
NUM_NEGATIVES  = settings.SASREC_NUM_NEGATIVES
MAX_SEQ_LEN    = settings.MAX_SEQ_LEN


class ContentProjector(nn.Module): ...
class DIFAttentionLayer(nn.Module): ...
class DIFSASRecBlock(nn.Module): ...
class DIFSASRecModel(nn.Module): ...
class DIFSASRecAgent: ...
```

---

## Step 4: Modify `app/repository/faiss_repo.py`

**One change only**: add `get_content_candidates()` method.

**Step 4a (HNSW auto-detection) is ALREADY DONE** — `_load_text_index()` already handles HNSW vs flat detection. `self.text_index` IS the HNSW index when `bge_index_hnsw.faiss` exists (it does). Do not touch the `__init__` or `_load_text_index`.

### Add `get_content_candidates()` method (append to end of class)

```python
    def get_content_candidates(self, query_vector, top_n: int = 200,
                                exclude_asins: set = None) -> list:
        """
        Personal Pipeline candidate generation via HNSW KNN search.

        Uses self.text_index (already HNSW if bge_index_hnsw.faiss exists)
        to find nearest neighbors to the query vector.
        COMPLETELY INDEPENDENT of Cleora.

        Used by the DIF-SASRec personal pipeline ("You Might Like") tab.
        The query_vector is typically the user's text_profile (1024-dim
        weighted average of their clicked items' BGE-M3 embeddings).

        Args:
            query_vector:  np.ndarray [1024] — user's BGE-M3 profile vector
            top_n:         number of candidates to return
            exclude_asins: set of ASINs to exclude (already clicked)
        Returns:
            list of ASIN strings, ordered by similarity
        """
        if exclude_asins is None:
            exclude_asins = set()

        q = query_vector.reshape(1, -1).astype("float32")
        fetch_n = top_n + len(exclude_asins) + 50
        D, I = self.text_index.search(q, fetch_n)

        results = []
        for faiss_i in I[0]:
            if faiss_i < 0 or faiss_i >= len(self.asins):
                continue
            asin = self.asins[faiss_i]
            if asin in exclude_asins:
                continue
            results.append(asin)
            if len(results) >= top_n:
                break

        return results
```

---

## Step 5: Modify `app/repository/profile_repo.py`

Three changes:

### 5a. Add `category_encoder` parameter to `__init__` (line 39)

Change:
```python
    def __init__(self, retriever=None, data_dir: str = None):
```
To:
```python
    def __init__(self, retriever=None, data_dir: str = None, category_encoder=None):
```

And add after `os.makedirs(self._profiles_dir, exist_ok=True)`:
```python
        self._category_encoder = category_encoder
```

### 5b. Populate `preferred_categories` in `log_click()` (after line 108 `profile.recent_interactions.append(item_id)`)

Insert:
```python
        # Update category preferences for the "You Might Like" personal pipeline
        if self._category_encoder:
            cat_id = self._category_encoder.get_category_id(item_id)
            cat_name = self._category_encoder.get_category_name(cat_id)
            if cat_name and cat_name not in ("PAD", "UNK"):
                profile.preferred_categories[cat_name] += 1
```

### 5c. Add `get_click_sequence_with_categories()` method (after `get_click_sequence`, after line 83)

Insert this new **async** method:

```python
    async def get_click_sequence_with_categories(self, user_id: str,
                                                  max_len: int = settings.MAX_RECENT_INTERACTIONS):
        """
        Return (asin_list, category_id_list) for the DIF-SASRec model.

        Both lists are the same length, in chronological order (most recent last).
        Category IDs come from the CategoryEncoder vocabulary.
        """
        profile = await self.get_profile(user_id)
        asins = [c["item_id"] for c in profile.clicks
                 if c.get("action", "click") in ("click", "cart")]
        asins = asins[-max_len:]

        if self._category_encoder:
            cat_ids = self._category_encoder.encode_sequence(asins)
        else:
            cat_ids = [1] * len(asins)  # UNK fallback

        return asins, cat_ids
```

---

## Step 6: Modify `app/services/passive_recommend.py`

**This is the biggest change.** Replace the dual-tab recommendation logic with the dual-pipeline architecture.

### 6a. Replace import at the top

Change:
```python
from app.services.rl_filter import RLSequentialFilter
```
To:
```python
from app.services.dif_sasrec import DIFSASRecAgent
```

### 6b. Change `__init__` signature and body

Change:
```python
    def __init__(self, retriever, profile_manager):
        self.retriever       = retriever
        self.profile_manager = profile_manager
        self.rl_cf           = RLSequentialFilter(retriever)
```
To:
```python
    def __init__(self, retriever, profile_manager, category_encoder=None):
        self.retriever        = retriever
        self.profile_manager  = profile_manager
        self.category_encoder = category_encoder
        self.sasrec           = DIFSASRecAgent(retriever, category_encoder)
```

### 6c. Replace `_dqn_path`, `save_rl_weights`, `load_rl_weights` with personal equivalents

Replace these three methods:
```python
    def _dqn_path(self, data_dir: str, user_id: str) -> str: ...
    def save_rl_weights(self, user_id: str, data_dir: str): ...
    def load_rl_weights(self, user_id: str, data_dir: str): ...
```
With:
```python
    def _sasrec_path(self, data_dir: str, user_id: str) -> str:
        profiles_dir = os.path.join(data_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id)
        return os.path.join(profiles_dir, f"{safe_id}_dif_sasrec.pt")

    def save_personal_weights(self, user_id: str, data_dir: str):
        self.sasrec.save(self._sasrec_path(data_dir, user_id))

    def load_personal_weights(self, user_id: str, data_dir: str):
        path = self._sasrec_path(data_dir, user_id)
        if os.path.exists(path):
            self.sasrec.load(path)
```

### 6d. Replace `recommend_for_user` with dual-pipeline logic

Replace the existing `recommend_for_user` async method with:

```python
    async def recommend_for_user(self, user_id: str, top_k: int = TOP_K):
        """
        Dual-pipeline personalised recommendations.

        Tab 1 "People Also Buy" → Cleora pipeline (unchanged)
        Tab 2 "You Might Like"  → DIF-SASRec + HNSW KNN (zero Cleora dependency)
        """
        profile = await self.profile_manager.get_profile(user_id)
        if len(profile.clicks) < COLD_START_THRESHOLD:
            return None

        # ── Pipeline A: People Also Buy (Cleora-dependent) ────────────────────
        candidates = self.collaborative_filter(profile, top_n=BEHAVIORAL_CANDIDATES)
        if not candidates:
            people_also_buy = []
        else:
            verified = self.content_verify(
                candidates,
                user_text_profile=profile.text_profile,
                user_visual_profile=profile.visual_profile,
            )
            pab_ranked = sorted(
                verified,
                key=lambda x: max(x["text_score"], x["visual_score"]),
                reverse=True,
            )
            people_also_buy = [
                (item["asin"], float(max(item["text_score"], item["visual_score"])), "Retrieval")
                for item in pab_ranked[:top_k]
            ]

        # ── Pipeline B: You Might Like (DIF-SASRec, zero Cleora) ─────────────
        you_might_like = await self._personal_recommend(profile, user_id, top_k)

        if not people_also_buy and not you_might_like:
            return None

        return {"people_also_buy": people_also_buy, "you_might_like": you_might_like}
```

### 6e. Add `_personal_recommend()` method

```python
    async def _personal_recommend(self, profile, user_id: str, top_k: int) -> list:
        """
        Pipeline B: DIF-SASRec intent → HNSW KNN → content veto → DIF-SASRec scoring.
        Uses text_profile (BGE-M3 weighted average) for HNSW retrieval.
        Zero dependency on Cleora — works even if cleora_index is None.
        """
        if profile.text_profile is None:
            return []

        # Step 1: HNSW KNN retrieval from user's BGE-M3 text profile
        seen = {c["item_id"] for c in profile.clicks}
        candidates = self.retriever.get_content_candidates(
            profile.text_profile,
            top_n=PERSONAL_CANDIDATES,
            exclude_asins=seen,
        )
        if not candidates:
            return []

        # Step 2: Content veto (same threshold as Pipeline A)
        verified = self.content_verify(
            candidates,
            user_text_profile=profile.text_profile,
            user_visual_profile=profile.visual_profile,
        )
        if not verified:
            return []

        # Step 3: DIF-SASRec scoring
        asins, cat_ids = await self.profile_manager.get_click_sequence_with_categories(user_id)
        candidate_asins = [item["asin"] for item in verified]
        scores = self.sasrec.get_candidate_scores(asins, cat_ids, candidate_asins)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(asin, float(score), "DIF-SASRec") for asin, score in ranked[:top_k]]
```

Also add at the top of the file (after other settings imports):
```python
PERSONAL_CANDIDATES = settings.PERSONAL_CANDIDATES
```

### 6f. Replace `train_rl()` with `train_personal()`

Replace:
```python
    def train_rl(self, user_id: str, item_asin: str, reward: float,
                 click_seq_before: list = None, click_seq_after: list = None) -> float | None:
        return self.rl_cf.train_step(
            click_seq_before or [],
            item_asin,
            reward,
            click_seq_after or [],
        )
```
With:
```python
    def train_personal(self, user_id: str, item_asin: str,
                       click_seq_before: list = None) -> float | None:
        """Train the DIF-SASRec model on the latest click event."""
        if not click_seq_before:
            return None
        cat_id = (self.category_encoder.get_category_id(item_asin)
                  if self.category_encoder else 1)
        all_asins = list(self.retriever.asin_to_idx.keys())
        return self.sasrec.train_step(
            click_seq_before,
            item_asin,
            cat_id,
            all_asins,
        )
```

### 6g. Keep unchanged

- `collaborative_filter()` — unchanged (Pipeline A still uses it)
- `content_verify()` — unchanged (both pipelines use it)
- `rrf_fusion()` — keep as-is (not currently called, preserved for future use)

---

## Step 7: Modify `app/core/lifespan.py`

### 7a. Add `CategoryEncoder` initialization (between FAISS load and pipeline objects)

In the `lifespan` function, replace:
```python
    # 3. Pipeline objects
    profile_manager  = UserProfileManager(retriever=retriever, data_dir=settings.DATA_DIR)
    recommend_engine = PassiveRecommendationEngine(retriever, profile_manager)
```
With:
```python
    # 3. Category encoder (for DIF-SASRec personal pipeline)
    from app.services.category_encoder import CategoryEncoder
    cat_encoder = CategoryEncoder()
    cat_vocab_path = os.path.join(settings.DATA_DIR, "category_vocab.json")
    if os.path.exists(cat_vocab_path):
        cat_encoder.load(cat_vocab_path)
    else:
        log.info("Building category vocabulary from item_metadata.parquet ...")
        cat_encoder.build_from_parquet(
            os.path.join(settings.DATA_DIR, "item_metadata.parquet")
        )
        cat_encoder.save(cat_vocab_path)

    # 3b. Pipeline objects
    profile_manager  = UserProfileManager(
        retriever=retriever,
        data_dir=settings.DATA_DIR,
        category_encoder=cat_encoder,
    )
    recommend_engine = PassiveRecommendationEngine(
        retriever, profile_manager, category_encoder=cat_encoder
    )
```

Also add `import os` if not already at the top (check first — it may already be imported).

### 7b. Add `category_encoder` to `AppContainer`

In `app/core/container.py`, add a field to `AppContainer`:
```python
    category_encoder: Any = None          # app.services.category_encoder.CategoryEncoder
```

And in `lifespan.py`, after building `cat_encoder`:
```python
    container.category_encoder = cat_encoder
```

---

## Step 8: Modify `app/api/routes/recommend.py`

### 8a. Update `/recommend` endpoint

Change:
```python
        recommend_engine.load_rl_weights(user_id, settings.DATA_DIR)
```
To:
```python
        recommend_engine.load_personal_weights(user_id, settings.DATA_DIR)
```

### 8b. Update `/rl_metrics` endpoint

Replace the entire `/rl_metrics` endpoint:
```python
@router.get("/rl_metrics")
async def rl_metrics(user_id: str, container: AppContainer = Depends(require_ready)):
    """Return real-time DIF-SASRec model metrics."""
    recommend_engine = container.recommend_engine
    recommend_engine.load_personal_weights(user_id, settings.DATA_DIR)
    agent = recommend_engine.sasrec

    return {
        "user_id": user_id,
        "step":    agent._step,
        "arch":    "DIF-SASRec",
    }
```

---

## Step 9: Modify `app/api/routes/interact.py`

Replace the RL training block at the bottom of `interact()`:

Change:
```python
    loss = None
    if click_seq_before:
        loss = recommend_engine.train_rl(
            req.user_id, req.item_id, reward,
            click_seq_before=click_seq_before,
            click_seq_after=click_seq_after,
        )
        recommend_engine.save_rl_weights(req.user_id, settings.DATA_DIR)
```
To:
```python
    # ── Train the DIF-SASRec personal model ──────────────────────────────────
    loss = None
    if click_seq_before and req.action in ("click", "cart"):
        loss = recommend_engine.train_personal(
            req.user_id, req.item_id,
            click_seq_before=click_seq_before,
        )
        recommend_engine.save_personal_weights(req.user_id, settings.DATA_DIR)
```

Also remove the `click_seq_after` capture — it's no longer needed:

Remove:
```python
    # Capture s_{t+1} AFTER profile update
    click_seq_after = await profile_manager.get_click_sequence(req.user_id)
```

---

## Step 10: Create `scripts/train/pretrain_dif_sasrec.py` (NEW FILE)

**Purpose**: Offline pre-training on `evaluation/eval_users.json` to create a warm global checkpoint.

**Create file**: `scripts/train/pretrain_dif_sasrec.py`

```python
"""
scripts/train/pretrain_dif_sasrec.py — Offline pre-training for DIF-SASRec

Trains a global DIF-SASRec checkpoint using existing user interaction histories
from eval_users.json. This checkpoint is loaded for ALL new users as their
starting point, then per-user weights diverge via online fine-tuning.

Usage:
    python scripts/train/pretrain_dif_sasrec.py

Output:
    data/dif_sasrec_pretrained.pt   (model checkpoint)
    data/category_vocab.json        (category vocabulary — if not already built)
"""
import json
import os
import sys
import time

# Add project root to path so we can import app.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.services.category_encoder import CategoryEncoder
from app.services.dif_sasrec import DIFSASRecAgent

DATA_DIR = settings.DATA_DIR
EVAL_PATH = os.path.join(ROOT, "evaluation", "eval_users.json")
PRETRAINED_PATH = os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")
CAT_VOCAB_PATH  = os.path.join(DATA_DIR, "category_vocab.json")

EPOCHS = 30


def main():
    t0 = time.time()

    # 1. Load Retriever
    print("Loading FAISS indices and Cleora embeddings ...")
    cleora_data = np.load(os.path.join(DATA_DIR, "cleora_embeddings.npz"))
    retriever = Retriever(DATA_DIR, cleora_data)

    # 2. Load / build CategoryEncoder
    cat_encoder = CategoryEncoder()
    if os.path.exists(CAT_VOCAB_PATH):
        cat_encoder.load(CAT_VOCAB_PATH)
    else:
        cat_encoder.build_from_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))
        cat_encoder.save(CAT_VOCAB_PATH)

    # 3. Build DIFSASRecAgent
    agent = DIFSASRecAgent(retriever, cat_encoder)
    print(f"Model params: {sum(p.numel() for p in agent.model.parameters()):,}")

    # 4. Load eval users
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_users = json.load(f)

    all_asins = list(retriever.asin_to_idx.keys())

    # 5. Training loop
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_steps = 0

        for user in eval_users:
            train_clicks = user.get("train_clicks", [])
            if len(train_clicks) < 3:
                continue

            for t in range(2, len(train_clicks)):
                input_seq  = train_clicks[:t]
                target     = train_clicks[t]
                target_cat = cat_encoder.get_category_id(target)

                loss = agent.train_step(input_seq, target, target_cat, all_asins)
                if loss is not None:
                    total_loss += loss
                    n_steps    += 1

        avg = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch:2d}/{EPOCHS}  steps={n_steps:,}  avg_loss={avg:.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

    # 6. Save checkpoint
    agent.save(PRETRAINED_PATH)
    print(f"\nPre-trained checkpoint saved to {PRETRAINED_PATH}")
    print(f"Total training time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
```

---

## Step 11: Create `scripts/benchmark/evaluate_recommendation.py` (NEW FILE)

**Purpose**: Offline evaluation comparing GRU-SeqDQN vs DIF-SASRec vs Content baselines.

Note: The original plan referenced `scripts/evaluate_recommendation.py` but no such file exists. The scripts are organized into subdirectories — place it in `scripts/benchmark/`.

```python
"""
scripts/benchmark/evaluate_recommendation.py — Offline recommendation evaluation.

Evaluates multiple recommendation strategies on evaluation/eval_users.json.
Metrics: HR@5, HR@10, NDCG@10

Usage:
    python scripts/benchmark/evaluate_recommendation.py

Strategies evaluated:
    - Content Baseline (BGE-M3 profile → HNSW KNN, no model)
    - GRU-SeqDQN (existing RL model)
    - DIF-SASRec (new personal model, pre-trained checkpoint if available)
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

from app.config import settings
from app.repository.faiss_repo import Retriever
from app.services.category_encoder import CategoryEncoder

DATA_DIR  = settings.DATA_DIR
EVAL_PATH = os.path.join(ROOT, "evaluation", "eval_users.json")

# ─── Strategy implementations ────────────────────────────────────────────────


class ContentBaseline:
    """BGE-M3 profile → HNSW KNN. No model — pure retrieval baseline."""

    def __init__(self, retriever):
        self.retriever = retriever

    def recommend(self, train_clicks, k=10):
        if not train_clicks:
            return []
        vecs = []
        for asin in train_clicks:
            if asin in self.retriever.asin_to_idx:
                idx = self.retriever.asin_to_idx[asin]
                vecs.append(self.retriever.text_flat.reconstruct(idx))
        if not vecs:
            return []
        profile_vec = np.mean(vecs, axis=0)
        seen = set(train_clicks)
        return self.retriever.get_content_candidates(
            profile_vec, top_n=k, exclude_asins=seen
        )


class GRUSeqDQNStrategy:
    """Existing GRU-Sequential DQN (Cleora-dependent)."""

    def __init__(self, retriever):
        from app.services.rl_filter import RLSequentialFilter
        self.agent = RLSequentialFilter(retriever)
        self.retriever = retriever

    def recommend(self, train_clicks, k=10):
        if not train_clicks:
            return []
        vecs = []
        for asin in train_clicks:
            if asin in self.retriever.asin_to_idx:
                idx = self.retriever.asin_to_idx[asin]
                vecs.append(self.retriever.text_flat.reconstruct(idx))
        if not vecs:
            return []
        profile_vec = np.mean(vecs, axis=0)
        seen = set(train_clicks)
        candidates = self.retriever.get_content_candidates(
            profile_vec, top_n=settings.PERSONAL_CANDIDATES, exclude_asins=seen
        )
        scores = self.agent.get_candidate_scores(train_clicks, candidates)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [asin for asin, _ in ranked[:k]]


class DIFSASRecStrategy:
    """DIF-SASRec personal recommendation (zero Cleora dependency)."""

    def __init__(self, retriever, category_encoder, pretrained_path=None):
        from app.services.dif_sasrec import DIFSASRecAgent
        self.agent       = DIFSASRecAgent(retriever, category_encoder, pretrained_path)
        self.retriever   = retriever
        self.cat_encoder = category_encoder

    def recommend(self, train_clicks, k=10):
        if not train_clicks:
            return []
        cat_ids = self.cat_encoder.encode_sequence(train_clicks)
        vecs = []
        for asin in train_clicks:
            if asin in self.retriever.asin_to_idx:
                idx = self.retriever.asin_to_idx[asin]
                vecs.append(self.retriever.text_flat.reconstruct(idx))
        if not vecs:
            return []
        profile_vec = np.mean(vecs, axis=0)
        seen = set(train_clicks)
        candidates = self.retriever.get_content_candidates(
            profile_vec, top_n=settings.PERSONAL_CANDIDATES, exclude_asins=seen
        )
        scores = self.agent.get_candidate_scores(train_clicks, cat_ids, candidates)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [asin for asin, _ in ranked[:k]]


# ─── Metrics ─────────────────────────────────────────────────────────────────


def hit_rate(recommended, ground_truth):
    return 1.0 if any(a in ground_truth for a in recommended) else 0.0


def ndcg(recommended, ground_truth, k=10):
    for i, a in enumerate(recommended[:k]):
        if a in ground_truth:
            import math
            return 1.0 / math.log2(i + 2)
    return 0.0


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print("Loading FAISS indices ...")
    cleora_data = np.load(os.path.join(DATA_DIR, "cleora_embeddings.npz"))
    retriever   = Retriever(DATA_DIR, cleora_data)

    cat_encoder = CategoryEncoder()
    cat_vocab_path = os.path.join(DATA_DIR, "category_vocab.json")
    if os.path.exists(cat_vocab_path):
        cat_encoder.load(cat_vocab_path)
    else:
        cat_encoder.build_from_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))

    pretrained_path = os.path.join(DATA_DIR, "dif_sasrec_pretrained.pt")

    strategies = {
        "Content Baseline":       ContentBaseline(retriever),
        "GRU-SeqDQN":            GRUSeqDQNStrategy(retriever),
        "DIF-SASRec (pre-trained)": DIFSASRecStrategy(
            retriever, cat_encoder,
            pretrained_path=pretrained_path if os.path.exists(pretrained_path) else None,
        ),
    }

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_users = json.load(f)

    for name, strategy in strategies.items():
        hr5, hr10, ndcg10 = [], [], []
        t0 = time.time()
        for user in eval_users:
            train = user.get("train_clicks", [])
            test  = set(user.get("test_clicks", []))
            if not train or not test:
                continue
            recs10 = strategy.recommend(train, k=10)
            hr5.append(hit_rate(recs10[:5], test))
            hr10.append(hit_rate(recs10, test))
            ndcg10.append(ndcg(recs10, test, k=10))

        n = len(hr5) or 1
        print(f"\n{name}:")
        print(f"  HR@5={sum(hr5)/n:.4f}  HR@10={sum(hr10)/n:.4f}  "
              f"NDCG@10={sum(ndcg10)/n:.4f}  n={n}  t={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
```

---

## Verification Steps (Run After Implementation)

### 1. Category encoder test
```bash
cd "DATN (1)"
python -c "
from app.services.category_encoder import CategoryEncoder
from app.config import settings
import os
e = CategoryEncoder()
e.build_from_parquet(os.path.join(settings.DATA_DIR, 'item_metadata.parquet'))
print(f'Vocab: {e.num_categories}')
e.save(os.path.join(settings.DATA_DIR, 'category_vocab.json'))
"
```
Expected: prints vocab size (~50-200 categories) and saves JSON.

### 2. DIF-SASRec model import test
```bash
python -c "
from app.services.category_encoder import CategoryEncoder
from app.repository.faiss_repo import Retriever
from app.services.dif_sasrec import DIFSASRecAgent
from app.config import settings
import numpy as np, os

cat = CategoryEncoder()
cat.load(os.path.join(settings.DATA_DIR, 'category_vocab.json'))

cleora = np.load(os.path.join(settings.DATA_DIR, 'cleora_embeddings.npz'))
ret = Retriever(settings.DATA_DIR, cleora)

agent = DIFSASRecAgent(ret, cat)
print(f'Model params: {sum(p.numel() for p in agent.model.parameters()):,}')
print('PASS')
"
```

### 3. Pre-training
```bash
python scripts/train/pretrain_dif_sasrec.py
```
Expected: ~10 min on RTX 4060, loss should decrease over epochs.

### 4. Offline evaluation
```bash
python scripts/benchmark/evaluate_recommendation.py
```
Expected: DIF-SASRec (pre-trained) HR@10 ≥ 0.12 (vs Content baseline ~0.11).

### 5. End-to-end API test
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# In another terminal:
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend?user_id=test_user"
curl -X POST http://127.0.0.1:8000/interact \
     -H "Content-Type: application/json" \
     -d '{"user_id":"test_user","item_id":"B001","action":"click"}'
```

### 6. Cleora-independence test
Verify "You Might Like" works even when Cleora is missing:
```python
# Temporarily: retriever.cleora_index = None
# → Pipeline A ("People Also Buy") returns []
# → Pipeline B ("You Might Like") should still return results via HNSW KNN
```

---

## Files Summary

| # | File | Action | Key Changes |
|:--|:--|:--|:--|
| 1 | `app/services/category_encoder.py` | **NEW** | Category vocabulary from parquet |
| 2 | `app/config.py` | MODIFY | +9 fields inside `Settings` dataclass |
| 3 | `app/services/dif_sasrec.py` | **NEW** | Full DIF-SASRec model + agent (~350 lines) |
| 4 | `app/repository/faiss_repo.py` | MODIFY | +30 lines: `get_content_candidates()` only (HNSW already there) |
| 5 | `app/repository/profile_repo.py` | MODIFY | +25 lines: `category_encoder` param + `get_click_sequence_with_categories()` |
| 6 | `app/services/passive_recommend.py` | MODIFY | Major rewrite: dual pipeline (Cleora + DIF-SASRec) |
| 7 | `app/core/lifespan.py` | MODIFY | +15 lines: CategoryEncoder init, wire into constructors |
| 7b | `app/core/container.py` | MODIFY | +1 field: `category_encoder` |
| 8 | `app/api/routes/recommend.py` | MODIFY | `load_rl_weights` → `load_personal_weights`, update `/rl_metrics` |
| 9 | `app/api/routes/interact.py` | MODIFY | `train_rl` → `train_personal`, remove `click_seq_after` |
| 10 | `scripts/train/pretrain_dif_sasrec.py` | **NEW** | Offline pre-training script |
| 11 | `scripts/benchmark/evaluate_recommendation.py` | **NEW** | Offline evaluation with 3 strategies |

**DO NOT DELETE**: `app/services/sequential_dqn.py`, `app/services/rl_filter.py` — kept for thesis comparison.
