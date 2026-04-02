# NBA Multimodal Book Recommendation System — Thesis Improvement Roadmap

> **Context**: This roadmap is derived from advisor feedback received on March 25, 2026, and maps directly onto the current system state as documented in `Project_update_25-3.md`, `item_metadata_update_19-3.md`, and `RL_UI_Inspection_update_25-3.md`. All tasks reference specific files in the existing codebase.

---

## Overview

| Phase | Focus | Priority | Estimated Effort |
|---|---|---|---|
| Phase 1 | Performance & Accuracy Fixes | 🔴 Critical | 4–5 days |
| Phase 2 | Model Comparison & Evaluation Metrics | 🟡 High | 4–5 days |
| Phase 3 | Missing Metadata Fallback via Qwen | 🔵 Medium | 3–4 days |
| **Total** | | | **~2 weeks** |

---

## Phase 1 — Performance & Accuracy Fixes

> **Advisor feedback**: *"Check the FAISS mechanism — response time under 1 second is acceptable. Check Vietnamese search accuracy. Set a threshold to block results with low similarity scores."*

---

### Task 1.1 — Benchmark End-to-End Response Time

**Goal**: Profile the full Mode 1 pipeline (API call → JSON response) and confirm it runs under 1 second.

**Files to touch**: `api.py`, `active_search_engine.py`

#### What to do

Add `time.perf_counter()` checkpoints at every major stage of the pipeline. Return the breakdown in the API response (dev mode only).

```python
# api.py — /search endpoint, dev profiling block
import time

@app.post("/search")
async def search(req: SearchRequest):
    t0 = time.perf_counter()

    # Stage 1: Text encoding (BLaIR)
    text_vec = blair_model.encode([req.query]).astype("float32")
    t1 = time.perf_counter()

    # Stage 2: Image encoding (CLIP) — if image provided
    image_vec = clip_model.get_image_features(**inputs).cpu().numpy() if req.image_base64 else None
    t2 = time.perf_counter()

    # Stage 3: FAISS search + RRF fusion
    results = search_engine.search(req.user_id, text_vec, image_vec)
    t3 = time.perf_counter()

    # Stage 4: BGE Reranker cross-encoding
    reranked = reranker.rerank(req.query, results[:20])
    t4 = time.perf_counter()

    # Stage 5: Metadata hydration
    enriched = hydrate_metadata(reranked)
    t5 = time.perf_counter()

    timings = {
        "encode_blair_ms": round((t1 - t0) * 1000, 1),
        "encode_clip_ms":  round((t2 - t1) * 1000, 1),
        "faiss_rrf_ms":    round((t3 - t2) * 1000, 1),
        "reranker_ms":     round((t4 - t3) * 1000, 1),
        "metadata_ms":     round((t5 - t4) * 1000, 1),
        "total_ms":        round((t5 - t0) * 1000, 1),
    }

    return {"results": enriched, "_debug_timings": timings}
```

#### Investigation Checklist

After running the profiler, investigate based on where time is being spent:

**If FAISS takes > 200ms:**
- Confirm that both the BLaIR and CLIP indices are loaded into RAM at startup (stored in `_state`) and not re-read from disk per request.
- Check whether `IndexFlatIP` (exact) is being used. For 3M items, this is `O(N·d)` — it may be worth switching to `IndexIVFFlat` with `nprobe` tuning for approximate but much faster search.

```python
# Switching to approximate IVF search — build time is offline, query is fast
import faiss
quantizer = faiss.IndexFlatIP(1024)
index = faiss.IndexIVFFlat(quantizer, 1024, nlist=4096, metric=faiss.METRIC_INNER_PRODUCT)
index.train(all_blair_vecs)  # one-time offline step
index.add(all_blair_vecs)
index.nprobe = 64  # tune: higher = more accurate, slower
```

**If BGE Reranker takes > 400ms (most likely bottleneck):**
- Reduce top-K candidates passed to the reranker from 20 → 10.
- Ensure the model is running on GPU. If CPU-only, VRAM contention with BLaIR/CLIP may be causing serialization.
- Consider caching reranker results for repeated identical queries (LRU cache keyed on `(query, frozenset(candidate_asins))`).

**Target breakdown (suggested budget for < 1s total):**

| Stage | Target |
|---|---|
| BLaIR encoding | < 100ms |
| CLIP encoding | < 80ms |
| FAISS search (both indices) | < 50ms |
| RRF fusion | < 5ms |
| BGE Reranker (top-10) | < 400ms |
| Metadata hydration | < 20ms |
| **Total** | **< 700ms** |

---

### Task 1.2 — Add Similarity Threshold to Block Low-Confidence Results

**Goal**: Prevent weak or irrelevant matches from reaching the user. This is especially important for Vietnamese queries where BLaIR scores are systematically lower.

**Files to touch**: `active_search_engine.py`, `api.py`

#### What to do

Add a post-fusion filter that drops candidates whose scores fall below a configurable threshold. Return a clean "no results" state to the frontend rather than showing garbage.

```python
# active_search_engine.py — after RRF fusion step, before metadata filter

# Configurable thresholds (tune empirically from profiling data)
DENSE_SCORE_THRESHOLD = 0.25   # cosine similarity in BLaIR/CLIP space
RERANKER_SCORE_THRESHOLD = 0.30  # BGE cross-encoder score (0–1 range)

# ── Step: Drop low-confidence dense results ───────────────────────────────
dense_ranking = [
    (asin, data) for asin, data in dense_ranking
    if data.get('score', 0.0) >= DENSE_SCORE_THRESHOLD
]

# ── Step: Drop low-confidence reranked results ────────────────────────────
# (applied after BGEReranker.rerank() returns)
reranked_results = [
    item for item in reranked_results
    if item.get('reranker_score', 0.0) >= RERANKER_SCORE_THRESHOLD
]
```

```python
# api.py — /search endpoint response
if not enriched:
    return {
        "results": [],
        "message": "No sufficiently relevant results found. Try rephrasing your query."
    }
```

**Frontend update (`App.jsx`)**: Handle the empty results state and display a user-friendly "No results found" message instead of a blank list.

#### Tuning the Threshold

Don't set it and forget it. Log how often thresholds fire:

```python
# In active_search_engine.py
dropped_count = original_count - len(dense_ranking)
logger.info(f"Threshold filter dropped {dropped_count}/{original_count} candidates")
```

Run 50–100 representative queries and adjust thresholds until precision improves without destroying recall for legitimate queries.

---

### Task 1.3 — Audit and Fix Vietnamese Search Accuracy

**Goal**: Identify whether BLaIR's degraded performance on Vietnamese text is causing poor results, and apply a targeted fix.

**Files to touch**: `active_search_engine.py`, `api.py`

#### Diagnosis First

Create a small test set of Vietnamese queries with known expected results:

```python
VI_TEST_QUERIES = [
    {"query": "tiểu thuyết trinh thám", "expected_title_keywords": ["detective", "mystery", "crime"]},
    {"query": "sách phát triển bản thân", "expected_title_keywords": ["self-help", "habits", "mindset"]},
    {"query": "fantasy phép thuật", "expected_title_keywords": ["magic", "wizard", "fantasy"]},
    {"query": "lịch sử thế chiến", "expected_title_keywords": ["world war", "history", "1939"]},
]
```

Run each query through BLaIR encoding and inspect:
1. The raw cosine similarity scores of the top-10 results
2. Whether the expected books appear in the top-10 at all

**If scores are consistently below 0.3 for all results**: BLaIR cannot handle Vietnamese. Apply the translation fallback below.

#### Fix Option A — Language Detection + Query Translation

```bash
pip install langdetect
pip install transformers sentencepiece  # for Helsinki-NLP model
```

```python
# active_search_engine.py — Step 0.5: Language detection & translation
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Load once at startup in api.py
_vi_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
_vi_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")

def translate_if_vietnamese(query: str) -> str:
    try:
        lang = detect(query)
        if lang == 'vi':
            inputs = _vi_en_tokenizer([query], return_tensors="pt", padding=True)
            outputs = _vi_en_model.generate(**inputs)
            return _vi_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        pass
    return query  # fallback: return original query unchanged

# In search():
query_for_encoding = translate_if_vietnamese(raw_query)
text_vec = blair_model.encode([query_for_encoding]).astype("float32")
```

#### Fix Option B — Vietnamese Keyword Scan Fallback (Lighter)

If the translation model is too heavy for the deployment environment:

```python
# active_search_engine.py
from langdetect import detect

def search(self, user_id, text_query, image_base64=None):
    lang = detect(text_query) if text_query else 'en'

    # For Vietnamese: force keyword scan regardless of query length
    force_keyword_scan = (lang == 'vi')

    if len(text_query.split()) <= 4 or force_keyword_scan:
        keyword_hits = self._keyword_scan(text_query)
    ...
```

This guarantees Vietnamese proper-noun queries (author names, series names) always surface via substring match even if dense retrieval fails.

---

### Task 1.4 — Verify CLIP Version and Evaluate Upgrade Options

**Goal**: Confirm the current CLIP variant, assess whether upgrading to a stronger model meaningfully improves visual retrieval, and document the decision.

**Current model**: `openai/clip-vit-base-patch32` → 512-dim embeddings, ViT-B/32 backbone.

#### Quick Audit (0.5 days)

Run a set of cover image queries and manually inspect the top-5 visual nearest neighbors:

```python
# diagnostic_clip_audit.py
test_images = [
    "dark_noir_cover.jpg",   # expect: dark, moody, crime novels
    "bright_childrens.jpg",  # expect: colorful, illustrated children's books
    "sci_fi_cover.jpg",      # expect: space, futuristic imagery
]
for img_path in test_images:
    image_vec = encode_image(img_path)
    D, I = clip_index.search(image_vec, 5)
    print_results(I, D)  # inspect titles and covers of top-5
```

#### Upgrade Candidates

| Model | Dim | Relative Quality | Notes |
|---|---|---|---|
| `openai/clip-vit-base-patch32` *(current)* | 512 | Baseline | Fast, widely tested |
| `openai/clip-vit-large-patch14` | 768 | ~15% better | OpenAI's stronger release |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 1024 | State-of-art | Very large, best recall |

**Important caveat**: Upgrading CLIP changes embedding dimensionality. This requires:
1. Re-embedding all ~3M book covers (compute-intensive)
2. Rebuilding the FAISS CLIP index from scratch
3. Updating all 90 NPZ chunks to the new dimension
4. Updating the DQN input dimension (currently 1536 = 1024 + 512; would become 1024 + 768 = 1792 if upgrading to ViT-L)

**Recommendation**: If the ViT-B/32 audit shows reasonable visual matching (correct genre/aesthetic in top-5), do not upgrade — document the justification instead. Only upgrade if the baseline is clearly broken. Thesis timeline is the constraint.

---

## Phase 2 — Model Comparison & Evaluation Metrics

> **Advisor feedback**: *"Prepare comparison metrics between models."*

---

### Task 2.1 — Define the Evaluation Protocol

**Goal**: Establish a rigorous, reproducible evaluation setup before running any experiments. The protocol must be written up in the thesis.

#### Metrics to Report

| Metric | Formula | What it measures |
|---|---|---|
| **Hit@K** | Fraction of queries where a relevant item appears in top-K | Raw recall quality |
| **NDCG@K** | Normalized Discounted Cumulative Gain at K | Ranking quality — rewards relevant items at higher positions |
| **MRR** | Mean Reciprocal Rank | How high the first relevant item appears on average |
| **Recall@50** | Fraction of relevant items captured in the 50 FAISS candidates | Retrieval stage recall before reranking |
| **Response time (ms)** | Wall-clock end-to-end latency | System performance |

#### Ground Truth Construction

**Option A — Interaction-based (faster)**: Use purchase/cart interactions from the Amazon Reviews dataset as positive labels. A query is the text of items a user previously interacted with; the positives are other items they purchased in the same session.

**Option B — Manual annotation (more defensible for thesis)**: Manually create 50–100 test queries with a human-curated list of relevant books per query.

```python
# Example ground truth format
GROUND_TRUTH = {
    "gritty detective novels": ["B001234", "B005678", "B009012"],
    "fantasy magic systems": ["B003456", "B007890"],
    ...
}
```

#### Train/Test Split for RL Evaluation

Use an 80/20 temporal split on interaction history:
- First 80% of interactions (chronologically) → used for DQN training
- Last 20% → held out for evaluation (predict what the user would interact with next)

---

### Task 2.2 — Run Ablation Study Across Retrieval Configurations

**Goal**: Produce a comparison table showing the incremental contribution of each model component. This is the core model evaluation section of the thesis.

#### Ablation Configurations

| ID | Configuration | Description |
|---|---|---|
| **A** | BLaIR only | Semantic text retrieval, no visual, no reranking |
| **B** | CLIP only | Visual retrieval only |
| **C** | BLaIR + CLIP, RRF fusion (β=0.8) | Multimodal fusion, no reranking |
| **D** | BLaIR + CLIP + BGE Reranker | Full Mode 1 pipeline |
| **E** | Mode 2: Cleora + DQN | Passive recommendation pipeline |
| **F** | Full system (Mode 1 + Mode 2 combined) | Complete NBA system |

> **Note**: You already have results for A (29.6% Top-1) and C (66.5% Top-1) from the Phase 1 evaluation. You need to run D, E, and F fresh.

#### Expected Results Table (template to fill in)

| Config | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | MRR | Response time |
|---|---|---|---|---|---|---|
| A — BLaIR only | 29.6% | — | — | — | — | ~Xms |
| B — CLIP only | — | — | — | — | — | ~Xms |
| C — BLaIR+CLIP, RRF | 66.5% | — | — | — | — | ~Xms |
| D — Full Mode 1 (+Reranker) | **?** | **?** | **?** | **?** | **?** | **?** |
| E — Mode 2 (Cleora+DQN) | **?** | **?** | **?** | **?** | **?** | **?** |
| F — Full NBA system | **?** | **?** | **?** | **?** | **?** | **?** |

#### Implementation

```python
# evaluation/ablation_runner.py
import json
from typing import List, Dict

def evaluate_config(
    config_name: str,
    search_fn,            # callable: query -> List[asin]
    ground_truth: Dict,   # {query: [relevant_asins]}
    K_values: List[int] = [1, 5, 10]
) -> Dict:
    hits = {k: 0 for k in K_values}
    ndcg_scores = []
    reciprocal_ranks = []

    for query, relevant_asins in ground_truth.items():
        results = search_fn(query)  # returns list of ASINs, ranked

        for k in K_values:
            top_k = results[:k]
            if any(asin in relevant_asins for asin in top_k):
                hits[k] += 1

        # NDCG@5
        dcg = sum(
            1 / (i + 1) if asin in relevant_asins else 0  # simplified: binary relevance
            for i, asin in enumerate(results[:5])
        )
        idcg = sum(1 / (i + 1) for i in range(min(len(relevant_asins), 5)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        # MRR
        for i, asin in enumerate(results):
            if asin in relevant_asins:
                reciprocal_ranks.append(1 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0)

    n = len(ground_truth)
    return {
        "config": config_name,
        **{f"Hit@{k}": round(hits[k] / n * 100, 2) for k in K_values},
        "NDCG@5": round(sum(ndcg_scores) / n * 100, 2),
        "MRR": round(sum(reciprocal_ranks) / n * 100, 2),
    }
```

---

### Task 2.3 — Report DQN Training Convergence

**Goal**: Demonstrate that the RL agent actually learns — show loss convergence and a measurable improvement in recommendation quality before vs. after training.

**Files to touch**: `rl_collaborative_filter.py`, evaluation scripts

#### What to produce

**1. Loss convergence plot**

Run a longer simulation (200–500 interactions) using the existing simulation harness. Plot training step vs. MSE loss using the `loss_history` array already tracked by `RLCollaborativeFilter`.

```python
# evaluation/plot_dqn_convergence.py
import matplotlib.pyplot as plt
import json

with open("data/profiles/user_demo_01.json") as f:
    profile = json.load(f)

loss_history = profile['rl_metrics']['loss_history']  # or load from the /rl_metrics endpoint

plt.figure(figsize=(10, 4))
plt.plot(loss_history, color='steelblue', linewidth=1.2)
plt.xlabel("Training step")
plt.ylabel("MSE Loss")
plt.title("DQN Training Loss Convergence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/dqn_loss_convergence.pdf", dpi=150)
```

**2. Before/after recommendation quality comparison**

```python
# Compare Hit@5 before training (random DQN weights) vs. after 300 interactions
before_score = evaluate_config("DQN untrained", untrained_recommend_fn, ground_truth)
# ... run 300 simulated interactions ...
after_score  = evaluate_config("DQN trained (300 steps)", trained_recommend_fn, ground_truth)

print(f"Hit@5 improvement: {before_score['Hit@5']}% → {after_score['Hit@5']}%")
```

**3. Reward signal distribution**

Report the breakdown of rewards seen during the simulation:

| Action | Count | Proportion | Total reward |
|---|---|---|---|
| Click (+1.0) | — | — | — |
| Cart (+5.0) | — | — | — |
| Skip (0.0) | — | — | — |

This validates that the reward shaping is behaving as intended (cart actions dominating gradient updates).

---

## Phase 3 — Missing Metadata Fallback via Qwen

> **Advisor feedback**: *"If a column like author, genre, or description is missing — you can use Qwen to fetch/generate that data."*

---

### Task 3.1 — Audit Current Missing Metadata Rates

**Goal**: Quantify exactly how many items are missing each field before building any pipeline.

```python
# scripts/audit_metadata.py
import pandas as pd

df = pd.read_parquet("data/item_metadata.parquet")
print("=== Metadata Audit ===")
print(f"Total rows: {len(df):,}")
print()

for col in ['title', 'author_name', 'main_category', 'image_url', 'description']:
    missing = df[col].isnull().sum() + (df[col] == '').sum() + (df[col] == 'Unknown Author').sum()
    pct = missing / len(df) * 100
    print(f"{col:20s}: {missing:>10,} missing ({pct:.1f}%)")
```

Expected output (based on current docs):

```
title               :           ~0 missing (0.0%)
author_name         :       ~??,??? missing (??.?%)
main_category       :       ~??,??? missing (??.?%)
image_url           :       ~??,??? missing (??.?%)
description         :    912,226 missing (52.6%)
```

After running the audit, **prioritize by impact**:
1. `description` — highest impact (directly feeds BGE Reranker quality)
2. `author_name` — medium impact (displayed in UI, used in keyword scan)
3. `main_category` — lower impact (UI badge only)
4. `image_url` — Qwen cannot generate images; log these separately as "cover not available"

Also sample 50 items with missing descriptions to understand the pattern:

```python
missing_desc = df[df['description'].isnull() | (df['description'] == '')].sample(50)
print(missing_desc[['title', 'author_name', 'main_category']].to_string())
```

---

### Task 3.2 — Build Qwen-Powered Metadata Enrichment Pipeline

**Goal**: Use the existing `Qwen2.5-1.5B-Instruct` model (already loaded in `api.py`) to generate synthetic metadata for items missing key fields. This runs as an **offline batch job**, not at query time.

**Files to create**: `scripts/enrich_metadata_qwen.py`

#### Prompt Templates

```python
# scripts/enrich_metadata_qwen.py

DESCRIPTION_PROMPT = """You are a book cataloger. Write a concise 2-sentence description for the following book. Be factual and informative. Do not use phrases like "This book..." — start directly with the content.

Book title: {title}
Author: {author}

Description:"""

AUTHOR_PROMPT = """What is the author of the book titled "{title}"? 
Answer with ONLY the author's full name. If unknown, respond with exactly: Unknown Author"""

GENRE_PROMPT = """Classify the book "{title}" by {author} into exactly ONE of these genres:
Fiction, Non-fiction, Mystery, Fantasy, Science Fiction, Self-help, Romance, History, Biography, Children's, Other

Respond with ONLY the genre name, nothing else."""
```

#### Batch Processing Script

```python
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

BATCH_SIZE = 16
MAX_NEW_TOKENS = 80
OUTPUT_PATH = "data/item_metadata_enriched.parquet"

# Load model (reuse the same weights as api.py)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

df = pd.read_parquet("data/item_metadata.parquet")

# Add tracking columns
df['qwen_generated'] = False  # flag: True = Qwen-generated, False = original Amazon data

# ── Enrich descriptions ───────────────────────────────────────────────────
missing_desc_idx = df[(df['description'].isnull()) | (df['description'] == '')].index

print(f"Enriching {len(missing_desc_idx):,} missing descriptions...")

# Process top-N most frequently retrieved ASINs first (prioritize by query hit count)
# If you log FAISS lookups, sort by frequency here. Otherwise process sequentially.

for i in tqdm(range(0, len(missing_desc_idx), BATCH_SIZE)):
    batch_idx = missing_desc_idx[i:i+BATCH_SIZE]
    prompts = [
        DESCRIPTION_PROMPT.format(
            title=df.loc[idx, 'title'],
            author=df.loc[idx, 'author_name'] or "Unknown"
        )
        for idx in batch_idx
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    for j, idx in enumerate(batch_idx):
        generated = tokenizer.decode(outputs[j], skip_special_tokens=True)
        # Strip the prompt prefix from the output
        description = generated[len(prompts[j]):].strip()
        # Cap at 1000 characters (matching existing schema)
        df.loc[idx, 'description'] = description[:1000]
        df.loc[idx, 'qwen_generated'] = True

# Save enriched parquet
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved enriched metadata to {OUTPUT_PATH}")
```

#### Performance Estimate

| Field | Missing count | Est. rate (GPU) | Est. time |
|---|---|---|---|
| `description` | ~912,000 | ~15 items/sec | ~17 hours |
| `author_name` | TBD | ~20 items/sec | TBD |
| `main_category` | TBD | ~20 items/sec | TBD |

**Practical strategy**: Don't enrich all 912K at once. Enrich the top 50K–100K most-retrieved ASINs first (log FAISS lookup frequency from Task 1.1), then run the rest overnight.

#### Thesis Contribution Note

The `qwen_generated: bool` flag is important for academic honesty. In the thesis, report separately:
- Reranker quality on items with **original** Amazon descriptions
- Reranker quality on items with **Qwen-generated** descriptions

This is a valid and interesting comparison — it shows the enrichment is beneficial even if Qwen-generated text is slightly noisier.

---

### Task 3.3 — Add Runtime Graceful Degradation

**Goal**: Even before the enrichment batch job completes, the live system should handle missing fields cleanly — no placeholder strings visible to the user.

**Files to touch**: `api.py`, `active_search_engine.py`, `frontend/src/App.jsx`

#### Backend — Reranker Fallback (verify existing behavior)

Per the March 19 docs, the BGE Reranker already falls back to title-only when description is empty. Verify this is actually happening:

```python
# active_search_engine.py — BGEReranker input construction
def build_reranker_input(query: str, title: str, description: str) -> list:
    if description and len(description.strip()) > 0:
        doc_text = f"Title: {title}. Description: {description}"
    else:
        doc_text = f"Title: {title}"  # ← confirm this branch is hit
    return [[query, doc_text]]
```

Add a log line to confirm the fallback is firing as expected:

```python
if not description:
    logger.debug(f"Reranker title-only fallback for ASIN: {asin}")
```

#### Backend — Author / Genre Fallback

```python
# api.py — metadata hydration, clean up display values before returning to frontend
def clean_metadata(row: dict) -> dict:
    return {
        "title":       row.get('title', 'Unknown Title'),
        "author":      row['author_name'] if row.get('author_name') and row['author_name'] != 'Unknown Author' else None,
        "genre":       row['main_category'] if row.get('main_category') and row['main_category'] != 'Books' else None,
        "description": row.get('description') or None,
        "image_url":   row.get('image_url') or None,
    }
```

#### Frontend — Display Fallbacks (`App.jsx`)

```jsx
// BookCard component — handle null fields gracefully
const BookCard = ({ book }) => (
  <div className="book-card">
    {book.image_url
      ? <img src={book.image_url} alt={book.title} />
      : <div className="cover-placeholder">No cover available</div>
    }
    <h3>{book.title}</h3>
    {book.author
      ? <p className="author">{book.author}</p>
      : <p className="author muted">Author unknown</p>   // ← muted gray, not "Unknown Author"
    }
    {book.genre && <span className="genre-badge">{book.genre}</span>}  // ← hide entirely if null
    {book.description
      ? <p className="description">{book.description.slice(0, 300)}...</p>
      : null  // ← show nothing, not a placeholder string
    }
  </div>
)
```

#### Ask AI / RAG Fallback (Qwen)

If the Wikipedia lookup in `/ask_llm` returns no article, Qwen should answer from title alone with a disclaimer:

```python
# api.py — /ask_llm endpoint
if not wikipedia_context:
    prompt = f"""The book "{title}" by {author or "an unknown author"} has no Wikipedia article available.
Based on the title alone, provide a brief 2-sentence summary of what this book is likely about.
Begin your response with: "Based on the title alone: " """
else:
    prompt = f"""Context: {wikipedia_context}\n\nQuestion: Tell me about the book "{title}"."""
```

---

## Summary & Suggested Execution Order

```
Week 1
├── Day 1–2:  Task 1.1 — Profile pipeline, find bottleneck
├── Day 2–3:  Task 1.2 — Add similarity thresholds
├── Day 3–4:  Task 1.3 — Vietnamese search audit + fix
└── Day 4–5:  Task 1.4 — CLIP version audit (quick)

Week 2
├── Day 1:    Task 2.1 — Define evaluation protocol + build ground truth set
├── Day 2–4:  Task 2.2 — Run ablation study (A → F configs)
├── Day 4:    Task 2.3 — DQN convergence plots + before/after comparison
├── Day 4–5:  Task 3.1 — Audit missing metadata rates
└── Day 5:    Task 3.3 — Add frontend/backend graceful degradation (quick wins)

Week 3 (ongoing, can run in background)
└── Task 3.2 — Run Qwen enrichment batch job overnight on top-100K ASINs
```

---

## Open Issues (Not Covered by Advisor Feedback, but Worth Noting)

These are pre-existing gaps identified in the March 25 status review. They are lower priority than the advisor's feedback but should be documented in the thesis as future work:

| Issue | Description | Suggested mitigation |
|---|---|---|
| **Cold start** | New users have no click history → Cleora scouting returns empty seeds | Fall back to popularity-based recommendations (top-100 most interacted items globally) |
| **DQN training scale** | Only 35 interactions tested; real convergence needs 500+ | Run extended simulation (Task 2.3 addresses this) |
| **FAISS coverage gap** | ~270K items exist in embedding indices but have no metadata; currently silently dropped | Either enrich via Qwen (Task 3.2) or add a "data quality" field to the parquet to track intentionally excluded items |

---

*Last updated: March 31, 2026. This document maps directly to advisor feedback from March 25, 2026 and should be read alongside `Project_update_25-3.md` and `RL_UI_Inspection_update_25-3.md`.*