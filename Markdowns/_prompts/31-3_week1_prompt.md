# Agent Prompt — Week 1: Performance & Accuracy Fixes
# NBA Multimodal Book Recommendation System (DATN Thesis)

---

## Your Role

You are a senior ML systems engineer helping improve a thesis project — an NBA (Next Best Action) multimodal book recommendation system. You have full access to the local codebase, terminal, and can run Python scripts. The FastAPI backend is fully set up and runnable.

Your job is to complete **4 sequential tasks** this session. Each task builds on the findings of the previous one. Work methodically: **read before you edit, verify after every change, never guess at file structure**.

---

## Project Architecture (Read This First)

Before touching any file, read the following source files in order to ground yourself:

1. `api.py` — FastAPI routes: `/search`, `/recommend`, `/interact`, `/ask_llm`, `/rl_metrics`
2. `src/active_search_engine.py` — Mode 1: keyword scan → BLaIR/CLIP FAISS → RRF → BGE Reranker
3. `src/retriever.py` — FAISS index wrapper (BLaIR 1024-dim, CLIP 512-dim, Cleora 1024-dim)
4. `src/user_profile_manager.py` — exponential-decay interest fingerprint
5. `src/rl_collaborative_filter.py` — PyTorch DQN
6. `src/passive_recommendation_engine.py` — Mode 2: 3-layer recommendation funnel

Key data facts:
- `item_metadata.parquet`: 1,732,910 rows, 6 columns (`parent_asin`, `title`, `author_name`, `main_category`, `image_url`, `description`)
- Description fill rate: 47.36% (820,684 populated)
- FAISS indices cover ~3M items; metadata covers ~1.73M → ~1.27M are "ghost" items (no metadata, already filtered in active search as of March 25)
- BGE Reranker: `BAAI/bge-reranker-v2-m3`, cross-encoder, re-scores top-20 candidates
- CLIP model: `openai/clip-vit-base-patch32`, 512-dim
- BLaIR model: `hyp1231/blair-roberta-large`, 1024-dim

---

## Task 1.1 — Profile End-to-End Response Time

**Goal**: Instrument the full Mode 1 search pipeline and identify which stage is the bottleneck. Target: total response time < 1 second.

### Step 1 — Read and understand the search pipeline

Read `api.py` and `src/active_search_engine.py` fully. Map out the exact call chain from when a POST `/search` request arrives to when the JSON response is returned. Note the order of: encoding → FAISS search → RRF/bypass → metadata filter → BGE reranker → response assembly.

### Step 2 — Add timing instrumentation to `api.py`

Add `time.perf_counter()` checkpoints at every major stage boundary inside the `/search` endpoint. Store results in a `_timings` dict. Append it to the API response under a `"_debug_timings"` key (you can make this conditional on a `?debug=true` query parameter if you prefer clean production responses).

The stages to measure individually:
- Text encoding (BLaIR)
- Image encoding (CLIP) — record 0ms if no image provided
- FAISS search execution (both indices combined)
- RRF fusion OR single-modality bypass
- Metadata filter + keyword injection
- BGE Reranker cross-encoding
- Metadata hydration (parquet lookup)
- Response serialization

All timing values must be in **milliseconds**, rounded to 1 decimal place.

### Step 3 — Create a benchmark script

Create `scripts/benchmark_search.py`. This script should:

1. Start the FastAPI server as a subprocess (or connect to it if already running on `localhost:8000`)
2. Run the following test queries and record `_debug_timings` for each:
   - Text-only, short query: `"jojo's bizarre adventure"` (triggers keyword scan)
   - Text-only, long query: `"gritty detective novels set in noir cities"` (pure dense retrieval)
   - Text-only, Vietnamese: `"tiểu thuyết trinh thám"`
   - Image-only: use any small book cover JPG from the project assets or generate a 224x224 gray placeholder
   - Hybrid text + image: `"dark fantasy"` + same image
3. Print a formatted timing table per query showing each stage in ms and the total
4. Print a summary: average total latency across all queries, and flag any stage averaging > 200ms as a **bottleneck**

Run the script and paste the output into a new file: `profiling/benchmark_results_<date>.txt`

### Step 4 — Investigate and fix the bottleneck

Based on the benchmark output, apply the appropriate fix:

**If FAISS search > 150ms:**
- Check whether indices are loaded into RAM at startup or re-read per request. They must be held in `_state` or a module-level variable.
- If using `IndexFlatIP` on 3M vectors: consider switching to `IndexIVFFlat`. Document the tradeoff (exact vs approximate) and implement only if the current index is confirmed as the bottleneck.

**If BGE Reranker > 400ms (most likely):**
- Check whether it runs on CPU or GPU. Log the device.
- Reduce the number of candidates passed to the reranker from 20 → 10 if quality impact is acceptable.
- Implement an LRU cache for reranker results keyed on `(query_text, tuple(sorted(candidate_asins)))` using `functools.lru_cache` or a manual dict with max size 256.

**If BLaIR/CLIP encoding > 150ms:**
- Check whether the model is in eval mode (`model.eval()`) and whether `torch.no_grad()` is used during inference. Add both if missing.

After applying fixes, re-run `scripts/benchmark_search.py` and save the new results as `profiling/benchmark_results_<date>_after_fix.txt`. Report the before/after comparison.

**Deliverables for Task 1.1:**
- [ ] Timing instrumentation added to `api.py`
- [ ] `scripts/benchmark_search.py` created and runnable
- [ ] `profiling/benchmark_results_*.txt` files saved (before and after)
- [ ] At least one concrete fix applied if any stage exceeded its budget
- [ ] Brief summary comment in `api.py` near the `/search` route noting the bottleneck finding

---

## Task 1.2 — Add Similarity Threshold to Block Low-Confidence Results

**Goal**: Add configurable score thresholds that drop weak candidates before they reach the user. Return a clean "no results" state rather than low-quality matches.

### Step 1 — Understand the current score structure

Read `src/active_search_engine.py` carefully. Understand:
- What score values the RRF fusion produces and their typical range
- What score values the BGE Reranker produces and their range
- Where in the code results are currently passed back to `api.py`

Print 10 example score values from a live query to understand the real distribution before setting any threshold.

### Step 2 — Add the threshold filter in `active_search_engine.py`

Add two configurable constants near the top of the file:

```python
# Similarity thresholds — tune based on profiling data
DENSE_SCORE_THRESHOLD = 0.25    # post-RRF or post-single-modality score (cosine sim range)
RERANKER_SCORE_THRESHOLD = 0.30  # BGE cross-encoder output (0.0–1.0 range)
```

Apply `DENSE_SCORE_THRESHOLD` after the fusion/bypass step and before the metadata filter. Apply `RERANKER_SCORE_THRESHOLD` after the BGE Reranker returns scores.

Add a logging line each time the threshold drops a candidate:

```python
import logging
logger = logging.getLogger(__name__)
# ...
n_before = len(dense_ranking)
dense_ranking = [item for item in dense_ranking if item[1].get('score', 0) >= DENSE_SCORE_THRESHOLD]
logger.debug(f"Dense threshold dropped {n_before - len(dense_ranking)}/{n_before} candidates")
```

### Step 3 — Update `api.py` to handle the empty results case

After the full search pipeline completes, if the result list is empty, return a structured response:

```python
if not results:
    return {
        "results": [],
        "query": req.query,
        "message": "No sufficiently relevant results found. Try rephrasing your query or using different keywords.",
        "total": 0
    }
```

### Step 4 — Verify threshold behavior

Run these test cases and confirm behavior is correct:

1. A normal, clear English query → results returned as usual
2. A nonsense query (`"xzqwerty blorp"`) → empty results with the message
3. A Vietnamese query (`"tiểu thuyết trinh thám"`) → observe whether threshold fires more aggressively than for English (it likely will — document this)

Record what thresholds cause the Vietnamese query to return zero results vs. at least some results. This informs Task 1.3.

**Deliverables for Task 1.2:**
- [ ] `DENSE_SCORE_THRESHOLD` and `RERANKER_SCORE_THRESHOLD` constants added to `active_search_engine.py`
- [ ] Threshold filter applied at both the dense and reranker stages
- [ ] Empty results response added to `api.py`
- [ ] Test cases verified and behavior documented in a comment block near the constants

---

## Task 1.3 — Audit and Fix Vietnamese Search Accuracy

**Goal**: Determine whether BLaIR degrades on Vietnamese input, quantify the degradation, and apply a targeted fix.

### Step 1 — Create the Vietnamese test set

Create `evaluation/vi_test_queries.py` with the following ground truth:

```python
VI_TEST_QUERIES = [
    {
        "query": "tiểu thuyết trinh thám",
        "expected_genres": ["mystery", "detective", "crime", "thriller"],
        "note": "detective/mystery novels"
    },
    {
        "query": "sách phát triển bản thân",
        "expected_genres": ["self-help", "personal development", "motivation", "habits"],
        "note": "self-help books"
    },
    {
        "query": "fantasy phép thuật",
        "expected_genres": ["fantasy", "magic", "wizard", "sorcery"],
        "note": "fantasy/magic books"
    },
    {
        "query": "lịch sử thế chiến thứ hai",
        "expected_genres": ["world war", "history", "ww2", "military"],
        "note": "World War II history"
    },
    {
        "query": "khoa học vũ trụ",
        "expected_genres": ["astronomy", "space", "physics", "cosmos"],
        "note": "astronomy/space science"
    },
]

# Equivalent English queries for comparison
EN_EQUIVALENT_QUERIES = [
    "detective mystery novels",
    "self-help personal development books",
    "fantasy magic wizard novels",
    "world war two history",
    "astronomy space science",
]
```

### Step 2 — Run the diagnostic

Create `scripts/audit_vietnamese_search.py`. For each Vietnamese query and its English equivalent:

1. Encode both with BLaIR
2. Search the FAISS BLaIR index, retrieve top-10
3. Look up titles from `item_metadata.parquet`
4. Record:
   - The top-10 titles for each query
   - The cosine similarity scores for each result
   - The **mean score** for Vietnamese vs. English
   - Whether any result title/category matches the `expected_genres` (simple substring check, case-insensitive)

Print a comparison table:

```
Query pair: "tiểu thuyết trinh thám" vs "detective mystery novels"
  VI mean score: 0.21  |  EN mean score: 0.38  |  Delta: -0.17
  VI genre hit rate (top-10): 2/10  |  EN genre hit rate: 7/10
```

Save output to `profiling/vi_search_audit.txt`.

### Step 3 — Apply the fix based on findings

**If mean VI score is < 0.30 or genre hit rate is < 3/10** (clear degradation): implement translation fallback.

Install the required packages if not already present:
```bash
pip install langdetect
pip install transformers sentencepiece
```

Add a language detection + translation utility to `src/utils.py` (create this file if it doesn't exist):

```python
# src/utils.py
from langdetect import detect, LangDetectException

_vi_en_tokenizer = None
_vi_en_model = None

def get_vi_en_translator():
    global _vi_en_tokenizer, _vi_en_model
    if _vi_en_tokenizer is None:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        _vi_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
        _vi_en_model = MarianMTModel.from_pretrained(model_name)
    return _vi_en_tokenizer, _vi_en_model

def translate_vi_to_en(text: str) -> str:
    """Translate Vietnamese text to English. Returns original if not Vietnamese or on error."""
    try:
        lang = detect(text)
        if lang != 'vi':
            return text
        tokenizer, model = get_vi_en_translator()
        import torch
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except (LangDetectException, Exception):
        return text  # always fall back to original
```

In `src/active_search_engine.py`, import and apply translation at the top of the `search()` method, before BLaIR encoding:

```python
from src.utils import translate_vi_to_en

def search(self, user_id, text_query=None, image_base64=None):
    if text_query:
        text_query_for_encoding = translate_vi_to_en(text_query)
        # Keep original for keyword scan (substring match works better in native language)
        keyword_scan_query = text_query  # unchanged
    ...
    # Use text_query_for_encoding for BLaIR encoding
    # Use keyword_scan_query for _keyword_scan()
```

**If degradation is mild (mean VI score 0.28–0.35)**: skip translation, instead lower `DENSE_SCORE_THRESHOLD` specifically for detected Vietnamese queries:

```python
from langdetect import detect
lang = detect(text_query) if text_query else 'en'
threshold = 0.18 if lang == 'vi' else DENSE_SCORE_THRESHOLD
```

Document which branch you took and why in a comment block.

### Step 4 — Re-run the audit after the fix

Run `scripts/audit_vietnamese_search.py` again and compare before/after. Save to `profiling/vi_search_audit_after_fix.txt`.

**Deliverables for Task 1.3:**
- [ ] `evaluation/vi_test_queries.py` created
- [ ] `scripts/audit_vietnamese_search.py` created and run
- [ ] `profiling/vi_search_audit.txt` (before) and `vi_search_audit_after_fix.txt` (after) saved
- [ ] Fix applied (translation fallback OR adjusted threshold) with reasoning documented in code comments
- [ ] Before/after comparison printed and meaningful improvement shown

---

## Task 1.4 — Verify CLIP Version and Evaluate Upgrade Options

**Goal**: Confirm the current CLIP model version, run a quick quality audit on real book cover images, and make a documented go/no-go decision on upgrading to a stronger variant.

### Step 1 — Confirm what's actually running

Read `api.py` and wherever CLIP is initialized. Print the exact model name string, embedding dimension, patch size, and whether it runs on CPU or GPU. Confirm it matches `openai/clip-vit-base-patch32` (512-dim).

Add a startup log line in `api.py` if not already present:

```python
logger.info(f"CLIP model loaded: {clip_model_name} | dim={clip_dim} | device={clip_device}")
```

### Step 2 — Run a visual quality audit

Create `scripts/audit_clip_quality.py`. The script must:

1. Pick 5 test images. Use real book cover images from your project assets if available. If not, download 5 small representative images (a dark noir cover, a bright children's book, a sci-fi cover, a romance cover, a history textbook). Save them to `evaluation/test_covers/`.

2. For each test image:
   - Encode with the current CLIP model
   - Search the FAISS CLIP index for top-5 visual nearest neighbors
   - Look up their titles and `main_category` from `item_metadata.parquet`
   - Print results with similarity scores

3. Manually assess: do the top-5 results look visually similar to the input? (Dark/moody books for noir input, colorful for children's, etc.)

4. Record the **mean similarity score** across top-5 for each test image.

Save output to `profiling/clip_audit.txt`.

### Step 3 — Research upgrade candidates (use web search)

Search for current benchmarks comparing these CLIP variants on image-text retrieval tasks:
- `openai/clip-vit-base-patch32` (current)
- `openai/clip-vit-large-patch14`
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

Find: zero-shot retrieval recall numbers on standard benchmarks (MS-COCO, Flickr30k). Note the relative improvement of ViT-L over ViT-B/32.

### Step 4 — Make and document the upgrade decision

Create `profiling/clip_upgrade_decision.md` with this structure:

```markdown
# CLIP Upgrade Decision — <date>

## Current model
- Name: openai/clip-vit-base-patch32
- Embedding dim: 512
- Device: <GPU/CPU>
- Mean visual similarity score across test covers: <X.XX>

## Quality audit findings
<Paste the audit results — do the top-5 results make visual sense?>

## Upgrade candidate benchmarks
| Model | Dim | Flickr30k R@1 | MS-COCO R@1 | Relative gain vs current |
|---|---|---|---|---|
| clip-vit-base-patch32 (current) | 512 | ? | ? | baseline |
| clip-vit-large-patch14 | 768 | ? | ? | +?% |
| CLIP-ViT-H-14 (OpenCLIP) | 1024 | ? | ? | +?% |

## Upgrade cost (if proceeding)
- Re-embed ~3M items → estimated time: <X> hours
- Rebuild FAISS CLIP index
- Update DQN input dim: 1024 (BLaIR) + <new_dim> = <new_total>
- Update all 90 NPZ chunks

## Decision
**GO / NO-GO**

Reason: <1–2 sentences. Key factors: quality gap from audit, compute cost, thesis timeline.>
```

**Decision heuristic**: If the current model's top-5 results are visually coherent (right aesthetic genre) for 4+ out of 5 test images AND mean similarity > 0.25 → **NO-GO** (document and move on). Only proceed with upgrade if the baseline is clearly broken (random-looking results, mean similarity < 0.15).

If GO: do not implement the upgrade in this session. Create a separate task card and schedule it. The re-embedding job takes hours and should not block Week 1 work.

**Deliverables for Task 1.4:**
- [ ] Exact CLIP model name, dim, and device confirmed and logged
- [ ] `scripts/audit_clip_quality.py` created and run
- [ ] `profiling/clip_audit.txt` saved
- [ ] `profiling/clip_upgrade_decision.md` written with GO/NO-GO decision and full rationale

---

## End-of-Session Checklist

When all 4 tasks are complete, do the following before closing:

1. **Run the full test suite** (if one exists). If not, manually hit the `/search` endpoint with these 3 queries and confirm all return valid JSON without errors:
   - `POST /search` with `{"query": "jojo's bizarre adventure", "user_id": "test_user"}`
   - `POST /search` with `{"query": "tiểu thuyết trinh thám", "user_id": "test_user"}`
   - `POST /search` with `{"query": "xzqwerty blorp nonsense", "user_id": "test_user"}` → must return empty results with message

2. **Commit all changes** with a clear commit message:
   ```
   feat(week1): performance profiling, similarity thresholds, VI search fix, CLIP audit
   
   - Task 1.1: Added timing instrumentation to /search, benchmark script
   - Task 1.2: Dense + reranker score thresholds, empty results handling
   - Task 1.3: Vietnamese query audit + translation fallback (or threshold adjustment)
   - Task 1.4: CLIP quality audit + GO/NO-GO upgrade decision
   ```

3. **Write a brief session summary** to `profiling/week1_summary.md`:
   - What was the bottleneck found in Task 1.1? What was fixed?
   - What thresholds were set in Task 1.2?
   - What was the Vietnamese search finding and what fix was applied?
   - What was the CLIP upgrade decision?
   - Any surprises, unexpected issues, or follow-up items

---

## Important Constraints

- **Read before writing**: Always read the full relevant file before making any edits. Never assume file structure from the architecture description alone.
- **Verify after every change**: After each code edit, either run the affected script or hit the relevant API endpoint to confirm nothing is broken before moving to the next task.
- **Do not re-architect**: The goal is instrumentation and targeted fixes, not rewrites. If you find a major structural issue, document it in `profiling/week1_summary.md` and leave it for a separate session.
- **Preserve existing behavior**: The `_keyword_scan()` logic, RRF bypass, metadata filter, and cart reward (+5.0) implemented on March 22–25 must remain intact. Do not accidentally revert these.
- **Log, don't print**: Use `logger.info()` / `logger.debug()` in production code paths. Use `print()` only in standalone scripts under `scripts/` and `evaluation/`.