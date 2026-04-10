# Translation Pipeline Improvement Proposal

## Current State

```
Component          Avg     P95
─────────────────────────────────
NLLB Translation   55ms    274ms   ← dominant bottleneck
BGE-M3 Encoding    24ms     27ms
FAISS HNSW Search   2ms      3ms
Tantivy BM25        6ms     12ms
─────────────────────────────────
Total API          95ms    305ms
```

Translation accounts for **58% of average latency** and **90% of P95**.

---

## Root Cause Analysis

### 1. ~~Language detection is unreliable on short queries~~ — RESOLVED (2026-04-10)

`langdetect` (the original detector) was probabilistic, non-deterministic, and failed
frequently on queries shorter than 20 characters. It has been replaced with **lingua**,
a deterministic, high-accuracy detector based on a trained statistical model.

Benchmark run 2026-04-10 (`scripts/benchmark/language_detection.py`):

| Metric | lingua | langdetect |
|---|---|---|
| Accuracy (23-case corpus) | **95.7%** (22/23) | 78.3% (18/23) |
| Load time | **0 ms** | 478 ms |
| Per-call latency | **0.40 ms** | 1.13 ms |
| Throughput | **2 519 calls/s** | 885 calls/s |
| Memory delta | **+0 MB** | +59 MB |

Lingua wins on every dimension. The one miss (`"best fantasy novels 2024"` → `da`)
is a short ambiguous phrase; Danish would pass through untranslated (correct behaviour).

> **fasttext (`lid.176.ftz`)** was also evaluated as part of the original proposal.
> Lingua matches or beats it on accuracy for the query lengths this system sees,
> with zero additional download (fasttext requires a 917KB model file). Lingua stays.

### 2. Model is oversized AND locked to Vietnamese only

```python
# translation.py — current
NLLB_MODEL_ID = "facebook/nllb-200-distilled-1.3B"   # 2.5 GB on disk

def translate_vi_to_en(text: str) -> str:  # ← VI only; FR/DE/ZH get no translation
    if detect_language(text) != "vi":
        return text
    # tokenizer called without setting src_lang → NLLB uses its default
```

`nllb-200-distilled-1.3B` supports 200 languages but the pipeline only ever handles
one. Users typing French, Korean, or Japanese get their query passed through
untranslated, degrading search quality silently.

### 3. No caching

Popular queries translate from scratch every time.

### 4. No startup warmup

First inference triggers CUDA kernel compilation → P95 spike to **274ms**.

---

## Proposed Architecture

```
User query
    │
    ▼
┌─────────────────────────────┐
│  lingua language detection  │  < 1ms  (already in place ✓)
│  → ISO 639-1 code           │
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │ English?           │ Non-English?
    ▼                    ▼
  passthrough     ┌─────────────────────────────┐
  (0ms)           │  LRU cache lookup           │  0ms on hit
                  │  → NLLB-600M translate      │  ~30ms on miss
                  │  → cache result             │
                  └─────────────────────────────┘
                              │
                              ▼
                      English query → BGE-M3
```

---

## Implementation Steps

### Step 1 — Replace language detector ✅ DONE (2026-04-10)

**lingua** is already in `app/infrastructure/translation.py`.
Benchmark confirms it is the correct choice.
No further action required on this step.

---

### Step 2 — Generalize translation to any non-English language

**Files:** `app/infrastructure/translation.py`, `app/api/routes/search.py`

Add `_ISO_TO_NLLB` mapping and replace the VI-only guard with a dynamic lookup:

```python
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"   # Step 3 change

_ISO_TO_NLLB = {
    "vi": "vie_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "hi": "hin_Deva",
    "sv": "swe_Latn",
}

def translate_to_en(text: str) -> str:
    """Translate any supported non-English language to English."""
    lang = detect_language(text)
    if lang == "en":
        return text
    nllb_code = _ISO_TO_NLLB.get(lang)
    if nllb_code is None:
        log.debug(f"[NLLB] Unsupported lang '{lang}' — passing through untranslated")
        return text
    return _cached_translate(text, nllb_code)
```

Update call site in `app/api/routes/search.py`:
`translate_vi_to_en` → `translate_to_en`

**Verification:** call `translate_to_en` manually with VI, FR, JA strings,
confirm each returns coherent English.

---

### Step 3 — Downgrade to NLLB-600M

**Files:** `app/infrastructure/translation.py`

```python
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"   # was: 1.3B
```

The 600M distilled model covers the same 200 languages at roughly half the
inference time and half the disk footprint (~1.2 GB vs ~2.5 GB). Quality is
sufficient for short search query translation.

Model downloads automatically from HuggingFace on first startup.
Pre-download with: `huggingface-cli download facebook/nllb-200-distilled-600M`

**Verification:** server starts, log shows `[NLLB] Translation model ready ✓`.

---

### Step 4 — Add LRU cache

**Files:** `app/infrastructure/translation.py`

```python
from functools import lru_cache

@lru_cache(maxsize=2048)
def _cached_translate(text: str, nllb_src_lang: str) -> str:
    """Calls NLLB. Result is cached — repeat queries are free."""
    _nllb_tokenizer.src_lang = nllb_src_lang
    inputs = _nllb_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(_nllb_device)
    forced_bos = _nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
    with torch.no_grad():
        out = _nllb_model.generate(
            **inputs, forced_bos_token_id=forced_bos,
            num_beams=1, max_new_tokens=64,
        )
    return _nllb_tokenizer.decode(out[0], skip_special_tokens=True)
```

**Verification:** call `translate_to_en("sách hay")` twice — second call
returns in < 0.1ms.

---

### Step 5 — Add startup warmup

**Files:** `app/core/lifespan.py`

Add after the NLLB load block (currently lazy — move to eager for the warmup):

```python
log.info("Warming up translation model (2 language paths)...")
from app.infrastructure.translation import translate_to_en
translate_to_en("sách hay")        # VI path — forces CUDA kernel compile
translate_to_en("livre de magie")  # FR path — second NLLB src_lang path
log.info("Translation warmup done ✓")
```

Eliminates the 274ms cold-start P95 spike.

**Verification:** restart server, run `scripts/benchmark/search_timing.py`
immediately — P95 should be < 40ms instead of 274ms.

---

### Step 6 — Benchmark and commit

```bash
python scripts/benchmark/search_timing.py "post_translation_improvement"
```

Compare `profiling/benchmark_post_*.json` against baseline. Commit if targets met.

---

## Expected Results

| Metric | Current | After Steps 2–5 |
|---|---|---|
| Supported languages | 1 (VI only) | 19+ (extendable to 200) |
| Language detection accuracy | **95.7%** (lingua, done ✓) | 95.7% (no change) |
| Translation latency (cold) | ~55ms avg / 274ms P95 | ~30ms avg / ~35ms P95 |
| Translation latency (cached repeat) | ~55ms | < 1ms |
| P95 latency (after warmup) | 274ms | ~35ms |
| Avg total API latency | 95ms | ~50ms |
| Model disk size | ~2.5 GB | ~1.2 GB |

---

## Step Status Summary

| Step | Description | Status |
|---|---|---|
| 1 | Replace language detector with lingua | ✅ Done — 2026-04-10 |
| 2 | Generalize to 19 languages, rename function | ✅ Done — 2026-04-10 |
| 3 | Downgrade NLLB 1.3B → 600M | ✅ Done — 2026-04-10 |
| 4 | LRU cache (maxsize=2048) | ✅ Done — 2026-04-10 |
| 5 | Startup warmup (eliminate P95 spike) | ✅ Done — 2026-04-10 |
| 6 | Benchmark + commit | ✅ Done — 2026-04-10 (run_016) |

Steps 2–5 touch **two files** (`translation.py`, `lifespan.py`) and one call site
(`search.py`). Each step is independently testable and reversible.

---

## What is NOT proposed

- **Removing NLLB** — it is the right tool; we are keeping it, just downsizing and generalising it.
- **Per-language specialist models** — simpler to maintain one model than 19 pipelines.
- **Client-side language detection** — adds frontend complexity; server-side is more robust.
- **Quantization (INT8)** — adds complexity; NLLB-600M at fp16 already meets the latency target.

---

## Periodic Benchmark Cadence

Run `scripts/benchmark/search_timing.py` at these trigger points:

| When | Why |
|---|---|
| After any translation change | Catch regressions before merge |
| After server hardware change | GPU upgrade/downgrade affects all timings |
| After BGE-M3 or NLLB model update | New weights can shift latency |
| Weekly during active development | Catch drift from unrelated changes |
| Before thesis demo / evaluation | Confirm the system is in peak shape |

```bash
python scripts/benchmark/search_timing.py "stage_name_here"
```

Results accumulate in `profiling/`. To compare two runs:

```bash
python -c "
import json, glob
files = sorted(glob.glob('profiling/*.json'))[-2:]
for f in files:
    d = json.load(open(f))
    t = d['results']
    print(f\"{d['metadata']['stage']:35}  translate={t.get('translate_ms',{}).get('avg','?')}ms  total={t.get('total_ms',{}).get('avg','?')}ms\")
"
```
