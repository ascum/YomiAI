# BGE-M3 Encoding Optimization Proposal

## Current State

After the translation pipeline improvements (2026-04-10), NLLB translation dropped from
the dominant bottleneck to under 1ms. BGE-M3 text encoding is now the single largest
component in the search pipeline:

```
Component                    Avg      P95      Max    % of total
─────────────────────────────────────────────────────────────────
NLLB Translation             1.2ms    1.6ms    1.7ms  < 1%
Text Encoding (BGE-M3)      34.4ms   42.5ms   42.9ms  ~61%     ← bottleneck
Tantivy Keyword Search       5.8ms   11.8ms   15.5ms  ~10%
FAISS Text Search            2.3ms    3.2ms    4.3ms
Metadata Parquet Lookup      2.5ms    4.2ms    6.7ms
─────────────────────────────────────────────────────────────────
Total API Internal          56.4ms   69.3ms   72.1ms
E2E Wall Clock              74.0ms   89.2ms   89.8ms
```

Source: `run_016 / post_translation_improvement` — 10 queries × 3 runs, warmup discarded.

---

## Root Cause Analysis

BGE-M3 (`BAAI/bge-m3`, 568M parameters) is loaded via `sentence_transformers` in
`app/core/models.py:load_text_encoder()`. Three inefficiencies compound:

### 1. Sequence length is uncapped at encode time

`SentenceTransformer.encode()` is called without `max_length`. The tokenizer uses
BGE-M3's default of **8192 tokens**. A typical search query is 3–15 tokens.

Transformer attention complexity is O(n²) in sequence length. Padding 10 real tokens
to 8192 means the attention matrix is ~500× larger than necessary. Even with padding
masks, the matrix allocation and kernel launch cost scales with `max_length`.

### 2. No graph compilation

The model runs in standard eager PyTorch mode. Each forward pass re-dispatches through
the Python→C++ bridge, re-allocates intermediate tensors, and re-plans the CUDA kernel
grid. `torch.compile()` fuses these into a static computation graph compiled once on
first call — subsequent calls skip all Python overhead.

### 3. No embedding cache

Every query, including popular ones that recur constantly ("fantasy novels",
"mystery books", "romance"), triggers a full forward pass. The translation module
gained a `@lru_cache` that reduced translation from 55ms to 1ms; the same pattern
applies here.

---

## Proposed Optimizations

Four independent steps, ordered by effort-to-impact ratio. Each is independently
testable and reversible.

---

### Step 1 — Cap `max_length` at 64 tokens

**File:** `app/core/models.py:encode_text()`

**Estimated gain:** 5–10ms off avg latency

Search queries in this system are short by nature — genre names, author names, short
descriptions. 64 tokens accommodates even the longest realistic query with room to spare.

```python
# app/core/models.py — current
def encode_text(text: str, text_encoder) -> np.ndarray | None:
    if text_encoder is None or not text.strip():
        return None
    try:
        vec = text_encoder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype("float32")
    except Exception as e:
        log.warning(f"Text encode failed: {e}")
        return None
```

```python
# app/core/models.py — proposed
_TEXT_ENCODE_MAX_LEN = 64   # search queries are ≤ 15 tokens; 64 is safe headroom

def encode_text(text: str, text_encoder) -> np.ndarray | None:
    if text_encoder is None or not text.strip():
        return None
    try:
        vec = text_encoder.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            max_length=_TEXT_ENCODE_MAX_LEN,
        )
        return vec.astype("float32")
    except Exception as e:
        log.warning(f"Text encode failed: {e}")
        return None
```

**Verification:** confirm embedding dim is still 1024, run cosine similarity spot-check
between uncapped and capped vectors on a 10-query sample — should be > 0.99 for all.

---

### Step 2 — `torch.compile()` on the underlying transformer

**File:** `app/core/models.py:load_text_encoder()`

**Estimated gain:** 3–7ms off avg latency (PyTorch 2.0+, CUDA path)

`torch.compile(mode="reduce-overhead")` traces the forward pass on first call and emits
a fused CUDA graph. Subsequent calls bypass Python dispatch entirely.

`SentenceTransformer` wraps a HuggingFace model at `model[0].auto_model`. That inner
module is what we compile — the `SentenceTransformer` wrapper itself stays in Python.

```python
# app/core/models.py — proposed addition inside load_text_encoder()
def load_text_encoder(device: torch.device):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            settings.TEXT_ENCODER_MODEL, device=str(device), trust_remote_code=True
        )
        if device.type == "cuda":
            model.half()
            # Compile the inner transformer for fused CUDA kernels.
            # mode="reduce-overhead" optimises for repeated same-shape inputs
            # (single short query), trading a longer first-call for faster steady-state.
            try:
                model[0].auto_model = torch.compile(
                    model[0].auto_model, mode="reduce-overhead"
                )
                log.info("BGE-M3 inner transformer compiled with torch.compile ✓")
            except Exception as e:
                log.warning(f"torch.compile skipped (non-fatal): {e}")
        log.info(
            f"Text encoder ready ✓  model={settings.TEXT_ENCODER_MODEL}  "
            f"dtype={'fp16' if device.type == 'cuda' else 'fp32'}"
        )
        return model
    except Exception as e:
        log.warning(f"Text encoder failed to load — text search will use proxy mode: {e}")
        return None
```

**Caveats:**
- First call after startup triggers a recompilation warm-up (~2–5s). Handle by running
  a warmup encode in `lifespan.py` alongside the NLLB warmup (same pattern already used).
- Requires PyTorch ≥ 2.0. Degrades gracefully: the `try/except` means it skips on
  older environments without breaking startup.
- `reduce-overhead` mode requires static shapes. Since all queries are padded to
  `max_length=64` (Step 1), this condition is met.

**Verification:** benchmark shows `encode_text_ms` avg dropped; embedding values
unchanged (compile is numerically identical).

---

### Step 3 — LRU embedding cache

**File:** `app/core/models.py`

**Estimated gain:** 0ms on cache hits (34ms → 0ms for repeat queries)

Identical pattern to the translation LRU cache. Popular search queries recur constantly
in a book recommendation system. A 4096-entry cache holds ~350 KB of float32 vectors
(4096 × 1024 × 4 bytes) — negligible memory footprint.

```python
# app/core/models.py — proposed

from functools import lru_cache
import numpy as np

@lru_cache(maxsize=4096)
def _cached_encode(text: str, encoder_id: str) -> tuple:
    """
    Cache wrapper around the text encoder. Keyed on (text, encoder_id) so the
    cache is automatically invalidated if the model is swapped in config.

    Returns a tuple (not ndarray) because lru_cache requires hashable return
    values — callers convert back with np.array(..., dtype='float32').
    """
    vec = _text_encoder_ref.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        max_length=_TEXT_ENCODE_MAX_LEN,
    )
    return tuple(vec.flatten().tolist())


# Module-level reference set by load_text_encoder(); avoids passing model as arg
# (non-hashable) while keeping lru_cache on the function.
_text_encoder_ref = None


def load_text_encoder(device: torch.device):
    global _text_encoder_ref
    # ... existing load logic ...
    _text_encoder_ref = model
    return model


def encode_text(text: str, text_encoder) -> np.ndarray | None:
    if text_encoder is None or not text.strip():
        return None
    try:
        result = _cached_encode(text, settings.TEXT_ENCODER_MODEL)
        return np.array(result, dtype="float32").reshape(1, -1)
    except Exception as e:
        log.warning(f"Text encode failed: {e}")
        return None
```

**Cache sizing rationale:**

| Cache size | Memory | Coverage |
|---|---|---|
| 1024 | ~87 MB | top-1024 recurring queries |
| 4096 | ~349 MB | top-4096 recurring queries |
| 8192 | ~698 MB | top-8192 recurring queries |

4096 is the recommended default. Increase if memory permits; decrease if VRAM is tight
(relevant on CPU-only deployments where RAM is also used for FAISS).

**Verification:** call `encode_text` twice with the same string — second call returns
in < 0.1ms. Call `_cached_encode.cache_info()` to inspect hit rate after a benchmark run.

---

### Step 4 — ONNX Runtime export (highest effort, highest ceiling)

**Files:** new `scripts/export/export_bge_onnx.py`, `app/core/models.py`

**Estimated gain:** 1.5–2× throughput over PyTorch (15–20ms avg instead of 34ms)

ONNX Runtime with `CUDAExecutionProvider` compiles the model to a highly optimised
CUDA graph at the ONNX level, with constant folding, operator fusion, and memory
planning that PyTorch's JIT cannot match.

**Export script outline:**

```python
# scripts/export/export_bge_onnx.py
from sentence_transformers import SentenceTransformer
from transformers.onnx import export
import torch

model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
dummy = torch.zeros(1, 64, dtype=torch.long)   # (batch=1, seq=64)

torch.onnx.export(
    model[0].auto_model,
    (dummy, dummy, dummy),                       # input_ids, attention_mask, token_type_ids
    "data/bge_m3.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
    opset_version=17,
)
```

**Inference loader outline:**

```python
# app/core/models.py — ONNX path (alternative to SentenceTransformer)
import onnxruntime as ort

def load_text_encoder_onnx(device: torch.device):
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.type == "cuda" else ["CPUExecutionProvider"])
    session = ort.InferenceSession("data/bge_m3.onnx", providers=providers)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return session, tokenizer
```

**Why this is Step 4, not Step 1:**

- Requires exporting and validating the ONNX graph (edge cases around pooling, normalization layers).
- `sentence_transformers` pooling + normalization logic lives outside the HuggingFace model and must be replicated in the ONNX wrapper or applied post-inference.
- Loses `torch.compile` compatibility (they are mutually exclusive paths).
- Must be re-exported any time the model weights are updated.

Recommended only if Steps 1–3 still leave BGE-M3 as the bottleneck after measurement.

---

## Expected Results

| Step | Effort | Avg encoding latency | P95 encoding latency |
|---|---|---|---|
| Baseline (current) | — | 34.4ms | 42.5ms |
| + Step 1 (max_length=64) | 10 min | ~26ms | ~33ms |
| + Step 2 (torch.compile) | 15 min | ~20ms | ~26ms |
| + Step 3 (LRU cache) | 45 min | **0ms** on hit | **0ms** on hit |
| + Step 4 (ONNX) | 2–4 hours | ~15ms | ~20ms |

Steps 1–3 on their own are expected to reduce total API avg from **56ms → ~35ms**
and P95 from **69ms → ~50ms** — all repeat queries cost near zero.

---

## What Is NOT Proposed

- **Replacing BGE-M3 with a smaller model** — BGE-M3 is the only model in this class
  that supports multilingual queries (VI, FR, ZH, JA, KO, …) at 1024-dim quality.
  Smaller models (`bge-small-en`, `MiniLM`) are English-only and would break the
  translation pipeline integration.
- **Quantization (INT8)** — FP16 is already in place on CUDA. INT8 requires calibration
  data and introduces non-trivial accuracy risk for semantic retrieval; not worth it
  while Steps 1–3 headroom remains.
- **Batching across requests** — adds queueing complexity and latency jitter; not
  appropriate for a low-concurrency thesis demo system.

---

## Step Status

| Step | Description | Status |
|---|---|---|
| 1 | Cap `max_length=64` in `encode_text()` | ⏳ Pending |
| 2 | `torch.compile()` on `model[0].auto_model` | ⏳ Pending |
| 3 | LRU embedding cache (maxsize=4096) | ⏳ Pending |
| 4 | ONNX Runtime export + loader | ⏳ Pending |

---

## Benchmark Cadence

Run after each step:

```bash
python scripts/benchmark/search_timing.py "bge_opt_step_N"
```

Compare against `run_016` (baseline after translation improvement) using:

```bash
python scripts/benchmark/search_timing.py --compare run_016
```

---

*Proposal written April 10, 2026. Baseline: run_016 / post_translation_improvement.*
