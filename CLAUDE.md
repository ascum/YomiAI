# CLAUDE.md

## Stack

- **Backend**: Python 3.11, FastAPI, uvicorn
- **Frontend**: React 19, Vite 8, Tailwind CSS
- **Infra services**: MongoDB (logs), Redis (caching) — via Docker
- **ML**: PyTorch, FAISS (HNSW/flat indices), sentence-transformers, HuggingFace transformers
- **Search**: Tantivy BM25, NLLB translation for Vietnamese queries

---

## Run Commands

### Backend (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Vite + React)
```bash
cd frontend
npm install        # first time
npm run dev        # dev server at http://localhost:5173
npm run build      # production build → frontend/dist/
```

### Infrastructure (MongoDB + Redis)
```bash
docker-compose up -d
```

### Evaluation
```bash
# Main recommendation metrics (HR@10, NDCG@10)
python scripts/benchmark/evaluate_recommendation.py

# BLaIR vs BGE-M3 encoder comparison (sampled eval + query retrieval test)
python scripts/benchmark/compare_encoders.py --max-users 20000 --seed 42

# Encoder comparison — query test only, skip heavy steps
python scripts/benchmark/compare_encoders.py --no-latency --overlap-queries 50 --hnsw-queries 50 --max-users 1000
```

### Training / Setup
```bash
# Pre-process dataset and build eval_users.json
python scripts/setup_dif_sasrec.py
```

---

## Architecture Overview

The system is a **dual-mode multimodal book recommendation** API.

### Pipeline A — Behavioral + Semantic (proactive recommendations)
1. Cleora co-purchase graph embeddings narrow candidates to behaviorally related items (~375k item index).
2. BGE-M3 profile vector (mean of clicked item embeddings) re-ranks candidates by cosine similarity.

### Pipeline B — Sequential Intent (DIF-SASRec)
- DIF-SASRec transformer with decoupled content/category attention streams.
- Pretrained on 100k users; per-user online fine-tuning at inference time.
- Content veto (cosine threshold τ=0.3) filters semantically unrelated candidates.

### Final output
Union of top-10 from Pipeline A + Pipeline B, deduplicated (HR@10 = 0.7886).

### Text search (active query)
- Query → NLLB translation (if Vietnamese) → BGE-M3 encoding → FAISS HNSW search (3M items)
- Re-ranked by Tantivy BM25 via RRF fusion.

### Visual search
- CLIP HNSW index (~3M items), searched by image embedding.

### Key data files (`data/`)
| File | Contents |
|---|---|
| `bge_index_hnsw.faiss` | BGE-M3 HNSW (1.7M vectors, production search) |
| `bge_index_flat.faiss` | BGE-M3 flat (3M vectors, exact eval) |
| `blair_index_hnsw_legacy.faiss` | Legacy BLaIR HNSW (3M vectors) |
| `cleora_embeddings.npz` | Behavioral graph embeddings (375k items) |
| `clip_index_hnsw.faiss` | CLIP visual index |
| `item_metadata.parquet` | Title, author, genre, image URL (3M items) |
| `dif_sasrec_pretrained.pt` | Pretrained DIF-SASRec weights |

### App layout (`app/`)
```
app/
├── api/routes/      # FastAPI routers: search, recommend, interact, profile, llm, auth
├── core/            # lifespan (index loading), startup logic
├── infrastructure/  # FAISS, Tantivy, encoder wrappers
├── repository/      # metadata_repo (parquet), user profile repo
├── services/        # pipeline A, pipeline B, fusion logic
└── config.py        # all paths and model settings (pydantic Settings)
```
