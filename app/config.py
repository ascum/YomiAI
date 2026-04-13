"""
app/config.py — Typed settings for the NBA Recommendation System.

All constants previously scattered via `from config import *` are now
collected here as a Settings dataclass instance. Import with:

    from app.config import settings
    settings.TOP_K
"""
import os
from dataclasses import dataclass, field

# Project root is two levels above this file (app/config.py → app/ → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    # ── Paths ────────────────────────────────────────────────────────────────
    PROJECT_ROOT: str = _PROJECT_ROOT
    DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")

    # ── Model Names (single source of truth — change here to swap models) ────────
    TEXT_ENCODER_MODEL: str = "BAAI/bge-m3"
    CLIP_MODEL_NAME:    str = "openai/clip-vit-base-patch32"

    # ── Embedding Dimensions ──────────────────────────────────────────────────
    TEXT_EMBED_DIM: int = 1024   # BGE-M3 output dim
    CLIP_DIM:       int = 512

    # ── Index File Names (relative to DATA_DIR) ───────────────────────────────
    # File names encode which model built them; keys encode the role.
    # When you re-embed with a new model, point these to the new files.
    TEXT_INDEX_HNSW:        str = "bge_index_hnsw.faiss"       # BGE-M3, fast ANN
    TEXT_INDEX_FLAT:        str = "bge_index_flat.faiss"       # BGE-M3, exact + reconstruct
    TEXT_INDEX_HNSW_LEGACY: str = "blair_index_hnsw_legacy.faiss"  # BLaIR, fallback
    TEXT_INDEX_FLAT_LEGACY: str = "blair_index_flat_legacy.faiss"  # BLaIR, last resort
    CLIP_INDEX_HNSW:        str = "clip_index_hnsw.faiss"
    CLIP_INDEX_FLAT:        str = "clip_index.faiss"

    # ── Keyword Index ─────────────────────────────────────────────────────────
    KEYWORD_INDEX_DIR: str = "tantivy_index"

    # ── Retrieval & Ranking ───────────────────────────────────────────────────
    TOP_K: int = 10
    BEHAVIORAL_CANDIDATES: int = 50
    RRF_K: int = 60

    # ── Content Sanity Check ──────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.3

    # ── Simulation ────────────────────────────────────────────────────────────
    STEPS: int = 2000

    # ── User Profile Management ───────────────────────────────────────────────
    TEMPORAL_DECAY: float = 0.1
    MAX_RECENT_INTERACTIONS: int = 50
    COLD_START_THRESHOLD: int = 5
    WARM_USER_THRESHOLD: int = 20

    # ── RL / DQN Hyperparameters ──────────────────────────────────────────────
    REPLAY_BUFFER_SIZE: int = 5000
    REPLAY_BATCH_SIZE: int = 32
    TARGET_NET_UPDATE_FREQ: int = 50
    TARGET_NET_TAU: float = 0.01
    RL_GAMMA: float = 0.95
    RL_ITEM_PROJ_DIM: int = 256

    # ── GRU-Sequential DQN ────────────────────────────────────────────────────
    SEQ_ITEM_PROJ_DIM: int = 512
    GRU_HIDDEN_DIM: int = 512
    GRU_NUM_LAYERS: int = 3
    GRU_DROPOUT: float = 0.2
    MAX_SEQ_LEN: int = 50

    # ── ε-greedy exploration ──────────────────────────────────────────────────
    EPSILON_START: float = 0.3
    EPSILON_END: float = 0.05
    EPSILON_DECAY_STEPS: int = 200
    NEG_SAMPLE_SIZE: int = 10

    # ── Keyword Search Tuning (Tantivy/BM25 parameters) ──────────────────────
    BM25_TOP_N: int = 30
    BM25_MIN_SCORE: float = 0.5
    BM25_HIGH_CONF: float = 0.6
    BM25_VISUAL_BOOST: float = 0.5

    # ── DIF-SASRec (Personal Pipeline — "You Might Like") ────────────────────
    # Replaces GRU-SeqDQN for the personal "You Might Like" tab.
    # Uses decoupled category attention to model individual user taste evolution.
    # Candidates come from HNSW BGE-M3 index — zero Cleora dependency.
    SASREC_HIDDEN_DIM:     int   = 512    # model internal dimension (projects 1024→512)
    SASREC_N_BLOCKS:       int   = 4      # DIF-attention transformer layers
    SASREC_N_HEADS:        int   = 8      # attention heads (head_dim = 512/8 = 64)
    SASREC_DROPOUT:        float = 0.2    # dropout in attention and FFN
    SASREC_LR:             float = 1e-3   # peak learning rate (cosine schedule)
    SASREC_WEIGHT_DECAY:   float = 0.01   # AdamW weight decay
    SASREC_WARMUP_EPOCHS:  int   = 2      # linear LR warmup before cosine decay
    SASREC_ALPHA_INIT:     float = 0.7    # initial content vs category attention balance
    SASREC_CAT_AUX_WEIGHT: float = 0.1   # category prediction auxiliary loss weight
    SASREC_NUM_NEGATIVES:  int   = 512    # sampled softmax negatives per step
    PERSONAL_CANDIDATES:   int   = 200    # HNSW KNN retrieval count (no Cleora)


settings = Settings()
