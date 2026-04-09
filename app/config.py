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

    # ── Multimodal Embeddings ─────────────────────────────────────────────────
    BLAIR_DIM: int = 1024
    CLIP_DIM: int = 512

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

    # ── BM25 Adaptive Search ──────────────────────────────────────────────────
    BM25_TOP_N: int = 30
    BM25_MIN_SCORE: float = 0.5
    BM25_HIGH_CONF: float = 0.6
    BM25_VISUAL_BOOST: float = 0.5


settings = Settings()
