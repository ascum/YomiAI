# Multimodal Embeddings
BLAIR_DIM = 1024
CLIP_DIM = 512

# Retrieval & Ranking
TOP_K = 10
BEHAVIORAL_CANDIDATES = 50
RRF_K = 60

# Content Sanity Check
SIMILARITY_THRESHOLD = 0.3

# Simulation Parameters
STEPS = 2000

# User Profile Management
TEMPORAL_DECAY = 0.1
MAX_RECENT_INTERACTIONS = 50
COLD_START_THRESHOLD = 5
WARM_USER_THRESHOLD = 20

# ─── RL / DQN Hyperparameters ────────────────────────────────────────────────
# Experience replay
REPLAY_BUFFER_SIZE  = 5000   # max transitions stored
REPLAY_BATCH_SIZE   = 32     # mini-batch drawn per train_step when buffer is full

# Target network — stabilises Q-learning by decoupling the update target
TARGET_NET_UPDATE_FREQ = 50  # soft-update the target net every N train steps
TARGET_NET_TAU         = 0.01 # Polyak averaging: θ_target ← τ·θ_online + (1-τ)·θ_target

# Bellman discount factor
RL_GAMMA = 0.95

# Dual-stream item projection: each modality projected to this dim before fusion
# Final item representation = PROJ_DIM * 2  (BLaIR-proj || CLIP-proj)
RL_ITEM_PROJ_DIM = 256

# ─── BM25 Adaptive Search ────────────────────────────────────────────────────
# max number of keyword hits to pull from BM25 and feed into the fusion
BM25_TOP_N        = 30
# BM25 scores below this are treated as "no meaningful keyword match"
BM25_MIN_SCORE    = 0.5
# confidence ≥ this → keyword channel dominates the fusion (clear title/author hit)
BM25_HIGH_CONF    = 0.6
# extra weight added to the visual (CLIP) channel when an image is present.
# keeps the visual signal influential even when BM25 is also confident.
BM25_VISUAL_BOOST = 0.5
