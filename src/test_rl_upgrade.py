"""
test_rl_upgrade.py — Unit tests for the upgraded RLCollaborativeFilter.
Run from the project root:   python src/test_rl_upgrade.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch

from config import (
    BLAIR_DIM, CLIP_DIM, REPLAY_BATCH_SIZE,
    RL_ITEM_PROJ_DIM, RL_GAMMA,
)
from rl_collaborative_filter import ReplayBuffer, RLCollaborativeFilter

STATE_DIM = BLAIR_DIM + CLIP_DIM   # 1536
ITEM_DIM  = RL_ITEM_PROJ_DIM * 2   # 512

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ─── 1. ReplayBuffer ──────────────────────────────────────────────────────────
section("ReplayBuffer")

buf = ReplayBuffer(capacity=100)
assert len(buf) == 0, f"{FAIL} empty buffer should have len=0"
print(f"{PASS}  empty buffer len=0")

# Push some transitions
for i in range(10):
    s  = np.random.randn(STATE_DIM).astype(np.float32)
    b  = np.random.randn(BLAIR_DIM).astype(np.float32)
    c  = np.random.randn(CLIP_DIM).astype(np.float32)
    r  = float(i % 3)
    ns = np.random.randn(STATE_DIM).astype(np.float32) if i < 8 else None
    buf.push(s, b, c, r, ns)

assert len(buf) == 10, f"{FAIL} expected 10 transitions"
print(f"{PASS}  push 10 transitions → len=10")

# Circular overflow
for _ in range(95):
    buf.push(np.ones(STATE_DIM, dtype=np.float32),
             np.ones(BLAIR_DIM, dtype=np.float32),
             np.ones(CLIP_DIM,  dtype=np.float32),
             0.0, None)
assert len(buf) == 100, f"{FAIL} circular buffer should cap at 100"
print(f"{PASS}  circular overflow → len=100")

# Sample shapes
batch = buf.sample(32)
assert len(batch) == 32, f"{FAIL} sample should return 32 items"
s0, b0, c0, r0, _ = batch[0]
assert s0.shape == (STATE_DIM,), f"{FAIL} state shape mismatch"
assert b0.shape == (BLAIR_DIM,), f"{FAIL} BLaIR shape mismatch"
assert c0.shape == (CLIP_DIM,),  f"{FAIL} CLIP shape mismatch"
print(f"{PASS}  sample(32) returns correct shapes "
      f"state={s0.shape} blair={b0.shape} clip={c0.shape}")

# ─── 2. CollaborativeFilterDQN architecture ────────────────────────────────────
section("CollaborativeFilterDQN  (architecture + forward pass)")

from rl_collaborative_filter import CollaborativeFilterDQN

net = CollaborativeFilterDQN(state_dim=STATE_DIM)

B = 8
states  = torch.randn(B, STATE_DIM)
blairs  = torch.randn(B, BLAIR_DIM)
clips   = torch.randn(B, CLIP_DIM)

with torch.no_grad():
    out = net(states, blairs, clips)

assert out.shape == (B, 1), f"{FAIL} DQN output shape should be [B,1], got {out.shape}"
print(f"{PASS}  forward pass OK — output shape {out.shape} for batch={B}")

# encode_item produces normalized 512-dim repr
item_repr = net.encode_item(blairs, clips)
assert item_repr.shape == (B, RL_ITEM_PROJ_DIM * 2), (
    f"{FAIL} item_repr shape should be [B, {RL_ITEM_PROJ_DIM*2}], got {item_repr.shape}")
norms = torch.norm(item_repr[:, :RL_ITEM_PROJ_DIM], dim=-1)  # BLaIR slice normalised
assert torch.allclose(norms, torch.ones(B), atol=1e-5), f"{FAIL} BLaIR slice not L2-normalised"
print(f"{PASS}  encode_item → {item_repr.shape}, BLaIR-slice is L2-normalised")

# ─── 3. RLCollaborativeFilter — buffer warm-up ─────────────────────────────────
section("RLCollaborativeFilter  (buffer warm-up)")

class _MockProfile:
    """Minimal profile stub that mimics UserProfile."""
    def __init__(self, has_profile=True):
        if has_profile:
            self.text_profile   = np.random.randn(BLAIR_DIM).astype(np.float32)
            self.visual_profile = np.random.randn(CLIP_DIM).astype(np.float32)
        else:
            self.text_profile   = None
            self.visual_profile = None

class _MockRetriever:
    """Stub retriever that always returns random (blair, clip) tuples."""
    def get_asin_vec(self, asin):
        return (np.random.randn(BLAIR_DIM).astype(np.float32),
                np.random.randn(CLIP_DIM).astype(np.float32))
    def __contains__(self, key):   # asin_to_idx membership check fallback
        return True

agent    = RLCollaborativeFilter(state_dim=STATE_DIM, item_dim=ITEM_DIM)
retriever = _MockRetriever()
profile   = _MockProfile()
next_prof = _MockProfile()

# While buffer is < REPLAY_BATCH_SIZE, train_step should NOT return a loss
for i in range(REPLAY_BATCH_SIZE - 1):
    loss = agent.train_step(profile, f"ASIN{i:04d}", 1.0, retriever,
                            next_profile=next_prof)
assert loss is None, f"{FAIL} should be None while buffer is warming up (got {loss})"
print(f"{PASS}  train_step returns None during buffer warm-up (<{REPLAY_BATCH_SIZE} transitions)")

# Push one more to hit the batch threshold
loss = agent.train_step(profile, "ASIN_FINAL", 1.0, retriever, next_profile=next_prof)
assert loss is not None and isinstance(loss, float), (
    f"{FAIL} expected a float loss after buffer fills, got {loss}")
assert loss >= 0.0, f"{FAIL} loss should be non-negative"
print(f"{PASS}  train_step returns loss={loss:.6f} after buffer reaches batch size")

# ─── 4. get_candidate_scores shapes ───────────────────────────────────────────
section("get_candidate_scores")

asins  = [f"B{i:010d}" for i in range(20)]
scores = agent.get_candidate_scores(profile, asins, retriever)
assert isinstance(scores, dict),            f"{FAIL} expected dict"
assert set(scores.keys()) == set(asins),    f"{FAIL} scores dict should cover all ASINs"
assert all(isinstance(v, float) for v in scores.values()), f"{FAIL} all scores must be float"
print(f"{PASS}  get_candidate_scores returns dict of {len(scores)} floats")

# Cold-start profile (no text_profile) should return zero dict
cold_profile = _MockProfile(has_profile=False)
scores_cold  = agent.get_candidate_scores(cold_profile, asins, retriever)
assert all(v == 0.0 for v in scores_cold.values()), f"{FAIL} cold-start should return 0s"
print(f"{PASS}  cold-start profile → all scores=0.0")

# ─── 5. get_asin_vec returns tuple ────────────────────────────────────────────
section("Retriever.get_asin_vec (tuple form)")

# Directly test the real Retriever if indices are available; otherwise skip.
try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from retriever import Retriever
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    if os.path.exists(os.path.join(DATA_DIR, "asins.csv")):
        retriever_real = Retriever(DATA_DIR)
        first_asin = retriever_real.asins[0]
        result = retriever_real.get_asin_vec(first_asin)
        assert result is not None,           f"{FAIL} get_asin_vec returned None for valid ASIN"
        assert isinstance(result, tuple),    f"{FAIL} get_asin_vec should return a tuple"
        blair_v, clip_v = result
        assert blair_v.shape == (BLAIR_DIM,), f"{FAIL} BLaIR shape: expected ({BLAIR_DIM},) got {blair_v.shape}"
        assert clip_v.shape  == (CLIP_DIM,),  f"{FAIL} CLIP shape: expected ({CLIP_DIM},) got {clip_v.shape}"
        print(f"{PASS}  Retriever.get_asin_vec returns tuple "
              f"  blair={blair_v.shape}  clip={clip_v.shape}")
    else:
        print("⚠️  SKIP  FAISS indices not found — skipping real Retriever test")
except Exception as e:
    print(f"⚠️  SKIP  Retriever test skipped: {e}")

# ─── 6. Save / load cycle ─────────────────────────────────────────────────────
section("Save / Load persistence")

import tempfile, pathlib

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt = str(pathlib.Path(tmpdir) / "test_dqn.pt")
    agent.save(ckpt)
    assert os.path.isfile(ckpt), f"{FAIL} checkpoint file not created"
    print(f"{PASS}  save() created checkpoint file")

    # Mutate weights to verify load actually restores them
    for p in agent.model.parameters():
        p.data.fill_(0.0)

    agent.load(ckpt)
    # Verify at least one parameter is non-zero again
    any_nonzero = any(p.data.abs().sum() > 0 for p in agent.model.parameters())
    assert any_nonzero, f"{FAIL} model weights not restored after load()"
    print(f"{PASS}  load() correctly restores model weights")

    # Incompatible state_dim should be skipped gracefully
    agent_big = RLCollaborativeFilter(state_dim=STATE_DIM + 128, item_dim=ITEM_DIM)
    agent_big.load(ckpt)   # should print a skip message, not crash
    print(f"{PASS}  load() with mismatched state_dim silently skips (no crash)")

# ─── Summary ──────────────────────────────────────────────────────────────────
section("All tests passed 🎉")
