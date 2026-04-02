"""
rl_collaborative_filter.py — Upgraded DQN-based Recommendation Ranker

Architecture improvements over v1:
  1. Dual-stream projection: BLaIR (1024) and CLIP (512) are each projected to
     a shared 256-dim space and L2-normalised before concatenation, resolving
     the heterogeneous embedding space problem.
  2. Experience replay buffer (circular, capacity=REPLAY_BUFFER_SIZE) enables
     mini-batch gradient updates, vastly reducing noise vs online single-sample.
  3. Target network (soft-updated via Polyak averaging every TARGET_NET_UPDATE_FREQ
     steps) decouples Q-targets from the online network, preventing oscillation.
  4. Proper Bellman-based Q-target:
       Q_target = r + γ · max_a' Q_frozen(next_state, a')
     instead of the previous naive MSE regression against raw reward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import deque
import random
import os
import pickle
from config import (
    BLAIR_DIM, CLIP_DIM,
    REPLAY_BUFFER_SIZE, REPLAY_BATCH_SIZE,
    TARGET_NET_UPDATE_FREQ, TARGET_NET_TAU,
    RL_GAMMA, RL_ITEM_PROJ_DIM,
)

# ─── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular experience replay buffer.
    Each transition stores:
        state      — concatenated user profile  [state_dim]
        blair_vec  — raw BLaIR item embedding   [1024]
        clip_vec   — raw CLIP item embedding    [512]
        reward     — scalar immediate reward
        next_state — updated user profile after interaction  [state_dim]
                     (None if terminal / no next state available)
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, blair_vec, clip_vec, reward, next_state):
        self.buffer.append((
            np.array(state,     dtype=np.float32),
            np.array(blair_vec, dtype=np.float32),
            np.array(clip_vec,  dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32) if next_state is not None else None,
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─── Network ───────────────────────────────────────────────────────────────────

class CollaborativeFilterDQN(nn.Module):
    """
    Dual-stream projected DQN for recommendation ranking.

    Item representation pipeline:
        BLaIR (1024) → blair_proj (256) → L2-norm ──┐
                                                      cat → item_repr [512]
        CLIP  (512)  → clip_proj  (256) → L2-norm ──┘

    Final DQN input: L2-norm(user_state [1536]) || item_repr [512] = [2048]
    Output: scalar Q-value (predicted preference score)
    """
    def __init__(self, state_dim: int, proj_dim: int = RL_ITEM_PROJ_DIM):
        super().__init__()
        self.proj_dim   = proj_dim
        item_repr_dim   = proj_dim * 2          # BLaIR-proj || CLIP-proj
        dqn_input_dim   = state_dim + item_repr_dim

        # Dual-stream item encoder
        self.blair_proj = nn.Linear(BLAIR_DIM, proj_dim)
        self.clip_proj  = nn.Linear(CLIP_DIM,  proj_dim)

        # Main Q-network
        self.fc1        = nn.Linear(dqn_input_dim, 512)
        self.fc2        = nn.Linear(512, 256)
        self.score_head = nn.Linear(256, 1)

        # Weight initialisation (Xavier uniform is suitable for linear layers)
        for layer in [self.blair_proj, self.clip_proj, self.fc1, self.fc2, self.score_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode_item(self, blair_vecs: torch.Tensor, clip_vecs: torch.Tensor) -> torch.Tensor:
        """Project and fuse both item modalities into a normalised 512-dim repr."""
        b = F.normalize(self.blair_proj(blair_vecs), p=2, dim=-1)  # [N, 256]
        c = F.normalize(self.clip_proj(clip_vecs),  p=2, dim=-1)  # [N, 256]
        return torch.cat([b, c], dim=-1)                            # [N, 512]

    def forward(self, state: torch.Tensor, blair_vecs: torch.Tensor,
                clip_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state      [batch, state_dim]  — L2-normalised user profile
            blair_vecs [batch, 1024]       — raw BLaIR item embeddings
            clip_vecs  [batch, 512]        — raw CLIP item embeddings
        Returns:
            Q-values   [batch, 1]
        """
        item_repr   = self.encode_item(blair_vecs, clip_vecs)       # [batch, 512]
        state_normd = F.normalize(state, p=2, dim=-1)               # [batch, state_dim]
        x = torch.cat([state_normd, item_repr], dim=-1)             # [batch, 2048]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.score_head(x)                                    # [batch, 1]


# ─── RL Agent ─────────────────────────────────────────────────────────────────

class RLCollaborativeFilter:
    """
    Stateful DQN agent with:
    - Experience replay (ReplayBuffer)
    - Target network (Polyak soft-update)
    - Bellman-based Q-targets with discount factor γ
    - Per-user model persistence via save() / load()
    """

    def __init__(self, state_dim: int, item_dim: int):
        """
        Args:
            state_dim: dimensionality of the user state vector
                       (BLAIR_DIM + CLIP_DIM = 1536)
            item_dim:  kept for backwards-compatible API — not directly used
                       (item representation is computed internally via projections)
        """
        self.state_dim  = state_dim
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online + frozen target networks
        self.model        = CollaborativeFilterDQN(state_dim).to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.optimizer  = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion  = nn.MSELoss()
        self.buffer     = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.loss_history = []
        self._step      = 0   # counts train_step() calls for target update scheduling

        print(f"[RLCollaborativeFilter] device={self.device}  "
              f"replay_capacity={REPLAY_BUFFER_SIZE}  batch={REPLAY_BATCH_SIZE}  "
              f"γ={RL_GAMMA}  τ={TARGET_NET_TAU}")

    # ─── Soft target update ───────────────────────────────────────────────────
    def _soft_update_target(self):
        """Polyak average: θ_target ← τ·θ_online + (1-τ)·θ_target"""
        tau = TARGET_NET_TAU
        for p_online, p_target in zip(self.model.parameters(),
                                      self.target_model.parameters()):
            p_target.data.mul_(1.0 - tau)
            p_target.data.add_(tau * p_online.data)

    # ─── Inference ────────────────────────────────────────────────────────────
    def get_candidate_scores(self, user_profile, candidate_asins, retriever) -> dict:
        """
        Score a list of candidate ASINs using the online DQN.
        Returns {asin: q_value} for each valid ASIN.
        """
        if user_profile.text_profile is None:
            return {asin: 0.0 for asin in candidate_asins}

        state = np.concatenate([user_profile.text_profile, user_profile.visual_profile])
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, 1536]

        blair_vecs, clip_vecs, valid_asins = [], [], []
        for asin in candidate_asins:
            embs = retriever.get_asin_vec(asin)
            if embs is not None:
                blair_vec, clip_vec = embs
                blair_vecs.append(blair_vec)
                clip_vecs.append(clip_vec)
                valid_asins.append(asin)

        if not valid_asins:
            return {}

        blair_t = torch.FloatTensor(np.array(blair_vecs)).to(self.device)  # [N, 1024]
        clip_t  = torch.FloatTensor(np.array(clip_vecs)).to(self.device)   # [N, 512]
        state_exp = state_t.expand(blair_t.size(0), -1)                    # [N, 1536]

        self.model.eval()
        with torch.no_grad():
            scores = self.model(state_exp, blair_t, clip_t).squeeze(-1)    # [N]

        return {asin: float(s) for asin, s in zip(valid_asins, scores.cpu().numpy())}

    # ─── Training ─────────────────────────────────────────────────────────────
    def train_step(self, user_profile, item_asin, reward, retriever,
                   next_profile=None) -> float | None:
        """
        Record one transition and (when the buffer has enough data) perform a
        mini-batch gradient update using a Bellman Q-target.

        Args:
            user_profile: UserProfile snapshot BEFORE the interaction
            item_asin:    The item the user interacted with
            reward:       Immediate scalar reward (+1 click, +5 cart, 0 skip)
            retriever:    Retriever instance (for embedding lookup)
            next_profile: UserProfile snapshot AFTER the profile manager has
                          updated it.  Pass None if unavailable (treated as
                          terminal state — target = r with no future discount).
        Returns:
            MSE loss as a float, or None if the buffer is still warming up.
        """
        if user_profile.text_profile is None:
            return None

        embs = retriever.get_asin_vec(item_asin)
        if embs is None:
            return None
        blair_vec, clip_vec = embs

        # Build current state
        state = np.concatenate([user_profile.text_profile, user_profile.visual_profile])

        # Build next state (for Bellman target)
        if next_profile is not None and next_profile.text_profile is not None:
            next_state = np.concatenate([next_profile.text_profile,
                                         next_profile.visual_profile])
        else:
            next_state = None

        # Push transition to replay buffer
        self.buffer.push(state, blair_vec, clip_vec, reward, next_state)

        # ── Mini-batch update ──────────────────────────────────────────────────
        actual_batch_size = min(len(self.buffer), REPLAY_BATCH_SIZE)
        if actual_batch_size == 0:
            return None   # still warming up — do not train yet

        batch      = self.buffer.sample(actual_batch_size)
        states     = torch.FloatTensor(np.stack([t[0] for t in batch])).to(self.device)
        blairs     = torch.FloatTensor(np.stack([t[1] for t in batch])).to(self.device)
        clips      = torch.FloatTensor(np.stack([t[2] for t in batch])).to(self.device)
        rewards    = torch.FloatTensor([t[3] for t in batch]).to(self.device)    # [B]
        next_states_raw = [t[4] for t in batch]

        # ── Bellman Q-target computation ───────────────────────────────────────
        # For terminal transitions (next_state is None) the target is just r.
        # For non-terminal, we approximate max_a' Q(s', a') using the *same*
        # item embedding (re-predicting on the same item in the new state) since
        # we don't store the full candidate pool in the buffer.  This is a
        # contextual-bandit simplification appropriate for our setting.
        self.target_model.eval()
        with torch.no_grad():
            q_next = torch.zeros(actual_batch_size, device=self.device)
            non_terminal_mask = [i for i, ns in enumerate(next_states_raw) if ns is not None]
            if non_terminal_mask:
                ns_arr = np.stack([next_states_raw[i] for i in non_terminal_mask])
                ns_t   = torch.FloatTensor(ns_arr).to(self.device)
                bl_sub = blairs[non_terminal_mask]
                cl_sub = clips[non_terminal_mask]
                q_next_vals = self.target_model(ns_t, bl_sub, cl_sub).squeeze(-1)
                for out_i, in_i in enumerate(non_terminal_mask):
                    q_next[in_i] = q_next_vals[out_i]

            q_targets = rewards + RL_GAMMA * q_next   # [B]

        # ── Online network forward + loss ──────────────────────────────────────
        self.model.train()
        self.optimizer.zero_grad()
        q_pred = self.model(states, blairs, clips).squeeze(-1)  # [B]
        loss   = self.criterion(q_pred, q_targets)
        loss.backward()
        # Gradient clipping prevents exploding gradients during early training
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Periodic target network soft-update ───────────────────────────────
        self._step += 1
        if self._step % TARGET_NET_UPDATE_FREQ == 0:
            self._soft_update_target()

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]

        return loss_val

    # ─── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        """Persist model weights, optimizer state, and step counter to disk."""
        torch.save({
            "model_state":     self.model.state_dict(),
            "target_state":    self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step":            self._step,
            "loss_history":    self.loss_history,
            # store state_dim so we can detect incompatible saved weights
            "state_dim":       self.state_dim,
        }, path)
        
        # Save the replay buffer as well so history survives server restarts
        buffer_path = path.replace("_dqn.pt", "_buffer.pkl")
        if not buffer_path.endswith("_buffer.pkl"):
            buffer_path = path + "_buffer.pkl"
        try:
            with open(buffer_path, "wb") as f:
                pickle.dump(self.buffer.buffer, f)
        except Exception as e:
            print(f"[RLCollaborativeFilter] Warning: could not save replay buffer: {e}")

    def load(self, path: str):
        """Restore weights from disk. Silently skips if dimensions changed."""
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("state_dim") != self.state_dim:
            print(f"[RLCollaborativeFilter] Skipping {path} — "
                  f"saved state_dim={checkpoint.get('state_dim')} "
                  f"!= current {self.state_dim}. Starting fresh.")
            return
        self.model.load_state_dict(checkpoint["model_state"])
        self.target_model.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._step = checkpoint.get("step", 0)
        self.loss_history = checkpoint.get("loss_history", [])
        self.model.eval()
        print(f"[RLCollaborativeFilter] Loaded weights from {path} "
              f"(step={self._step})")
              
        # Load the replay buffer if it exists
        buffer_path = path.replace("_dqn.pt", "_buffer.pkl")
        if not buffer_path.endswith("_buffer.pkl"):
            buffer_path = path + "_buffer.pkl"
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, "rb") as f:
                    saved_buffer = pickle.load(f)
                    self.buffer.buffer = saved_buffer
                print(f"[RLCollaborativeFilter] Loaded replay buffer ({len(self.buffer)} transitions)")
            except Exception as e:
                print(f"[RLCollaborativeFilter] Warning: could not load replay buffer: {e}")

