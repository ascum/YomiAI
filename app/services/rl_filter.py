"""
app/services/rl_filter.py — GRU-Sequential DQN agent.

Moved from src/rl_collaborative_filter.py.
Imports updated: from app.config / app.services.sequential_dqn
"""
import copy
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.config import settings
from app.services.sequential_dqn import SequentialDQN

BLAIR_DIM           = settings.BLAIR_DIM
CLIP_DIM            = settings.CLIP_DIM
REPLAY_BUFFER_SIZE  = settings.REPLAY_BUFFER_SIZE
REPLAY_BATCH_SIZE   = settings.REPLAY_BATCH_SIZE
TARGET_NET_UPDATE_FREQ = settings.TARGET_NET_UPDATE_FREQ
TARGET_NET_TAU      = settings.TARGET_NET_TAU
RL_GAMMA            = settings.RL_GAMMA
GRU_HIDDEN_DIM      = settings.GRU_HIDDEN_DIM
MAX_SEQ_LEN         = settings.MAX_SEQ_LEN
EPSILON_START       = settings.EPSILON_START
EPSILON_END         = settings.EPSILON_END
EPSILON_DECAY_STEPS = settings.EPSILON_DECAY_STEPS
NEG_SAMPLE_SIZE     = settings.NEG_SAMPLE_SIZE
SEQ_ITEM_PROJ_DIM   = settings.SEQ_ITEM_PROJ_DIM


class SequentialReplayBuffer:
    """Index-based replay buffer (stores ASINs, reconstructs embeddings at train time)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, click_seq_asins: list, item_asin: str,
             reward: float, next_seq_asins: list):
        self.buffer.append((
            list(click_seq_asins),
            str(item_asin),
            float(reward),
            list(next_seq_asins),
        ))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLSequentialFilter:
    """GRU-based Sequential DQN agent for recommendation re-ranking."""

    def __init__(self, retriever):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = retriever

        self.model        = SequentialDQN().to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.optimizer    = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion    = nn.SmoothL1Loss()
        self.buffer       = SequentialReplayBuffer(REPLAY_BUFFER_SIZE)
        self.loss_history = []
        self._step        = 0
        self._epsilon     = EPSILON_START
        self._all_asins   = list(retriever.asin_to_idx.keys()) if retriever else []

        print(f"[RLSequentialFilter] device={self.device}  "
              f"GRU_hidden={GRU_HIDDEN_DIM}  replay={REPLAY_BUFFER_SIZE}  "
              f"ε={EPSILON_START}→{EPSILON_END}/{EPSILON_DECAY_STEPS}")

    # ── Epsilon scheduling ────────────────────────────────────────────────────

    def _update_epsilon(self):
        frac = min(1.0, self._step / max(1, EPSILON_DECAY_STEPS))
        self._epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)

    @property
    def epsilon(self):
        return self._epsilon

    # ── Soft target update ────────────────────────────────────────────────────

    def _soft_update_target(self):
        for p_online, p_target in zip(self.model.parameters(),
                                      self.target_model.parameters()):
            p_target.data.mul_(1.0 - TARGET_NET_TAU)
            p_target.data.add_(TARGET_NET_TAU * p_online.data)

    # ── Sequence tensor builders ──────────────────────────────────────────────

    def _build_seq_tensors(self, click_seq_asins: list, device=None):
        if device is None:
            device = self.device
        blair_list, clip_list = [], []
        for asin in click_seq_asins[-MAX_SEQ_LEN:]:
            embs = self.retriever.get_asin_vec(asin)
            if embs is not None:
                b, c = embs
                blair_list.append(b)
                clip_list.append(c)

        actual_len = len(blair_list)
        if actual_len == 0:
            return (
                torch.zeros(1, MAX_SEQ_LEN, BLAIR_DIM, device=device),
                torch.zeros(1, MAX_SEQ_LEN, CLIP_DIM,  device=device),
                torch.tensor([0], device=device),
            )

        blair_arr = np.zeros((MAX_SEQ_LEN, BLAIR_DIM), dtype=np.float32)
        clip_arr  = np.zeros((MAX_SEQ_LEN, CLIP_DIM),  dtype=np.float32)
        blair_arr[:actual_len] = np.array(blair_list)
        clip_arr[:actual_len]  = np.array(clip_list)

        return (
            torch.FloatTensor(blair_arr).unsqueeze(0).to(device),
            torch.FloatTensor(clip_arr).unsqueeze(0).to(device),
            torch.tensor([actual_len], device=device),
        )

    def _build_batch_seq_tensors(self, batch_seqs: list, device=None):
        if device is None:
            device = self.device
        B = len(batch_seqs)
        blair_batch = np.zeros((B, MAX_SEQ_LEN, BLAIR_DIM), dtype=np.float32)
        clip_batch  = np.zeros((B, MAX_SEQ_LEN, CLIP_DIM),  dtype=np.float32)
        lengths     = np.zeros(B, dtype=np.int64)

        for i, seq_asins in enumerate(batch_seqs):
            truncated = seq_asins[-MAX_SEQ_LEN:]
            blair_list, clip_list = [], []
            for asin in truncated:
                embs = self.retriever.get_asin_vec(asin)
                if embs is not None:
                    b, c = embs
                    blair_list.append(b)
                    clip_list.append(c)
            n = len(blair_list)
            if n > 0:
                blair_batch[i, :n] = np.array(blair_list)
                clip_batch[i, :n]  = np.array(clip_list)
            lengths[i] = n

        return (
            torch.FloatTensor(blair_batch).to(device),
            torch.FloatTensor(clip_batch).to(device),
            torch.LongTensor(lengths).to(device),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_candidate_scores(self, click_seq_asins: list,
                              candidate_asins: list) -> dict:
        if not click_seq_asins or not candidate_asins:
            return {asin: 0.0 for asin in candidate_asins}

        if random.random() < self._epsilon:
            return {asin: random.random() for asin in candidate_asins}

        blair_seq, clip_seq, length = self._build_seq_tensors(click_seq_asins)
        if length.item() == 0:
            return {asin: 0.0 for asin in candidate_asins}

        self.model.eval()
        with torch.no_grad():
            user_state = self.model.encode_user(blair_seq, clip_seq, length)

        blair_vecs, clip_vecs, valid_asins = [], [], []
        for asin in candidate_asins:
            embs = self.retriever.get_asin_vec(asin)
            if embs is not None:
                b, c = embs
                blair_vecs.append(b)
                clip_vecs.append(c)
                valid_asins.append(asin)

        if not valid_asins:
            return {}

        blair_t  = torch.FloatTensor(np.array(blair_vecs)).to(self.device)
        clip_t   = torch.FloatTensor(np.array(clip_vecs)).to(self.device)
        user_exp = user_state.expand(len(valid_asins), -1)

        with torch.no_grad():
            scores = self.model(user_exp, blair_t, clip_t).squeeze(-1)

        return {asin: float(s) for asin, s in zip(valid_asins, scores.cpu().numpy())}

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self, click_seq_before: list, item_asin: str,
                   reward: float, click_seq_after: list) -> float | None:
        if self.retriever.get_asin_vec(item_asin) is None:
            return None

        self.buffer.push(click_seq_before, item_asin, reward, click_seq_after)

        actual_batch = min(len(self.buffer), REPLAY_BATCH_SIZE)
        if actual_batch == 0:
            return None

        batch = self.buffer.sample(actual_batch)
        seqs_before = [t[0] for t in batch]
        item_asins  = [t[1] for t in batch]
        rewards_raw = [t[2] for t in batch]
        seqs_after  = [t[3] for t in batch]

        blair_seqs, clip_seqs, lengths = self._build_batch_seq_tensors(seqs_before)

        item_blairs, item_clips = [], []
        for asin in item_asins:
            embs = self.retriever.get_asin_vec(asin)
            if embs is not None:
                b, c = embs
                item_blairs.append(b)
                item_clips.append(c)
            else:
                item_blairs.append(np.zeros(BLAIR_DIM, dtype=np.float32))
                item_clips.append(np.zeros(CLIP_DIM,  dtype=np.float32))

        item_blair_t = torch.FloatTensor(np.array(item_blairs)).to(self.device)
        item_clip_t  = torch.FloatTensor(np.array(item_clips)).to(self.device)
        rewards_t    = torch.FloatTensor(rewards_raw).to(self.device)

        self.target_model.eval()
        with torch.no_grad():
            blair_next, clip_next, len_next = self._build_batch_seq_tensors(seqs_after)
            h_next = self.target_model.encode_user(blair_next, clip_next, len_next)

            neg_asins = random.sample(self._all_asins, min(NEG_SAMPLE_SIZE, len(self._all_asins)))
            neg_blairs, neg_clips = [], []
            for asin in neg_asins:
                embs = self.retriever.get_asin_vec(asin)
                if embs is not None:
                    neg_blairs.append(embs[0])
                    neg_clips.append(embs[1])

            if neg_blairs:
                neg_b_t = torch.FloatTensor(np.array(neg_blairs)).to(self.device)
                neg_c_t = torch.FloatTensor(np.array(neg_clips)).to(self.device)
                K = neg_b_t.size(0)

                h_exp      = h_next.unsqueeze(1).expand(-1, K, -1).reshape(-1, GRU_HIDDEN_DIM)
                neg_b_exp  = neg_b_t.unsqueeze(0).expand(actual_batch, -1, -1).reshape(-1, BLAIR_DIM)
                neg_c_exp  = neg_c_t.unsqueeze(0).expand(actual_batch, -1, -1).reshape(-1, CLIP_DIM)

                q_neg      = self.target_model(h_exp, neg_b_exp, neg_c_exp).reshape(actual_batch, K)
                max_q_next = q_neg.max(dim=1).values
            else:
                max_q_next = torch.zeros(actual_batch, device=self.device)

            terminal_mask = (len_next == 0).float()
            q_targets = rewards_t + RL_GAMMA * max_q_next * (1.0 - terminal_mask)

        self.model.train()
        self.optimizer.zero_grad()
        h_t    = self.model.encode_user(blair_seqs, clip_seqs, lengths)
        q_pred = self.model(h_t, item_blair_t, item_clip_t).squeeze(-1)
        loss   = self.criterion(q_pred, q_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step += 1
        self._update_epsilon()
        if self._step % TARGET_NET_UPDATE_FREQ == 0:
            self._soft_update_target()

        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]

        return loss_val

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state":     self.model.state_dict(),
            "target_state":    self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step":            self._step,
            "epsilon":         self._epsilon,
            "loss_history":    self.loss_history,
            "arch":            "sequential_dqn_v1",
        }, path)

        buffer_path = path.replace("_seq_dqn.pt", "_seq_buffer.pkl")
        if not buffer_path.endswith("_seq_buffer.pkl"):
            buffer_path = path + "_seq_buffer.pkl"
        try:
            with open(buffer_path, "wb") as f:
                pickle.dump(self.buffer.buffer, f)
        except Exception as e:
            print(f"[RLSequentialFilter] Warning: could not save buffer: {e}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint.get("arch") != "sequential_dqn_v1":
            print(f"[RLSequentialFilter] Skipping {path} — arch mismatch")
            return

        self.model.load_state_dict(checkpoint["model_state"])
        self.target_model.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._step        = checkpoint.get("step", 0)
        self._epsilon     = checkpoint.get("epsilon", EPSILON_START)
        self.loss_history = checkpoint.get("loss_history", [])
        self.model.eval()
        print(f"[RLSequentialFilter] Loaded from {path} (step={self._step}, ε={self._epsilon:.3f})")

        buffer_path = path.replace("_seq_dqn.pt", "_seq_buffer.pkl")
        if not buffer_path.endswith("_seq_buffer.pkl"):
            buffer_path = path + "_seq_buffer.pkl"
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, "rb") as f:
                    self.buffer.buffer = pickle.load(f)
                print(f"[RLSequentialFilter] Replay buffer loaded ({len(self.buffer)} transitions)")
            except Exception as e:
                print(f"[RLSequentialFilter] Warning: could not load buffer: {e}")
