"""
app/services/sequential_dqn.py — GRU-based Sequential DQN model.

Moved from src/sequential_dqn.py.
Import updated: from app.config import settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import settings

TEXT_EMBED_DIM  = settings.TEXT_EMBED_DIM
CLIP_DIM        = settings.CLIP_DIM
SEQ_ITEM_PROJ_DIM = settings.SEQ_ITEM_PROJ_DIM
GRU_HIDDEN_DIM  = settings.GRU_HIDDEN_DIM
GRU_NUM_LAYERS  = settings.GRU_NUM_LAYERS
GRU_DROPOUT     = settings.GRU_DROPOUT


class DualStreamItemEncoder(nn.Module):
    """Projects text (TEXT_EMBED_DIM) and CLIP (CLIP_DIM) embeddings into a shared space."""

    def __init__(self, proj_dim: int = SEQ_ITEM_PROJ_DIM):
        super().__init__()
        self.proj_dim   = proj_dim
        self.output_dim = proj_dim * 2

        self.text_proj = nn.Linear(TEXT_EMBED_DIM, proj_dim)
        self.clip_proj = nn.Linear(CLIP_DIM,       proj_dim)

        for layer in [self.text_proj, self.clip_proj]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, text_vecs: torch.Tensor, clip_vecs: torch.Tensor) -> torch.Tensor:
        t = F.normalize(self.text_proj(text_vecs), p=2, dim=-1)
        c = F.normalize(self.clip_proj(clip_vecs), p=2, dim=-1)
        return torch.cat([t, c], dim=-1)


class GRUUserEncoder(nn.Module):
    """Recurrent encoder: ordered click sequence → user intent h_t."""

    def __init__(self, item_encoder: DualStreamItemEncoder,
                 hidden_dim: int = GRU_HIDDEN_DIM,
                 num_layers: int = GRU_NUM_LAYERS,
                 dropout: float = GRU_DROPOUT):
        super().__init__()
        self.item_encoder = item_encoder
        self.hidden_dim   = hidden_dim
        self.gru = nn.GRU(
            input_size=item_encoder.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, text_seqs: torch.Tensor, clip_seqs: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = text_seqs.shape
        text_flat   = text_seqs.reshape(B * T, -1)
        clip_flat   = clip_seqs.reshape(B * T, -1)
        item_embeds = self.item_encoder(text_flat, clip_flat).reshape(B, T, -1)

        packed = nn.utils.rnn.pack_padded_sequence(
            item_embeds, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return h_n[-1]  # [B, hidden_dim]


class SequentialDQN(nn.Module):
    """Full Sequential DQN: sequence encoder + candidate scorer."""

    def __init__(self, hidden_dim: int = GRU_HIDDEN_DIM,
                 proj_dim: int = SEQ_ITEM_PROJ_DIM):
        super().__init__()
        self.hidden_dim   = hidden_dim
        item_repr_dim     = proj_dim * 2
        scorer_input      = hidden_dim + item_repr_dim  # 1536

        self.item_encoder = DualStreamItemEncoder(proj_dim)
        self.user_encoder = GRUUserEncoder(self.item_encoder, hidden_dim)

        self.fc1        = nn.Linear(scorer_input, 1024)
        self.fc2        = nn.Linear(1024, 512)
        self.fc3        = nn.Linear(512, 256)
        self.score_head = nn.Linear(256, 1)

        for layer in [self.fc1, self.fc2, self.fc3, self.score_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode_user(self, text_seqs: torch.Tensor, clip_seqs: torch.Tensor,
                    lengths: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(text_seqs, clip_seqs, lengths)

    def encode_item(self, text_vecs: torch.Tensor,
                    clip_vecs: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(text_vecs, clip_vecs)

    def forward(self, user_state: torch.Tensor,
                text_vecs: torch.Tensor,
                clip_vecs: torch.Tensor) -> torch.Tensor:
        item_repr = self.item_encoder(text_vecs, clip_vecs)
        x = torch.cat([user_state, item_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.score_head(x)

    def forward_from_sequence(self, text_seqs, clip_seqs, lengths,
                               cand_text, cand_clip) -> torch.Tensor:
        h_t = self.encode_user(text_seqs, clip_seqs, lengths)
        return self.forward(h_t, cand_text, cand_clip)
