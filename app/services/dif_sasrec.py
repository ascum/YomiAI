"""
app/services/dif_sasrec.py — DIF-SASRec model and online training agent.

Architecture: Decoupled-Information-Feature Sequential Recommendation.
Two parallel attention streams (content + category) fused via a learnable α,
capturing both semantic content evolution and genre preference shifts.

Key design choices:
  - ContentProjector: reduces BGE-M3 1024-dim → 256-dim hidden space
  - DIFAttentionLayer: content Q/K/V + category Q/K (no category V), fused with sigmoid-α
  - Causal mask: prevents attending to future positions (autoregressive)
  - Sampled softmax loss: scales to 3M-item catalog without full softmax
  - Category auxiliary loss: improves category-level intent signal

Usage:
    agent = DIFSASRecAgent(retriever, category_encoder)
    scores = agent.get_candidate_scores(click_asins, cat_ids, candidate_asins)
    loss   = agent.train_step(click_asins, target_asin, target_cat_id, all_asins)
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from app.config import settings

# ── Constants from settings ───────────────────────────────────────────────────
TEXT_EMBED_DIM  = settings.TEXT_EMBED_DIM       # 1024 — BGE-M3 output dim
HIDDEN_DIM      = settings.SASREC_HIDDEN_DIM    # 512
N_BLOCKS        = settings.SASREC_N_BLOCKS      # 4
N_HEADS         = settings.SASREC_N_HEADS       # 8
HEAD_DIM        = HIDDEN_DIM // N_HEADS         # 64
DROPOUT         = settings.SASREC_DROPOUT       # 0.2
LR              = settings.SASREC_LR            # 1e-3
WEIGHT_DECAY    = settings.SASREC_WEIGHT_DECAY  # 0.01
ALPHA_INIT      = settings.SASREC_ALPHA_INIT    # 0.7
CAT_AUX_WEIGHT  = settings.SASREC_CAT_AUX_WEIGHT   # 0.1
NUM_NEGATIVES   = settings.SASREC_NUM_NEGATIVES # 512
MAX_SEQ_LEN     = settings.MAX_SEQ_LEN          # 50


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class ContentProjector(nn.Module):
    """
    Projects BGE-M3 embeddings (1024-dim) into the model's hidden space (256-dim).

    Used for both the sequence items and candidate items so they share the same
    projection and can be scored via dot product.

    Shape: [*, 1024] → [*, 256]
    """

    def __init__(self, in_dim: int = TEXT_EMBED_DIM, out_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.proj(x)))


class DIFAttentionLayer(nn.Module):
    """
    Decoupled-Information-Feature attention.

    Two parallel attention streams:
      - Content stream: Q_c, K_c, V from projected BGE-M3 embeddings
      - Category stream: Q_k, K_k from category embeddings (NO category V)

    Fusion:
        A_content  = softmax(Q_c · K_c^T / √head_dim)
        A_category = softmax(Q_k · K_k^T / √head_dim)
        A_fused    = α·A_category + (1-α)·A_content
        output     = A_fused · V_content

    α is a learnable scalar initialised so that sigmoid(α_logit) ≈ ALPHA_INIT.
    A causal lower-triangular mask prevents attending to future positions.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads

        # Content stream projections
        self.q_content = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_content = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_content = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Category stream projections (no V — values always from content)
        self.q_cat = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_cat = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop     = nn.Dropout(DROPOUT)

        # Learnable fusion scalar: α = sigmoid(α_logit)
        # init so sigmoid(x) ≈ ALPHA_INIT → x = log(α/(1-α))
        import math
        init_logit = math.log(ALPHA_INIT / (1.0 - ALPHA_INIT))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, n_heads, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, content: torch.Tensor, category: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content:    [B, T, hidden_dim]  projected BGE-M3 embeddings
            category:   [B, T, hidden_dim]  category embeddings
            causal_mask:[T, T] bool — True where attention is BLOCKED (upper triangle)
        Returns:
            [B, T, hidden_dim]
        """
        scale = self.head_dim ** -0.5

        # Content stream
        Q_c = self._split_heads(self.q_content(content))   # [B, H, T, head_dim]
        K_c = self._split_heads(self.k_content(content))
        V   = self._split_heads(self.v_content(content))

        # Category stream
        Q_k = self._split_heads(self.q_cat(category))
        K_k = self._split_heads(self.k_cat(category))

        # Attention logits
        A_content  = (Q_c @ K_c.transpose(-2, -1)) * scale   # [B, H, T, T]
        A_category = (Q_k @ K_k.transpose(-2, -1)) * scale

        # Apply causal mask (set masked positions to -inf before softmax)
        A_content  = A_content.masked_fill(causal_mask, float("-inf"))
        A_category = A_category.masked_fill(causal_mask, float("-inf"))

        A_content  = torch.softmax(A_content,  dim=-1)
        A_category = torch.softmax(A_category, dim=-1)

        alpha   = torch.sigmoid(self.alpha_logit)
        A_fused = alpha * A_category + (1.0 - alpha) * A_content
        A_fused = self.drop(A_fused)

        out = (A_fused @ V)                                   # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(content.shape[0], -1, self.hidden_dim)
        return self.out_proj(out)


class DIFSASRecBlock(nn.Module):
    """
    One transformer block using DIF attention.

    Pre-norm architecture (LayerNorm before each sub-layer):
        x → LN → DIF-Attn(x, cat) → + residual
          → LN → FFN(256→512→256, GELU) → + residual
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn  = DIFAttentionLayer(hidden_dim)
        self.drop  = nn.Dropout(DROPOUT)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, content: torch.Tensor, category: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        # DIF-Attention sub-layer (pre-norm)
        h = content + self.drop(self.attn(self.norm1(content), category, causal_mask))
        # FFN sub-layer (pre-norm)
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h


class DIFSASRecModel(nn.Module):
    """
    Full DIF-SASRec model.

    Takes a click sequence of BGE-M3 embeddings + category IDs and produces:
      - hidden: per-position hidden states [B, T, hidden_dim]
      - intent: last-valid-position hidden state [B, hidden_dim]
      - cat_logits: per-position category predictions [B, T, num_categories]
    """

    def __init__(self, num_categories: int, hidden_dim: int = HIDDEN_DIM,
                 n_blocks: int = N_BLOCKS, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len    = max_len

        self.content_proj  = ContentProjector(TEXT_EMBED_DIM, hidden_dim)
        self.category_emb  = nn.Embedding(num_categories, hidden_dim, padding_idx=0)
        self.position_emb  = nn.Embedding(max_len, hidden_dim)
        self.blocks        = nn.ModuleList([DIFSASRecBlock(hidden_dim) for _ in range(n_blocks)])
        self.final_norm    = nn.LayerNorm(hidden_dim)

        # Candidate projection (shared weight space — dot product scoring)
        self.candidate_proj = ContentProjector(TEXT_EMBED_DIM, hidden_dim)

        # Auxiliary task: predict category from hidden state
        self.category_head = nn.Linear(hidden_dim, num_categories)

        # Pre-build causal mask buffer (will be trimmed to actual seq len at runtime)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask_full", mask)

    def forward(self, bge_seqs: torch.Tensor, cat_ids: torch.Tensor,
                lengths: torch.Tensor):
        """
        Args:
            bge_seqs: [B, T, 1024]  BGE-M3 embeddings
            cat_ids:  [B, T]        category IDs (0=PAD)
            lengths:  [B]           actual sequence lengths
        Returns:
            hidden     [B, T, hidden_dim]
            intent     [B, hidden_dim]    — hidden state at last valid position
            cat_logits [B, T, num_cats]
        """
        B, T, _ = bge_seqs.shape
        device   = bge_seqs.device

        # Content projection + positional encoding
        content  = self.content_proj(bge_seqs)              # [B, T, D]
        pos_ids  = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        content  = content + self.position_emb(pos_ids)

        # Category embedding
        category = self.category_emb(cat_ids)               # [B, T, D]

        # Causal mask trimmed to current sequence length
        causal_mask = self.causal_mask_full[:T, :T]         # [T, T]
        # Expand for multi-head: [1, 1, T, T]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        h = content
        for block in self.blocks:
            h = block(h, category, causal_mask)
        h = self.final_norm(h)                              # [B, T, D]

        # Extract intent vector at the last valid position for each sequence
        # lengths are 1-indexed: position index = length - 1
        idx = (lengths - 1).clamp(min=0)                   # [B]
        intent = h[torch.arange(B, device=device), idx]    # [B, D]

        cat_logits = self.category_head(h)                  # [B, T, num_cats]

        return h, intent, cat_logits

    def score_candidates(self, intent: torch.Tensor,
                         candidate_bge: torch.Tensor) -> torch.Tensor:
        """
        Score candidates against the user intent vector.

        Args:
            intent:        [B, hidden_dim]
            candidate_bge: [N, 1024]
        Returns:
            scores [B, N]
        """
        cand_proj = self.candidate_proj(candidate_bge)     # [N, D]
        return intent @ cand_proj.T                         # [B, N]


# ─────────────────────────────────────────────────────────────────────────────
# Agent (training + inference interface)
# ─────────────────────────────────────────────────────────────────────────────

class DIFSASRecAgent:
    """
    High-level interface for DIF-SASRec — drop-in replacement for RLSequentialFilter
    in the personal "You Might Like" pipeline.

    Handles:
      - Building BGE-M3 tensors from FAISS reconstruction
      - Online training via sampled softmax + category auxiliary loss
      - Checkpoint save/load
      - Candidate scoring for the recommendation funnel
    """

    def __init__(self, retriever, category_encoder, pretrained_path: str = None):
        self.retriever        = retriever
        self.category_encoder = category_encoder
        self.device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_cats = category_encoder.num_categories if category_encoder else 2
        self.model = DIFSASRecModel(num_categories=num_cats).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Mixed-precision training (no-op on CPU — GradScaler requires CUDA)
        self._amp_enabled = self.device.type == "cuda"
        self.scaler       = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        self.scheduler    = None   # set via configure_scheduler() before training

        self._step        = 0
        self.loss_history = []
        self._emb_cache: dict = {}   # populated by set_embedding_cache() during pretraining

        # Pool of all known ASINs for negative sampling (online training)
        self._all_asins = list(retriever.asin_to_idx.keys()) if retriever else []

        if pretrained_path and os.path.exists(pretrained_path):
            self.load(pretrained_path)
        else:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"[DIFSASRecAgent] Initialized fresh model — "
                  f"{param_count:,} params  device={self.device}  "
                  f"num_cats={num_cats}")

    # ── Tensor building ───────────────────────────────────────────────────────

    def _build_tensors(self, click_seq_asins: list, cat_ids: list = None):
        """
        Reconstruct BGE-M3 embeddings from FAISS and build model-ready tensors.

        Args:
            click_seq_asins: list of ASIN strings (ordered, most recent last)
            cat_ids:         list of int category IDs (same length); uses UNK if None

        Returns:
            bge_tensor  [1, T, 1024]
            cat_tensor  [1, T]         (long)
            lengths     [1]            actual sequence length T
        """
        seq = click_seq_asins[-MAX_SEQ_LEN:]
        bge_list, valid_cats = [], []

        for i, asin in enumerate(seq):
            if asin not in self.retriever.asin_to_idx:
                continue
            idx = self.retriever.asin_to_idx[asin]
            vec = self.retriever.text_flat.reconstruct(idx)   # [1024] BGE-M3
            bge_list.append(vec)
            if cat_ids is not None:
                # cat_ids may be shorter than seq if some ASINs were unknown
                valid_cats.append(cat_ids[i] if i < len(cat_ids) else self.category_encoder.UNK_ID)
            else:
                valid_cats.append(self.category_encoder.UNK_ID if self.category_encoder
                                  else 1)

        T = len(bge_list)
        if T == 0:
            return None, None, None

        bge_arr = np.zeros((MAX_SEQ_LEN, TEXT_EMBED_DIM), dtype=np.float32)
        cat_arr = np.zeros(MAX_SEQ_LEN,                   dtype=np.int64)
        bge_arr[:T] = np.array(bge_list)
        cat_arr[:T] = np.array(valid_cats)

        bge_t = torch.FloatTensor(bge_arr).unsqueeze(0).to(self.device)  # [1, MAX, 1024]
        cat_t = torch.LongTensor(cat_arr).unsqueeze(0).to(self.device)   # [1, MAX]
        len_t = torch.tensor([T], device=self.device)

        return bge_t, cat_t, len_t

    def set_embedding_cache(self, cache: dict):
        """
        Inject a pre-loaded {asin: np.ndarray[1024]} cache.
        When set, _get_asin_vec serves from RAM instead of FAISS mmap.
        Used by the pretraining script to eliminate per-step disk I/O.
        Online training (no cache) continues to use FAISS normally.
        """
        self._emb_cache = cache

    def configure_scheduler(self, total_steps: int, warmup_steps: int):
        """
        Attach a linear-warmup + cosine-decay LR scheduler.

        Call this once before the training loop starts, after the total number
        of gradient steps is known. The scheduler is stepped inside
        train_step_batch() automatically.

        Args:
            total_steps:  total gradient steps across all epochs
            warmup_steps: steps over which LR rises linearly from 0 → peak LR
        """
        import math
        from torch.optim.lr_scheduler import LambdaLR

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup: 0 → 1
                return step / max(1, warmup_steps)
            # Cosine decay: 1 → 0.05 (never fully zero — keeps fine-tuning stable)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, _lr_lambda)
        print(f"[DIFSASRecAgent] Scheduler: linear warmup {warmup_steps:,} steps "
              f"→ cosine decay to {total_steps:,} steps")

    def _get_asin_vec(self, asin: str):
        """Return BGE-M3 vector [1024] for an ASIN, or None."""
        # Fast path: serve from pre-loaded RAM cache (pretraining only)
        if self._emb_cache and asin in self._emb_cache:
            return self._emb_cache[asin]
        # Slow path: reconstruct from memory-mapped FAISS (online training)
        if asin not in self.retriever.asin_to_idx:
            return None
        idx = self.retriever.asin_to_idx[asin]
        return self.retriever.text_flat.reconstruct(idx)

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_intent_vector(self, click_seq_asins: list,
                          cat_ids: list = None) -> np.ndarray | None:
        """
        Encode a click sequence into a 256-dim intent vector.

        Returns None if no valid ASINs in the sequence.
        """
        bge_t, cat_t, len_t = self._build_tensors(click_seq_asins, cat_ids)
        if bge_t is None:
            return None

        self.model.eval()
        with torch.no_grad():
            _, intent, _ = self.model(bge_t, cat_t, len_t)
        return intent.squeeze(0).cpu().numpy()     # [256]

    def get_candidate_scores(self, click_seq_asins: list, cat_ids: list,
                              candidate_asins: list) -> dict:
        """
        Score a list of candidate ASINs against the user's current intent.

        Drop-in replacement for RLSequentialFilter.get_candidate_scores().

        Args:
            click_seq_asins: user's click history
            cat_ids:         category IDs for the click history
            candidate_asins: ASINs to score

        Returns:
            {asin: float_score}
        """
        if not click_seq_asins or not candidate_asins:
            return {asin: 0.0 for asin in candidate_asins}

        bge_t, cat_t, len_t = self._build_tensors(click_seq_asins, cat_ids)
        if bge_t is None:
            return {asin: 0.0 for asin in candidate_asins}

        self.model.eval()
        with torch.no_grad():
            _, intent, _ = self.model(bge_t, cat_t, len_t)   # intent [1, 256]

        # Gather candidate BGE-M3 vectors
        valid_asins, cand_vecs = [], []
        for asin in candidate_asins:
            vec = self._get_asin_vec(asin)
            if vec is not None:
                valid_asins.append(asin)
                cand_vecs.append(vec)

        if not valid_asins:
            return {}

        cand_t = torch.FloatTensor(np.array(cand_vecs)).to(self.device)  # [N, 1024]
        with torch.no_grad():
            scores = self.model.score_candidates(intent, cand_t)          # [1, N]
        scores_np = scores.squeeze(0).cpu().numpy()

        return {asin: float(s) for asin, s in zip(valid_asins, scores_np)}

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_batch_tensors(self, batch_seqs: list):
        """
        Build padded batch tensors from multiple sequences (all from emb_cache).

        Args:
            batch_seqs: list of ASIN lists (variable length, most recent last)
        Returns:
            bge_t   [B, MAX_SEQ_LEN, 1024]  float32
            cat_t   [B, MAX_SEQ_LEN]         int64
            len_t   [B]                       int64  actual sequence lengths
        """
        B = len(batch_seqs)
        bge_arr = np.zeros((B, MAX_SEQ_LEN, TEXT_EMBED_DIM), dtype=np.float32)
        cat_arr = np.zeros((B, MAX_SEQ_LEN),                  dtype=np.int64)
        lengths = np.zeros(B,                                  dtype=np.int64)

        for i, seq in enumerate(batch_seqs):
            truncated = seq[-MAX_SEQ_LEN:]
            vecs, cats = [], []
            for asin in truncated:
                vec = self._get_asin_vec(asin)
                if vec is not None:
                    vecs.append(vec)
                    cats.append(
                        self.category_encoder.get_category_id(asin)
                        if self.category_encoder else 1
                    )
            T = len(vecs)
            if T > 0:
                bge_arr[i, :T] = vecs
                cat_arr[i, :T] = cats
            lengths[i] = T

        bge_t = torch.FloatTensor(bge_arr).to(self.device)
        cat_t = torch.LongTensor(cat_arr).to(self.device)
        len_t = torch.tensor(lengths, device=self.device)
        return bge_t, cat_t, len_t

    def train_step_batch(self, batch_seqs: list, target_asins: list,
                         target_cat_ids: list,
                         neg_pool_vecs: np.ndarray) -> float | None:
        """
        Batched training step — processes B sequences in one GPU forward pass.

        Uses shared negatives across the batch (standard in large-scale rec systems):
          logits = [B, 1+K]  where col 0 is the positive score, cols 1..K are shared negatives.
          target = [0, 0, ..., 0]  (positive is always index 0 for every sample)

        Args:
            batch_seqs:     list of B ASIN sequences (input, before the target)
            target_asins:   list of B target ASIN strings
            target_cat_ids: list of B int category IDs for the targets
            neg_pool_vecs:  [M, 1024] numpy array — pre-loaded negative embeddings

        Returns:
            mean loss over the batch, or None if batch has no valid sequences
        """
        # Filter out sequences/targets that have no embeddings in cache
        valid = []
        for seq, tgt, cat in zip(batch_seqs, target_asins, target_cat_ids):
            if self._get_asin_vec(tgt) is not None:
                valid.append((seq, tgt, cat))
        if not valid:
            return None

        seqs       = [v[0] for v in valid]
        tgt_asins  = [v[1] for v in valid]
        tgt_cats   = [v[2] for v in valid]
        B          = len(valid)

        # ── Build sequence tensors ────────────────────────────────────────────
        bge_t, cat_t, len_t = self._build_batch_tensors(seqs)

        # Skip sequences the model can't encode (all items unknown)
        valid_seq_mask = (len_t > 0)
        if not valid_seq_mask.any():
            return None

        bge_t  = bge_t[valid_seq_mask]
        cat_t  = cat_t[valid_seq_mask]
        len_t  = len_t[valid_seq_mask]
        tgt_asins = [a for a, m in zip(tgt_asins, valid_seq_mask.cpu().tolist()) if m]
        tgt_cats  = [c for c, m in zip(tgt_cats,  valid_seq_mask.cpu().tolist()) if m]
        B = len(tgt_asins)

        # ── Positive + negative tensors (built outside autocast — float32 input) ─
        pos_vecs = np.array([self._get_asin_vec(a) for a in tgt_asins], dtype=np.float32)
        pos_t    = torch.FloatTensor(pos_vecs).to(self.device)     # [B, 1024]

        K       = min(NUM_NEGATIVES, len(neg_pool_vecs))
        neg_idx = np.random.choice(len(neg_pool_vecs), K, replace=False)
        neg_np  = neg_pool_vecs[neg_idx]                           # [K, 1024]
        neg_t   = torch.FloatTensor(neg_np).to(self.device)

        # ── Forward pass + loss under AMP autocast ────────────────────────────
        self.model.train()
        with torch.autocast(device_type=self.device.type, enabled=self._amp_enabled):
            _, intent, cat_logits = self.model(bge_t, cat_t, len_t)   # [B,512], [B,T,C]

            pos_proj = self.model.candidate_proj(pos_t)                # [B, 512]
            neg_proj = self.model.candidate_proj(neg_t)                # [K, 512]

            # Sampled softmax: positive is always index 0
            scores_pos = (intent * pos_proj).sum(dim=1, keepdim=True)  # [B, 1]
            scores_neg = intent @ neg_proj.T                            # [B, K]
            logits     = torch.cat([scores_pos, scores_neg], dim=1)    # [B, 1+K]
            targets    = torch.zeros(B, dtype=torch.long, device=self.device)
            loss_softmax = F.cross_entropy(logits, targets)

            # Category auxiliary loss
            last_idx    = (len_t - 1).clamp(min=0)
            last_logits = cat_logits[torch.arange(B, device=self.device), last_idx]
            cat_tgt     = torch.tensor(tgt_cats, dtype=torch.long, device=self.device)
            num_cats    = self.model.category_head.out_features
            valid_cat   = (cat_tgt >= 0) & (cat_tgt < num_cats)
            if valid_cat.any():
                loss_cat = F.cross_entropy(last_logits[valid_cat], cat_tgt[valid_cat])
            else:
                loss_cat = torch.tensor(0.0, device=self.device)

            total_loss = loss_softmax + CAT_AUX_WEIGHT * loss_cat

        # ── AMP-aware backprop ────────────────────────────────────────────────
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)                        # needed before clip
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Only advance the LR schedule when the optimizer actually stepped.
        # If AMP detected inf/nan gradients it skips the optimizer and reduces
        # the loss scale — detect that by a scale decrease.
        if self.scheduler is not None and self.scaler.get_scale() >= scale_before:
            self.scheduler.step()

        self._step += 1
        loss_val = float(total_loss.item())
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 500:
            self.loss_history = self.loss_history[-500:]
        return loss_val

    def train_step(self, click_seq_asins: list, target_asin: str,
                   target_cat_id: int, neg_pool_asins: list,
                   neg_pool_vecs: np.ndarray = None) -> float | None:
        """
        One online training step using sampled softmax loss.

        Loss = SampledSoftmax(intent, target_bge, neg_bge_samples)
             + CAT_AUX_WEIGHT * CrossEntropy(cat_logits[-1], target_cat_id)

        Args:
            click_seq_asins: input sequence (items seen before target)
            target_asin:     the item the user actually clicked next
            target_cat_id:   category ID of the target
            neg_pool_asins:  pool to sample negatives from (all_asins)

        Returns:
            total loss as float, or None if sequence has no valid ASINs
        """
        target_vec = self._get_asin_vec(target_asin)
        if target_vec is None:
            return None

        bge_t, cat_t, len_t = self._build_tensors(click_seq_asins)
        if bge_t is None:
            return None

        self.model.train()
        _, intent, cat_logits = self.model(bge_t, cat_t, len_t)   # intent [1, 256]

        # ── Sampled softmax loss ─────────────────────────────────────────────
        if neg_pool_vecs is not None:
            # Fast path (pretraining): sample rows from pre-loaded numpy array
            # neg_pool_vecs is [M, 1024] already in RAM — no FAISS reads needed
            n_neg    = min(NUM_NEGATIVES, len(neg_pool_vecs))
            neg_idx  = np.random.choice(len(neg_pool_vecs), size=n_neg, replace=False)
            neg_vecs = [neg_pool_vecs[i] for i in neg_idx]
        else:
            # Slow path (online training): reconstruct from FAISS per-step
            neg_pool  = [a for a in neg_pool_asins if a != target_asin]
            n_neg     = min(NUM_NEGATIVES, len(neg_pool))
            neg_asins = np.random.choice(neg_pool, size=n_neg, replace=False).tolist()
            neg_vecs  = [v for a in neg_asins
                         if (v := self._get_asin_vec(a)) is not None]

        if not neg_vecs:
            return None

        # Stack positive + negatives: [1+K, 1024]
        pos_t  = torch.FloatTensor(target_vec).unsqueeze(0).to(self.device)    # [1, 1024]
        neg_t  = torch.FloatTensor(np.array(neg_vecs)).to(self.device)         # [K, 1024]
        all_t  = torch.cat([pos_t, neg_t], dim=0)                              # [1+K, 1024]

        scores = self.model.score_candidates(intent, all_t).squeeze(0)         # [1+K]
        # Label 0 = positive (first element)
        target_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        softmax_loss = F.cross_entropy(scores.unsqueeze(0), target_idx)

        # ── Category auxiliary loss ──────────────────────────────────────────
        last_pos  = (len_t - 1).clamp(min=0)                                   # [1]
        last_logit = cat_logits[0, last_pos[0]]                                # [num_cats]
        cat_target = torch.tensor([target_cat_id], device=self.device)
        cat_loss   = F.cross_entropy(last_logit.unsqueeze(0), cat_target)

        total_loss = softmax_loss + CAT_AUX_WEIGHT * cat_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step += 1
        loss_val = float(total_loss.item())
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 200:
            self.loss_history = self.loss_history[-200:]

        return loss_val

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model state, optimizer state, and step counter."""
        torch.save({
            "arch":            "dif_sasrec_v1",
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step":            self._step,
            "loss_history":    self.loss_history,
            "num_categories":  self.model.category_emb.num_embeddings,
        }, path)
        print(f"[DIFSASRecAgent] Saved checkpoint to {path} (step={self._step})")

    def load(self, path: str):
        """Load checkpoint from disk. Skips if arch key mismatches."""
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if ckpt.get("arch") != "dif_sasrec_v1":
            print(f"[DIFSASRecAgent] Skipping {path} — arch mismatch "
                  f"(got '{ckpt.get('arch')}')")
            return

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._step        = ckpt.get("step", 0)
        self.loss_history = ckpt.get("loss_history", [])
        self.model.eval()
        print(f"[DIFSASRecAgent] Loaded from {path} (step={self._step})")
