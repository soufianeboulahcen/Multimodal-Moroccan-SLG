"""SignLLM model — text → 3D pose sequence.

Reproduces the SignLLM architecture from arXiv:2405.10718 §4.1–4.3 in PyTorch.
Hyperparameters are locked per docs/DECISIONS.md (2026-05-10):

    d_model=768, heads=12, d_ff=3072, dropout=0.1, layers=2 (enc) + 2 (dec)
    sinusoidal positional encoding (Vaswani 2017)
    pre-norm: False (Vaswani 2017 default = post-norm)
    activation: ReLU (Vaswani 2017 default)

For our single-language MoSL setting, MLSF and Prompt2LangGloss collapse to the
same model (no language-switch table, no language-prefix on tokens).  The
model is structured so multi-language support is a pure scale-up later if we
ever extend.

Forward pass:
    text_tokens (B, L)            text encoder        text_features (B, L, d)
                                       │
                                       ▼
                    + sinusoidal PE → 2-layer Transformer Enc

    pose_target (B, T, 150) ───┐
    time         (B, T)         ├── concat → (B, T, 151) → linear → (B, T, d)
                                │
                  shift right ──┘    + sinusoidal PE → 2-layer Transformer Dec
                                                          (cross-attn over text)
                                                          │
                                                          ▼
                                                   linear → pose_pred (B, T, 150)

    text_features → mean-pool over real tokens → length_head → log_T_pred (B,)

Heads (output):
    pose_pred  (B, T, 150)   — per-frame 50-joint × (x,y,z) prediction
    log_T_pred (B,)          — predicted log T (scalar per clip)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mosl.model.positional import SinusoidalPositionalEncoding


# Per-frame coord count from the .skels format: 50 joints × (x, y, z).  Matches
# COORDS_PER_FRAME in mosl/data/dataset.py.
COORDS_PER_FRAME = 150


@dataclass
class SignLLMConfig:
    """All hyperparameters in one place. Locked in docs/DECISIONS.md."""
    vocab_size: int                  # set by tokenizer
    d_model: int = 768
    nhead: int = 12
    d_ff: int = 3072
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1
    pose_dim: int = COORDS_PER_FRAME
    max_text_len: int = 32           # our labels are 1 token + bos/eos = 3; 32 is generous
    max_pose_len: int = 256          # observed max T = 236 (see docs/STATS.md); pad to 256
    pad_id: int = 0


class SignLLM(nn.Module):
    """Text-to-pose Sign Language Production model.

    The decoder is autoregressive in the time dimension during training (with
    teacher forcing + a causal mask) and at inference (one-step-at-a-time).
    """

    def __init__(self, config: SignLLMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # --- Text side -----------------------------------------------------
        self.text_embed = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_id)
        self.text_pos = SinusoidalPositionalEncoding(d, max_len=config.max_text_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,           # post-norm = Vaswani 2017
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_enc_layers)

        # --- Pose side -----------------------------------------------------
        # Decoder input: per-frame pose (150) + time marker (1) → linear → d.
        self.pose_proj_in = nn.Linear(config.pose_dim + 1, d)
        self.pose_pos = SinusoidalPositionalEncoding(d, max_len=config.max_pose_len)

        # Learnable start-of-sequence frame embedding (added in front of the
        # shifted decoder input; needed for the first-step prediction).
        self.start_frame = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.start_frame, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.pose_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_dec_layers)

        # --- Heads ---------------------------------------------------------
        self.pose_head = nn.Linear(d, config.pose_dim)
        self.length_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization (Vaswani 2017 doesn't pin a scheme; PyTorch default is
    # Xavier-uniform for Linear and N(0,1) for Embedding which is too large.
    # We use the same N(0, 0.02) that BERT and GPT use — empirically stable.)
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    def encode_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """text_ids (B, L) → text_features (B, L, d).
        text_mask (B, L) is True for real tokens, False for padding."""
        x = self.text_embed(text_ids) * (self.config.d_model ** 0.5)   # Vaswani §3.4
        x = self.text_pos(x)
        # PyTorch convention: src_key_padding_mask is True where padding.
        x = self.text_encoder(x, src_key_padding_mask=~text_mask)
        return x

    def predict_length(self, text_features: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over real text tokens, then MLP → log T prediction.
        Returns (B,) tensor of predicted log T."""
        mask_f = text_mask.unsqueeze(-1).to(text_features.dtype)        # (B, L, 1)
        pooled = (text_features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.length_head(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # Decoder (training: teacher-forced full sequence; inference: step)
    # ------------------------------------------------------------------
    def _embed_pose_frames(
        self, pose: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate time marker to each frame, project to d_model, add PE."""
        # pose (B, T, 150), time (B, T) → (B, T, 151)
        x = torch.cat([pose, time.unsqueeze(-1)], dim=-1)
        x = self.pose_proj_in(x)                                        # (B, T, d)
        x = self.pose_pos(x)
        return x

    def decode(
        self,
        decoder_input_emb: torch.Tensor,         # (B, T, d) already PE'd
        text_features: torch.Tensor,             # (B, L, d)
        text_mask: torch.Tensor,                 # (B, L)
        tgt_key_padding_mask: torch.Tensor | None = None,   # (B, T), True for pad
    ) -> torch.Tensor:
        """Run the transformer decoder + pose head.  Returns (B, T, 150).
        Caller is responsible for shift-right + start-frame prepending."""
        T = decoder_input_emb.size(1)
        # Causal mask: True at positions to be masked out (i.e., future).  Bool
        # dtype matches our padding masks — avoids torch's
        # "mismatched key_padding_mask and attn_mask" deprecation warning.
        causal = torch.ones(T, T, dtype=torch.bool, device=decoder_input_emb.device).triu(diagonal=1)
        h = self.pose_decoder(
            tgt=decoder_input_emb,
            memory=text_features,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=~text_mask,
        )
        return self.pose_head(h)

    # ------------------------------------------------------------------
    # Training forward pass — teacher forced
    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor,                  # (B, L)
        text_mask: torch.Tensor,                 # (B, L)
        pose_target: torch.Tensor,               # (B, T, 150)
        time: torch.Tensor,                      # (B, T)
        pose_mask: torch.Tensor,                 # (B, T) — True for real frames
    ) -> dict:
        """Returns predictions for both heads.  Loss computation is the caller's
        responsibility (so we can swap MSE / RL / RL+PLC without touching the model)."""
        text_features = self.encode_text(text_ids, text_mask)
        log_T_pred = self.predict_length(text_features, text_mask)

        # Build decoder input by shifting target right and prepending the
        # learned start frame.  Drop the last frame so length stays = T.
        # decoder_input[:, t] is what the decoder *sees* when predicting target[:, t].
        prev_pose = torch.zeros_like(pose_target)                       # (B, T, 150)
        prev_pose[:, 1:] = pose_target[:, :-1]
        prev_time = torch.zeros_like(time)
        prev_time[:, 1:] = time[:, :-1]
        dec_in_emb = self._embed_pose_frames(prev_pose, prev_time)
        # Replace position 0's embedding with the learnable start frame so the
        # decoder has a real signal there instead of the projection of zeros.
        start = self.start_frame.expand(dec_in_emb.size(0), 1, -1)
        dec_in_emb = torch.cat([start, dec_in_emb[:, 1:]], dim=1)

        pose_pred = self.decode(
            dec_in_emb, text_features, text_mask,
            tgt_key_padding_mask=~pose_mask,
        )

        return {
            "pose_pred": pose_pred,         # (B, T, 150)
            "log_T_pred": log_T_pred,       # (B,)
            "text_features": text_features, # (B, L, d) — kept for diagnostics
        }

    # ------------------------------------------------------------------
    # Inference — autoregressive decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,                  # (B, L)
        text_mask: torch.Tensor,                 # (B, L)
        max_T: int | None = None,
    ) -> dict:
        """Generate a pose sequence per text input.  Length T is predicted from
        the text by the length head and clamped to [1, max_pose_len]."""
        text_features = self.encode_text(text_ids, text_mask)
        log_T_pred = self.predict_length(text_features, text_mask)
        T_pred = log_T_pred.exp().round().long().clamp(min=1, max=self.config.max_pose_len)
        if max_T is not None:
            T_pred = T_pred.clamp(max=max_T)
        # We decode each clip up to its predicted T but unroll the longest in
        # the batch; shorter clips stop early via the returned `lengths`.
        T_max = int(T_pred.max().item())
        B = text_ids.size(0)
        device = text_ids.device

        pose_so_far = torch.zeros(B, 0, self.config.pose_dim, device=device)
        time_so_far = torch.zeros(B, 0, device=device)

        for t in range(T_max):
            # Build decoder input from frames generated so far, plus start frame.
            if pose_so_far.size(1) == 0:
                dec_in_emb = self.start_frame.expand(B, 1, -1)
            else:
                emb = self._embed_pose_frames(pose_so_far, time_so_far)
                start = self.start_frame.expand(B, 1, -1)
                dec_in_emb = torch.cat([start, emb], dim=1)

            preds = self.decode(dec_in_emb, text_features, text_mask)
            next_frame = preds[:, -1, :]                                # (B, 150)
            pose_so_far = torch.cat([pose_so_far, next_frame.unsqueeze(1)], dim=1)
            # Time at the *next* step (i+1)/T — 1-indexed per the .skels format.
            next_time = torch.full((B,), (t + 1) / max(T_max, 1), device=device)
            time_so_far = torch.cat([time_so_far, next_time.unsqueeze(1)], dim=1)

        return {
            "pose": pose_so_far,            # (B, T_max, 150)
            "lengths": T_pred,              # (B,) — actual per-clip length
            "log_T_pred": log_T_pred,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: build the locked config, count params, run a forward pass on
    # synthetic data of the right shape.
    import sys
    sys.path.insert(0, ".")
    from mosl.text.tokenizer import WordTokenizer

    tok = WordTokenizer.load("data/processed/vocab.json")
    cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    model = SignLLM(cfg)
    n = count_parameters(model)
    print(f"vocab_size={tok.vocab_size}")
    print(f"config: {cfg}")
    print(f"trainable parameters: {n:,}  (~{n / 1e6:.1f} M)")

    B, L, T = 4, 3, 100
    text_ids = torch.zeros(B, L, dtype=torch.long)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    pose = torch.randn(B, T, 150)
    time = torch.linspace(0.01, 1.0, T).unsqueeze(0).expand(B, -1)
    pose_mask = torch.ones(B, T, dtype=torch.bool)

    out = model(text_ids, text_mask, pose, time, pose_mask)
    print()
    print("forward shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>15}: {tuple(v.shape)}")

    gen = model.generate(text_ids, text_mask, max_T=10)
    print()
    print("generate shapes (max_T=10):")
    for k, v in gen.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>15}: {tuple(v.shape)}  values={v.tolist() if v.dim() == 1 else '...'}")
