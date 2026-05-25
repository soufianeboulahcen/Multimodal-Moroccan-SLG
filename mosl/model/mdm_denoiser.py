"""MDM-style Motion Diffusion denoiser for MoSL.

Architecture: transformer denoiser that predicts the clean pose x0 from a
noisy pose x_t, conditioned on Arabic text embeddings produced by the frozen
SignLLM text encoder and optionally on a signer style embedding.

Design decisions (see docs/avatar_pipeline.md):
  - x0-prediction (not epsilon): empirically better for motion quality and
    allows direct geometric losses on the predicted clean pose.
  - Reuses SignLLM.encode_text verbatim — weights are frozen during diffusion
    training so the text representation is stable.
  - Temporal self-attention across the full T dimension ensures frame-to-frame
    coherence without an explicit recurrence.
  - Cross-attention over text features conditions each frame on the full Arabic
    label context.
  - AdaLN (adaptive layer norm) injects both the diffusion timestep embedding
    and the optional signer style embedding into every transformer layer.
  - Pose dimension is 150 (50 joints × xyz) — same as the existing .skels
    format; no format change required.

Joint index conventions (from Prompt2Sign / OpenPose COCO-18 + MANO):
  Joints  0-17  → body (COCO-18 subset used by Prompt2Sign)
  Joints 18-38  → left hand (MANO 21 joints)
  Joints 39-49  → right hand (partial MANO, 11 joints from OpenPose 21)

  Hand joints receive a 3× loss weight because hands carry the primary
  lexical content of sign language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mosl.model.positional import SinusoidalPositionalEncoding
from mosl.model.signllm import SignLLM, SignLLMConfig

# Pose dimensionality — must match COORDS_PER_FRAME in dataset.py
POSE_DIM = 150

# Hand joint range in the 50-joint layout (joints 18–49 inclusive).
# Each joint contributes 3 coords (xyz), so the coord range is [54, 150).
HAND_JOINT_START = 18
HAND_JOINT_END = 50   # exclusive
HAND_COORD_START = HAND_JOINT_START * 3   # 54
HAND_COORD_END = HAND_JOINT_END * 3       # 150


def build_hand_weight(pose_dim: int = POSE_DIM, hand_weight: float = 3.0) -> torch.Tensor:
    """Return a (pose_dim,) weight tensor with hand coords upweighted."""
    w = torch.ones(pose_dim)
    w[HAND_COORD_START:HAND_COORD_END] = hand_weight
    return w


# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal → MLP, standard diffusion practice)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Maps integer diffusion timestep t → dense embedding of size out_dim."""

    def __init__(self, out_dim: int, max_steps: int = 1000) -> None:
        super().__init__()
        self.out_dim = out_dim
        # Sinusoidal base (same formula as positional encoding but 1-D input)
        half = out_dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        self.register_buffer("freqs", freqs)   # (half,)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, out_dim)."""
        t_f = t.float().unsqueeze(-1)                    # (B, 1)
        args = t_f * self.freqs.unsqueeze(0)             # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, out_dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# AdaLN modulation block
# ---------------------------------------------------------------------------

class AdaLNBlock(nn.Module):
    """Single transformer layer with AdaLN conditioning.

    Applies adaptive layer norm modulation from a conditioning vector
    (timestep + optional signer style) before each sub-layer, following
    DiT (Peebles & Xie 2023) and MDM (Tevet et al. 2022).
    """

    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 dropout: float = 0.1, cond_dim: int = 512) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: one MLP produces scale+shift for each of the 3 norms.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),   # 2 (scale+shift) × 3 norms
        )
        # Zero-init so the model starts as identity (DiT §3.3)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, d_model)
        context: torch.Tensor,                  # (B, L, d_model) — text features
        cond: torch.Tensor,                     # (B, cond_dim) — t_emb [+ style]
        context_key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=pad
    ) -> torch.Tensor:
        # Compute 6 modulation params from the conditioning vector
        mods = self.adaLN_modulation(cond)      # (B, 6*d_model)
        s1, b1, s2, b2, s3, b3 = mods.chunk(6, dim=-1)  # each (B, d_model)
        s1 = s1.unsqueeze(1); b1 = b1.unsqueeze(1)
        s2 = s2.unsqueeze(1); b2 = b2.unsqueeze(1)
        s3 = s3.unsqueeze(1); b3 = b3.unsqueeze(1)

        # Self-attention with AdaLN
        h = (1 + s1) * self.norm1(x) + b1
        attn_out, _ = self.self_attn(h, h, h)
        x = x + attn_out

        # Cross-attention over text with AdaLN
        h = (1 + s2) * self.norm2(x) + b2
        mem_mask = ~context_key_padding_mask if context_key_padding_mask is not None else None
        cross_out, _ = self.cross_attn(h, context, context,
                                        key_padding_mask=context_key_padding_mask)
        x = x + cross_out

        # Feed-forward with AdaLN
        h = (1 + s3) * self.norm3(x) + b3
        x = x + self.ff(h)
        return x


# ---------------------------------------------------------------------------
# MDM Denoiser
# ---------------------------------------------------------------------------

@dataclass
class MDMConfig:
    """Hyperparameters for the MDM denoiser.

    Defaults are sized to fit in 18 GB VRAM (RTX 3090) with batch_size=32,
    T=196, using bf16 + gradient checkpointing.
    """
    # Pose / motion
    pose_dim: int = POSE_DIM                  # 150 — must match .skels format
    max_pose_len: int = 256                   # max T; matches SignLLMConfig

    # Denoiser transformer
    d_model: int = 512                        # smaller than SignLLM (768) to save VRAM
    nhead: int = 8
    d_ff: int = 2048
    n_layers: int = 8
    dropout: float = 0.1

    # Conditioning
    cond_dim: int = 512                       # timestep + style embedding size
    text_d_model: int = 768                   # must match SignLLMConfig.d_model
    use_signer_style: bool = False            # enable once signer_id is annotated

    # Diffusion
    n_diffusion_steps: int = 1000
    hand_loss_weight: float = 3.0            # upweight hand coords in all losses

    # SignLLM text encoder checkpoint (frozen)
    signllm_checkpoint: Optional[str] = None  # path to best.pt from SignLLM training


class MDMDenoiser(nn.Module):
    """Motion Diffusion Model denoiser.

    Takes a noisy pose sequence x_t and predicts the clean pose x0.
    Conditioned on:
      - Arabic text via frozen SignLLM text encoder (cross-attention)
      - Diffusion timestep t (AdaLN)
      - Optional signer style embedding (AdaLN, added to timestep embedding)

    The SignLLM text encoder is loaded from a checkpoint and frozen.
    Only the denoiser parameters are trained.
    """

    def __init__(self, config: MDMConfig, signllm_config: SignLLMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # --- Frozen text encoder (SignLLM) ---------------------------------
        self._signllm_cfg = signllm_config
        self.text_encoder_model = SignLLM(signllm_config)
        # Project text features from SignLLM d_model (768) to denoiser d_model
        self.text_proj = nn.Linear(config.text_d_model, d)

        # --- Timestep embedding --------------------------------------------
        self.time_embed = TimestepEmbedding(config.cond_dim)

        # --- Pose input projection -----------------------------------------
        # Noisy pose (pose_dim) → d_model; no time marker needed (t is in cond)
        self.pose_proj_in = nn.Linear(config.pose_dim, d)
        self.pose_pos = SinusoidalPositionalEncoding(d, max_len=config.max_pose_len)

        # --- Denoiser transformer layers -----------------------------------
        self.layers = nn.ModuleList([
            AdaLNBlock(d, config.nhead, config.d_ff, config.dropout, config.cond_dim)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(d)

        # --- Output head ---------------------------------------------------
        self.pose_head = nn.Linear(d, config.pose_dim)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)

        # --- Hand loss weight buffer ---------------------------------------
        self.register_buffer(
            "hand_weight",
            build_hand_weight(config.pose_dim, config.hand_loss_weight),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.pose_head:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_text_encoder(self, checkpoint_path: str) -> None:
        """Load SignLLM weights into the text encoder and freeze it."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        self.text_encoder_model.load_state_dict(state, strict=False)
        for p in self.text_encoder_model.parameters():
            p.requires_grad = False
        self.text_encoder_model.eval()

    def freeze_text_encoder(self) -> None:
        """Freeze text encoder without loading a checkpoint (e.g. for testing)."""
        for p in self.text_encoder_model.parameters():
            p.requires_grad = False

    def encode_text(
        self,
        text_ids: torch.Tensor,    # (B, L)
        text_mask: torch.Tensor,   # (B, L) True=real
    ) -> torch.Tensor:
        """Run frozen SignLLM encoder → project to denoiser d_model.
        Returns (B, L, d_model)."""
        with torch.no_grad():
            text_features = self.text_encoder_model.encode_text(text_ids, text_mask)
        return self.text_proj(text_features)   # (B, L, d)

    def forward(
        self,
        x_noisy: torch.Tensor,                 # (B, T, 150) noisy pose at step t
        t: torch.Tensor,                       # (B,) integer diffusion timesteps
        text_ids: torch.Tensor,                # (B, L)
        text_mask: torch.Tensor,               # (B, L) True=real token
        style_emb: Optional[torch.Tensor] = None,  # (B, cond_dim) signer style
        pose_mask: Optional[torch.Tensor] = None,  # (B, T) True=real frame
    ) -> torch.Tensor:
        """Predict clean pose x0 from noisy pose x_t.

        Returns (B, T, 150) — the predicted clean pose sequence.
        """
        # Text conditioning
        text_ctx = self.encode_text(text_ids, text_mask)   # (B, L, d)
        text_pad_mask = ~text_mask                          # True=padding (PyTorch convention)

        # Timestep conditioning
        cond = self.time_embed(t)                           # (B, cond_dim)
        if style_emb is not None and self.config.use_signer_style:
            cond = cond + style_emb                         # additive fusion

        # Pose embedding
        x = self.pose_proj_in(x_noisy)                     # (B, T, d)
        x = self.pose_pos(x)

        # Denoiser transformer
        for layer in self.layers:
            x = layer(x, text_ctx, cond, context_key_padding_mask=text_pad_mask)

        x = self.final_norm(x)
        x0_pred = self.pose_head(x)                        # (B, T, 150)
        return x0_pred

    def diffusion_loss(
        self,
        x0_pred: torch.Tensor,     # (B, T, 150) predicted clean pose
        x0_target: torch.Tensor,   # (B, T, 150) ground-truth clean pose
        pose_mask: torch.Tensor,   # (B, T) True=real frame
    ) -> torch.Tensor:
        """Weighted MSE loss with hand upweighting, masked to real frames.

        Returns scalar loss.
        """
        diff = (x0_pred - x0_target) ** 2                  # (B, T, 150)
        # Apply hand weight: (150,) broadcast over (B, T, 150)
        diff = diff * self.hand_weight.unsqueeze(0).unsqueeze(0)
        per_frame = diff.mean(dim=-1)                       # (B, T)
        mask = pose_mask.to(per_frame.dtype)
        n_real = mask.sum(dim=1).clamp(min=1.0)            # (B,)
        per_sample = (per_frame * mask).sum(dim=1) / n_real  # (B,)
        return per_sample.mean()


def count_denoiser_parameters(model: MDMDenoiser) -> dict:
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen
    return {"total": total, "frozen": frozen, "trainable": trainable}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from mosl.text.tokenizer import WordTokenizer

    tok = WordTokenizer.load("data/processed/vocab.json")
    signllm_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    mdm_cfg = MDMConfig()

    model = MDMDenoiser(mdm_cfg, signllm_cfg)
    model.freeze_text_encoder()

    params = count_denoiser_parameters(model)
    print(f"total:     {params['total']:>12,}")
    print(f"frozen:    {params['frozen']:>12,}  (SignLLM text encoder)")
    print(f"trainable: {params['trainable']:>12,}  (denoiser only)")

    B, L, T = 4, 3, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_noisy = torch.randn(B, T, 150, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    text_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    text_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    pose_mask = torch.ones(B, T, dtype=torch.bool, device=device)

    x0_pred = model(x_noisy, t, text_ids, text_mask, pose_mask=pose_mask)
    print(f"\nforward: x_noisy {tuple(x_noisy.shape)} → x0_pred {tuple(x0_pred.shape)}")

    x0_target = torch.randn_like(x0_pred)
    loss = model.diffusion_loss(x0_pred, x0_target, pose_mask)
    loss.backward()
    print(f"loss: {loss.item():.4f}  (backward OK)")
