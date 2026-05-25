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

def _try_flash_attention() -> bool:
    """Return True if scaled_dot_product_attention with Flash Attention is available."""
    try:
        # PyTorch >= 2.0 has SDPA; Flash Attention backend is auto-selected when available
        import torch.nn.functional as F
        _ = F.scaled_dot_product_attention
        return True
    except AttributeError:
        return False


_HAS_SDPA = _try_flash_attention()


class AdaLNBlock(nn.Module):
    """Single transformer layer with AdaLN conditioning.

    Applies adaptive layer norm modulation from a conditioning vector
    (timestep + optional signer style) before each sub-layer, following
    DiT (Peebles & Xie 2023) and MDM (Tevet et al. 2022).

    Uses torch.nn.functional.scaled_dot_product_attention (Flash Attention
    backend) when available (PyTorch >= 2.0 + CUDA).
    """

    def __init__(self, d_model: int, nhead: int, d_ff: int,
                 dropout: float = 0.1, cond_dim: int = 512,
                 use_flash: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.use_flash = use_flash and _HAS_SDPA

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Fused QKV projections for self-attention
        self.self_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_out = nn.Linear(d_model, d_model)

        # Separate Q / KV projections for cross-attention
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model)

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

    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.self_qkv(x)                              # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.use_flash:
            # Reshape to (B, nhead, T, head_dim) for SDPA
            q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            drop = self.dropout if self.training else 0.0
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=False
            )
            out = out.transpose(1, 2).contiguous().view(B, T, D)
        else:
            # Fallback: standard MHA
            q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            if self.training and self.dropout > 0:
                attn = torch.nn.functional.dropout(attn, p=self.dropout)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)

        return self.self_out(out)

    def _cross_attn(
        self,
        x: torch.Tensor,           # (B, T, D)
        context: torch.Tensor,     # (B, L, D)
        context_pad_mask: Optional[torch.Tensor] = None,  # (B, L) True=padding
    ) -> torch.Tensor:
        B, T, D = x.shape
        L = context.shape[1]

        q = self.cross_q(x)                                 # (B, T, D)
        kv = self.cross_kv(context)                         # (B, L, 2D)
        k, v = kv.chunk(2, dim=-1)

        if self.use_flash:
            q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

            # Build attention bias from padding mask
            attn_bias = None
            if context_pad_mask is not None:
                # (B, 1, 1, L) — large negative for padding positions
                attn_bias = torch.zeros(B, 1, 1, L, device=x.device, dtype=x.dtype)
                attn_bias = attn_bias.masked_fill(
                    context_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

            drop = self.dropout if self.training else 0.0
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=drop
            )
            out = out.transpose(1, 2).contiguous().view(B, T, D)
        else:
            q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if context_pad_mask is not None:
                attn = attn.masked_fill(
                    context_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            attn = attn.softmax(dim=-1)
            if self.training and self.dropout > 0:
                attn = torch.nn.functional.dropout(attn, p=self.dropout)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)

        return self.cross_out(out)

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
        x = x + self._self_attn(h)

        # Cross-attention over text with AdaLN
        h = (1 + s2) * self.norm2(x) + b2
        x = x + self._cross_attn(h, context, context_key_padding_mask)

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

    # Classifier-free guidance
    cfg_dropout: float = 0.1                  # probability of dropping text conditioning during training
    cfg_scale: float = 2.5                    # guidance scale at inference (1.0 = no guidance)

    # Diffusion
    n_diffusion_steps: int = 1000
    hand_loss_weight: float = 3.0            # upweight hand coords in all losses

    # Flash Attention
    use_flash: bool = True                    # use SDPA / Flash Attention when available

    # Gradient checkpointing
    grad_checkpoint: bool = False             # trade compute for VRAM (~40% savings)

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

        # --- Null text embedding for classifier-free guidance --------------
        # Learned unconditional embedding (replaces text context when dropped)
        self.null_text_embed = nn.Parameter(torch.zeros(1, 1, d))

        # --- Timestep embedding --------------------------------------------
        self.time_embed = TimestepEmbedding(config.cond_dim)

        # --- Pose input projection -----------------------------------------
        # Noisy pose (pose_dim) → d_model; no time marker needed (t is in cond)
        self.pose_proj_in = nn.Linear(config.pose_dim, d)
        self.pose_pos = SinusoidalPositionalEncoding(d, max_len=config.max_pose_len)

        # --- Denoiser transformer layers -----------------------------------
        self.layers = nn.ModuleList([
            AdaLNBlock(
                d, config.nhead, config.d_ff, config.dropout,
                config.cond_dim, use_flash=config.use_flash,
            )
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
        drop_text: Optional[torch.Tensor] = None,  # (B,) bool — True=use null embed (CFG training)
    ) -> torch.Tensor:
        """Predict clean pose x0 from noisy pose x_t.

        Returns (B, T, 150) — the predicted clean pose sequence.

        drop_text: per-sample mask for classifier-free guidance training.
          When True for a sample, the null text embedding is used instead of
          the actual text features. At inference, run twice (conditional +
          unconditional) and interpolate with cfg_scale.
        """
        B = x_noisy.shape[0]

        # Text conditioning
        text_ctx = self.encode_text(text_ids, text_mask)   # (B, L, d)
        text_pad_mask = ~text_mask                          # True=padding (PyTorch convention)

        # Classifier-free guidance: replace text context with null embed for dropped samples
        if drop_text is not None and drop_text.any():
            null = self.null_text_embed.expand(B, text_ctx.shape[1], -1)  # (B, L, d)
            drop_mask = drop_text.view(B, 1, 1).to(text_ctx.dtype)
            text_ctx = text_ctx * (1 - drop_mask) + null * drop_mask
            # Null embed has no padding — clear the pad mask for dropped samples
            if text_pad_mask is not None:
                text_pad_mask = text_pad_mask & ~drop_text.unsqueeze(1)

        # Timestep conditioning
        cond = self.time_embed(t)                           # (B, cond_dim)
        if style_emb is not None and self.config.use_signer_style:
            cond = cond + style_emb                         # additive fusion

        # Pose embedding
        x = self.pose_proj_in(x_noisy)                     # (B, T, d)
        x = self.pose_pos(x)

        # Denoiser transformer (with optional gradient checkpointing)
        if self.config.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(
                    layer, x, text_ctx, cond, text_pad_mask,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                x = layer(x, text_ctx, cond, context_key_padding_mask=text_pad_mask)

        x = self.final_norm(x)
        x0_pred = self.pose_head(x)                        # (B, T, 150)
        return x0_pred

    @torch.no_grad()
    def forward_cfg(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        cfg_scale: float = 2.5,
        style_emb: Optional[torch.Tensor] = None,
        pose_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classifier-free guidance inference: conditional + unconditional forward.

        x0_cfg = x0_uncond + cfg_scale * (x0_cond - x0_uncond)

        cfg_scale=1.0 is equivalent to standard conditional generation.
        cfg_scale>1.0 amplifies the text conditioning signal.
        """
        B = x_noisy.shape[0]

        # Conditional prediction
        x0_cond = self.forward(x_noisy, t, text_ids, text_mask,
                               style_emb=style_emb, pose_mask=pose_mask)

        # Unconditional prediction (null text)
        drop_all = torch.ones(B, dtype=torch.bool, device=x_noisy.device)
        x0_uncond = self.forward(x_noisy, t, text_ids, text_mask,
                                 style_emb=style_emb, pose_mask=pose_mask,
                                 drop_text=drop_all)

        return x0_uncond + cfg_scale * (x0_cond - x0_uncond)

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
