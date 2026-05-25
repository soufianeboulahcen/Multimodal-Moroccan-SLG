"""Signer style encoder for identity-preserving motion generation.

Encodes a reference clip (or reference frames) from a specific signer into a
style embedding that is injected into the MDM denoiser via AdaLN conditioning.
This allows the diffusion model to generate motion that preserves the signing
style, rhythm, and motion identity of the target signer.

Architecture:
  - Pose style encoder: lightweight transformer over a reference pose sequence
    → style_embedding (B, style_dim)
  - Appearance encoder: frozen DINO-ViT features from reference video frames
    → appearance_embedding (B, appearance_dim)
  - Combined embedding: style + appearance → fused_embedding (B, cond_dim)
    injected into MDMDenoiser via AdaLN

The pose style encoder is trained jointly with the MDM denoiser.
The appearance encoder is frozen (DINO-ViT pretrained weights).

Activation path:
  - Pose style: always active (uses existing .skels data)
  - Appearance style: active only when reference video frames are available
    (requires signer_id annotation in labels.csv)

When signer_id is not yet annotated, the style encoder operates in
"pose-only" mode using the motion statistics of the reference clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pose style encoder
# ---------------------------------------------------------------------------

class PoseStyleEncoder(nn.Module):
    """Encodes a reference pose sequence into a signer style embedding.

    Uses a small transformer encoder over the reference clip's pose frames,
    then mean-pools to a fixed-size style vector.

    The style vector captures:
      - Signing rhythm (temporal statistics of joint velocities)
      - Motion amplitude (range of joint positions)
      - Hand dynamics (velocity patterns of hand joints)
      - Body posture tendencies (mean joint positions relative to root)
    """

    def __init__(
        self,
        pose_dim: int = 150,
        d_model: int = 256,
        nhead: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        style_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Project to style_dim
        self.style_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, style_dim),
        )

        # Motion statistics head (captures rhythm and amplitude)
        # Concatenated with transformer output before final projection
        self.stats_proj = nn.Linear(pose_dim * 4, d_model)   # mean, std, vel_mean, vel_std

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ref_pose: torch.Tensor,    # (B, T_ref, 150) reference clip pose
        ref_mask: Optional[torch.Tensor] = None,  # (B, T_ref) True=real frame
    ) -> torch.Tensor:
        """Encode reference pose sequence → style embedding (B, style_dim)."""
        B, T, D = ref_pose.shape

        # Compute motion statistics (rhythm + amplitude features)
        mask_f = ref_mask.unsqueeze(-1).float() if ref_mask is not None else torch.ones(B, T, 1, device=ref_pose.device)
        n_real = mask_f.sum(dim=1).clamp(min=1.0)

        mean_pose = (ref_pose * mask_f).sum(dim=1) / n_real          # (B, D)
        sq_diff = ((ref_pose - mean_pose.unsqueeze(1)) ** 2) * mask_f
        std_pose = (sq_diff.sum(dim=1) / n_real).sqrt()              # (B, D)

        vel = ref_pose[:, 1:] - ref_pose[:, :-1]                     # (B, T-1, D)
        vel_mask = mask_f[:, 1:] * mask_f[:, :-1]
        n_vel = vel_mask.sum(dim=1).clamp(min=1.0)
        mean_vel = (vel * vel_mask).sum(dim=1) / n_vel               # (B, D)
        sq_vel = ((vel - mean_vel.unsqueeze(1)) ** 2) * vel_mask
        std_vel = (sq_vel.sum(dim=1) / n_vel).sqrt()                 # (B, D)

        stats = torch.cat([mean_pose, std_pose, mean_vel, std_vel], dim=-1)  # (B, 4*D)
        stats_emb = self.stats_proj(stats)                           # (B, d_model)

        # Transformer encoding
        x = self.pose_proj(ref_pose)                                 # (B, T, d_model)
        pad_mask = ~ref_mask if ref_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean pool over real frames
        x_pooled = (x * mask_f).sum(dim=1) / n_real                 # (B, d_model)

        # Fuse transformer output with motion statistics
        fused = x_pooled + stats_emb                                 # (B, d_model)
        style = self.style_head(fused)                               # (B, style_dim)
        return style


# ---------------------------------------------------------------------------
# Appearance encoder (frozen DINO-ViT)
# ---------------------------------------------------------------------------

class AppearanceEncoder(nn.Module):
    """Extracts signer appearance embedding from reference video frames.

    Uses frozen DINO-ViT (self-supervised) features which capture visual
    identity without requiring labelled face data.

    Requires: pip install timm  (for DINO-ViT weights)
    Falls back to a zero embedding if timm is not available.
    """

    def __init__(self, appearance_dim: int = 256, vit_model: str = "vit_small_patch16_224") -> None:
        super().__init__()
        self.appearance_dim = appearance_dim
        self._vit_available = False

        try:
            import timm
            self.vit = timm.create_model(vit_model, pretrained=True, num_classes=0)
            vit_out_dim = self.vit.num_features
            self.proj = nn.Linear(vit_out_dim, appearance_dim)
            for p in self.vit.parameters():
                p.requires_grad = False
            self._vit_available = True
        except ImportError:
            # timm not installed — appearance conditioning disabled
            self.proj = nn.Linear(1, appearance_dim)   # placeholder

    def forward(self, frames: Optional[torch.Tensor]) -> torch.Tensor:
        """frames: (B, C, H, W) reference frame(s), normalised to [0,1].

        Returns (B, appearance_dim). Returns zeros if frames is None or
        timm is not available.
        """
        if frames is None or not self._vit_available:
            B = frames.shape[0] if frames is not None else 1
            return torch.zeros(B, self.appearance_dim,
                               device=frames.device if frames is not None else "cpu")

        with torch.no_grad():
            vit_features = self.vit(frames)   # (B, vit_out_dim)
        return self.proj(vit_features)        # (B, appearance_dim)


# ---------------------------------------------------------------------------
# Combined signer encoder
# ---------------------------------------------------------------------------

@dataclass
class SignerEncoderConfig:
    pose_dim: int = 150
    style_dim: int = 256
    appearance_dim: int = 256
    cond_dim: int = 512          # must match MDMConfig.cond_dim
    use_appearance: bool = False  # enable once signer_id is annotated


class SignerEncoder(nn.Module):
    """Combines pose style and appearance into a single conditioning vector.

    Output is added to the timestep embedding in MDMDenoiser, injecting
    signer identity into every AdaLN layer of the denoiser.

    Usage:
        style_emb = signer_encoder(ref_pose, ref_mask, ref_frames)
        x0_pred = denoiser(x_noisy, t, text_ids, text_mask, style_emb=style_emb)
    """

    def __init__(self, config: SignerEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.pose_encoder = PoseStyleEncoder(
            pose_dim=config.pose_dim,
            style_dim=config.style_dim,
        )

        if config.use_appearance:
            self.appearance_encoder = AppearanceEncoder(
                appearance_dim=config.appearance_dim,
            )
            fusion_in = config.style_dim + config.appearance_dim
        else:
            self.appearance_encoder = None
            fusion_in = config.style_dim

        # Project fused embedding to cond_dim (matches MDMDenoiser.cond_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )

    def forward(
        self,
        ref_pose: torch.Tensor,                    # (B, T_ref, 150)
        ref_mask: Optional[torch.Tensor] = None,   # (B, T_ref)
        ref_frames: Optional[torch.Tensor] = None, # (B, C, H, W) appearance frames
    ) -> torch.Tensor:
        """Returns style embedding (B, cond_dim) for injection into MDMDenoiser."""
        style = self.pose_encoder(ref_pose, ref_mask)   # (B, style_dim)

        if self.config.use_appearance and self.appearance_encoder is not None:
            appearance = self.appearance_encoder(ref_frames)   # (B, appearance_dim)
            fused = torch.cat([style, appearance], dim=-1)
        else:
            fused = style

        return self.fusion(fused)   # (B, cond_dim)


# ---------------------------------------------------------------------------
# Signer consistency loss
# ---------------------------------------------------------------------------

def signer_consistency_loss(
    style_emb_a: torch.Tensor,   # (B, cond_dim) — embedding from clip A of signer i
    style_emb_b: torch.Tensor,   # (B, cond_dim) — embedding from clip B of signer i
    style_emb_neg: Optional[torch.Tensor] = None,  # (B, cond_dim) — different signer
    margin: float = 1.0,
) -> torch.Tensor:
    """Contrastive loss to make same-signer embeddings similar.

    When style_emb_neg is provided, uses triplet margin loss.
    Otherwise uses cosine similarity loss (pulls same-signer embeddings together).

    Requires signer_id annotations to form positive/negative pairs.
    """
    if style_emb_neg is not None:
        # Triplet loss: same-signer pair should be closer than different-signer pair
        dist_pos = F.pairwise_distance(style_emb_a, style_emb_b)
        dist_neg = F.pairwise_distance(style_emb_a, style_emb_neg)
        loss = F.relu(dist_pos - dist_neg + margin).mean()
    else:
        # Cosine similarity: pull same-signer embeddings together
        cos_sim = F.cosine_similarity(style_emb_a, style_emb_b, dim=-1)
        loss = (1.0 - cos_sim).mean()

    return loss


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    cfg = SignerEncoderConfig(use_appearance=False)
    encoder = SignerEncoder(cfg)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"SignerEncoder trainable params: {n_params:,}")

    B, T_ref = 4, 80
    ref_pose = torch.randn(B, T_ref, 150)
    ref_mask = torch.ones(B, T_ref, dtype=torch.bool)
    ref_mask[0, 60:] = False   # simulate variable-length clip

    style_emb = encoder(ref_pose, ref_mask)
    print(f"style_emb: {tuple(style_emb.shape)}  (expected: ({B}, {cfg.cond_dim}))")
    assert style_emb.shape == (B, cfg.cond_dim)

    # Test consistency loss
    ref_pose_b = torch.randn(B, T_ref, 150)
    style_emb_b = encoder(ref_pose_b, ref_mask)
    loss = signer_consistency_loss(style_emb, style_emb_b)
    print(f"consistency loss (no negatives): {loss.item():.4f}")

    # Test with MDMDenoiser integration
    from mosl.model.mdm_denoiser import MDMConfig, MDMDenoiser
    from mosl.model.signllm import SignLLMConfig
    from mosl.text.tokenizer import WordTokenizer

    tok = WordTokenizer.load("data/processed/vocab.json")
    signllm_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    mdm_cfg = MDMConfig(use_signer_style=True, cond_dim=cfg.cond_dim)
    denoiser = MDMDenoiser(mdm_cfg, signllm_cfg)
    denoiser.freeze_text_encoder()

    B, L, T = 4, 3, 100
    x_noisy = torch.randn(B, T, 150)
    t = torch.randint(0, 1000, (B,))
    text_ids = torch.zeros(B, L, dtype=torch.long)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    pose_mask = torch.ones(B, T, dtype=torch.bool)

    x0_pred = denoiser(x_noisy, t, text_ids, text_mask,
                       style_emb=style_emb, pose_mask=pose_mask)
    print(f"denoiser with style: {tuple(x0_pred.shape)}  ✓")
