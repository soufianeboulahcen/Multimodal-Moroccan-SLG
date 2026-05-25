"""Losses for the SignLLM-on-MoSL training.

Three loss components, combinable in the three configurations from the paper's
Table 5 ablation:

    Base + Normal MSE Loss
    Base + RL Loss
    Base + RL Loss + Priority Learning Channel (PLC)

Plus a small length-prediction loss (`MSE(log_T_pred, log(n_frames))`) added to
all three so the model can decide when to stop generating.

What the paper specifies (verbatim from §4.4):

    r = -1/N Σ_i (y_i - ŷ_i)²
    θ* = argmax_θ E_θ[Σ r_t] = argmin_θ E_θ[Σ L(y_t, M(x_t))]      (eq 3)
    P(i) = r(i)^η / Σ_{j ∈ S} r(j)^η
    "if the reward is less than 50%, skip the batch"

What we have to interpret:

    1.  P(i) = r(i)^η is mathematically problematic when r(i) ≤ 0 (and r is
        negative MSE, so always ≤ 0).  We use exp(η · r) instead — softmax-style
        prioritization that gives higher weight to less-negative rewards (=
        lower MSE), is well-defined for all η ≥ 0, and matches the *spirit* of
        "prioritize high-reward samples".

    2.  "if reward < 50%, skip" — the most defensible interpretation is
        threshold = median of per-sample rewards in the current batch
        (skip the worst half).  Documented in docs/DECISIONS.md.

    3.  η value not specified.  We default to η = 1.0 (linear prioritization,
        matches the paper's "linear" reading).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Per-sample / per-batch primitives
# ----------------------------------------------------------------------------

def masked_per_sample_mse(
    pose_pred: torch.Tensor,        # (B, T, D)
    pose_target: torch.Tensor,      # (B, T, D)
    pose_mask: torch.Tensor,        # (B, T) bool
) -> torch.Tensor:
    """MSE per sample, averaged over real frames and all D coords.
    Returns (B,)."""
    if pose_pred.shape != pose_target.shape:
        raise ValueError(f"shape mismatch: pred {pose_pred.shape} vs target {pose_target.shape}")
    sq = (pose_pred - pose_target) ** 2                                    # (B, T, D)
    per_frame = sq.mean(dim=-1)                                            # (B, T)
    mask = pose_mask.to(per_frame.dtype)
    real_frames = mask.sum(dim=1).clamp(min=1.0)                           # (B,)
    return (per_frame * mask).sum(dim=1) / real_frames                     # (B,)


def masked_mse_loss(
    pose_pred: torch.Tensor,
    pose_target: torch.Tensor,
    pose_mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar MSE loss averaged across batch (uniform per-sample weighting)."""
    return masked_per_sample_mse(pose_pred, pose_target, pose_mask).mean()


def length_loss(log_T_pred: torch.Tensor, n_frames: torch.Tensor) -> torch.Tensor:
    """MSE between predicted log T and ground-truth log(n_frames).
    log-space regression keeps the loss well-conditioned across our 43 → 236
    range of clip lengths."""
    target = torch.log(n_frames.to(log_T_pred.dtype).clamp(min=1.0))
    return F.mse_loss(log_T_pred, target)


# ----------------------------------------------------------------------------
# RL Loss (paper §4.4)
# ----------------------------------------------------------------------------

def rl_per_sample_reward(
    pose_pred: torch.Tensor,
    pose_target: torch.Tensor,
    pose_mask: torch.Tensor,
) -> torch.Tensor:
    """r(i) = -MSE(i).  Always ≤ 0; closer to 0 = better."""
    return -masked_per_sample_mse(pose_pred, pose_target, pose_mask)


def rl_loss(
    pose_pred: torch.Tensor,
    pose_target: torch.Tensor,
    pose_mask: torch.Tensor,
) -> torch.Tensor:
    """RL Loss = -mean(reward) = mean(MSE).

    Per the paper's eq (3), this is mathematically identical to the standard
    MSE loss at the per-step level.  The interesting behaviour comes from
    wrapping this in PLC for batch prioritization.
    """
    return -rl_per_sample_reward(pose_pred, pose_target, pose_mask).mean()


# ----------------------------------------------------------------------------
# Priority Learning Channel (paper §4.4)
# ----------------------------------------------------------------------------

@dataclass
class PLCConfig:
    eta: float = 1.0                         # prioritization intensity
    skip_below_quantile: float = 0.5         # skip samples below this rank in batch
    enabled: bool = True


def plc_weights(
    per_sample_reward: torch.Tensor,         # (B,)
    cfg: PLCConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Priority Learning Channel keep-mask and per-sample weights.

    Returns
    -------
    keep_mask : (B,) bool
        True for samples to keep (skip the rest).  When the whole batch would
        be skipped (e.g. all rewards equal), we keep everything to avoid an
        empty-batch step.
    weights   : (B,) float
        Per-sample loss weights, normalised so the kept samples sum to the
        kept-sample count (i.e., uniform weighting recovers `weights = mask`).
    """
    if not cfg.enabled:
        keep = torch.ones_like(per_sample_reward, dtype=torch.bool)
        return keep, keep.to(per_sample_reward.dtype)

    # Threshold: samples below the configured quantile of batch rewards are skipped.
    # quantile=0.5 ⇒ keep upper half ("if reward < 50% skip").
    if per_sample_reward.numel() == 0:
        keep = torch.zeros_like(per_sample_reward, dtype=torch.bool)
        return keep, keep.to(per_sample_reward.dtype)
    thresh = per_sample_reward.quantile(cfg.skip_below_quantile)
    keep = per_sample_reward >= thresh
    if not keep.any():
        # Defensive: never produce an empty batch (would yield NaN gradients).
        keep = torch.ones_like(per_sample_reward, dtype=torch.bool)

    # Prioritization: exp(η · r) is the softmax-style equivalent of P(i) ∝ r^η
    # but well-defined for negative r.  η=0 ⇒ uniform; η=1 ⇒ linear.
    raw = torch.exp(cfg.eta * per_sample_reward)                           # (B,)
    raw = raw * keep.to(raw.dtype)
    raw_sum = raw.sum().clamp(min=1e-12)
    n_kept = keep.to(raw.dtype).sum().clamp(min=1.0)
    weights = raw / raw_sum * n_kept                                       # mean weight ≈ 1
    return keep, weights


# ----------------------------------------------------------------------------
# Top-level training step loss — combines everything per the chosen mode
# ----------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Selects which loss combination to use for a training run."""
    mode: str = "mse"                # one of "mse", "rl", "rl_plc"
    length_weight: float = 1.0       # weight on length-prediction MSE
    plc: PLCConfig = None            # only consulted when mode == "rl_plc"

    def __post_init__(self) -> None:
        if self.mode not in {"mse", "rl", "rl_plc"}:
            raise ValueError(f"mode must be mse / rl / rl_plc, got {self.mode!r}")
        if self.plc is None:
            self.plc = PLCConfig(enabled=(self.mode == "rl_plc"))


def signllm_step_loss(
    model_out: dict,                         # output of SignLLM.forward
    batch: dict,                             # output of mosl_collate
    cfg: LossConfig,
) -> dict:
    """Compute the full training loss for one batch.

    Returns a dict with keys:
        "loss"          (scalar)  — total loss to .backward() on
        "pose_loss"     (scalar)  — pose component (MSE / RL Loss)
        "length_loss"   (scalar)
        "n_kept"        (int)     — samples used (relevant for RL+PLC)
        "reward_mean"   (scalar)  — diagnostic: mean per-sample reward across batch
    """
    pose_pred = model_out["pose_pred"]
    log_T_pred = model_out["log_T_pred"]
    pose_target = batch["pose"]
    pose_mask = batch["pose_mask"]
    n_frames = batch["n_frames"]

    per_sample_mse = masked_per_sample_mse(pose_pred, pose_target, pose_mask)
    reward = -per_sample_mse                 # (B,)

    if cfg.mode == "mse":
        pose_loss = per_sample_mse.mean()
        n_kept = per_sample_mse.numel()
    elif cfg.mode == "rl":
        # Same value as MSE at per-step level (paper eq 3) but exposes the
        # per-sample reward for downstream uses (logging, future PLC enable).
        pose_loss = per_sample_mse.mean()
        n_kept = per_sample_mse.numel()
    elif cfg.mode == "rl_plc":
        keep_mask, weights = plc_weights(reward.detach(), cfg.plc)
        # Apply weights to per-sample MSE; only kept samples contribute.
        pose_loss = (per_sample_mse * weights).sum() / weights.sum().clamp(min=1e-12)
        n_kept = int(keep_mask.sum().item())
    else:                                     # already validated; defensive
        raise AssertionError

    len_loss = length_loss(log_T_pred, n_frames)
    total = pose_loss + cfg.length_weight * len_loss

    return {
        "loss": total,
        "pose_loss": pose_loss.detach(),
        "length_loss": len_loss.detach(),
        "n_kept": n_kept,
        "reward_mean": reward.mean().detach(),
    }


# ----------------------------------------------------------------------------
# Smoke test (runs inside the container)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from torch.utils.data import DataLoader
    from mosl.data.dataset import MoSLSkelsDataset, mosl_collate
    from mosl.text.tokenizer import WordTokenizer
    from mosl.model.signllm import SignLLM, SignLLMConfig

    tok = WordTokenizer.load("data/processed/vocab.json")
    ds = MoSLSkelsDataset("dev", tokenizer=tok)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=mosl_collate)
    batch = next(iter(dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    model = SignLLM(cfg).to(device)
    out = model(
        text_ids=batch["text_ids"], text_mask=batch["text_mask"],
        pose_target=batch["pose"], time=batch["time"], pose_mask=batch["pose_mask"],
    )
    print(f"pose_pred {tuple(out['pose_pred'].shape)}  log_T_pred {tuple(out['log_T_pred'].shape)}")

    print()
    for mode in ("mse", "rl", "rl_plc"):
        loss_cfg = LossConfig(mode=mode)
        m = signllm_step_loss(out, batch, loss_cfg)
        # Backward to verify gradients flow
        model.zero_grad(set_to_none=True)
        m["loss"].backward(retain_graph=True)
        any_grad = any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
        print(
            f"{mode:>6}  total={m['loss'].item():.4f}  "
            f"pose={m['pose_loss'].item():.4f}  len={m['length_loss'].item():.4f}  "
            f"n_kept={m['n_kept']}/8  reward={m['reward_mean'].item():+.4f}  "
            f"grads={'OK' if any_grad else 'NONE'}"
        )
