"""Diffusion noise schedule and DDIM sampler for MoSL motion generation.

Implements:
  - Cosine beta schedule (Nichol & Dhariwal 2021) — better than linear for
    motion data because it avoids over-noising at the end of the schedule.
  - DDPM forward process: q(x_t | x_0) = N(sqrt_alphas_cumprod * x0, (1-alphas_cumprod) * I)
  - DDIM deterministic reverse sampling (Song et al. 2020) — produces
    temporally consistent motion without stochastic frame-to-frame variation.
  - Savitzky-Golay post-processing for final jitter removal.

All tensors are registered as buffers so they move with .to(device) calls.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021) eq. 17.

    Returns beta_t for t = 1..n_steps as a (n_steps,) tensor.
    Clipped to [1e-4, 0.9999] to avoid numerical issues at the boundaries.
    """
    steps = n_steps + 1
    t = torch.linspace(0, n_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999).float()


def linear_beta_schedule(n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule (Ho et al. 2020). Kept for ablation comparison."""
    return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Noise schedule (precomputed buffers)
# ---------------------------------------------------------------------------

class NoiseSchedule(nn.Module):
    """Precomputed diffusion schedule buffers.

    Registers all derived quantities as buffers so they are:
      - moved to the correct device with .to(device)
      - saved/loaded with model checkpoints
      - not treated as trainable parameters

    Usage:
        schedule = NoiseSchedule(n_steps=1000)
        x_t, noise = schedule.q_sample(x0, t)   # forward process
        x0_pred = denoiser(x_t, t, ...)
        x_prev = schedule.ddim_step(x_t, x0_pred, t, t_prev)
    """

    def __init__(self, n_steps: int = 1000, schedule: str = "cosine") -> None:
        super().__init__()
        self.n_steps = n_steps

        if schedule == "cosine":
            betas = cosine_beta_schedule(n_steps)
        elif schedule == "linear":
            betas = linear_beta_schedule(n_steps)
        else:
            raise ValueError(f"unknown schedule {schedule!r}; use 'cosine' or 'linear'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Quantities used in q_sample (forward process)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # Quantities used in DDPM posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-12))
        self.register_buffer("posterior_log_variance_clipped",
            self.posterior_variance.clamp(min=1e-20).log())
        self.register_buffer("posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod).clamp(min=1e-12))
        self.register_buffer("posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod).clamp(min=1e-12))

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,          # (B, T, D) clean pose
        t: torch.Tensor,           # (B,) integer timesteps in [0, n_steps)
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from the forward process.

        Returns (x_t, noise) where noise ~ N(0, I).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omc = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ac * x0 + sqrt_omc * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # DDIM reverse step (deterministic)
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        x_t: torch.Tensor,         # (B, T, D) noisy pose at step t
        x0_pred: torch.Tensor,     # (B, T, D) predicted clean pose from denoiser
        t: torch.Tensor,           # (B,) current timestep indices
        t_prev: torch.Tensor,      # (B,) previous timestep indices (t - stride)
        eta: float = 0.0,          # 0 = fully deterministic DDIM
    ) -> torch.Tensor:
        """One DDIM reverse step: x_t → x_{t_prev}.

        With eta=0 this is fully deterministic (no added noise), which
        produces temporally consistent motion sequences.
        """
        ac_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        ac_prev = self._extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t.shape)

        # Predicted noise from x0 prediction
        sqrt_omc_t = (1.0 - ac_t).sqrt()
        eps_pred = (x_t - ac_t.sqrt() * x0_pred) / sqrt_omc_t.clamp(min=1e-8)

        # DDIM update
        sqrt_ac_prev = ac_prev.sqrt()
        dir_xt = (1.0 - ac_prev - eta ** 2 * (1.0 - ac_prev)).clamp(min=0).sqrt() * eps_pred

        if eta > 0:
            noise = torch.randn_like(x_t)
            sigma = eta * ((1.0 - ac_prev) / (1.0 - ac_t).clamp(min=1e-8) * (1.0 - ac_t / ac_prev.clamp(min=1e-8))).sqrt()
            x_prev = sqrt_ac_prev * x0_pred + dir_xt + sigma * noise
        else:
            x_prev = sqrt_ac_prev * x0_pred + dir_xt

        return x_prev

    # ------------------------------------------------------------------
    # Full DDIM sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser_fn: Callable,     # fn(x_noisy, t, text_ids, text_mask, **kw) → x0_pred
        shape: tuple,              # (B, T, D)
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        n_sample_steps: int = 50,  # DDIM uses far fewer steps than DDPM
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        **denoiser_kwargs,
    ) -> torch.Tensor:
        """Generate a clean pose sequence from pure noise via DDIM.

        Uses n_sample_steps uniformly spaced timesteps from n_steps → 0.
        Returns (B, T, D) clean pose sequence.
        """
        if device is None:
            device = text_ids.device

        # Uniformly spaced timestep sequence (descending)
        timesteps = torch.linspace(self.n_steps - 1, 0, n_sample_steps + 1,
                                   dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)

        for i in range(n_sample_steps):
            t_cur = timesteps[i].expand(shape[0])
            t_prev = timesteps[i + 1].expand(shape[0])

            x0_pred = denoiser_fn(x, t_cur, text_ids, text_mask, **denoiser_kwargs)
            x = self.ddim_step(x, x0_pred, t_cur, t_prev, eta=eta)

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Gather values from schedule tensor `a` at indices `t`, broadcast to `shape`."""
        B = t.shape[0]
        out = a.gather(0, t.clamp(0, a.shape[0] - 1))   # (B,)
        # Reshape to (B, 1, 1, ...) to broadcast over (B, T, D)
        return out.view(B, *([1] * (len(shape) - 1)))


# ---------------------------------------------------------------------------
# Savitzky-Golay temporal smoother (post-processing, no training required)
# ---------------------------------------------------------------------------

def savgol_smooth(
    pose_seq: torch.Tensor,    # (T, D) or (B, T, D)
    window: int = 5,
    polyorder: int = 2,
) -> torch.Tensor:
    """Apply Savitzky-Golay filter along the time axis.

    Reduces high-frequency jitter in generated motion without distorting
    the overall trajectory. Applied as post-processing after DDIM sampling.

    Requires scipy. Falls back to a simple moving average if unavailable.
    """
    try:
        from scipy.signal import savgol_filter
        arr = pose_seq.cpu().numpy()
        smoothed = savgol_filter(arr, window_length=window, polyorder=polyorder, axis=-2)
        return torch.from_numpy(smoothed).to(pose_seq.device)
    except ImportError:
        # Fallback: uniform moving average via 1D convolution
        batched = pose_seq.dim() == 3
        if not batched:
            pose_seq = pose_seq.unsqueeze(0)
        B, T, D = pose_seq.shape
        # Reshape to (B*D, 1, T) for conv1d
        x = pose_seq.permute(0, 2, 1).reshape(B * D, 1, T)
        kernel = torch.ones(1, 1, window, device=x.device) / window
        pad = window // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
        x = torch.nn.functional.conv1d(x, kernel)[:, :, :T]
        result = x.reshape(B, D, T).permute(0, 2, 1)
        return result if batched else result.squeeze(0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    schedule = NoiseSchedule(n_steps=1000, schedule="cosine")
    print(f"n_steps: {schedule.n_steps}")
    print(f"betas:   min={schedule.betas.min():.6f}  max={schedule.betas.max():.6f}")
    print(f"alphas_cumprod: t=0 → {schedule.alphas_cumprod[0]:.4f}  "
          f"t=500 → {schedule.alphas_cumprod[500]:.4f}  "
          f"t=999 → {schedule.alphas_cumprod[999]:.6f}")

    B, T, D = 4, 100, 150
    x0 = torch.randn(B, T, D)
    t = torch.randint(0, 1000, (B,))
    x_t, noise = schedule.q_sample(x0, t)
    print(f"\nq_sample: x0 {tuple(x0.shape)} → x_t {tuple(x_t.shape)}")

    # Verify SNR degrades with t
    snr_low = (schedule.alphas_cumprod[100] / (1 - schedule.alphas_cumprod[100])).item()
    snr_high = (schedule.alphas_cumprod[900] / (1 - schedule.alphas_cumprod[900])).item()
    print(f"SNR at t=100: {snr_low:.3f}  at t=900: {snr_high:.6f}  (should decrease)")

    # Test Savitzky-Golay smoother
    noisy_seq = torch.randn(T, D)
    smooth_seq = savgol_smooth(noisy_seq, window=5, polyorder=2)
    print(f"\nsavgol_smooth: {tuple(noisy_seq.shape)} → {tuple(smooth_seq.shape)}")
    print(f"  std before: {noisy_seq.std():.4f}  after: {smooth_seq.std():.4f}  (should decrease)")
