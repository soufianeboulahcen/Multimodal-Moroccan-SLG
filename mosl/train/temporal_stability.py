"""Temporal stability analysis and post-processing for generated motion.

Validates and improves the temporal coherence of generated pose sequences.
Applied as post-processing after DDIM sampling.

Metrics computed:
  - jerk_score:       mean magnitude of third-order finite differences (lower = smoother)
  - velocity_std:     standard deviation of per-joint velocities (lower = more consistent)
  - acceleration_std: standard deviation of per-joint accelerations
  - hand_flicker:     fraction of frames where hand velocity exceeds 3× median (flickering)
  - finger_instability: variance of finger joint positions relative to wrist
  - temporal_consistency: DTW distance between consecutive 10-frame windows (lower = smoother)

Post-processing:
  - Savitzky-Golay smoothing (configurable window + polynomial order)
  - Velocity clamping (hard limit on per-frame joint displacement)
  - Acceleration regularization (soft penalty applied iteratively)

Usage:
    from mosl.train.temporal_stability import TemporalStabilityAnalyzer, smooth_motion

    # Smooth a generated sequence
    smooth = smooth_motion(pose_seq, window=7, polyorder=3)

    # Validate stability
    analyzer = TemporalStabilityAnalyzer()
    report = analyzer.analyze(pose_seq)
    print(report.summary())

    # Validate a batch of generated sequences
    reports = analyzer.analyze_batch(pose_seqs)
    analyzer.print_batch_summary(reports)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Joint layout constants (matches losses_diffusion.py)
# ---------------------------------------------------------------------------

POSE_DIM = 150
N_JOINTS = 50

BODY_COORD_END = 54       # joints 0-17 × 3
LHAND_COORD_START = 54    # joints 18-38 × 3
LHAND_COORD_END = 117
RHAND_COORD_START = 117   # joints 39-49 × 3
RHAND_COORD_END = 150
HAND_COORD_START = 54
HAND_COORD_END = 150

# Finger joints within the hand (relative to hand start)
# Left hand: joints 18-38 (21 joints). Wrist = joint 18 (coord 54-56).
# Finger joints = 19-38 (coords 57-116)
LHAND_WRIST_COORDS = slice(54, 57)
LHAND_FINGER_COORDS = slice(57, 117)

# Right hand: joints 39-49 (11 joints). Wrist = joint 39 (coord 117-119).
RHAND_WRIST_COORDS = slice(117, 120)
RHAND_FINGER_COORDS = slice(120, 150)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_motion(
    pose_seq: np.ndarray,   # (T, 150) or (B, T, 150)
    window: int = 7,
    polyorder: int = 3,
    mode: str = "savgol",
) -> np.ndarray:
    """Apply temporal smoothing to a pose sequence.

    Args:
        pose_seq: (T, 150) or (B, T, 150) float32 pose sequence
        window:   Savitzky-Golay window length (must be odd, >= polyorder+1)
        polyorder: Savitzky-Golay polynomial order
        mode:     "savgol" (default) or "moving_avg"

    Returns:
        Smoothed array of same shape as input.
    """
    batched = pose_seq.ndim == 3
    if not batched:
        pose_seq = pose_seq[np.newaxis]   # (1, T, 150)

    B, T, D = pose_seq.shape

    # Ensure window is valid
    window = min(window, T)
    if window % 2 == 0:
        window -= 1
    window = max(window, polyorder + 1)

    if mode == "savgol":
        try:
            from scipy.signal import savgol_filter
            smoothed = savgol_filter(pose_seq, window_length=window,
                                     polyorder=polyorder, axis=1)
        except ImportError:
            # Fallback to moving average
            smoothed = _moving_avg(pose_seq, window)
    elif mode == "moving_avg":
        smoothed = _moving_avg(pose_seq, window)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return smoothed if batched else smoothed[0]


def _moving_avg(pose_seq: np.ndarray, window: int) -> np.ndarray:
    """Uniform moving average along the time axis."""
    B, T, D = pose_seq.shape
    pad = window // 2
    # Reflect-pad to avoid boundary artifacts
    padded = np.pad(pose_seq, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(window) / window
    result = np.zeros_like(pose_seq)
    for b in range(B):
        for d in range(D):
            result[b, :, d] = np.convolve(padded[b, :, d], kernel, mode="valid")[:T]
    return result


def clamp_velocity(
    pose_seq: np.ndarray,   # (T, 150)
    max_velocity: float = 0.1,
) -> np.ndarray:
    """Clamp per-frame joint displacement to prevent sudden jumps.

    Iteratively adjusts frames where the displacement from the previous frame
    exceeds max_velocity (in normalised coordinate units).
    """
    result = pose_seq.copy()
    for t in range(1, len(result)):
        delta = result[t] - result[t - 1]
        norm = np.linalg.norm(delta)
        if norm > max_velocity:
            result[t] = result[t - 1] + delta * (max_velocity / norm)
    return result


def iterative_acceleration_regularization(
    pose_seq: np.ndarray,   # (T, 150)
    n_iters: int = 3,
    alpha: float = 0.3,
) -> np.ndarray:
    """Reduce acceleration by iteratively blending each frame with its neighbours.

    Each iteration: x[t] = (1-alpha)*x[t] + alpha*0.5*(x[t-1] + x[t+1])
    Applied only to interior frames (not first/last).
    """
    result = pose_seq.copy()
    for _ in range(n_iters):
        prev = result[:-2]   # (T-2, 150)
        curr = result[1:-1]  # (T-2, 150)
        nxt = result[2:]     # (T-2, 150)
        result[1:-1] = (1 - alpha) * curr + alpha * 0.5 * (prev + nxt)
    return result


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    """Per-sequence temporal stability metrics."""
    n_frames: int = 0

    # Smoothness metrics (lower = smoother)
    jerk_score: float = 0.0           # mean |third-order diff|
    velocity_std: float = 0.0         # std of per-joint velocities
    acceleration_std: float = 0.0     # std of per-joint accelerations

    # Hand-specific metrics
    hand_flicker_rate: float = 0.0    # fraction of frames with velocity spike
    finger_instability: float = 0.0   # variance of finger positions rel. to wrist

    # Temporal consistency
    window_dtw: float = 0.0           # mean DTW between consecutive 10-frame windows

    # Pass/fail thresholds
    JERK_THRESHOLD: float = 0.05
    FLICKER_THRESHOLD: float = 0.10
    FINGER_THRESHOLD: float = 0.02

    def is_stable(self) -> bool:
        return (
            self.jerk_score < self.JERK_THRESHOLD
            and self.hand_flicker_rate < self.FLICKER_THRESHOLD
            and self.finger_instability < self.FINGER_THRESHOLD
        )

    def summary(self) -> str:
        status = "STABLE" if self.is_stable() else "UNSTABLE"
        return (
            f"[{status}] T={self.n_frames}  "
            f"jerk={self.jerk_score:.4f}  "
            f"vel_std={self.velocity_std:.4f}  "
            f"acc_std={self.acceleration_std:.4f}  "
            f"hand_flicker={self.hand_flicker_rate:.3f}  "
            f"finger_instab={self.finger_instability:.4f}"
        )


class TemporalStabilityAnalyzer:
    """Computes temporal stability metrics for generated pose sequences."""

    def analyze(self, pose_seq: np.ndarray) -> StabilityReport:
        """Analyze a single (T, 150) pose sequence.

        Returns a StabilityReport with all metrics.
        """
        if pose_seq.ndim != 2 or pose_seq.shape[1] != POSE_DIM:
            raise ValueError(f"expected (T, 150), got {pose_seq.shape}")

        T = pose_seq.shape[0]
        report = StabilityReport(n_frames=T)

        if T < 4:
            return report

        # --- Velocity (first-order diff) ---
        vel = np.diff(pose_seq, axis=0)           # (T-1, 150)
        vel_mag = np.linalg.norm(vel, axis=-1)    # (T-1,)
        report.velocity_std = float(vel_mag.std())

        # --- Acceleration (second-order diff) ---
        acc = np.diff(vel, axis=0)                # (T-2, 150)
        acc_mag = np.linalg.norm(acc, axis=-1)
        report.acceleration_std = float(acc_mag.std())

        # --- Jerk (third-order diff) ---
        jerk = np.diff(acc, axis=0)               # (T-3, 150)
        jerk_mag = np.linalg.norm(jerk, axis=-1)
        report.jerk_score = float(jerk_mag.mean())

        # --- Hand flicker ---
        hand_vel = vel[:, HAND_COORD_START:HAND_COORD_END]  # (T-1, 96)
        hand_vel_mag = np.linalg.norm(hand_vel, axis=-1)    # (T-1,)
        median_hv = np.median(hand_vel_mag)
        flicker_threshold = 3.0 * median_hv + 1e-6
        report.hand_flicker_rate = float((hand_vel_mag > flicker_threshold).mean())

        # --- Finger instability ---
        # Variance of finger positions relative to wrist position
        lhand = pose_seq[:, LHAND_COORD_START:LHAND_COORD_END].reshape(T, 21, 3)
        lhand_wrist = lhand[:, 0:1, :]                      # (T, 1, 3)
        lhand_rel = lhand[:, 1:, :] - lhand_wrist           # (T, 20, 3)
        lhand_instab = float(lhand_rel.var())

        rhand = pose_seq[:, RHAND_COORD_START:RHAND_COORD_END].reshape(T, 11, 3)
        rhand_wrist = rhand[:, 0:1, :]
        rhand_rel = rhand[:, 1:, :] - rhand_wrist
        rhand_instab = float(rhand_rel.var())

        report.finger_instability = (lhand_instab + rhand_instab) / 2.0

        # --- Window DTW (temporal consistency) ---
        report.window_dtw = self._window_dtw(pose_seq, window=10)

        return report

    def analyze_batch(self, pose_seqs: list[np.ndarray]) -> list[StabilityReport]:
        """Analyze a list of pose sequences."""
        return [self.analyze(seq) for seq in pose_seqs]

    def print_batch_summary(self, reports: list[StabilityReport]) -> None:
        """Print aggregate statistics for a batch of reports."""
        if not reports:
            print("no reports")
            return

        n_stable = sum(1 for r in reports if r.is_stable())
        jerk_vals = [r.jerk_score for r in reports]
        flicker_vals = [r.hand_flicker_rate for r in reports]
        finger_vals = [r.finger_instability for r in reports]

        print(f"\n=== Temporal Stability Summary ({len(reports)} sequences) ===")
        print(f"  stable:          {n_stable}/{len(reports)} ({100*n_stable/len(reports):.1f}%)")
        print(f"  jerk_score:      mean={np.mean(jerk_vals):.4f}  "
              f"max={np.max(jerk_vals):.4f}  (threshold={StabilityReport.JERK_THRESHOLD})")
        print(f"  hand_flicker:    mean={np.mean(flicker_vals):.3f}  "
              f"max={np.max(flicker_vals):.3f}  (threshold={StabilityReport.FLICKER_THRESHOLD})")
        print(f"  finger_instab:   mean={np.mean(finger_vals):.4f}  "
              f"max={np.max(finger_vals):.4f}  (threshold={StabilityReport.FINGER_THRESHOLD})")

    @staticmethod
    def _window_dtw(pose_seq: np.ndarray, window: int = 10) -> float:
        """Compute mean DTW distance between consecutive windows.

        A low value means the motion is temporally consistent (smooth transitions).
        """
        T = pose_seq.shape[0]
        if T < 2 * window:
            return 0.0

        distances = []
        for start in range(0, T - 2 * window, window):
            w1 = pose_seq[start:start + window]
            w2 = pose_seq[start + window:start + 2 * window]
            # Simple Euclidean distance between window means (fast proxy for DTW)
            dist = float(np.linalg.norm(w1.mean(axis=0) - w2.mean(axis=0)))
            distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------

def post_process_motion(
    pose_seq: np.ndarray,           # (T, 150) or (B, T, 150)
    savgol_window: int = 7,
    savgol_polyorder: int = 3,
    clamp_vel: Optional[float] = None,   # None = no clamping
    acc_reg_iters: int = 0,              # 0 = no acceleration regularization
    acc_reg_alpha: float = 0.3,
) -> np.ndarray:
    """Apply the full temporal post-processing pipeline.

    Order:
      1. Savitzky-Golay smoothing (primary jitter removal)
      2. Velocity clamping (prevent sudden jumps)
      3. Iterative acceleration regularization (secondary smoothing)

    Returns smoothed array of same shape as input.
    """
    batched = pose_seq.ndim == 3
    if not batched:
        pose_seq = pose_seq[np.newaxis]

    results = []
    for seq in pose_seq:
        s = smooth_motion(seq, window=savgol_window, polyorder=savgol_polyorder)
        if clamp_vel is not None:
            s = clamp_velocity(s, max_velocity=clamp_vel)
        if acc_reg_iters > 0:
            s = iterative_acceleration_regularization(s, n_iters=acc_reg_iters,
                                                      alpha=acc_reg_alpha)
        results.append(s)

    result = np.stack(results, axis=0)
    return result if batched else result[0]


# ---------------------------------------------------------------------------
# Torch-compatible wrapper (for use during DDIM sampling)
# ---------------------------------------------------------------------------

def smooth_motion_torch(
    pose_seq: torch.Tensor,   # (T, 150) or (B, T, 150)
    window: int = 7,
    polyorder: int = 3,
) -> torch.Tensor:
    """Torch wrapper around smooth_motion. Preserves device and dtype."""
    device = pose_seq.device
    dtype = pose_seq.dtype
    arr = pose_seq.float().cpu().numpy()
    smoothed = smooth_motion(arr, window=window, polyorder=polyorder)
    return torch.from_numpy(smoothed).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# CLI: validate a checkpoint's generated motion
# ---------------------------------------------------------------------------

def validate_checkpoint(
    checkpoint_path: str,
    n_samples: int = 20,
    device: str = "cuda",
    savgol_window: int = 7,
) -> None:
    """Generate n_samples from a trained MDM checkpoint and report stability."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import torch
    from mosl.text.tokenizer import WordTokenizer
    from mosl.model.signllm import SignLLMConfig
    from mosl.model.mdm_denoiser import MDMConfig, MDMDenoiser
    from mosl.train.noise_schedule import NoiseSchedule

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tok = WordTokenizer.load("data/processed/vocab.json")
    signllm_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    mdm_cfg = MDMConfig()
    model = MDMDenoiser(mdm_cfg, signllm_cfg).to(dev)

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = NoiseSchedule(n_steps=1000).to(dev)
    analyzer = TemporalStabilityAnalyzer()

    # Use a fixed test sign
    test_sign = "الأذان"
    text_ids = torch.tensor([[1, tok.encode(test_sign)[0] if tok.encode(test_sign) else 3, 2]],
                             dtype=torch.long, device=dev)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    reports_raw = []
    reports_smooth = []

    for i in range(n_samples):
        with torch.no_grad():
            pose = schedule.ddim_sample(
                denoiser_fn=lambda x, t, ti, tm: model(x, t, ti, tm),
                shape=(1, 64, 150),
                text_ids=text_ids,
                text_mask=text_mask,
                n_sample_steps=50,
                device=dev,
            )
        pose_np = pose[0].cpu().numpy()
        reports_raw.append(analyzer.analyze(pose_np))

        smooth_np = smooth_motion(pose_np, window=savgol_window)
        reports_smooth.append(analyzer.analyze(smooth_np))

    print("\n--- Raw generated motion ---")
    analyzer.print_batch_summary(reports_raw)
    print(f"\n--- After Savitzky-Golay (window={savgol_window}) ---")
    analyzer.print_batch_summary(reports_smooth)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate temporal stability of generated motion")
    parser.add_argument("checkpoint", help="Path to MDM checkpoint (best.pt)")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--savgol-window", type=int, default=7)
    args = parser.parse_args()

    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        n_samples=args.n_samples,
        device=args.device,
        savgol_window=args.savgol_window,
    )
