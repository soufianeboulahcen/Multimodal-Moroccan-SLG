"""Evaluation utilities for the SignLLM-on-MoSL training.

Two evaluation modes:

  * **Teacher-forced** — feed ground-truth previous frames to the decoder,
    compute per-frame MSE.  Fast (~one forward per batch).  Used for
    early-stopping during training.

  * **Autoregressive** — generate the full pose sequence from text only
    (using the model's predicted length), compute DTW between predicted and
    target.  Slow (~one forward per generated frame).  Used at the end of
    training for paper-comparable numbers.

DTW is hand-rolled in numpy with scipy.spatial.distance.cdist for the pairwise
frame distance matrix.  Standard symmetric step pattern; cumulative-distance
DP table; result divided by path length so DTW values are comparable across
clips of different durations.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from mosl.train.losses import length_loss, masked_per_sample_mse


# ----------------------------------------------------------------------------
# DTW
# ----------------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Path-length-normalised DTW distance between two pose sequences.

    Parameters
    ----------
    a, b : (T_a, D) and (T_b, D) float arrays
        Pose sequences.  D must match.

    Returns
    -------
    float : DTW(a, b) / path_length, where pairwise distance is L2.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2-D, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dim mismatch: {a.shape[1]} vs {b.shape[1]}")
    Ta, Tb = a.shape[0], b.shape[0]
    dist = cdist(a, b, metric="euclidean")                     # (Ta, Tb)
    # DP table (Ta+1, Tb+1): dp[i, j] = min cumulative distance to reach (i-1, j-1).
    dp = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            dp[i, j] = dist[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    # Backtrack to get path length (for normalisation).
    i, j = Ta, Tb
    path_len = 0
    while i > 0 and j > 0:
        path_len += 1
        prev = np.argmin([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    return dp[Ta, Tb] / max(path_len, 1)


# ----------------------------------------------------------------------------
# Evaluation drivers
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_teacher_forced(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """One forward per batch with teacher forcing.  Returns aggregated metrics."""
    model.eval()
    n_clips = 0
    sum_pose_mse = 0.0
    sum_len_mse = 0.0
    sum_log_T_abs_err = 0.0   # |log_T_pred − log T_true|

    for batch in loader:
        for k in ("text_ids", "text_mask", "pose", "time", "pose_mask", "n_frames"):
            batch[k] = batch[k].to(device)

        out = model(
            text_ids=batch["text_ids"], text_mask=batch["text_mask"],
            pose_target=batch["pose"], time=batch["time"], pose_mask=batch["pose_mask"],
        )
        per_sample = masked_per_sample_mse(out["pose_pred"], batch["pose"], batch["pose_mask"])
        len_loss_val = length_loss(out["log_T_pred"], batch["n_frames"])
        log_T_true = torch.log(batch["n_frames"].to(out["log_T_pred"].dtype).clamp(min=1.0))
        log_abs = (out["log_T_pred"] - log_T_true).abs()

        B = per_sample.size(0)
        n_clips += B
        sum_pose_mse += per_sample.sum().item()
        sum_len_mse += len_loss_val.item() * B
        sum_log_T_abs_err += log_abs.sum().item()

    return {
        "tf_pose_mse": sum_pose_mse / max(n_clips, 1),
        "tf_length_mse": sum_len_mse / max(n_clips, 1),
        "tf_log_T_abs_err": sum_log_T_abs_err / max(n_clips, 1),
        "n_clips": n_clips,
    }


@torch.no_grad()
def evaluate_autoregressive(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_clips: Optional[int] = None,
    progress: bool = False,
) -> dict:
    """Autoregressive generation per clip + DTW vs. target.  Slow.

    Parameters
    ----------
    max_clips : int, optional
        Cap the number of clips evaluated (for fast development feedback).
    """
    model.eval()
    seen = 0
    dtw_values: list[float] = []

    iter_dl = loader
    if progress:
        from tqdm import tqdm as _tqdm
        iter_dl = _tqdm(loader, desc="ar-eval")
    for batch in iter_dl:
        for k in ("text_ids", "text_mask"):
            batch[k] = batch[k].to(device)
        gen = model.generate(text_ids=batch["text_ids"], text_mask=batch["text_mask"])
        pose_pred = gen["pose"]                # (B, T_max, 150)
        lengths = gen["lengths"]               # (B,)

        # Compare each clip's predicted length-T_pred slice against its target
        # of length n_frames[b].  DTW handles the length mismatch naturally.
        target = batch["pose"]                 # (B, T_target_max, 150) on CPU still
        n_frames = batch["n_frames"]
        for b in range(pose_pred.size(0)):
            T_pred = int(lengths[b].item())
            T_true = int(n_frames[b].item())
            pred_np = pose_pred[b, :T_pred].detach().cpu().float().numpy()
            true_np = target[b, :T_true].detach().cpu().float().numpy()
            if pred_np.size == 0 or true_np.size == 0:
                continue
            dtw_values.append(dtw_distance(pred_np, true_np))
            seen += 1
            if max_clips is not None and seen >= max_clips:
                break
        if max_clips is not None and seen >= max_clips:
            break

    if not dtw_values:
        return {"ar_dtw_mean": math.nan, "ar_dtw_median": math.nan, "n_clips": 0}
    return {
        "ar_dtw_mean": float(np.mean(dtw_values)),
        "ar_dtw_median": float(np.median(dtw_values)),
        "n_clips": seen,
    }
