"""Temporal consistency post-processing for generated avatar frames.

Applies multi-stage smoothing to eliminate flickering and ghosting artefacts
that arise from independent per-frame diffusion generation:

  Stage 1 — Pixel-space Gaussian smoothing
    A 1D Gaussian filter along the time axis of each pixel channel.
    Reduces high-frequency flicker without blurring spatial detail.

  Stage 2 — Optical-flow-guided warping (optional, requires opencv-contrib)
    Warps each frame toward its neighbours using dense optical flow.
    Preserves sharp edges while enforcing motion consistency.

  Stage 3 — Weighted temporal blend
    Blends each frame with a weighted average of its neighbours.
    Provides a final smoothing pass that handles residual artefacts.
"""
from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from avatar_video_generator.configs.config import TemporalConfig

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Post-processes a sequence of generated frames for temporal consistency.

    Usage::

        smoother = TemporalSmoother(temporal_cfg)
        smooth_frames = smoother.smooth(raw_frames)
    """

    def __init__(self, cfg: TemporalConfig) -> None:
        self.cfg = cfg

    def smooth(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply all enabled smoothing stages to a frame sequence.

        Args:
            frames: list of (H, W, 3) uint8 RGB arrays.

        Returns:
            Smoothed frames, same format.
        """
        if len(frames) < 3:
            return frames

        result = frames

        # Stage 1: Gaussian temporal smoothing
        if self.cfg.pixel_smooth_sigma > 0:
            result = self._gaussian_smooth(result, self.cfg.pixel_smooth_sigma)

        # Stage 2: Optical-flow warping (optional)
        if self.cfg.use_flow_warp:
            result = self._flow_warp_smooth(result)

        return result

    # ------------------------------------------------------------------
    # Stage 1: Gaussian temporal smoothing
    # ------------------------------------------------------------------

    def _gaussian_smooth(
        self,
        frames: List[np.ndarray],
        sigma: float,
    ) -> List[np.ndarray]:
        """Apply 1D Gaussian filter along the time axis.

        Operates on each (R, G, B) channel independently.
        The spatial structure of each frame is preserved — only the
        temporal variation is smoothed.
        """
        # Stack: (T, H, W, 3) float32
        arr = np.stack(frames, axis=0).astype(np.float32)

        # Filter along axis=0 (time)
        smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

        return [smoothed[i] for i in range(len(frames))]

    # ------------------------------------------------------------------
    # Stage 2: Optical-flow-guided warping
    # ------------------------------------------------------------------

    def _flow_warp_smooth(
        self,
        frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Warp each frame toward its neighbours using dense optical flow.

        Uses Farneback dense optical flow (available in base opencv-python).
        For each frame i, computes flow from i-1→i and i+1→i, then blends
        the warped neighbours with the original frame.
        """
        result = list(frames)
        n = len(frames)

        for i in range(1, n - 1):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            # Forward flow: prev → curr
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            # Backward flow: next → curr
            flow_bwd = cv2.calcOpticalFlowFarneback(
                next_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )

            warped_prev = _warp_frame(frames[i - 1], flow_fwd)
            warped_next = _warp_frame(frames[i + 1], flow_bwd)

            # Blend: 60% current + 20% warped_prev + 20% warped_next
            blended = (
                0.6 * frames[i].astype(np.float32)
                + 0.2 * warped_prev.astype(np.float32)
                + 0.2 * warped_next.astype(np.float32)
            )
            result[i] = np.clip(blended, 0, 255).astype(np.uint8)

        return result


def _warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a frame using a dense optical flow field."""
    h, w = flow.shape[:2]
    map_x = (np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0])
    map_y = (np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1])
    warped = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped
