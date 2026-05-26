"""RIFE frame interpolation for smooth avatar video output.

RIFE (Real-Time Intermediate Flow Estimation) synthesises intermediate frames
between existing ones using learned optical flow, doubling or quadrupling the
effective frame rate without re-running diffusion.

For sign language video at 25 fps, 2× interpolation produces 50 fps output
that looks significantly smoother on screen while preserving all generated
motion detail.

Backends (tried in order):
  1. rife-ncnn-vulkan  — fastest, GPU-accelerated via Vulkan (no Python dep)
  2. ECCV2022-RIFE     — PyTorch implementation, requires torch
  3. Linear blend      — fallback: simple linear interpolation between frames

The linear blend fallback is always available and produces acceptable results
for slow-to-medium motion (typical of sign language).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from avatar_video_generator.configs.config import InterpolationConfig

logger = logging.getLogger(__name__)


class RIFEInterpolator:
    """Doubles (or quadruples) frame rate via RIFE optical-flow interpolation.

    Usage::

        interp = RIFEInterpolator(cfg)
        smooth_frames = interp.interpolate(frames_25fps)
        # smooth_frames: 2× as many frames at 50 fps
    """

    def __init__(self, cfg: InterpolationConfig) -> None:
        self.cfg = cfg
        self._backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Interpolate a frame sequence to a higher frame rate.

        Args:
            frames: list of (H, W, 3) uint8 RGB arrays at the source fps.

        Returns:
            Interpolated frames at ``multiplier`` × the source frame rate.
            Length = (len(frames) - 1) * multiplier + 1
        """
        if not self.cfg.enabled or len(frames) < 2:
            return frames

        multiplier = self.cfg.multiplier
        if multiplier not in (2, 4):
            logger.warning(f"Unsupported multiplier {multiplier}, using 2×.")
            multiplier = 2

        # Try backends in order of quality
        for backend in ("rife_ncnn", "rife_torch", "linear"):
            try:
                result = self._interpolate_with(frames, multiplier, backend)
                self._backend = backend
                logger.info(
                    f"RIFE interpolation: {len(frames)} → {len(result)} frames "
                    f"({multiplier}×) via {backend}"
                )
                return result
            except Exception as e:
                logger.debug(f"Backend {backend} failed: {e}")
                continue

        logger.warning("All RIFE backends failed — returning original frames.")
        return frames

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _interpolate_with(
        self,
        frames: List[np.ndarray],
        multiplier: int,
        backend: str,
    ) -> List[np.ndarray]:
        if backend == "rife_ncnn":
            return self._rife_ncnn(frames, multiplier)
        elif backend == "rife_torch":
            return self._rife_torch(frames, multiplier)
        elif backend == "linear":
            return self._linear_blend(frames, multiplier)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------
    # Backend 1: rife-ncnn-vulkan (CLI tool)
    # ------------------------------------------------------------------

    def _rife_ncnn(
        self, frames: List[np.ndarray], multiplier: int
    ) -> List[np.ndarray]:
        """Use rife-ncnn-vulkan CLI for GPU-accelerated interpolation."""
        rife_bin = shutil.which("rife-ncnn-vulkan")
        if rife_bin is None:
            raise FileNotFoundError("rife-ncnn-vulkan not found on PATH.")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "input"
            out_dir = tmp_path / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            # Write input frames as PNG
            for i, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(in_dir / f"{i:06d}.png"), bgr)

            # Run rife-ncnn-vulkan
            cmd = [
                rife_bin,
                "-i", str(in_dir),
                "-o", str(out_dir),
                "-m", f"rife-v{self.cfg.model_version}",
                "-n", str(multiplier),
                "-f", "%06d.png",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Read output frames
            out_files = sorted(out_dir.glob("*.png"))
            result: List[np.ndarray] = []
            for p in out_files:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    result.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if not result:
            raise RuntimeError("rife-ncnn-vulkan produced no output frames.")
        return result

    # ------------------------------------------------------------------
    # Backend 2: ECCV2022-RIFE (PyTorch)
    # ------------------------------------------------------------------

    def _rife_torch(
        self, frames: List[np.ndarray], multiplier: int
    ) -> List[np.ndarray]:
        """Use the ECCV2022-RIFE PyTorch model for interpolation."""
        try:
            import torch
            from torchvision.transforms.functional import to_tensor, to_pil_image
        except ImportError:
            raise ImportError("torch/torchvision required for RIFE PyTorch backend.")

        # Try to import the RIFE model from third_party or installed package
        try:
            from rife.model.RIFE_HDv3 import Model as RIFEModel  # type: ignore
        except ImportError:
            # Try the third_party path
            import sys
            rife_path = Path(__file__).parents[3] / "third_party" / "ECCV2022-RIFE"
            if rife_path.exists():
                sys.path.insert(0, str(rife_path))
                from model.RIFE_HDv3 import Model as RIFEModel  # type: ignore
            else:
                raise ImportError(
                    "RIFE PyTorch model not found. "
                    "Clone https://github.com/megvii-research/ECCV2022-RIFE "
                    "into third_party/ECCV2022-RIFE"
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RIFEModel()
        model.load_model(
            str(Path(__file__).parents[3] / "third_party" / "ECCV2022-RIFE" / "train_log"),
            -1,
        )
        model.eval()
        model.device()

        result: List[np.ndarray] = []

        for i in range(len(frames) - 1):
            f0 = frames[i]
            f1 = frames[i + 1]

            t0 = to_tensor(f0).unsqueeze(0).to(device)
            t1 = to_tensor(f1).unsqueeze(0).to(device)

            result.append(f0)

            # Generate intermediate frames
            n_mid = multiplier - 1
            for k in range(1, n_mid + 1):
                t = k / multiplier
                with torch.no_grad():
                    mid = model.inference(t0, t1, timestep=t)
                mid_np = (mid.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                result.append(mid_np)

        result.append(frames[-1])
        return result

    # ------------------------------------------------------------------
    # Backend 3: Linear blend (always available)
    # ------------------------------------------------------------------

    def _linear_blend(
        self, frames: List[np.ndarray], multiplier: int
    ) -> List[np.ndarray]:
        """Synthesise intermediate frames via linear pixel interpolation.

        This is a simple but effective fallback. For sign language motion
        (which is relatively smooth), linear interpolation produces
        visually acceptable results at 2× frame rate.
        """
        result: List[np.ndarray] = []
        n_mid = multiplier - 1

        for i in range(len(frames) - 1):
            f0 = frames[i].astype(np.float32)
            f1 = frames[i + 1].astype(np.float32)
            result.append(frames[i])

            for k in range(1, n_mid + 1):
                alpha = k / multiplier
                mid = (1.0 - alpha) * f0 + alpha * f1
                result.append(np.clip(mid, 0, 255).astype(np.uint8))

        result.append(frames[-1])
        return result
