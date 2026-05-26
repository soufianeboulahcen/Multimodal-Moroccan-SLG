"""Pose extraction and ControlNet conditioning preparation.

Reads the existing OpenPose PNG frames produced by the SignLLM pipeline
(outputs/pose_control/<sign>_keypoints/) or skeleton MP4 videos and
prepares them as PIL Images ready for ControlNet conditioning.

This module is read-only with respect to the SignLLM outputs — it never
modifies .skels files, keypoint JSON, or any upstream pipeline artifact.

Supported sources (in priority order):
  1. Directory of pose_*.png OpenPose frames  (outputs/pose_control/)
  2. Skeleton MP4 video                       (outputs/videos/skeleton/)
  3. Mosaic/overlay MP4 video                 (outputs/videos/mosaic/)
  4. Raw .skels file → rendered on-the-fly
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from avatar_video_generator.utils.image_utils import resize_frame


# ---------------------------------------------------------------------------
# COCO-18 skeleton rendering constants (matches existing pipeline exactly)
# ---------------------------------------------------------------------------

COCO_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]

# OpenPose canonical colours (RGB) — same as controlnet-aux DWPose output
COCO_JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

COCO_LIMB_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]


# ---------------------------------------------------------------------------
# PoseExtractor
# ---------------------------------------------------------------------------

class PoseExtractor:
    """Loads and normalises pose conditioning frames for ControlNet.

    The extractor is stateless — call ``load()`` to get a list of PIL Images
    ready to pass directly to the ControlNet pipeline.

    Example::

        extractor = PoseExtractor(resolution=512)
        pose_images = extractor.load(
            "outputs/pose_control/أَنْتِ_keypoints"
        )
        # pose_images: List[PIL.Image] (H=512, W=512, RGB)
    """

    def __init__(self, resolution: int = 512) -> None:
        self.resolution = resolution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str | Path) -> List[Image.Image]:
        """Load pose conditioning frames from any supported source.

        Args:
            source: path to a pose PNG directory, skeleton MP4, or .skels file.

        Returns:
            List of PIL Images (resolution × resolution, RGB) ready for ControlNet.
        """
        source = Path(source)

        if source.is_dir():
            return self._load_from_png_dir(source)
        elif source.suffix.lower() == ".mp4":
            return self._load_from_video(source)
        elif source.suffix.lower() == ".skels":
            return self._load_from_skels(source)
        else:
            raise ValueError(
                f"Unsupported pose source: {source}\n"
                "Expected: directory of pose_*.png, .mp4 skeleton video, or .skels file."
            )

    def load_numpy(self, source: str | Path) -> List[np.ndarray]:
        """Same as ``load()`` but returns numpy (H, W, 3) uint8 arrays."""
        return [np.array(img) for img in self.load(source)]

    # ------------------------------------------------------------------
    # PNG directory loader (primary path — reuses existing pipeline output)
    # ------------------------------------------------------------------

    def _load_from_png_dir(self, directory: Path) -> List[Image.Image]:
        """Load sorted pose_*.png files from an existing keypoints directory."""
        # Try multiple naming patterns used by the existing pipeline
        for pattern in ("pose_*.png", "frame_*.png", "*.png"):
            candidates = sorted(directory.glob(pattern))
            if candidates:
                break

        if not candidates:
            raise FileNotFoundError(
                f"No PNG frames found in {directory}\n"
                "Run the SignLLM pipeline first to generate pose frames."
            )

        frames: List[Image.Image] = []
        for p in candidates:
            img = Image.open(p).convert("RGB")
            if img.size != (self.resolution, self.resolution):
                img = img.resize(
                    (self.resolution, self.resolution), Image.LANCZOS
                )
            frames.append(img)

        return frames

    # ------------------------------------------------------------------
    # MP4 skeleton video loader
    # ------------------------------------------------------------------

    def _load_from_video(self, video_path: Path) -> List[Image.Image]:
        """Extract frames from a skeleton/mosaic/overlay MP4."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames: List[Image.Image] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb, (self.resolution, self.resolution),
                interpolation=cv2.INTER_LANCZOS4,
            )
            frames.append(Image.fromarray(resized))

        cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        return frames

    # ------------------------------------------------------------------
    # .skels file loader (renders keypoints on-the-fly)
    # ------------------------------------------------------------------

    def _load_from_skels(self, skels_path: Path) -> List[Image.Image]:
        """Render OpenPose frames from a .skels keypoint file.

        The .skels format stores one pose per line as space-separated floats
        (T × 150 values: 18 body joints × (x, y, conf) + hand joints).
        This matches the Prompt2Sign / SignLLM format exactly.
        """
        sequences = _parse_skels(skels_path)
        if not sequences:
            raise ValueError(f"No sequences found in {skels_path}")

        # Use the first sequence
        pose_seq = sequences[0]  # (T, 150)
        frames: List[Image.Image] = []

        for t in range(pose_seq.shape[0]):
            kp = pose_seq[t, :54].reshape(18, 3)  # body keypoints only
            img = self._render_openpose(kp)
            frames.append(img)

        return frames

    def _render_openpose(self, keypoints: np.ndarray) -> Image.Image:
        """Render a single OpenPose frame from (18, 3) keypoints [x, y, conf].

        Coordinates are assumed normalised [0, 1]. Produces the same visual
        style as the existing pipeline (coloured joints + limbs on black bg).
        """
        W = H = self.resolution
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        kp = keypoints.copy()
        # Normalise if needed
        if kp[:, :2].max() <= 1.0:
            kp[:, 0] *= W
            kp[:, 1] *= H

        # Draw limbs
        for idx, (i, j) in enumerate(COCO_CONNECTIONS):
            if i >= len(kp) or j >= len(kp):
                continue
            ci = kp[i, 2] if kp.shape[1] > 2 else 1.0
            cj = kp[j, 2] if kp.shape[1] > 2 else 1.0
            if ci < 0.05 or cj < 0.05:
                continue
            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
            color = COCO_LIMB_COLORS[idx % len(COCO_LIMB_COLORS)]
            # OpenCV uses BGR
            cv2.line(canvas, pt1, pt2,
                     (color[2], color[1], color[0]), 3, cv2.LINE_AA)

        # Draw joints
        for k in range(min(len(kp), 18)):
            conf = kp[k, 2] if kp.shape[1] > 2 else 1.0
            if conf < 0.05:
                continue
            x, y = int(kp[k, 0]), int(kp[k, 1])
            color = COCO_JOINT_COLORS[k % len(COCO_JOINT_COLORS)]
            cv2.circle(canvas, (x, y), 5,
                       (color[2], color[1], color[0]), -1, cv2.LINE_AA)

        # canvas is BGR → convert to RGB for PIL
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# .skels parser
# ---------------------------------------------------------------------------

def _parse_skels(path: Path) -> List[np.ndarray]:
    """Parse a .skels file into a list of (T, D) float32 arrays.

    Format: each line is one sign sequence, values space-separated.
    Each sequence has T*D values where D=150 (SignLLM convention).
    """
    sequences: List[np.ndarray] = []
    D = 150

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.array(line.split(), dtype=np.float32)
            if len(vals) % D != 0:
                continue
            T = len(vals) // D
            sequences.append(vals.reshape(T, D))

    return sequences


# ---------------------------------------------------------------------------
# Convenience: auto-discover pose source for a sign name
# ---------------------------------------------------------------------------

def find_pose_source(
    sign_name: str,
    pose_control_dir: str | Path = "outputs/pose_control",
    skeleton_dir: str | Path = "outputs/videos/skeleton",
    mosaic_dir: str | Path = "outputs/videos/mosaic",
) -> Optional[Path]:
    """Find the best available pose source for a given sign name.

    Search order:
      1. outputs/pose_control/<sign>_keypoints/  (PNG frames — best)
      2. outputs/videos/skeleton/<sign>_skeleton.mp4
      3. outputs/videos/mosaic/<sign>_mosaic.mp4

    Returns None if no source is found.
    """
    pose_dir = Path(pose_control_dir) / f"{sign_name}_keypoints"
    if pose_dir.exists() and list(pose_dir.glob("*.png")):
        return pose_dir

    skel = Path(skeleton_dir) / f"{sign_name}_skeleton.mp4"
    if skel.exists():
        return skel

    mosaic = Path(mosaic_dir) / f"{sign_name}_mosaic.mp4"
    if mosaic.exists():
        return mosaic

    return None
