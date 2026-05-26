"""Image manipulation utilities for the avatar pipeline."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def resize_frame(
    frame: np.ndarray,
    size: int | Tuple[int, int],
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """Resize a (H, W, 3) frame to (size, size) or (w, h)."""
    if isinstance(size, int):
        size = (size, size)
    return cv2.resize(frame, size, interpolation=interpolation)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] → float32 [0, 1]."""
    return frame.astype(np.float32) / 255.0


def denormalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert float32 [0, 1] → uint8 [0, 255]."""
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def frames_to_tensor(frames: List[np.ndarray], device: str = "cpu"):
    """Convert list of (H, W, 3) uint8 RGB → torch (T, C, H, W) float32 [0,1]."""
    import torch
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)              # (T, 3, H, W)
    return t.to(device)


def tensor_to_frames(tensor) -> List[np.ndarray]:
    """Convert torch (T, C, H, W) float32 [0,1] → list of (H, W, 3) uint8 RGB."""
    arr = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()    # (T, H, W, 3)
    return [np.clip(arr[i] * 255, 0, 255).astype(np.uint8) for i in range(arr.shape[0])]


def crop_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    size: int = 224,
    margin: float = 0.3,
) -> np.ndarray:
    """Crop and resize a face region from a frame.

    Args:
        frame: (H, W, 3) uint8 RGB.
        bbox: (x1, y1, x2, y2) face bounding box in pixels.
        size: output square size.
        margin: fractional padding around the bbox.

    Returns:
        (size, size, 3) uint8 RGB face crop.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert PIL.Image (RGB) → numpy (H, W, 3) uint8."""
    return np.array(pil_image.convert("RGB"))


def numpy_to_pil(arr: np.ndarray):
    """Convert numpy (H, W, 3) uint8 RGB → PIL.Image."""
    from PIL import Image
    return Image.fromarray(arr.astype(np.uint8))
