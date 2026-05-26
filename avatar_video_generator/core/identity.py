"""Signer identity extraction from reference video frames.

Extracts a face embedding and best face crop from the MoSL dataset video
to lock the generated avatar's appearance to the real signer.

Backend cascade:
  1. InsightFace ArcFace (buffalo_l) — 512-d embedding, most accurate
  2. OpenCV Haar cascade — face crop only, no embedding
  3. Centre-crop fallback — always works

The face embedding is used by IP-Adapter FaceID.
The face crop is used as a reference image for IP-Adapter image conditioning.
The appearance description augments the text prompt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IdentityResult:
    face_embedding: Optional[np.ndarray]   # (512,) ArcFace, or None
    face_crop: Optional[np.ndarray]        # (224,224,3) uint8 RGB best crop
    appearance_prompt: str                 # text description for prompt augmentation
    backend: str


def extract_identity(
    frames: List[np.ndarray],
    crop_size: int = 224,
    n_frames: int = 8,
) -> IdentityResult:
    """Extract signer identity from a list of reference frames.

    Args:
        frames: list of (H,W,3) uint8 RGB frames from the dataset video
        crop_size: output face crop resolution
        n_frames: max frames to process (uses uniform sampling)

    Returns:
        IdentityResult with face_embedding, face_crop, appearance_prompt
    """
    if not frames:
        return IdentityResult(None, None, "", "none")

    # Sample uniformly
    indices = _uniform_sample(len(frames), n_frames)
    sampled = [frames[i] for i in indices]

    # Try InsightFace first
    try:
        return _insightface_extract(sampled, crop_size)
    except Exception as e:
        logger.debug(f"InsightFace failed: {e}")

    # Fallback: OpenCV Haar
    try:
        return _haar_extract(sampled, crop_size)
    except Exception as e:
        logger.debug(f"Haar failed: {e}")

    # Last resort: centre crop
    return _centre_crop(frames[0], crop_size)


def _uniform_sample(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    return [int(i * (total - 1) / (n - 1)) for i in range(n)]


# ── InsightFace backend ───────────────────────────────────────────────────────

def _insightface_extract(frames: List[np.ndarray], crop_size: int) -> IdentityResult:
    import insightface  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings: List[np.ndarray] = []
    best_crop: Optional[np.ndarray] = None
    best_score: float = -1.0

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score > best_score:
            best_score = face.det_score
            best_crop = _crop_face(frame, face.bbox.astype(int), crop_size)
        if face.normed_embedding is not None:
            embeddings.append(face.normed_embedding.copy())

    if not embeddings:
        raise ValueError("InsightFace: no faces detected")

    avg_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb /= norm

    desc = _describe_skin(best_crop)
    logger.info(f"InsightFace: {len(embeddings)} faces, score={best_score:.3f}")
    return IdentityResult(avg_emb.astype(np.float32), best_crop, desc, "insightface")


def _crop_face(frame: np.ndarray, bbox: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * 0.3), int(bh * 0.3)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)


# ── OpenCV Haar fallback ──────────────────────────────────────────────────────

def _haar_extract(frames: List[np.ndarray], crop_size: int) -> IdentityResult:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(cascade_path)
    best_crop: Optional[np.ndarray] = None
    best_area = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        for (x, y, fw, fh) in faces:
            if fw * fh > best_area:
                best_area = fw * fh
                best_crop = _crop_face(frame,
                    np.array([x, y, x+fw, y+fh]), crop_size)

    if best_crop is None:
        raise ValueError("Haar: no faces detected")

    desc = _describe_skin(best_crop)
    logger.info("Haar cascade: face detected")
    return IdentityResult(None, best_crop, desc, "haar")


# ── Centre-crop fallback ──────────────────────────────────────────────────────

def _centre_crop(frame: np.ndarray, crop_size: int) -> IdentityResult:
    h, w = frame.shape[:2]
    # Face is typically in upper-centre of frame
    cx, cy = w // 2, h // 5
    half = min(w, h) // 5
    x1 = max(0, cx - half); y1 = max(0, cy - half)
    x2 = min(w, cx + half); y2 = min(h, cy + half)
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
    desc = _describe_skin(crop)
    logger.info("Identity: centre-crop fallback")
    return IdentityResult(None, crop, desc, "centre_crop")


# ── Appearance description ────────────────────────────────────────────────────

def _describe_skin(crop: Optional[np.ndarray]) -> str:
    """Generate appearance tokens from face crop colour statistics."""
    base = "Moroccan adult, dark hair, professional appearance"
    if crop is None:
        return base

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2].mean()

    if brightness > 175:
        tone = "light skin tone"
    elif brightness > 120:
        tone = "medium skin tone"
    else:
        tone = "medium-dark skin tone"

    return f"Moroccan adult, {tone}, dark hair, professional appearance"
