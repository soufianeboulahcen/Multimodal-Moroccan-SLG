"""Identity preservation encoder for the MoSL avatar pipeline.

Extracts a face embedding from reference frames of the real signer and
packages it for injection into the diffusion pipeline via IP-Adapter FaceID
or InsightFace ArcFace conditioning.

The identity embedding ensures the generated avatar is visually identical
to the real person in the MoSL dataset — same face shape, skin tone,
hairstyle, and body proportions.

Backends
--------
insightface (default, recommended)
    Uses InsightFace ArcFace (buffalo_l model) to extract a 512-d face
    embedding. Fast, accurate, no extra model download beyond InsightFace.
    The embedding is used to condition IP-Adapter FaceID at inference time.

ip_adapter
    Uses the IP-Adapter FaceID pipeline directly. Requires downloading
    h94/IP-Adapter-FaceID from HuggingFace (~1 GB). Provides richer
    identity conditioning but is slower.

none
    Disables identity conditioning. The pipeline runs as a standard
    ControlNet + AnimateDiff generation without face locking.
    Use only for ablation studies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity embedding container
# ---------------------------------------------------------------------------

@dataclass
class IdentityEmbedding:
    """Holds the extracted identity information for one signer.

    Attributes:
        face_embedding: (512,) float32 ArcFace embedding, or None.
        face_image: (224, 224, 3) uint8 RGB best face crop, or None.
        reference_frames: list of (H, W, 3) uint8 RGB reference frames.
        backend: which backend produced this embedding.
        signer_description: human-readable description for prompt augmentation.
    """
    face_embedding: Optional[np.ndarray]       # (512,) ArcFace
    face_image: Optional[np.ndarray]           # (224, 224, 3) best crop
    reference_frames: List[np.ndarray]         # raw reference frames
    backend: str
    signer_description: str = ""

    def is_valid(self) -> bool:
        return self.face_embedding is not None or self.face_image is not None

    def save(self, path: str | Path) -> None:
        """Persist embedding to disk for reuse across runs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {"backend": self.backend, "description": self.signer_description}
        if self.face_embedding is not None:
            data["face_embedding"] = self.face_embedding
        if self.face_image is not None:
            data["face_image"] = self.face_image
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> "IdentityEmbedding":
        """Load a previously saved embedding."""
        d = np.load(path, allow_pickle=True)
        return cls(
            face_embedding=d.get("face_embedding"),
            face_image=d.get("face_image"),
            reference_frames=[],
            backend=str(d.get("backend", "loaded")),
            signer_description=str(d.get("description", "")),
        )


# ---------------------------------------------------------------------------
# IdentityEncoder
# ---------------------------------------------------------------------------

class IdentityEncoder:
    """Extracts signer identity from reference video frames.

    Usage::

        encoder = IdentityEncoder(backend="insightface")
        embedding = encoder.encode(reference_frames)
        # embedding.face_embedding: (512,) ArcFace vector
        # embedding.face_image:     (224, 224, 3) best face crop
    """

    def __init__(
        self,
        backend: str = "insightface",
        face_crop_size: int = 224,
        multi_frame_average: bool = True,
        cache_dir: str = "outputs/avatar_photorealistic/.cache",
    ) -> None:
        self.backend = backend
        self.face_crop_size = face_crop_size
        self.multi_frame_average = multi_frame_average
        self.cache_dir = Path(cache_dir)
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        reference_frames: List[np.ndarray],
        cache_key: Optional[str] = None,
    ) -> IdentityEmbedding:
        """Extract identity embedding from a list of reference frames.

        Args:
            reference_frames: list of (H, W, 3) uint8 RGB frames from the
                              MoSL dataset video of the target signer.
            cache_key: if provided, cache the result to disk and reuse on
                       subsequent calls (avoids re-running face detection).

        Returns:
            IdentityEmbedding with face_embedding and face_image populated.
        """
        if not reference_frames:
            logger.warning("No reference frames provided — identity disabled.")
            return IdentityEmbedding(
                face_embedding=None,
                face_image=None,
                reference_frames=[],
                backend=self.backend,
            )

        # Check cache
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"Identity loaded from cache: {cache_key}")
                cached.reference_frames = reference_frames
                return cached

        if self.backend == "insightface":
            embedding = self._encode_insightface(reference_frames)
        elif self.backend == "ip_adapter":
            embedding = self._encode_ip_adapter(reference_frames)
        elif self.backend == "none":
            embedding = IdentityEmbedding(
                face_embedding=None,
                face_image=None,
                reference_frames=reference_frames,
                backend="none",
            )
        else:
            raise ValueError(f"Unknown identity backend: {self.backend!r}")

        embedding.reference_frames = reference_frames

        # Save to cache
        if cache_key and embedding.is_valid():
            self._save_cache(cache_key, embedding)

        return embedding

    # ------------------------------------------------------------------
    # InsightFace backend
    # ------------------------------------------------------------------

    def _encode_insightface(
        self, frames: List[np.ndarray]
    ) -> IdentityEmbedding:
        """Extract ArcFace embedding using InsightFace buffalo_l model.

        Processes up to ``multi_frame_count`` frames and averages the
        embeddings for robustness against pose/lighting variation.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.warning(
                "InsightFace not installed. Falling back to face-crop-only mode.\n"
                "Install with: pip install insightface onnxruntime"
            )
            return self._encode_face_crop_only(frames)

        if self._model is None:
            logger.info("Loading InsightFace buffalo_l model...")
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._model = app

        embeddings: List[np.ndarray] = []
        best_face_img: Optional[np.ndarray] = None
        best_det_score: float = -1.0

        for frame in frames:
            # InsightFace expects BGR
            import cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            faces = self._model.get(bgr)

            if not faces:
                continue

            # Pick the largest / most confident face
            face = max(faces, key=lambda f: f.det_score)

            if face.det_score > best_det_score:
                best_det_score = face.det_score
                # Crop face region
                x1, y1, x2, y2 = face.bbox.astype(int)
                margin = int((x2 - x1) * 0.3)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)
                crop = frame[y1:y2, x1:x2]
                best_face_img = cv2.resize(
                    crop, (self.face_crop_size, self.face_crop_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if face.normed_embedding is not None:
                embeddings.append(face.normed_embedding.copy())

        if not embeddings:
            logger.warning(
                "InsightFace detected no faces in reference frames. "
                "Check that the reference video contains a visible face."
            )
            return self._encode_face_crop_only(frames)

        # Average embeddings and re-normalise
        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm

        desc = self._describe_appearance(best_face_img)

        logger.info(
            f"InsightFace: {len(embeddings)} face(s) detected, "
            f"embedding shape={avg_emb.shape}, det_score={best_det_score:.3f}"
        )

        return IdentityEmbedding(
            face_embedding=avg_emb.astype(np.float32),
            face_image=best_face_img,
            reference_frames=[],
            backend="insightface",
            signer_description=desc,
        )

    # ------------------------------------------------------------------
    # IP-Adapter FaceID backend
    # ------------------------------------------------------------------

    def _encode_ip_adapter(
        self, frames: List[np.ndarray]
    ) -> IdentityEmbedding:
        """Extract face embedding for IP-Adapter FaceID conditioning.

        IP-Adapter FaceID uses the same InsightFace ArcFace embedding
        internally, but the injection mechanism is different (cross-attention
        instead of AdaIN). We extract the embedding here and let the
        diffusion renderer handle the injection.
        """
        # IP-Adapter FaceID uses InsightFace embeddings under the hood
        embedding = self._encode_insightface(frames)
        embedding.backend = "ip_adapter"
        return embedding

    # ------------------------------------------------------------------
    # Fallback: face crop without embedding
    # ------------------------------------------------------------------

    def _encode_face_crop_only(
        self, frames: List[np.ndarray]
    ) -> IdentityEmbedding:
        """Extract a face crop without computing an embedding.

        Used as fallback when InsightFace is unavailable. The face crop
        is passed to IP-Adapter as a reference image for visual conditioning.
        """
        import cv2

        # Use OpenCV Haar cascade as a lightweight fallback detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)

        best_crop: Optional[np.ndarray] = None
        best_area: int = 0

        for frame in frames[:10]:  # check first 10 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

            for (x, y, w, h) in faces:
                area = w * h
                if area > best_area:
                    best_area = area
                    margin = int(w * 0.3)
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    crop = frame[y1:y2, x1:x2]
                    best_crop = cv2.resize(
                        crop, (self.face_crop_size, self.face_crop_size),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

        if best_crop is None and frames:
            # Last resort: use centre crop of first frame
            f = frames[0]
            h, w = f.shape[:2]
            cx, cy = w // 2, h // 4  # face is typically in upper quarter
            half = min(w, h) // 4
            crop = f[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            best_crop = cv2.resize(
                crop, (self.face_crop_size, self.face_crop_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return IdentityEmbedding(
            face_embedding=None,
            face_image=best_crop,
            reference_frames=[],
            backend="face_crop",
            signer_description="",
        )

    # ------------------------------------------------------------------
    # Appearance description for prompt augmentation
    # ------------------------------------------------------------------

    def _describe_appearance(
        self, face_crop: Optional[np.ndarray]
    ) -> str:
        """Generate a text description of the signer's appearance.

        Used to augment the diffusion prompt with identity-specific tokens
        that help the model maintain consistent appearance across frames.

        For the MoSL dataset, the signer is a Moroccan adult with:
        - Medium-dark skin tone
        - Dark hair
        - Professional appearance
        """
        # Base description for the MoSL dataset signer
        # This is refined by analysing the face crop's colour statistics
        base = "Moroccan adult, medium skin tone, dark hair"

        if face_crop is None:
            return base

        # Analyse skin tone from face crop
        import cv2
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_RGB2HSV)
        mean_v = hsv[:, :, 2].mean()  # brightness

        if mean_v > 180:
            tone = "light skin tone"
        elif mean_v > 130:
            tone = "medium skin tone"
        else:
            tone = "medium-dark skin tone"

        return f"Moroccan adult, {tone}, dark hair, professional appearance"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")[:80]
        return self.cache_dir / f"identity_{safe}.npz"

    def _load_cache(self, key: str) -> Optional[IdentityEmbedding]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return IdentityEmbedding.load(p)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, embedding: IdentityEmbedding) -> None:
        p = self._cache_path(key)
        try:
            embedding.save(p)
        except Exception as e:
            logger.debug(f"Could not save identity cache: {e}")
