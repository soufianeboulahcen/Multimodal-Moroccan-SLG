"""Master orchestration pipeline for photorealistic avatar video generation.

Connects all subsystems in the correct order:

  1. PoseExtractor       — load OpenPose PNG frames (reuses SignLLM outputs)
  2. IdentityEncoder     — extract face embedding from reference dataset video
  3. DiffusionRenderer   — ControlNet + AnimateDiff + SDXL rendering
  4. TemporalSmoother    — pixel-space Gaussian + optional flow warping
  5. RIFEInterpolator    — 2× frame rate via optical-flow interpolation
  6. VideoExporter       — H.264 MP4 + optional side-by-side comparison

The SignLLM pipeline is never touched. All inputs are read-only.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from avatar_video_generator.configs.config import AvatarConfig
from avatar_video_generator.identity.encoder import IdentityEncoder, IdentityEmbedding
from avatar_video_generator.interpolation.rife_interpolator import RIFEInterpolator
from avatar_video_generator.pipelines.pose_extractor import PoseExtractor, find_pose_source
from avatar_video_generator.rendering.diffusion_renderer import DiffusionRenderer
from avatar_video_generator.rendering.temporal_smoother import TemporalSmoother
from avatar_video_generator.utils.video_io import (
    extract_reference_frames,
    make_comparison_video,
    write_frames,
    write_video,
)

logger = logging.getLogger(__name__)


class AvatarPipeline:
    """End-to-end photorealistic avatar generation pipeline.

    Usage::

        from avatar_video_generator import AvatarPipeline, AvatarConfig

        cfg = AvatarConfig()
        pipeline = AvatarPipeline(cfg)
        pipeline.load_models()

        result = pipeline.run(
            pose_source="outputs/pose_control/أَنْتِ_keypoints",
            reference_video=".devcontainer/Dataset/mosl_videos_dataset_Pronouns/أَنْتِ.mp4",
            output_path="outputs/avatar/أَنْتِ_photorealistic.mp4",
        )
        print(f"Generated: {result.output_path}")
    """

    def __init__(self, cfg: Optional[AvatarConfig] = None) -> None:
        self.cfg = cfg or AvatarConfig()
        self._models_loaded = False

        # Subsystem instances (lazy-initialised)
        self._pose_extractor: Optional[PoseExtractor] = None
        self._identity_encoder: Optional[IdentityEncoder] = None
        self._renderer: Optional[DiffusionRenderer] = None
        self._smoother: Optional[TemporalSmoother] = None
        self._interpolator: Optional[RIFEInterpolator] = None

        # Configure logging
        level = logging.INFO if self.cfg.verbose else logging.WARNING
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            level=level,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Initialise all subsystems and load diffusion models into GPU memory.

        Call this once before running the pipeline. Model loading downloads
        weights from HuggingFace on first run (~5–15 GB depending on config).
        """
        if self._models_loaded:
            return

        logger.info("Initialising avatar pipeline subsystems...")

        self._pose_extractor = PoseExtractor(
            resolution=self.cfg.diffusion.resolution
        )

        self._identity_encoder = IdentityEncoder(
            backend=self.cfg.identity.backend,
            face_crop_size=self.cfg.identity.face_crop_size,
            multi_frame_average=self.cfg.identity.multi_frame_average,
            cache_dir=self.cfg.cache_dir,
        )

        self._renderer = DiffusionRenderer(
            diffusion_cfg=self.cfg.diffusion,
            temporal_cfg=self.cfg.temporal,
            device=self.cfg.device,
        )
        self._renderer.load_models()

        self._smoother = TemporalSmoother(self.cfg.temporal)

        self._interpolator = RIFEInterpolator(self.cfg.interpolation)

        self._models_loaded = True
        logger.info("All models loaded. Pipeline ready.")

    # ------------------------------------------------------------------
    # Single-sign generation
    # ------------------------------------------------------------------

    def run(
        self,
        pose_source: str | Path,
        reference_video: str | Path | None,
        output_path: str | Path,
        sign_name: Optional[str] = None,
        frames_dir: str | Path | None = None,
        reference_frames: Optional[List[np.ndarray]] = None,
    ) -> "GenerationResult":
        """Generate a photorealistic avatar video for one sign.

        Args:
            pose_source: path to pose PNG directory or skeleton MP4.
                         Typically ``outputs/pose_control/<sign>_keypoints/``.
            reference_video: optional path to the MoSL dataset video of the same sign.
                             Used to extract the signer's face identity.
                             Typically ``.devcontainer/Dataset/.../أَنْتِ.mp4``.
                             If omitted, the first prototype still runs with
                             identity conditioning disabled.
            output_path: destination MP4 path.
            sign_name: optional sign label for logging and cache keys.
            frames_dir: optional directory where generated avatar PNG frames
                        are written. Defaults to ``<output_stem>_frames``.
            reference_frames: optional pre-extracted RGB frames for InstantID /
                              IP-Adapter style identity conditioning.

        Returns:
            GenerationResult with paths and timing information.
        """
        if not self._models_loaded:
            self.load_models()

        t_start = time.time()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        label = sign_name or Path(pose_source).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating avatar: {label}")
        logger.info(f"{'='*60}")

        # ── Step 1: Load pose conditioning frames ──────────────────────
        t0 = time.time()
        logger.info("Step 1/6 — Loading pose frames...")
        pose_images = self._pose_extractor.load(pose_source)
        pose_numpy = [np.array(img) for img in pose_images]
        logger.info(f"  {len(pose_images)} pose frames loaded ({time.time()-t0:.1f}s)")

        # ── Step 2: Extract signer identity ────────────────────────────
        t0 = time.time()
        logger.info("Step 2/6 — Extracting signer identity...")
        if reference_frames is not None:
            ref_frames = reference_frames
        elif reference_video is not None and Path(reference_video).exists():
            ref_frames = extract_reference_frames(
                reference_video,
                n_frames=self.cfg.identity.multi_frame_count,
                resize=(self.cfg.diffusion.resolution, self.cfg.diffusion.resolution),
            )
        else:
            ref_frames = []
            logger.warning(
                "  No source identity video provided; generating prototype "
                "without InstantID/IP-Adapter identity locking."
            )
        identity = self._identity_encoder.encode(
            ref_frames,
            cache_key=label,
        )
        logger.info(
            f"  Identity: backend={identity.backend}, "
            f"valid={identity.is_valid()}, "
            f"desc='{identity.signer_description}' ({time.time()-t0:.1f}s)"
        )

        # ── Step 3: Diffusion rendering ─────────────────────────────────
        t0 = time.time()
        logger.info("Step 3/6 — Diffusion rendering (ControlNet + AnimateDiff)...")
        raw_frames = self._renderer.render(pose_images, identity)
        logger.info(f"  {len(raw_frames)} frames rendered ({time.time()-t0:.1f}s)")

        # ── Step 4: Temporal smoothing ──────────────────────────────────
        t0 = time.time()
        logger.info("Step 4/6 — Temporal smoothing...")
        smooth_frames = self._smoother.smooth(raw_frames)
        logger.info(f"  Smoothing complete ({time.time()-t0:.1f}s)")

        # ── Step 5: RIFE interpolation ──────────────────────────────────
        t0 = time.time()
        if self.cfg.interpolation.enabled:
            logger.info(
                f"Step 5/6 — RIFE {self.cfg.interpolation.multiplier}× interpolation..."
            )
            final_frames = self._interpolator.interpolate(smooth_frames)
            output_fps = self.cfg.interpolation.output_fps
            logger.info(
                f"  {len(smooth_frames)} → {len(final_frames)} frames "
                f"@ {output_fps:.0f} fps ({time.time()-t0:.1f}s)"
            )
        else:
            final_frames = smooth_frames
            output_fps = self.cfg.export.fps
            logger.info("Step 5/6 — RIFE interpolation skipped.")

        # ── Step 6: Export MP4 + generated frames ───────────────────────
        t0 = time.time()
        logger.info("Step 6/6 — Exporting MP4...")
        frames_path: Optional[Path] = None
        if self.cfg.export.export_frames:
            frames_path = Path(frames_dir) if frames_dir is not None else (
                out_path.parent / f"{out_path.stem}{self.cfg.export.frames_path_suffix}"
            )
            write_frames(final_frames, frames_path)
            logger.info(f"  Avatar frames: {frames_path}")

        write_video(
            final_frames,
            out_path,
            fps=output_fps,
            crf=self.cfg.export.crf,
            codec=self.cfg.export.codec,
            pixel_format=self.cfg.export.pixel_format,
        )

        # Optional side-by-side comparison
        comparison_path: Optional[Path] = None
        if self.cfg.export.export_comparison:
            stem = out_path.stem + self.cfg.export.comparison_path_suffix
            comparison_path = out_path.parent / f"{stem}.mp4"
            try:
                make_comparison_video(
                    pose_numpy, smooth_frames, comparison_path,
                    fps=self.cfg.export.fps,
                )
                logger.info(f"  Comparison: {comparison_path}")
            except Exception as e:
                logger.warning(f"  Comparison export failed: {e}")
                comparison_path = None

        elapsed = time.time() - t_start
        logger.info(f"\nDone in {elapsed:.1f}s → {out_path}")

        return GenerationResult(
            output_path=out_path,
            frames_dir=frames_path,
            comparison_path=comparison_path,
            n_pose_frames=len(pose_images),
            n_output_frames=len(final_frames),
            output_fps=output_fps,
            elapsed_seconds=elapsed,
            identity_backend=identity.backend,
            renderer_backend=self._renderer._pipe_type or "unknown",
        )

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def run_batch(
        self,
        signs: List[dict],
        output_dir: str | Path = "outputs/avatar_photorealistic",
    ) -> List["GenerationResult"]:
        """Generate avatar videos for multiple signs.

        Args:
            signs: list of dicts, each with keys:
                   - ``pose_source``: path to pose PNG dir or skeleton MP4
                   - ``reference_video``: path to MoSL dataset video
                   - ``sign_name``: (optional) label for the sign
            output_dir: directory for all output MP4 files.

        Returns:
            List of GenerationResult objects.
        """
        if not self._models_loaded:
            self.load_models()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: List[GenerationResult] = []
        n = len(signs)

        for i, sign in enumerate(signs):
            sign_name = sign.get("sign_name") or Path(sign["pose_source"]).stem
            safe_name = sign_name.replace("/", "_").replace("\\", "_")[:60]
            out_path = Path(sign.get("output_path") or (out_dir / f"{safe_name}_photorealistic.mp4"))
            frames_dir = sign.get("frames_dir")

            logger.info(f"\n[{i+1}/{n}] {sign_name}")

            if out_path.exists():
                logger.info(f"  [SKIP] already exists: {out_path}")
                continue

            try:
                result = self.run(
                    pose_source=sign["pose_source"],
                    reference_video=sign.get("reference_video"),
                    output_path=out_path,
                    sign_name=sign_name,
                    frames_dir=frames_dir,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"  [FAIL] {sign_name}: {e}")

        logger.info(f"\nBatch complete: {len(results)}/{n} generated in {out_dir}")
        return results

    # ------------------------------------------------------------------
    # Auto-discover and run for a sign name
    # ------------------------------------------------------------------

    def run_sign(
        self,
        sign_name: str,
        dataset_dir: str | Path = ".devcontainer/Dataset",
        output_dir: str | Path = "outputs/avatar_photorealistic",
    ) -> "GenerationResult":
        """High-level convenience: generate avatar for a sign by name.

        Automatically discovers the pose source and reference video from
        the standard project directory layout.

        Args:
            sign_name: Arabic sign name (e.g. "أَنْتِ").
            dataset_dir: root of the MoSL dataset directory.
            output_dir: output directory.

        Returns:
            GenerationResult.
        """
        # Find pose source
        pose_source = find_pose_source(sign_name)
        if pose_source is None:
            raise FileNotFoundError(
                f"No pose source found for '{sign_name}'.\n"
                f"Expected one of:\n"
                f"  outputs/pose_control/{sign_name}_keypoints/\n"
                f"  outputs/videos/skeleton/{sign_name}_skeleton.mp4\n"
                f"  outputs/videos/mosaic/{sign_name}_mosaic.mp4\n"
                "Run the SignLLM pipeline first to generate pose frames."
            )

        # Find reference video
        reference_video = _find_reference_video(sign_name, Path(dataset_dir))
        if reference_video is None:
            raise FileNotFoundError(
                f"No reference video found for '{sign_name}' in {dataset_dir}.\n"
                "The reference video is needed to extract the signer's identity."
            )

        safe_name = sign_name.replace("/", "_").replace("\\", "_")[:60]
        out_path = Path(output_dir) / f"{safe_name}_photorealistic.mp4"

        return self.run(
            pose_source=pose_source,
            reference_video=reference_video,
            output_path=out_path,
            sign_name=sign_name,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory and unload all models."""
        if self._renderer is not None:
            self._renderer.unload()
        self._models_loaded = False
        logger.info("Pipeline unloaded.")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class GenerationResult:
    """Holds the output of a single avatar generation run."""

    def __init__(
        self,
        output_path: Path,
        frames_dir: Optional[Path],
        comparison_path: Optional[Path],
        n_pose_frames: int,
        n_output_frames: int,
        output_fps: float,
        elapsed_seconds: float,
        identity_backend: str,
        renderer_backend: str,
    ) -> None:
        self.output_path = output_path
        self.frames_dir = frames_dir
        self.comparison_path = comparison_path
        self.n_pose_frames = n_pose_frames
        self.n_output_frames = n_output_frames
        self.output_fps = output_fps
        self.elapsed_seconds = elapsed_seconds
        self.identity_backend = identity_backend
        self.renderer_backend = renderer_backend

    def __repr__(self) -> str:
        return (
            f"GenerationResult(\n"
            f"  output={self.output_path}\n"
            f"  frames_dir={self.frames_dir}\n"
            f"  frames={self.n_pose_frames} → {self.n_output_frames} @ {self.output_fps:.0f}fps\n"
            f"  identity={self.identity_backend}  renderer={self.renderer_backend}\n"
            f"  elapsed={self.elapsed_seconds:.1f}s\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            "output_path": str(self.output_path),
            "frames_dir": str(self.frames_dir) if self.frames_dir else None,
            "comparison_path": str(self.comparison_path) if self.comparison_path else None,
            "n_pose_frames": self.n_pose_frames,
            "n_output_frames": self.n_output_frames,
            "output_fps": self.output_fps,
            "elapsed_seconds": self.elapsed_seconds,
            "identity_backend": self.identity_backend,
            "renderer_backend": self.renderer_backend,
        }


# ---------------------------------------------------------------------------
# Reference video discovery
# ---------------------------------------------------------------------------

def _find_reference_video(
    sign_name: str,
    dataset_dir: Path,
) -> Optional[Path]:
    """Search the MoSL dataset directory tree for a video matching sign_name."""
    # Direct match: <sign_name>.mp4
    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{sign_name}.mp4"
        if candidate.exists():
            return candidate

    # Partial match: video filename contains sign_name
    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue
        for mp4 in subdir.glob("*.mp4"):
            if sign_name in mp4.stem:
                return mp4

    return None
