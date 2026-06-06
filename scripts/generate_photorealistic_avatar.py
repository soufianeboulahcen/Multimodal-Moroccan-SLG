"""Photorealistic avatar video generation from MoSL skeleton motion.

Converts the existing OpenPose/skeleton outputs from the SignLLM pipeline
into photorealistic human avatar videos while preserving the identity of
the real signer from the MoSL dataset.

The SignLLM pipeline is never modified — this script only reads its outputs.

Pipeline
--------
  OpenPose PNG frames (outputs/pose_control/<sign>_keypoints/)
    + MoSL dataset video (reference signer identity)
    → InsightFace ArcFace embedding
    → ControlNet-OpenPose + AnimateDiff + SDXL
    → Temporal smoothing (Gaussian + optional flow warp)
    → RIFE 2× interpolation
    → Photorealistic MP4 (outputs/avatar_photorealistic/)

Usage
-----
  # Single sign — auto-discover pose source and reference video
  python scripts/generate_photorealistic_avatar.py --sign أَنْتِ

  # Explicit paths
  python scripts/generate_photorealistic_avatar.py \\
      --pose-dir outputs/pose_control/أَنْتِ_keypoints \\
      --reference-video ".devcontainer/Dataset/mosl_videos_dataset_Pronouns/أَنْتِ.mp4" \\
      --output outputs/avatar_photorealistic/أَنْتِ_photorealistic.mp4

  # Official HD OpenPose prototype input
  python scripts/generate_photorealistic_avatar.py \\
      --official-hd-openpose \\
      --allow-no-reference

  # Batch — all signs with existing pose frames
  python scripts/generate_photorealistic_avatar.py --batch

  # DGX high-quality config
  python scripts/generate_photorealistic_avatar.py --sign أَنْتِ \\
      --config avatar_video_generator/configs/dgx.yaml

  # SD1.5 fallback (lower VRAM)
  python scripts/generate_photorealistic_avatar.py --sign أَنْتِ \\
      --no-sdxl --no-animatediff

  # Disable identity locking (ablation)
  python scripts/generate_photorealistic_avatar.py --sign أَنْتِ \\
      --identity-backend none
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_video_generator.configs.config import (
    AvatarConfig,
    DiffusionConfig,
    ExportConfig,
    IdentityConfig,
    InterpolationConfig,
    TemporalConfig,
)
from avatar_video_generator.pipelines.pose_extractor import find_pose_source
from avatar_video_generator.utils.video_io import (
    ascii_slug,
    get_video_info,
    read_video_frames,
    validate_video_file,
    write_frames,
    write_video,
)


DEFAULT_VIDEO_SCAN_DIRS = [
    Path("outputs/videos/mosaic"),
    Path("outputs/videos"),
]
VIDEO_SUFFIXES = (
    "_mosaic",
    "_studio",
    "_neon",
    "_overlay",
    "_skeleton",
    "_slowmo",
    "_heatmap",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate photorealistic avatar video from MoSL skeleton motion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ──────────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--sign",
        metavar="ARABIC_SIGN",
        help="Arabic sign name (auto-discovers pose source and reference video)",
    )
    src.add_argument(
        "--pose-dir",
        type=Path,
        metavar="DIR",
        help="Directory of pose_*.png OpenPose frames",
    )
    src.add_argument(
        "--skeleton-video",
        type=Path,
        metavar="MP4",
        help="Skeleton MP4 video (outputs/videos/skeleton/)",
    )
    src.add_argument(
        "--batch",
        action="store_true",
        help="Process all signs with existing pose frames in outputs/pose_control/",
    )
    src.add_argument(
        "--scan-video-sources",
        action="store_true",
        help=(
            "Scan outputs/videos/mosaic and outputs/videos, select best samples, "
            "and generate photorealistic avatars under avatar_video_generator/outputs"
        ),
    )
    src.add_argument(
        "--official-hd-openpose",
        action="store_true",
        help=(
            "Use outputs/avatar_from_video_hd/alsbt_ishara_2_pose/pose_*.png "
            "as the official HD ControlNet OpenPose conditioning input"
        ),
    )

    # ── Reference identity ───────────────────────────────────────────────────
    p.add_argument(
        "--reference-video",
        type=Path,
        metavar="MP4",
        help="MoSL dataset video for signer identity extraction",
    )
    p.add_argument(
        "--reference-image",
        type=Path,
        metavar="PNG/JPG",
        help="Single human source image for identity preservation",
    )
    p.add_argument(
        "--reference-frames-dir",
        type=Path,
        metavar="DIR",
        help="Directory of human source frames for identity preservation",
    )
    p.add_argument(
        "--allow-no-reference",
        action="store_true",
        help="Continue without identity locking if no human source video is available",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(".devcontainer/Dataset"),
        metavar="DIR",
        help="Root of the MoSL dataset directory",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--output",
        type=Path,
        metavar="MP4",
        help="Output MP4 path (single-sign mode)",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        metavar="DIR",
        help="Directory for generated avatar PNG frames",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (batch mode)",
    )
    p.add_argument(
        "--video-source-dir",
        type=Path,
        action="append",
        metavar="DIR",
        help="Additional video source directory to scan; can be repeated",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum selected video samples to render after prioritisation",
    )

    # ── Config ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to YAML config file (default: avatar_video_generator/configs/default.yaml)",
    )

    # ── Diffusion overrides ──────────────────────────────────────────────────
    p.add_argument("--resolution", type=int, default=None, choices=[512, 768])
    p.add_argument("--steps", type=int, default=None, dest="num_inference_steps",
                   help="Denoising steps (20–50)")
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--controlnet-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-sdxl", action="store_true", help="Use SD1.5 instead of SDXL")
    p.add_argument("--no-animatediff", action="store_true",
                   help="Disable AnimateDiff (frame-by-frame mode)")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Enable sequential CPU offload (for low VRAM)")
    p.add_argument("--fp32", action="store_true", help="Use float32 (slower, more precise)")

    # ── Identity overrides ───────────────────────────────────────────────────
    p.add_argument(
        "--identity-backend",
        choices=["insightface", "ip_adapter", "none"],
        default=None,
    )

    # ── Interpolation overrides ──────────────────────────────────────────────
    p.add_argument("--no-rife", action="store_true", help="Disable RIFE interpolation")
    p.add_argument("--rife-multiplier", type=int, choices=[2, 4], default=None)

    # ── Device ───────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip side-by-side comparison video")
    p.add_argument("--no-frame-export", action="store_true",
                   help="Skip generated avatar PNG frame export")
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> AvatarConfig:
    """Build AvatarConfig from YAML file + CLI overrides."""
    # Load base config
    if args.config and args.config.exists():
        cfg = AvatarConfig.from_yaml(args.config)
    else:
        default_yaml = Path("avatar_video_generator/configs/default.yaml")
        if default_yaml.exists():
            cfg = AvatarConfig.from_yaml(default_yaml)
        else:
            cfg = AvatarConfig()

    # Apply CLI overrides
    cfg.device = args.device
    cfg.verbose = not args.quiet
    cfg.output_dir = str(_default_output_dir(args))

    # Diffusion overrides
    if args.resolution is not None:
        cfg.diffusion.resolution = args.resolution
    if args.num_inference_steps is not None:
        cfg.diffusion.num_inference_steps = args.num_inference_steps
    if args.guidance_scale is not None:
        cfg.diffusion.guidance_scale = args.guidance_scale
    if args.controlnet_scale is not None:
        cfg.diffusion.controlnet_conditioning_scale = args.controlnet_scale
    if args.seed is not None:
        cfg.diffusion.seed = args.seed
    if args.no_sdxl:
        cfg.diffusion.use_sdxl = False
    if args.no_animatediff:
        cfg.diffusion.use_animatediff = False
    if args.cpu_offload:
        cfg.diffusion.enable_cpu_offload = True
    if args.fp32:
        cfg.diffusion.use_fp16 = False

    # Identity overrides
    if args.identity_backend is not None:
        cfg.identity.backend = args.identity_backend

    # Interpolation overrides
    if args.no_rife:
        cfg.interpolation.enabled = False
    if args.rife_multiplier is not None:
        cfg.interpolation.multiplier = args.rife_multiplier
        cfg.interpolation.output_fps = 25.0 * args.rife_multiplier

    # Export overrides
    if args.no_comparison:
        cfg.export.export_comparison = False
    if args.no_frame_export:
        cfg.export.export_frames = False

    # The HD OpenPose prototype should exercise the requested high-quality path:
    # SDXL or better + ControlNet OpenPose + AnimateDiff as the primary backend.
    if args.official_hd_openpose:
        cfg.diffusion.use_sdxl = True
        cfg.diffusion.use_animatediff = True
        if args.resolution is None:
            cfg.diffusion.resolution = 768
        if args.num_inference_steps is None:
            cfg.diffusion.num_inference_steps = 30
        cfg.output_dir = str(_default_output_dir(args))

    return cfg


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def discover_batch_signs(
    pose_control_dir: Path,
    dataset_dir: Path,
) -> list:
    """Discover all signs with existing pose frames and matching dataset videos."""
    signs = []
    if not pose_control_dir.exists():
        return signs

    for d in sorted(pose_control_dir.iterdir()):
        if not d.is_dir():
            continue
        if not list(d.glob("*.png")):
            continue

        sign_name = d.name.replace("_keypoints", "")

        # Find reference video
        ref_video = _find_reference_video(sign_name, dataset_dir)
        if ref_video is None:
            print(f"  [SKIP] No reference video for '{sign_name}'")
            continue

        signs.append({
            "sign_name": sign_name,
            "pose_source": str(d),
            "reference_video": str(ref_video),
        })

    return signs


def discover_video_avatar_sources(
    scan_dirs: list[Path],
    dataset_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> list[dict]:
    """Select best available video samples for avatar rendering.

    Priority:
      1. outputs/videos/mosaic/*
      2. demo/*/*_mosaic.mp4
      3. studio/neon/overlay/skeleton/slowmo/heatmap fallbacks

    If an existing OpenPose conditioning directory is available in
    outputs/pose_control/<sign>_keypoints, it is used instead of extracting
    conditioning from the selected MP4.
    """
    candidates: dict[str, dict] = {}
    seen_paths: set[Path] = set()

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for video_path in sorted(scan_dir.rglob("*.mp4")):
            video_path = video_path.resolve()
            if video_path in seen_paths:
                continue
            seen_paths.add(video_path)

            sign_name = _sign_name_from_video(video_path)
            score = _video_priority(video_path)
            existing = candidates.get(sign_name)
            if existing is not None and existing["priority"] <= score:
                continue

            pose_source = _existing_pose_source(sign_name) or video_path
            safe_name = _safe_name(sign_name)
            reference_video = _find_reference_video(sign_name, dataset_dir)
            candidates[sign_name] = {
                "sign_name": sign_name,
                "pose_source": str(pose_source),
                "selected_video": str(video_path),
                "reference_video": str(reference_video) if reference_video else None,
                "output_path": str(output_dir / f"{safe_name}_photorealistic.mp4"),
                "frames_dir": str(output_dir / f"{safe_name}_frames"),
                "priority": score,
                "conditioning_source": "existing_openpose" if Path(pose_source).is_dir() else "selected_video",
            }

    selected = sorted(
        candidates.values(),
        key=lambda item: (item["priority"], item["sign_name"]),
    )
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if getattr(args, "scan_video_sources", False):
        return Path("avatar_video_generator/outputs")
    return Path("outputs/avatar_photorealistic")


def _scan_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = list(DEFAULT_VIDEO_SCAN_DIRS)
    if args.video_source_dir:
        dirs.extend(args.video_source_dir)
    # Deduplicate while preserving priority order.
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        key = d.resolve() if d.exists() else d
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out


def _sign_name_from_video(video_path: Path) -> str:
    stem = video_path.stem
    for suffix in VIDEO_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _video_priority(video_path: Path) -> int:
    parts = set(video_path.parts)
    stem = video_path.stem
    if "mosaic" in parts:
        return 0
    if stem.endswith("_mosaic"):
        return 1
    if "studio" in parts or stem.endswith("_studio"):
        return 2
    if "neon" in parts or stem.endswith("_neon"):
        return 3
    if "overlay" in parts or stem.endswith("_overlay"):
        return 4
    if "skeleton" in parts or stem.endswith("_skeleton"):
        return 5
    if "slowmo" in parts or stem.endswith("_slowmo"):
        return 6
    if "heatmap" in parts or stem.endswith("_heatmap"):
        return 7
    return 8


def _existing_pose_source(sign_name: str) -> Path | None:
    direct = Path("outputs/pose_control") / f"{sign_name}_keypoints"
    if direct.exists() and list(direct.glob("*.png")):
        return direct

    pose_root = Path("outputs/pose_control")
    if not pose_root.exists():
        return None
    for d in sorted(pose_root.iterdir()):
        if d.is_dir() and d.name.replace("_keypoints", "") == sign_name and list(d.glob("*.png")):
            return d
    return None


def _safe_name(sign_name: str) -> str:
    return ascii_slug(sign_name, fallback="avatar")


def _find_reference_video(sign_name: str, dataset_dir: Path) -> Path | None:
    """Search dataset directory for a video matching sign_name."""
    if not dataset_dir.exists():
        return None

    # Prefer exact stem matches across the full dataset. Short sign names such
    # as "1" otherwise match many labels that merely contain "إشارة 1".
    for mp4 in sorted(dataset_dir.rglob("*.mp4")):
        if mp4.stem == sign_name:
            return mp4

    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Exact match
        candidate = subdir / f"{sign_name}.mp4"
        if candidate.exists():
            return candidate
        # Partial match
        for mp4 in subdir.glob("*.mp4"):
            if sign_name in mp4.stem:
                return mp4
    return None


def _load_reference_frames(args: argparse.Namespace) -> list[np.ndarray] | None:
    """Load optional human source frames for identity conditioning."""
    if args.reference_image is not None:
        if not args.reference_image.exists():
            raise FileNotFoundError(f"Reference image not found: {args.reference_image}")
        return [np.array(Image.open(args.reference_image).convert("RGB"))]

    if args.reference_frames_dir is not None:
        if not args.reference_frames_dir.exists():
            raise FileNotFoundError(f"Reference frames dir not found: {args.reference_frames_dir}")
        frames = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for p in sorted(args.reference_frames_dir.glob(pattern)):
                frames.append(np.array(Image.open(p).convert("RGB")))
        if not frames:
            raise FileNotFoundError(f"No reference frames found in {args.reference_frames_dir}")
        return frames

    return None


def _diffusion_runtime_error(device: str) -> str | None:
    """Return a user-readable reason if diffusion generation cannot run."""
    try:
        import torch  # noqa: F401
    except Exception as e:
        return f"torch unavailable: {e}"

    try:
        import diffusers  # noqa: F401
    except Exception as e:
        return f"diffusers unavailable: {e}"

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                return "CUDA requested but no CUDA GPU is available"
        except Exception as e:
            return f"CUDA runtime check failed: {e}"
    return None


def _video_is_readable(video_path: str | Path | None) -> bool:
    if video_path is None:
        return False
    try:
        info = get_video_info(video_path)
    except Exception:
        return False
    return info.get("frame_count", 0) > 0 and info.get("width", 0) > 0 and info.get("height", 0) > 0


def _fallback_source_for_item(item: dict) -> Path | None:
    """Prefer the real dataset signer video when generated project media is corrupt."""
    for key in ("reference_video", "selected_video", "pose_source"):
        candidate = item.get(key)
        if candidate and Path(candidate).is_file() and _video_is_readable(candidate):
            return Path(candidate)
    return None


def _export_validated_video_fallback(
    signs: list[dict],
    output_dir: Path,
    fps_default: float = 25.0,
) -> list[dict]:
    """Create compatible avatar MP4s from readable human source videos.

    This path is intentionally limited to media repair/export. It is used when
    diffusion dependencies are not installed, or when project-generated MP4s
    are corrupted and the matching dataset source is the only valid human video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for item in signs:
        source = _fallback_source_for_item(item)
        if source is None:
            print(f"  [SKIP] No readable video source for {item.get('sign_name', 'unknown')}")
            continue

        safe_name = _safe_name(item.get("sign_name") or source.stem)
        output_path = Path(item.get("output_path") or output_dir / f"{safe_name}_avatar.mp4")
        if output_path.name.endswith("_photorealistic.mp4"):
            output_path = output_path.with_name(output_path.name.replace("_photorealistic.mp4", "_avatar.mp4"))
        frames_dir = Path(item.get("frames_dir") or output_dir / f"{safe_name}_avatar_frames")
        source_info = get_video_info(source)
        fps = float(source_info.get("fps") or fps_default or 25.0)

        frames = read_video_frames(source)
        if not frames:
            print(f"  [SKIP] Source produced no frames: {source}")
            continue

        frame_path = write_frames(frames, frames_dir)
        video_path = write_video(frames, output_path, fps=fps, verbose=True)
        validation = validate_video_file(video_path, expected_min_frames=len(frames))
        result = {
            "sign_name": item.get("sign_name"),
            "source": str(source),
            "output_path": str(video_path),
            "frames_dir": str(frame_path),
            "frame_count": len(frames),
            "fps": fps,
            "codec": validation.get("codec_name"),
            "pix_fmt": validation.get("pix_fmt"),
            "fallback": True,
        }
        results.append(result)
        print(
            f"  [OK] {video_path} "
            f"({len(frames)} frames, {result['codec']}/{result['pix_fmt']})"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if (
        not args.batch
        and not args.scan_video_sources
        and args.sign is None
        and args.pose_dir is None
        and args.skeleton_video is None
        and not args.official_hd_openpose
    ):
        print("Error: specify --sign, --pose-dir, --skeleton-video, --official-hd-openpose, or --batch")
        return 1

    output_dir = _default_output_dir(args)
    cfg = build_config(args)

    # ── Batch mode ───────────────────────────────────────────────────────────
    if args.batch or args.scan_video_sources:
        if args.scan_video_sources:
            signs = discover_video_avatar_sources(
                _scan_dirs(args),
                args.dataset_dir,
                output_dir,
                max_samples=args.max_samples,
            )
        else:
            signs = discover_batch_signs(
                Path("outputs/pose_control"),
                args.dataset_dir,
            )
        if not signs:
            print("No renderable avatar sources found.")
            return 1

        print(f"Batch: {len(signs)} source(s) to generate")
        if args.scan_video_sources:
            for item in signs:
                print(
                    f"  - {item['sign_name']}: {item['conditioning_source']} "
                    f"from {item['selected_video']}"
                )

        runtime_error = _diffusion_runtime_error(args.device)
        if runtime_error:
            print(
                "Diffusion backend unavailable; exporting validated human avatar "
                f"videos from readable source media instead ({runtime_error})."
            )
            fallback_results = _export_validated_video_fallback(signs, output_dir)
            summary = {
                "total": len(signs),
                "sources": signs,
                "generated": len(fallback_results),
                "results": fallback_results,
                "fallback": True,
                "runtime_error": runtime_error,
            }
            summary_path = output_dir / "generation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\nSummary: {summary_path}")
            return 0 if fallback_results else 1

        from avatar_video_generator import AvatarPipeline

        pipeline = AvatarPipeline(cfg)
        pipeline.load_models()
        results = pipeline.run_batch(signs, output_dir=output_dir)

        summary = {
            "total": len(signs),
            "sources": signs,
            "generated": len(results),
            "results": [r.to_dict() for r in results],
        }
        summary_path = output_dir / "generation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary: {summary_path}")
        return 0

    # ── Single-sign mode ─────────────────────────────────────────────────────
    runtime_error = _diffusion_runtime_error(args.device)
    if not runtime_error:
        from avatar_video_generator import AvatarPipeline

        pipeline = AvatarPipeline(cfg)
    else:
        pipeline = None

    if args.sign:
        # Auto-discover
        if pipeline is None:
            reference_video = _find_reference_video(args.sign, args.dataset_dir)
            if reference_video is None:
                print(f"Error: diffusion backend unavailable ({runtime_error}) and no source video found.")
                return 1
            safe_name = _safe_name(args.sign)
            signs = [{
                "sign_name": args.sign,
                "reference_video": str(reference_video),
                "selected_video": str(reference_video),
                "output_path": str((args.output or output_dir / f"{safe_name}_avatar.mp4")),
                "frames_dir": str((args.frames_dir or output_dir / f"{safe_name}_avatar_frames")),
            }]
            print(
                "Diffusion backend unavailable; exporting validated human avatar "
                f"video from readable source media instead ({runtime_error})."
            )
            fallback_results = _export_validated_video_fallback(signs, output_dir)
            return 0 if fallback_results else 1
        result = pipeline.run_sign(
            sign_name=args.sign,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
        )
    else:
        # Explicit paths
        if args.official_hd_openpose:
            pose_source = Path("outputs/avatar_from_video_hd/alsbt_ishara_2_pose")
            sign_name = "alsbt_ishara_2"
            output_path = args.output or Path("outputs/avatar_from_video_hd/alsbt_ishara_2_avatar_photorealistic.mp4")
            frames_dir = args.frames_dir or Path("outputs/avatar_from_video_hd/alsbt_ishara_2_avatar_frames")
        else:
            pose_source = args.pose_dir or args.skeleton_video
            sign_name = pose_source.stem.replace("_keypoints", "").replace("_skeleton", "") if pose_source else ""
            safe_name = _safe_name(sign_name)
            output_path = args.output or (output_dir / f"{safe_name}_photorealistic.mp4")
            frames_dir = args.frames_dir

        if pose_source is None:
            print("Error: --pose-dir or --skeleton-video required.")
            return 1
        if not Path(pose_source).exists():
            print(f"Error: pose source not found: {pose_source}")
            return 1

        reference_video = args.reference_video
        if reference_video is None:
            reference_video = _find_reference_video(sign_name, args.dataset_dir)
            if reference_video is None and not (args.allow_no_reference or args.official_hd_openpose):
                print(
                    f"Error: --reference-video required (could not auto-discover "
                    f"for '{sign_name}' in {args.dataset_dir})"
                )
                return 1

        try:
            reference_frames = _load_reference_frames(args)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

        if pipeline is None:
            signs = [{
                "sign_name": sign_name,
                "pose_source": str(pose_source),
                "reference_video": str(reference_video) if reference_video else None,
                "selected_video": str(pose_source) if Path(pose_source).is_file() else None,
                "output_path": str(output_path),
                "frames_dir": str(frames_dir or output_dir / f"{_safe_name(sign_name)}_avatar_frames"),
            }]
            print(
                "Diffusion backend unavailable; exporting validated human avatar "
                f"video from readable source media instead ({runtime_error})."
            )
            fallback_results = _export_validated_video_fallback(signs, output_dir)
            return 0 if fallback_results else 1

        result = pipeline.run(
            pose_source=pose_source,
            reference_video=reference_video,
            output_path=output_path,
            sign_name=sign_name,
            frames_dir=frames_dir,
            reference_frames=reference_frames,
        )

    print(f"\n{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
