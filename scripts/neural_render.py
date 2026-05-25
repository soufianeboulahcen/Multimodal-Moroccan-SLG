"""Neural video rendering pipeline for photorealistic signer generation.

Converts SMPL-X skeleton renders into photorealistic signer videos by
conditioning a video diffusion model on a reference signer image.

Supported backends (in priority order):
  1. Champ    — SMPL-X-conditioned video diffusion (best quality)
  2. AnimateAnyone — DWPose-conditioned video diffusion
  3. MagicAnimate  — DensePose-conditioned video diffusion

Pipeline:
  SMPL-X skeleton render (T, H, W, 3)
    + reference signer image (H, W, 3)
    → neural renderer
    → photorealistic signer video (T, H, W, 3)

The neural renderer preserves:
  - Signer appearance (from reference image)
  - Signer motion (from SMPL-X skeleton sequence)
  - Temporal consistency (video diffusion temporal attention)
  - Signer identity (reference image conditioning)

Usage:
    # Champ backend (recommended)
    python scripts/neural_render.py \
        --skeleton-video outputs/phase2_generation/الأذان_generated.mp4 \
        --reference-image data/reference_signers/signer_0.jpg \
        --backend champ \
        --champ-dir /path/to/Champ \
        --output outputs/neural/الأذان_photorealistic.mp4

    # AnimateAnyone backend
    python scripts/neural_render.py \
        --skeleton-video outputs/phase2_generation/الأذان_generated.mp4 \
        --reference-image data/reference_signers/signer_0.jpg \
        --backend animate_anyone \
        --animate-anyone-dir /path/to/AnimateAnyone

    # Batch: process all generated videos
    python scripts/neural_render.py \
        --skeleton-dir outputs/phase2_generation \
        --reference-image data/reference_signers/signer_0.jpg \
        --backend champ \
        --champ-dir /path/to/Champ

Setup:
    # Champ
    git clone https://github.com/fudan-generative-vision/champ third_party/Champ
    cd third_party/Champ && pip install -r requirements.txt
    # Download Champ checkpoints to third_party/Champ/pretrained_models/

    # AnimateAnyone
    git clone https://github.com/HumanAIGC/AnimateAnyone third_party/AnimateAnyone
    cd third_party/AnimateAnyone && pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backend(
    champ_dir: Optional[str] = None,
    animate_anyone_dir: Optional[str] = None,
    magic_animate_dir: Optional[str] = None,
) -> Optional[str]:
    """Auto-detect which neural rendering backend is available."""
    if champ_dir and Path(champ_dir).exists():
        inference_script = Path(champ_dir) / "inference.py"
        if inference_script.exists():
            return "champ"

    if animate_anyone_dir and Path(animate_anyone_dir).exists():
        inference_script = Path(animate_anyone_dir) / "run_net.py"
        if inference_script.exists():
            return "animate_anyone"

    if magic_animate_dir and Path(magic_animate_dir).exists():
        return "magic_animate"

    # Check third_party directory
    repo_root = Path(__file__).resolve().parents[1]
    for name, backend in [
        ("Champ", "champ"),
        ("AnimateAnyone", "animate_anyone"),
        ("MagicAnimate", "magic_animate"),
    ]:
        candidate = repo_root / "third_party" / name
        if candidate.exists():
            return backend

    return None


# ---------------------------------------------------------------------------
# Champ backend
# ---------------------------------------------------------------------------

def render_champ(
    skeleton_video: Path,
    reference_image: Path,
    output_path: Path,
    champ_dir: Path,
    smplx_params_path: Optional[Path] = None,
    width: int = 512,
    height: int = 768,
    fps: float = 25.0,
    guidance_scale: float = 3.5,
    n_inference_steps: int = 20,
) -> None:
    """Run Champ neural rendering.

    Champ uses SMPL-X body parameters + DWPose skeleton as conditioning signals.
    Reference image provides appearance (clothing, skin tone, face).

    Args:
        skeleton_video:    Path to skeleton overlay video (used as pose guide)
        reference_image:   Path to reference signer image
        output_path:       Output video path
        champ_dir:         Path to Champ repository
        smplx_params_path: Optional SMPL-X params NPZ (for richer conditioning)
        width, height:     Output resolution
        fps:               Output frame rate
        guidance_scale:    Classifier-free guidance scale (higher = more faithful to pose)
        n_inference_steps: Denoising steps (more = better quality, slower)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build Champ inference command
    # Champ expects: reference image + driving video (skeleton/SMPL-X render)
    inference_script = champ_dir / "inference.py"

    cmd = [
        sys.executable, str(inference_script),
        "--reference-image", str(reference_image),
        "--driving-video", str(skeleton_video),
        "--output-path", str(output_path),
        "--width", str(width),
        "--height", str(height),
        "--guidance-scale", str(guidance_scale),
        "--num-inference-steps", str(n_inference_steps),
    ]

    # If SMPL-X params are available, pass them for richer conditioning
    if smplx_params_path and smplx_params_path.exists():
        cmd += ["--smplx-params", str(smplx_params_path)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(champ_dir) + ":" + env.get("PYTHONPATH", "")

    print(f"  Running Champ: {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            cwd=str(champ_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"Champ inference failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    print(f"  Champ output: {output_path}")


# ---------------------------------------------------------------------------
# AnimateAnyone backend
# ---------------------------------------------------------------------------

def render_animate_anyone(
    skeleton_video: Path,
    reference_image: Path,
    output_path: Path,
    animate_anyone_dir: Path,
    width: int = 512,
    height: int = 768,
    fps: float = 25.0,
    guidance_scale: float = 3.5,
    n_inference_steps: int = 20,
) -> None:
    """Run AnimateAnyone neural rendering.

    AnimateAnyone uses DWPose skeleton as conditioning + reference image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # AnimateAnyone uses a config-based inference script
    inference_script = animate_anyone_dir / "run_net.py"
    if not inference_script.exists():
        # Try alternative entry points
        for alt in ["inference.py", "animate.py", "demo.py"]:
            candidate = animate_anyone_dir / alt
            if candidate.exists():
                inference_script = candidate
                break

    if not inference_script.exists():
        raise FileNotFoundError(
            f"AnimateAnyone inference script not found in {animate_anyone_dir}"
        )

    # Write a temporary config for this inference run
    config = {
        "reference_image": str(reference_image),
        "driving_video": str(skeleton_video),
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": n_inference_steps,
        "fps": fps,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(config, f)
        config_path = f.name

    cmd = [sys.executable, str(inference_script), "--config", config_path]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(animate_anyone_dir) + ":" + env.get("PYTHONPATH", "")

    print(f"  Running AnimateAnyone ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            cwd=str(animate_anyone_dir))
    Path(config_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"AnimateAnyone inference failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    print(f"  AnimateAnyone output: {output_path}")


# ---------------------------------------------------------------------------
# MagicAnimate backend
# ---------------------------------------------------------------------------

def render_magic_animate(
    skeleton_video: Path,
    reference_image: Path,
    output_path: Path,
    magic_animate_dir: Path,
    width: int = 512,
    height: int = 768,
    fps: float = 25.0,
) -> None:
    """Run MagicAnimate neural rendering (DensePose conditioning)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inference_script = magic_animate_dir / "demo" / "animate.py"
    if not inference_script.exists():
        inference_script = magic_animate_dir / "inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(
            f"MagicAnimate inference script not found in {magic_animate_dir}"
        )

    cmd = [
        sys.executable, str(inference_script),
        "--reference", str(reference_image),
        "--motion", str(skeleton_video),
        "--output", str(output_path),
        "--W", str(width), "--H", str(height),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(magic_animate_dir) + ":" + env.get("PYTHONPATH", "")

    print(f"  Running MagicAnimate ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            cwd=str(magic_animate_dir))

    if result.returncode != 0:
        raise RuntimeError(
            f"MagicAnimate inference failed:\n{result.stderr[-2000:]}"
        )
    print(f"  MagicAnimate output: {output_path}")


# ---------------------------------------------------------------------------
# Unified render function
# ---------------------------------------------------------------------------

def neural_render(
    skeleton_video: Path,
    reference_image: Path,
    output_path: Path,
    backend: str,
    backend_dir: Path,
    smplx_params_path: Optional[Path] = None,
    width: int = 512,
    height: int = 768,
    fps: float = 25.0,
    guidance_scale: float = 3.5,
    n_inference_steps: int = 20,
) -> None:
    """Dispatch to the appropriate neural rendering backend."""
    if not skeleton_video.exists():
        raise FileNotFoundError(f"skeleton video not found: {skeleton_video}")
    if not reference_image.exists():
        raise FileNotFoundError(f"reference image not found: {reference_image}")

    print(f"Neural rendering [{backend}]")
    print(f"  skeleton:  {skeleton_video.name}")
    print(f"  reference: {reference_image.name}")
    print(f"  output:    {output_path.name}")

    if backend == "champ":
        render_champ(
            skeleton_video, reference_image, output_path, backend_dir,
            smplx_params_path=smplx_params_path,
            width=width, height=height, fps=fps,
            guidance_scale=guidance_scale,
            n_inference_steps=n_inference_steps,
        )
    elif backend == "animate_anyone":
        render_animate_anyone(
            skeleton_video, reference_image, output_path, backend_dir,
            width=width, height=height, fps=fps,
            guidance_scale=guidance_scale,
            n_inference_steps=n_inference_steps,
        )
    elif backend == "magic_animate":
        render_magic_animate(
            skeleton_video, reference_image, output_path, backend_dir,
            width=width, height=height, fps=fps,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}. "
                         f"Choose: champ, animate_anyone, magic_animate")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_neural_render(
    skeleton_dir: Path,
    reference_image: Path,
    output_dir: Path,
    backend: str,
    backend_dir: Path,
    smplx_params_dir: Optional[Path] = None,
    pattern: str = "*_generated.mp4",
    **render_kwargs,
) -> dict:
    """Process all skeleton videos in a directory."""
    skeleton_videos = sorted(skeleton_dir.glob(pattern))
    if not skeleton_videos:
        print(f"No videos matching {pattern} in {skeleton_dir}")
        return {"processed": 0, "failed": 0, "total": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "failed": 0, "total": len(skeleton_videos)}

    for video_path in skeleton_videos:
        stem = video_path.stem.replace("_generated", "")
        output_path = output_dir / f"{stem}_photorealistic.mp4"

        if output_path.exists():
            print(f"  [SKIP] {output_path.name} already exists")
            stats["processed"] += 1
            continue

        # Find matching SMPL-X params if available
        smplx_path = None
        if smplx_params_dir:
            candidate = smplx_params_dir / f"{stem}_smplx.npz"
            if candidate.exists():
                smplx_path = candidate

        try:
            neural_render(
                skeleton_video=video_path,
                reference_image=reference_image,
                output_path=output_path,
                backend=backend,
                backend_dir=backend_dir,
                smplx_params_path=smplx_path,
                **render_kwargs,
            )
            stats["processed"] += 1
        except Exception as e:
            print(f"  [FAIL] {video_path.name}: {e}")
            stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

SETUP_INSTRUCTIONS = {
    "champ": """
Champ Setup:
  git clone https://github.com/fudan-generative-vision/champ third_party/Champ
  cd third_party/Champ
  pip install -r requirements.txt
  # Download pretrained models:
  # https://huggingface.co/fudan-generative-vision/champ
  # Place in third_party/Champ/pretrained_models/
""",
    "animate_anyone": """
AnimateAnyone Setup:
  git clone https://github.com/HumanAIGC/AnimateAnyone third_party/AnimateAnyone
  cd third_party/AnimateAnyone
  pip install -r requirements.txt
  # Download pretrained models from HuggingFace
  # https://huggingface.co/patrolli/AnimateAnyone
""",
    "magic_animate": """
MagicAnimate Setup:
  git clone https://github.com/magic-research/magic-animate third_party/MagicAnimate
  cd third_party/MagicAnimate
  pip install -r requirements.txt
  # Download pretrained models:
  # https://huggingface.co/zcxu-eric/MagicAnimate
""",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Neural video rendering: skeleton → photorealistic signer video"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--skeleton-video", type=Path,
                             help="Single skeleton video to process")
    input_group.add_argument("--skeleton-dir", type=Path,
                             help="Directory of skeleton videos to batch process")

    parser.add_argument("--reference-image", type=Path, required=True,
                        help="Reference signer image (appearance conditioning)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (single video mode)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/neural_render"),
                        help="Output directory (batch mode)")

    # Backend
    parser.add_argument("--backend", default=None,
                        choices=["champ", "animate_anyone", "magic_animate"],
                        help="Neural rendering backend (auto-detected if not specified)")
    parser.add_argument("--champ-dir", type=Path, default=None,
                        help="Path to Champ repository")
    parser.add_argument("--animate-anyone-dir", type=Path, default=None,
                        help="Path to AnimateAnyone repository")
    parser.add_argument("--magic-animate-dir", type=Path, default=None,
                        help="Path to MagicAnimate repository")

    # SMPL-X params (optional, for Champ)
    parser.add_argument("--smplx-params", type=Path, default=None,
                        help="SMPL-X params NPZ for richer Champ conditioning")
    parser.add_argument("--smplx-params-dir", type=Path, default=None,
                        help="Directory of SMPL-X params NPZ files (batch mode)")

    # Render settings
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--n-inference-steps", type=int, default=20)

    # Info
    parser.add_argument("--setup", action="store_true",
                        help="Print setup instructions for all backends")

    args = parser.parse_args()

    if args.setup:
        for backend, instructions in SETUP_INSTRUCTIONS.items():
            print(instructions)
        return 0

    # Resolve backend
    backend = args.backend
    backend_dir = None

    if backend is None:
        backend = detect_backend(
            champ_dir=str(args.champ_dir) if args.champ_dir else None,
            animate_anyone_dir=str(args.animate_anyone_dir) if args.animate_anyone_dir else None,
            magic_animate_dir=str(args.magic_animate_dir) if args.magic_animate_dir else None,
        )
        if backend is None:
            print("No neural rendering backend found.")
            print("Run with --setup to see installation instructions.")
            return 1
        print(f"Auto-detected backend: {backend}")

    # Resolve backend directory
    repo_root = Path(__file__).resolve().parents[1]
    if backend == "champ":
        backend_dir = args.champ_dir or (repo_root / "third_party" / "Champ")
    elif backend == "animate_anyone":
        backend_dir = args.animate_anyone_dir or (repo_root / "third_party" / "AnimateAnyone")
    elif backend == "magic_animate":
        backend_dir = args.magic_animate_dir or (repo_root / "third_party" / "MagicAnimate")

    if not backend_dir or not backend_dir.exists():
        print(f"Backend directory not found: {backend_dir}")
        print(f"Run with --setup to see installation instructions.")
        return 1

    render_kwargs = dict(
        width=args.width,
        height=args.height,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        n_inference_steps=args.n_inference_steps,
    )

    if args.skeleton_video:
        # Single video mode
        output_path = args.output or (
            args.output_dir / (args.skeleton_video.stem + "_photorealistic.mp4")
        )
        neural_render(
            skeleton_video=args.skeleton_video,
            reference_image=args.reference_image,
            output_path=output_path,
            backend=backend,
            backend_dir=backend_dir,
            smplx_params_path=args.smplx_params,
            **render_kwargs,
        )
    else:
        # Batch mode
        stats = batch_neural_render(
            skeleton_dir=args.skeleton_dir,
            reference_image=args.reference_image,
            output_dir=args.output_dir,
            backend=backend,
            backend_dir=backend_dir,
            smplx_params_dir=args.smplx_params_dir,
            **render_kwargs,
        )
        print(f"\nBatch complete: processed={stats['processed']}  "
              f"failed={stats['failed']}  total={stats['total']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
