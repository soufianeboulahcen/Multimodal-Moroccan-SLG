"""CPU-compatible photorealistic avatar generation from OpenPose skeleton frames.

Generates a photorealistic avatar video from the existing pose conditioning
frames using ControlNet-OpenPose + SD1.5 on CPU.

Strategy for CPU (no GPU):
  - Generate keyframes (every N-th pose frame) via diffusion
  - Fill gaps with linear interpolation (fast, acceptable for sign language)
  - Export final MP4 at 25 fps

On GPU this script runs the full pipeline at full resolution.

Usage:
    # Full pipeline (keyframe mode for CPU)
    python scripts/generate_avatar_cpu.py --sign أَنْتِ

    # All frames (slow on CPU, use on GPU)
    python scripts/generate_avatar_cpu.py --sign أَنْتِ --all-frames

    # Custom keyframe stride
    python scripts/generate_avatar_cpu.py --sign أَنْتِ --stride 2

    # Quick test (3 frames only)
    python scripts/generate_avatar_cpu.py --sign أَنْتِ --test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SIGN_HEX_MAP = {
    "أَنْتِ": "d8a7d98ed994d986d992d8aad990",
}

PROMPT = (
    "a Moroccan sign language interpreter, upper body portrait, "
    "natural skin tone, dark hair, professional appearance, "
    "studio lighting, sharp focus, photorealistic, "
    "realistic skin texture, clear expressive hands, neutral background"
)
NEG_PROMPT = (
    "skeleton, stick figure, cartoon, anime, illustration, "
    "blurry, low quality, deformed hands, extra fingers, missing fingers, "
    "watermark, text, logo, nsfw, ugly, distorted face, "
    "ghosting, flickering, artifacts, CGI, 3D render, plastic skin"
)


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(resolution: int, device: str = "cpu"):
    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading ControlNet-OpenPose SD1.5...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=dtype,
    )

    print("Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if device == "cuda":
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    else:
        # CPU: enable attention slicing to reduce peak memory
        pipe.enable_attention_slicing()

    print(f"Pipeline ready on {device}.")
    return pipe


# ---------------------------------------------------------------------------
# Single frame generation
# ---------------------------------------------------------------------------

def generate_frame(
    pipe,
    pose_img: Image.Image,
    resolution: int,
    steps: int,
    guidance: float,
    controlnet_scale: float,
    seed: int,
) -> np.ndarray:
    import torch

    pose_resized = pose_img.resize((resolution, resolution))
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=pose_resized,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=controlnet_scale,
        generator=torch.Generator().manual_seed(seed),
    )
    return np.array(result.images[0])


# ---------------------------------------------------------------------------
# Temporal interpolation
# ---------------------------------------------------------------------------

def interpolate_frames(
    keyframes: dict[int, np.ndarray],
    total_frames: int,
) -> list[np.ndarray]:
    """Fill gaps between keyframes with linear interpolation.

    Args:
        keyframes: dict mapping frame index → RGB array
        total_frames: total number of frames in the output

    Returns:
        List of total_frames RGB arrays
    """
    indices = sorted(keyframes.keys())
    result = [None] * total_frames

    # Place keyframes
    for idx, frame in keyframes.items():
        if idx < total_frames:
            result[idx] = frame

    # Interpolate between consecutive keyframes
    for i in range(len(indices) - 1):
        a_idx = indices[i]
        b_idx = indices[i + 1]
        a = keyframes[a_idx].astype(np.float32)
        b = keyframes[b_idx].astype(np.float32)
        gap = b_idx - a_idx

        for j in range(1, gap):
            t = j / gap
            interp = (1 - t) * a + t * b
            frame_idx = a_idx + j
            if frame_idx < total_frames:
                result[frame_idx] = np.clip(interp, 0, 255).astype(np.uint8)

    # Fill any remaining None slots (before first or after last keyframe)
    first_valid = next((f for f in result if f is not None), None)
    last_valid = None
    for f in result:
        if f is not None:
            last_valid = f

    for i in range(total_frames):
        if result[i] is None:
            result[i] = first_valid if i < indices[0] else last_valid

    return result


# ---------------------------------------------------------------------------
# Temporal smoothing (Gaussian along time axis)
# ---------------------------------------------------------------------------

def temporal_smooth(frames: list[np.ndarray], sigma: float = 0.8) -> list[np.ndarray]:
    if len(frames) < 3 or sigma <= 0:
        return frames
    from scipy.ndimage import gaussian_filter1d
    arr = np.stack(frames, axis=0).astype(np.float32)
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    return [np.clip(smoothed[i], 0, 255).astype(np.uint8) for i in range(len(frames))]


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def write_mp4(frames: list[np.ndarray], out_path: Path, fps: float = 25.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")


def write_comparison_mp4(
    pose_frames: list[np.ndarray],
    avatar_frames: list[np.ndarray],
    out_path: Path,
    fps: float = 25.0,
) -> None:
    """Side-by-side: pose skeleton | photorealistic avatar."""
    n = min(len(pose_frames), len(avatar_frames))
    h = pose_frames[0].shape[0]
    w = pose_frames[0].shape[1]
    combined = []
    for i in range(n):
        left = cv2.resize(pose_frames[i], (w, h))
        right = cv2.resize(avatar_frames[i], (w, h))
        # Add labels
        for arr, label in [(left, "OpenPose"), (right, "Avatar")]:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, label, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            arr[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        combined.append(np.concatenate([left, right], axis=1))
    write_mp4(combined, out_path, fps=fps)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pose_dir: Path,
    out_path: Path,
    resolution: int = 256,
    steps: int = 8,
    guidance: float = 7.5,
    controlnet_scale: float = 1.0,
    stride: int = 4,
    all_frames: bool = False,
    test_mode: bool = False,
    seed: int = 42,
    fps: float = 25.0,
    smooth_sigma: float = 0.8,
    device: str = "cpu",
) -> dict:
    t_start = time.time()

    # Load pose frames
    pose_paths = sorted(pose_dir.glob("pose_*.png"))
    if not pose_paths:
        raise FileNotFoundError(f"No pose frames in {pose_dir}")

    pose_images = [Image.open(str(p)).convert("RGB") for p in pose_paths]
    pose_numpy = [np.array(img.resize((resolution, resolution))) for img in pose_images]
    total = len(pose_images)
    print(f"Pose frames: {total} @ {resolution}px")

    # Determine which frames to generate via diffusion
    if test_mode:
        keyframe_indices = [0, total // 2, total - 1]
    elif all_frames:
        keyframe_indices = list(range(total))
    else:
        # Keyframe mode: every `stride` frames + first + last
        keyframe_indices = list(range(0, total, stride))
        if (total - 1) not in keyframe_indices:
            keyframe_indices.append(total - 1)

    print(f"Generating {len(keyframe_indices)}/{total} keyframes "
          f"(stride={stride}, {'all' if all_frames else 'interpolating gaps'})")

    # Load pipeline
    pipe = load_pipeline(resolution=resolution, device=device)

    # Generate keyframes
    keyframes: dict[int, np.ndarray] = {}
    for n, idx in enumerate(keyframe_indices):
        t0 = time.time()
        frame = generate_frame(
            pipe, pose_images[idx], resolution, steps, guidance, controlnet_scale, seed + idx
        )
        elapsed = time.time() - t0
        print(f"  [{n+1}/{len(keyframe_indices)}] frame {idx:02d} → {elapsed:.1f}s")
        keyframes[idx] = frame

        # Save intermediate keyframe
        kf_dir = out_path.parent / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(str(kf_dir / f"kf_{idx:04d}.png"))

    # Interpolate gaps
    print("Interpolating gaps...")
    all_avatar_frames = interpolate_frames(keyframes, total)

    # Temporal smoothing
    print("Temporal smoothing...")
    all_avatar_frames = temporal_smooth(all_avatar_frames, sigma=smooth_sigma)

    # Export main video
    print(f"Exporting {out_path}...")
    write_mp4(all_avatar_frames, out_path, fps=fps)
    size_mb = out_path.stat().st_size / 1e6

    # Export comparison video
    comp_path = out_path.parent / (out_path.stem + "_comparison.mp4")
    print(f"Exporting comparison {comp_path}...")
    write_comparison_mp4(pose_numpy, all_avatar_frames, comp_path, fps=fps)

    elapsed_total = time.time() - t_start
    result = {
        "output": str(out_path),
        "comparison": str(comp_path),
        "total_frames": total,
        "keyframes_generated": len(keyframe_indices),
        "resolution": resolution,
        "steps": steps,
        "elapsed_s": round(elapsed_total, 1),
        "size_mb": round(size_mb, 2),
    }
    print(f"\nDone in {elapsed_total:.0f}s → {out_path} ({size_mb:.1f} MB)")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_pose_dir(sign_name: str) -> Path | None:
    candidates = [
        Path(f"outputs/pose_control/{sign_name}_keypoints"),
    ]
    for c in candidates:
        if c.exists() and list(c.glob("pose_*.png")):
            return c
    return None


def main():
    p = argparse.ArgumentParser(
        description="Generate photorealistic avatar video from OpenPose skeleton frames"
    )
    p.add_argument("--sign", default="أَنْتِ", help="Sign name (Arabic)")
    p.add_argument("--pose-dir", type=Path, help="Explicit pose PNG directory")
    p.add_argument("--output", type=Path, help="Output MP4 path")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/avatar_photorealistic"))
    p.add_argument("--resolution", type=int, default=256, choices=[256, 384, 512])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--controlnet-scale", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=4,
                   help="Generate every N-th frame via diffusion, interpolate rest")
    p.add_argument("--all-frames", action="store_true",
                   help="Generate all frames (slow on CPU)")
    p.add_argument("--test", action="store_true",
                   help="Quick test: generate 3 frames only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    # Resolve sign name from hex if needed
    sign_name = args.sign
    if sign_name in SIGN_HEX_MAP:
        sign_name = bytes.fromhex(SIGN_HEX_MAP[sign_name]).decode("utf-8")

    # Find pose directory
    pose_dir = args.pose_dir
    if pose_dir is None:
        pose_dir = find_pose_dir(sign_name)
        if pose_dir is None:
            # Try hex-decoded name
            for hex_val in SIGN_HEX_MAP.values():
                decoded = bytes.fromhex(hex_val).decode("utf-8")
                candidate = Path(f"outputs/pose_control/{decoded}_keypoints")
                if candidate.exists() and list(candidate.glob("pose_*.png")):
                    pose_dir = candidate
                    sign_name = decoded
                    break
        if pose_dir is None:
            print(f"ERROR: No pose frames found for '{sign_name}'")
            print("Run: python scripts/extract_pose_frames.py --sign أَنْتِ")
            sys.exit(1)

    print(f"Pose dir: {pose_dir} ({len(list(pose_dir.glob('pose_*.png')))} frames)")

    # Output path
    safe_name = sign_name.replace("/", "_").replace("\\", "_")[:60]
    out_path = args.output or (args.output_dir / f"{safe_name}_photorealistic.mp4")

    result = run_pipeline(
        pose_dir=pose_dir,
        out_path=out_path,
        resolution=args.resolution,
        steps=args.steps,
        guidance=args.guidance,
        controlnet_scale=args.controlnet_scale,
        stride=args.stride,
        all_frames=args.all_frames,
        test_mode=args.test,
        seed=args.seed,
        fps=args.fps,
        device=args.device,
    )

    import json
    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
