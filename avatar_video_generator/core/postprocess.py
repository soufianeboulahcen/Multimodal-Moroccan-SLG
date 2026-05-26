"""Post-processing: temporal smoothing, RIFE interpolation, MP4 export."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ── Temporal smoothing ────────────────────────────────────────────────────────

def smooth_temporal(frames: List[np.ndarray], sigma: float = 0.8) -> List[np.ndarray]:
    """1D Gaussian filter along time axis — reduces flicker without spatial blur."""
    if len(frames) < 3 or sigma <= 0:
        return frames
    arr = np.stack(frames, axis=0).astype(np.float32)   # (T,H,W,3)
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    return [np.clip(smoothed[i], 0, 255).astype(np.uint8) for i in range(len(frames))]


# ── RIFE interpolation ────────────────────────────────────────────────────────

def interpolate_frames(
    frames: List[np.ndarray],
    multiplier: int = 2,
) -> List[np.ndarray]:
    """Increase frame rate by `multiplier` via linear blend (RIFE fallback).

    On GPU environments with rife-ncnn-vulkan on PATH, uses that instead.
    Linear blend is always available and works well for sign language motion.
    """
    if multiplier <= 1 or len(frames) < 2:
        return frames

    # Try rife-ncnn-vulkan CLI
    try:
        return _rife_ncnn(frames, multiplier)
    except Exception:
        pass

    # Linear blend fallback
    return _linear_blend(frames, multiplier)


def _rife_ncnn(frames: List[np.ndarray], multiplier: int) -> List[np.ndarray]:
    import shutil
    rife = shutil.which("rife-ncnn-vulkan")
    if not rife:
        raise FileNotFoundError("rife-ncnn-vulkan not on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir(); out_dir.mkdir()

        for i, f in enumerate(frames):
            cv2.imwrite(str(in_dir / f"{i:06d}.png"),
                        cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        subprocess.run(
            [rife, "-i", str(in_dir), "-o", str(out_dir), "-n", str(multiplier)],
            check=True, capture_output=True,
        )
        result = []
        for p in sorted(out_dir.glob("*.png")):
            bgr = cv2.imread(str(p))
            if bgr is not None:
                result.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return result


def _linear_blend(frames: List[np.ndarray], multiplier: int) -> List[np.ndarray]:
    result = []
    n_mid = multiplier - 1
    for i in range(len(frames) - 1):
        f0 = frames[i].astype(np.float32)
        f1 = frames[i + 1].astype(np.float32)
        result.append(frames[i])
        for k in range(1, n_mid + 1):
            alpha = k / multiplier
            mid = (1 - alpha) * f0 + alpha * f1
            result.append(np.clip(mid, 0, 255).astype(np.uint8))
    result.append(frames[-1])
    return result


# ── MP4 export ────────────────────────────────────────────────────────────────

def write_mp4(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 25.0,
    crf: int = 18,
) -> Path:
    """Encode frames to H.264 MP4 via ffmpeg pipe."""
    if not frames:
        raise ValueError("No frames to write.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]

    # Try ffmpeg pipe (best quality)
    try:
        _write_ffmpeg(frames, output_path, fps, crf, w, h)
    except Exception:
        _write_imageio(frames, output_path, fps)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {output_path}  ({len(frames)} frames @ {fps:.0f}fps, {size_mb:.1f}MB)")
    return output_path


def _write_ffmpeg(frames, out, fps, crf, w, h):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.returncode}")


def _write_imageio(frames, out, fps):
    import imageio  # type: ignore
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ── Side-by-side comparison ───────────────────────────────────────────────────

def write_comparison(
    pose_frames: List[np.ndarray],
    avatar_frames: List[np.ndarray],
    output_path: Path,
    fps: float = 25.0,
) -> Path:
    """Write pose | avatar side-by-side comparison MP4."""
    n = min(len(pose_frames), len(avatar_frames))
    h, w = avatar_frames[0].shape[:2]
    combined = []
    for i in range(n):
        left = cv2.resize(pose_frames[i], (w, h), interpolation=cv2.INTER_LANCZOS4)
        right = avatar_frames[i]
        # Label bars
        left = _label(left, "OpenPose")
        right = _label(right, "Photorealistic Avatar")
        combined.append(np.concatenate([left, right], axis=1))
    return write_mp4(combined, output_path, fps=fps, crf=20)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    out[h-26:h] = (out[h-26:h].astype(np.float32) * 0.4).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (6, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255,255,255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
