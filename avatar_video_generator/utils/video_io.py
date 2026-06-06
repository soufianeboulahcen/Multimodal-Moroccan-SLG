"""Video I/O utilities shared across the avatar pipeline.

All functions operate on numpy arrays (H, W, 3) uint8 in RGB colour space.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_video_frames(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    rgb: bool = True,
) -> List[np.ndarray]:
    """Read all frames from an MP4 into a list of (H, W, 3) uint8 arrays.

    Args:
        video_path: path to the MP4 file.
        max_frames: cap the number of frames read (None = all).
        resize: (width, height) to resize each frame, or None.
        rgb: if True, convert BGR→RGB (default). Set False to keep BGR.

    Returns:
        List of numpy arrays (H, W, 3) uint8.

    Raises:
        IOError: if the file cannot be opened.
    """
    path = Path(video_path)
    if not path.exists():
        raise IOError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LANCZOS4)
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def get_video_info(video_path: str | Path) -> dict:
    """Return basic metadata for a video file."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Reference frame extraction
# ---------------------------------------------------------------------------

def extract_reference_frames(
    video_path: str | Path,
    n_frames: int = 5,
    resize: Optional[Tuple[int, int]] = None,
    strategy: str = "uniform",
) -> List[np.ndarray]:
    """Extract representative frames from a video for identity conditioning.

    Args:
        video_path: source video (MoSL dataset clip of the target signer).
        n_frames: number of frames to extract.
        resize: optional (width, height) resize.
        strategy: "uniform" (evenly spaced) | "first" (first N frames).

    Returns:
        List of (H, W, 3) uint8 RGB arrays.
    """
    all_frames = read_video_frames(video_path, resize=resize)
    if not all_frames:
        return []

    total = len(all_frames)
    if strategy == "first" or total <= n_frames:
        return all_frames[:n_frames]

    # Uniform sampling — skip first and last 10% to avoid fade-in/out
    margin = max(1, total // 10)
    usable = all_frames[margin: total - margin]
    if len(usable) < n_frames:
        usable = all_frames

    indices = [int(i * (len(usable) - 1) / (n_frames - 1)) for i in range(n_frames)]
    return [usable[i] for i in indices]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_video(
    frames: List[np.ndarray],
    output_path: str | Path,
    fps: float = 25.0,
    crf: int = 18,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
    verbose: bool = True,
) -> Path:
    """Encode a list of RGB frames to an H.264 MP4 via ffmpeg.

    Falls back to imageio if ffmpeg is not on PATH.

    Args:
        frames: list of (H, W, 3) uint8 RGB arrays (all same size).
        output_path: destination .mp4 path.
        fps: output frame rate.
        crf: H.264 quality (0=lossless, 18=high, 23=default, 51=worst).
        codec: video codec (libx264 recommended).
        pixel_format: yuv420p for maximum compatibility.
        verbose: print progress.

    Returns:
        Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to write.")

    h, w = frames[0].shape[:2]

    # Try ffmpeg pipe (best quality + compatibility)
    try:
        _write_via_ffmpeg(frames, out, fps=fps, crf=crf,
                          codec=codec, pixel_format=pixel_format, w=w, h=h)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback: imageio
        _write_via_imageio(frames, out, fps=fps)

    if verbose:
        size_mb = out.stat().st_size / 1e6
        print(f"  Saved: {out}  ({len(frames)} frames @ {fps:.0f} fps, {size_mb:.1f} MB)")

    return out


def write_frames(
    frames: List[np.ndarray],
    output_dir: str | Path,
    pattern: str = "avatar_{:06d}.png",
) -> Path:
    """Write RGB avatar frames as PNG files.

    Args:
        frames: list of (H, W, 3) uint8 RGB arrays.
        output_dir: destination directory.
        pattern: filename pattern containing one integer replacement field.

    Returns:
        Path to the directory containing the generated frames.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write.")

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out / pattern.format(i))
    return out


def _write_via_ffmpeg(
    frames: List[np.ndarray],
    out: Path,
    fps: float,
    crf: int,
    codec: str,
    pixel_format: str,
    w: int,
    h: int,
) -> None:
    """Write frames via ffmpeg stdin pipe."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", codec,
        "-crf", str(crf),
        "-pix_fmt", pixel_format,
        "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _write_via_imageio(frames: List[np.ndarray], out: Path, fps: float) -> None:
    """Write frames via imageio (fallback when ffmpeg is unavailable)."""
    import imageio  # type: ignore
    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def make_comparison_video(
    pose_frames: List[np.ndarray],
    avatar_frames: List[np.ndarray],
    output_path: str | Path,
    fps: float = 25.0,
    label_left: str = "OpenPose",
    label_right: str = "Photorealistic Avatar",
) -> Path:
    """Create a side-by-side comparison MP4: pose | avatar.

    Both sequences are padded/trimmed to the same length.
    """
    n = min(len(pose_frames), len(avatar_frames))
    if n == 0:
        raise ValueError("Empty frame sequences.")

    h = pose_frames[0].shape[0]
    w = pose_frames[0].shape[1]

    combined: List[np.ndarray] = []
    for i in range(n):
        left = cv2.resize(pose_frames[i], (w, h), interpolation=cv2.INTER_LANCZOS4)
        right = cv2.resize(avatar_frames[i], (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Add text labels
        left = _add_label(left, label_left)
        right = _add_label(right, label_right)

        row = np.concatenate([left, right], axis=1)
        combined.append(row)

    return write_video(combined, output_path, fps=fps, verbose=False)


def _add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Overlay a text label at the bottom of a frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Draw semi-transparent bar
    bar = out[h - 28:h, :].astype(np.float32)
    bar = bar * 0.4
    out[h - 28:h, :] = bar.astype(np.uint8)
    # Draw text (OpenCV uses BGR internally but our array is RGB)
    rgb_copy = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(
        rgb_copy, text,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2RGB)
