"""Video I/O utilities shared across the avatar pipeline.

All functions operate on numpy arrays (H, W, 3) uint8 in RGB colour space.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


ASCII_STEM_MAX_LEN = 80


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def ascii_slug(value: str, fallback: str = "avatar", max_len: int = ASCII_STEM_MAX_LEN) -> str:
    """Return a filesystem-safe ASCII slug.

    Non-ASCII labels, including Arabic sign names, collapse to a deterministic
    ``fallback_<hash>`` name. This keeps generated media portable across video
    players, archives, and file shares that mishandle Unicode paths.
    """
    normalised = unicodedata.normalize("NFKD", value)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    if slug:
        return slug[:max_len].strip("_")
    return f"{fallback}_{digest}"


def ensure_ascii_media_path(path: str | Path, default_stem: str = "avatar") -> Path:
    """Return an equivalent path whose filename stem is simple ASCII."""
    p = Path(path)
    suffix = p.suffix.lower()
    stem = ascii_slug(p.stem or default_stem, fallback=default_stem)
    return p.with_name(f"{stem}{suffix}")


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

    ffmpeg is required so every MP4 is finalised as H.264/yuv420p.

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
    out = ensure_ascii_media_path(output_path, default_stem="avatar_video")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to write.")

    repaired = repair_frame_sequence(frames)
    h, w = repaired[0].shape[:2]

    # Always force broadly compatible MP4 output. If validation fails, re-export
    # once from the repaired in-memory frames before surfacing the failure.
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            _write_via_ffmpeg(
                repaired,
                out,
                fps=fps,
                crf=crf,
                codec="libx264",
                pixel_format="yuv420p",
                w=w,
                h=h,
            )
            validate_video_file(out, expected_min_frames=len(repaired))
            break
        except (FileNotFoundError, subprocess.CalledProcessError, IOError, ValueError) as e:
            last_error = e
            out.unlink(missing_ok=True)
            if attempt == 1:
                raise RuntimeError(f"Could not export a valid H.264/yuv420p MP4: {out}") from last_error

    if verbose:
        size_mb = out.stat().st_size / 1e6
        print(f"  Saved: {out}  ({len(repaired)} frames @ {fps:.0f} fps, {size_mb:.1f} MB)")

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
    out = ensure_ascii_media_path(output_dir, default_stem="avatar_frames").with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write.")

    repaired = repair_frame_sequence(frames)
    for i, frame in enumerate(repaired):
        frame_path = out / pattern.format(i)
        frame_path = ensure_ascii_media_path(frame_path, default_stem=f"avatar_{i:06d}")
        suffix = frame_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            frame_path = frame_path.with_suffix(".png")
        image = Image.fromarray(frame)
        if frame_path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(frame_path, format="JPEG", quality=95, optimize=False)
        else:
            image.save(frame_path, format="PNG", optimize=False)
        validate_image_file(frame_path)
    return out


def repair_frame_sequence(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Normalise generated frames and replace broken frames deterministically."""
    if not frames:
        raise ValueError("No frames to repair.")

    target_h: int | None = None
    target_w: int | None = None
    previous: np.ndarray | None = None
    repaired: List[np.ndarray] = []

    for frame in frames:
        fixed = _coerce_frame(frame)
        if fixed is None:
            if previous is not None:
                fixed = previous.copy()
            else:
                fixed = np.zeros((512, 512, 3), dtype=np.uint8)

        if target_h is None or target_w is None:
            target_h, target_w = fixed.shape[:2]
            target_h -= target_h % 2
            target_w -= target_w % 2
            target_h = max(target_h, 2)
            target_w = max(target_w, 2)

        if fixed.shape[:2] != (target_h, target_w):
            fixed = cv2.resize(fixed, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        fixed = fixed[:target_h, :target_w, :3].copy()
        previous = fixed
        repaired.append(fixed)

    return repaired


def validate_image_file(path: str | Path) -> None:
    """Raise if an exported PNG/JPG cannot be opened and verified."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise IOError(f"Invalid image output: {p}")
    with Image.open(p) as img:
        img.verify()
    with Image.open(p) as img:
        img.load()
        if img.format not in {"PNG", "JPEG"}:
            raise ValueError(f"Unsupported image format for {p}: {img.format}")
        if img.width <= 0 or img.height <= 0:
            raise ValueError(f"Invalid image dimensions for {p}: {img.size}")


def validate_video_file(path: str | Path, expected_min_frames: int = 1) -> dict:
    """Validate that an MP4 is readable and encoded as H.264/yuv420p."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise IOError(f"Invalid video output: {p}")

    info = _probe_video_with_ffprobe(p)
    if info:
        codec = info.get("codec_name")
        pix_fmt = info.get("pix_fmt")
        frames = int(float(info.get("nb_read_frames") or info.get("nb_frames") or 0))
        if codec != "h264":
            raise ValueError(f"Video is not H.264: {p} codec={codec}")
        if pix_fmt != "yuv420p":
            raise ValueError(f"Video is not yuv420p: {p} pix_fmt={pix_fmt}")
        if frames and frames < expected_min_frames:
            raise ValueError(f"Video frame count is invalid: {p} frames={frames}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open video: {p}")
    readable = 0
    while readable < max(1, min(expected_min_frames, 3)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        readable += 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if readable == 0 or total <= 0:
        raise ValueError(f"Video contains no readable frames: {p}")
    return {"path": str(p), "frame_count": total, **info}


def _coerce_frame(frame: object) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.size == 0 or arr.ndim not in {2, 3}:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] < 3:
        return None
    else:
        arr = arr[:, :, :3]

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


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
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-map_metadata", "-1",
        "-metadata", "major_brand=mp42",
        "-c:v", codec,
        "-profile:v", "high",
        "-level", "4.1",
        "-crf", str(crf),
        "-pix_fmt", pixel_format,
        "-movflags", "+faststart",
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        message = stderr.decode("utf-8", errors="replace").strip()
        if message:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=message)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    tmp.replace(out)


def _probe_video_with_ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=codec_name,pix_fmt,nb_frames,nb_read_frames",
        "-of", "default=nokey=0:noprint_wrappers=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


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
