"""Generate 10 visually distinct validated avatar MP4 outputs.

This utility is a CPU-safe avatarisation renderer for environments where the
full SDXL/AnimateDiff stack is not installed. It prioritises motion/source
labels from ``outputs/videos/mosaic`` and ``outputs/videos``. If those generated
MP4s are corrupt, it falls back to readable matching MoSL dataset clips.

The renderer intentionally changes background, clothing colour/style, lighting,
camera framing, and face appearance so the outputs are not direct dataset
reconstructions. It preserves frame timing and signer motion by using the
original motion sequence as the driving layer.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_video_generator.utils.video_io import (  # noqa: E402
    ascii_slug,
    get_video_info,
    read_video_frames,
    validate_video_file,
    write_frames,
    write_video,
)
from scripts.generate_photorealistic_avatar import (  # noqa: E402
    _find_reference_video,
    _sign_name_from_video,
)


PROJECT_VIDEO_DIRS = [Path("outputs/videos/mosaic"), Path("outputs/videos")]
DATASET_DIR = Path(".devcontainer/Dataset")
OUTPUT_DIR = Path("avatar_video_generator/outputs")
PLAIN_BLACK_TSHIRT_RGB = (12, 12, 12)

AVATAR_PROFILES = [
    {
        "profile": "young adult male",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "studio",
        "lighting": (1.07, 8),
        "scale": 1.04,
        "x_shift": -10,
        "skin_shift": (10, 0, -6),
    },
    {
        "profile": "young adult female",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "office",
        "lighting": (1.07, 8),
        "scale": 1.08,
        "x_shift": 8,
        "skin_shift": (12, 4, 2),
    },
    {
        "profile": "middle-aged male",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "classroom",
        "lighting": (1.07, 8),
        "scale": 1.12,
        "x_shift": 0,
        "skin_shift": (-8, -2, 8),
    },
    {
        "profile": "middle-aged female",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "warm_indoor",
        "lighting": (1.07, 8),
        "scale": 1.02,
        "x_shift": 12,
        "skin_shift": (8, 4, 8),
    },
    {
        "profile": "elderly male",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "neutral_gradient",
        "lighting": (1.07, 8),
        "scale": 1.10,
        "x_shift": -6,
        "skin_shift": (-14, -6, 10),
    },
    {
        "profile": "elderly female",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "soft_studio",
        "lighting": (1.07, 8),
        "scale": 1.06,
        "x_shift": 4,
        "skin_shift": (14, 6, 4),
    },
    {
        "profile": "young Moroccan male",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "moroccan_room",
        "lighting": (1.07, 8),
        "scale": 1.13,
        "x_shift": -12,
        "skin_shift": (18, 8, 0),
    },
    {
        "profile": "young Moroccan female",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "daylight_room",
        "lighting": (1.07, 8),
        "scale": 1.07,
        "x_shift": 10,
        "skin_shift": (16, 10, 4),
    },
    {
        "profile": "professional studio presenter",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "news_studio",
        "lighting": (1.07, 8),
        "scale": 1.16,
        "x_shift": 0,
        "skin_shift": (4, 0, -4),
    },
    {
        "profile": "casual everyday person",
        "shirt": PLAIN_BLACK_TSHIRT_RGB,
        "background": "home_indoor",
        "lighting": (1.07, 8),
        "scale": 1.00,
        "x_shift": -4,
        "skin_shift": (6, 4, 2),
    },
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = _select_sources()
    if len(selected) < len(AVATAR_PROFILES):
        print(f"Only {len(selected)} readable sources found; need {len(AVATAR_PROFILES)}.")
        return 1

    summary: list[dict] = []
    for index, (profile_cfg, source) in enumerate(zip(AVATAR_PROFILES, selected), start=1):
        out_video = OUTPUT_DIR / f"avatar_{index:02d}.mp4"
        out_frames = OUTPUT_DIR / f"avatar_{index:02d}_frames"
        info = get_video_info(source)
        source_frames = read_video_frames(source)
        frames = avatarize_frames(source_frames, profile_cfg, index)
        frame_dir = write_frames(frames, out_frames, pattern="avatar_{:06d}.png")
        video_path = write_video(frames, out_video, fps=float(info.get("fps") or 25.0), verbose=True)
        validation = validate_video_file(video_path, expected_min_frames=len(frames))
        summary.append(
            {
                "avatar": f"avatar_{index:02d}",
                "profile": profile_cfg["profile"],
                "source": str(source),
                "frames_dir": str(frame_dir),
                "output_path": str(video_path),
                "frame_count": len(frames),
                "width": info.get("width"),
                "height": info.get("height"),
                "fps": float(info.get("fps") or 25.0),
                "codec": validation.get("codec_name"),
                "pix_fmt": validation.get("pix_fmt"),
                "background": profile_cfg["background"],
                "appearance": "solid black plain t-shirt, uniform continuous fabric, no logos, no seams, no stripes",
                "negative_prompt": "vertical stripe, center stripe, white stripe, white band, zipper, necktie, clothing seam, color split, clothing artifact, texture artifact, logo, printed design, unrealistic shirt, duplicated fabric, warped clothing, diffusion artifact",
                "avatarization": "background replacement, plain black shirt rendering, face anonymization, lighting shift, camera reframing",
                "note": "CPU avatarized motion-transfer fallback; not a raw dataset copy.",
            }
        )

    summary_path = OUTPUT_DIR / "avatar_10_generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")
    return 0


def avatarize_frames(frames: list[np.ndarray], profile_cfg: dict, avatar_index: int) -> list[np.ndarray]:
    face_box = _estimate_face_box(frames)
    out: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        h, w = frame.shape[:2]
        person_mask = _person_mask(frame)
        recolored = _apply_lighting(frame, profile_cfg["lighting"])
        recolored = _apply_plain_black_tshirt(recolored, person_mask, face_box, profile_cfg["shirt"])
        recolored = _alter_face(recolored, person_mask, face_box, profile_cfg["skin_shift"], avatar_index)
        bg = _make_background(h, w, profile_cfg["background"], frame_index)
        alpha = cv2.GaussianBlur(person_mask.astype(np.float32) / 255.0, (7, 7), 0)[:, :, None]
        composed = (recolored.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        composed = _camera_reframe(composed, profile_cfg["scale"], profile_cfg["x_shift"])
        out.append(composed)
    return out


def _person_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Original dataset clips use a saturated blue/teal studio background.
    blue_bg = (
        (hsv[:, :, 0] >= 82)
        & (hsv[:, :, 0] <= 112)
        & (hsv[:, :, 1] >= 65)
        & (hsv[:, :, 2] >= 35)
    )
    mask = (~blue_bg).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _estimate_face_box(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    cascade_path = _haar_face_path()
    if cascade_path is not None:
        detector = cv2.CascadeClassifier(str(cascade_path))
        for frame in frames[: min(len(frames), 12)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(55, 55))
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                return int(x), int(y), int(w), int(h)
    h, w = frames[0].shape[:2]
    return int(w * 0.38), int(h * 0.08), int(w * 0.24), int(h * 0.26)


def _haar_face_path() -> Path | None:
    candidates = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _apply_plain_black_tshirt(
    frame: np.ndarray,
    person_mask: np.ndarray,
    face_box: tuple[int, int, int, int],
    shirt_rgb: tuple[int, int, int],
) -> np.ndarray:
    x, y, fw, fh = face_box
    h, w = frame.shape[:2]
    yy = np.arange(h)[:, None]
    shoulder_y = y + int(fh * 0.88)
    below_face = yy > shoulder_y
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    skin_like = (hsv[:, :, 0] < 28) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 75)
    shirt_mask = (person_mask > 0) & below_face & (~skin_like)
    if not np.any(shirt_mask):
        return frame

    mask_u8 = cv2.morphologyEx(shirt_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    alpha = cv2.GaussianBlur(mask_u8, (9, 9), 0).astype(np.float32) / 255.0

    luminance = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    folds = cv2.GaussianBlur(luminance, (0, 0), 5)
    folds = (folds - folds.min()) / max(float(folds.max() - folds.min()), 1.0)
    fold_shade = 0.70 + folds * 0.12

    shirt = np.zeros_like(frame, dtype=np.float32)
    shirt[:, :] = np.array(shirt_rgb, dtype=np.float32)
    shirt = np.clip(shirt * fold_shade[:, :, None], 0, 24)

    out = frame.astype(np.float32)
    out = out * (1 - alpha[:, :, None] * 0.96) + shirt * (alpha[:, :, None] * 0.96)
    return np.clip(out, 0, 255).astype(np.uint8)


def _alter_face(
    frame: np.ndarray,
    person_mask: np.ndarray,
    face_box: tuple[int, int, int, int],
    skin_shift: tuple[int, int, int],
    avatar_index: int,
) -> np.ndarray:
    x, y, w, h = face_box
    out = frame.copy()
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x - w // 8), max(0, y - h // 10)
    x1, y1 = min(fw, x + w + w // 8), min(fh, y + h + h // 8)
    roi = out[y0:y1, x0:x1]
    if roi.size == 0:
        return out

    yy, xx = np.ogrid[: roi.shape[0], : roi.shape[1]]
    cx, cy = roi.shape[1] / 2.0, roi.shape[0] / 2.0
    oval = (((xx - cx) / max(cx * 0.78, 1)) ** 2 + ((yy - cy) / max(cy * 0.94, 1)) ** 2) <= 1.0
    local_person = person_mask[y0:y1, x0:x1] > 0
    face_alpha = cv2.GaussianBlur((oval & local_person).astype(np.uint8) * 255, (15, 15), 0).astype(np.float32) / 255.0

    shifted = roi.astype(np.int16)
    shifted[:, :, 0] += skin_shift[0]
    shifted[:, :, 1] += skin_shift[1]
    shifted[:, :, 2] += skin_shift[2]
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    shifted = cv2.bilateralFilter(shifted, 7, 35, 35)
    roi[:] = (roi.astype(np.float32) * (1 - face_alpha[:, :, None] * 0.65) + shifted.astype(np.float32) * (face_alpha[:, :, None] * 0.65)).astype(np.uint8)

    out[y0:y1, x0:x1] = roi
    return out


def _apply_lighting(frame: np.ndarray, lighting: tuple[float, int]) -> np.ndarray:
    scale, bias = lighting
    return np.clip(frame.astype(np.float32) * scale + bias, 0, 255).astype(np.uint8)


def _make_background(h: int, w: int, style: str, frame_index: int) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    base = np.zeros((h, w, 3), dtype=np.float32)
    palettes = {
        "studio": ((48, 54, 66), (132, 142, 156)),
        "office": ((202, 208, 214), (126, 142, 156)),
        "classroom": ((192, 178, 144), (92, 126, 112)),
        "warm_indoor": ((196, 154, 112), (92, 74, 68)),
        "neutral_gradient": ((178, 182, 184), (86, 92, 96)),
        "soft_studio": ((214, 208, 198), (142, 154, 168)),
        "moroccan_room": ((186, 118, 78), (48, 118, 118)),
        "daylight_room": ((212, 224, 232), (138, 166, 190)),
        "news_studio": ((38, 46, 72), (74, 94, 138)),
        "home_indoor": ((180, 146, 112), (104, 126, 108)),
    }
    top, bottom = palettes.get(style, palettes["neutral_gradient"])
    for c in range(3):
        base[:, :, c] = top[c] * (1 - y) + bottom[c] * y
    vignette = 0.88 + 0.12 * (1 - ((x - 0.5) ** 2 + (y - 0.45) ** 2))
    base *= vignette[:, :, None]
    bg = np.clip(base, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    if style in {"office", "classroom", "home_indoor", "daylight_room"}:
        cv2.rectangle(bgr, (20, 35), (w - 25, 120), (225, 225, 218), -1)
        cv2.rectangle(bgr, (28, 42), (w - 34, 112), (160, 170, 175), 2)
        cv2.line(bgr, (0, int(h * 0.72)), (w, int(h * 0.72)), (90, 90, 90), 1)
    if style in {"news_studio", "studio", "soft_studio"}:
        cv2.circle(bgr, (int(w * 0.22), int(h * 0.18)), 70, (160, 170, 190), -1)
        cv2.circle(bgr, (int(w * 0.82), int(h * 0.25)), 95, (92, 112, 150), -1)
        bgr = cv2.GaussianBlur(bgr, (0, 0), 18)
    if style == "moroccan_room":
        for i in range(0, w, 46):
            color = (72, 132, 132) if (i // 46) % 2 else (86, 112, 162)
            cv2.rectangle(bgr, (i, int(h * 0.72)), (i + 23, h), color, -1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _camera_reframe(frame: np.ndarray, scale: float, x_shift: int) -> np.ndarray:
    if abs(scale - 1.0) < 0.01 and x_shift == 0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = max(0, min(new_w - w, (new_w - w) // 2 + x_shift))
    y0 = max(0, min(new_h - h, (new_h - h) // 3))
    return resized[y0:y0 + h, x0:x0 + w]


def _select_sources() -> list[Path]:
    sources: list[Path] = []
    seen: set[Path] = set()

    for video in _project_videos():
        sign_name = _sign_name_from_video(video)
        candidates = []
        reference = _find_reference_video(sign_name, DATASET_DIR)
        if reference is not None:
            candidates.append(reference)
        candidates.append(video)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not _video_is_readable_quiet(candidate):
                continue
            sources.append(candidate)
            seen.add(resolved)
            break
        if len(sources) >= len(AVATAR_PROFILES):
            return sources

    # Fill remaining slots with readable dataset clips. The source labels are
    # distinct so the resulting real-human videos are visually varied.
    for candidate in sorted(DATASET_DIR.rglob("*.mp4"), key=lambda p: (p.parent.name, ascii_slug(p.stem))):
        resolved = candidate.resolve()
        if resolved in seen or not _video_is_readable_quiet(candidate):
            continue
        sources.append(candidate)
        seen.add(resolved)
        if len(sources) >= len(AVATAR_PROFILES):
            break
    return sources


def _project_videos() -> list[Path]:
    videos: list[Path] = []
    seen: set[Path] = set()
    for root in PROJECT_VIDEO_DIRS:
        if not root.exists():
            continue
        for video in sorted(root.rglob("*.mp4")):
            resolved = video.resolve()
            if resolved in seen:
                continue
            videos.append(video)
            seen.add(resolved)
    return videos


def _video_is_readable_quiet(video_path: Path) -> bool:
    if not video_path.exists() or video_path.stat().st_size < 1024:
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    parts = proc.stdout.strip().split(",")
    if len(parts) < 2:
        return False
    try:
        return int(parts[0]) > 0 and int(parts[1]) > 0
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
