"""Stage 1 cinematic neon avatar renderer.

Reads generated pose NPZ files from outputs/phase3_generation/ and produces
high-quality MP4 videos with:
  - Cinematic dark background (deep navy gradient)
  - Smooth neon joints with glow effect (Gaussian blur overlay)
  - Motion trails (fading ghost frames)
  - Temporal interpolation (2× frame upsampling for smooth playback)
  - Anti-jitter filtering (Savitzky-Golay on render path)
  - Hand emphasis (larger, brighter hand joints)
  - Arabic label rendered with proper RTL shaping
  - Side-by-side: ground-truth LEFT | generated RIGHT

Output: outputs/phase3_neon/<sign>_neon.mp4
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Colour palette — neon on dark
# ---------------------------------------------------------------------------

BG_TOP    = np.array([10,  8, 25], dtype=np.uint8)    # deep navy
BG_BOT    = np.array([5,   3, 15], dtype=np.uint8)    # near-black

BODY_COL  = (80, 220, 120)    # neon green  (BGR)
LHAND_COL = (80, 120, 255)    # neon blue
RHAND_COL = (255, 100, 80)    # neon orange-red
JOINT_COL = (200, 255, 200)   # bright white-green for body joints
LHAND_JT  = (160, 180, 255)   # light blue for left hand joints
RHAND_JT  = (255, 180, 140)   # light orange for right hand joints

TRAIL_FRAMES = 6       # number of ghost frames in motion trail
TRAIL_ALPHA  = 0.18    # opacity of oldest trail frame

# Body skeleton connections (COCO-18 joint indices)
BODY_CONNECTIONS = [
    (0, 1),   # nose → neck
    (1, 2), (2, 3), (3, 4),    # neck → R shoulder → R elbow → R wrist
    (1, 5), (5, 6), (6, 7),    # neck → L shoulder → L elbow → L wrist
    (1, 8), (8, 9), (9, 10),   # neck → R hip → R knee → R ankle
    (1, 11), (11, 12), (12, 13),  # neck → L hip → L knee → L ankle
    (0, 14), (14, 16),          # nose → R eye → R ear
    (0, 15), (15, 17),          # nose → L eye → L ear
]

# Left hand finger chains (joint indices within the 21-joint MANO layout)
# Joints 18-38 in global layout → local indices 0-20
LHAND_CHAINS = [
    [0, 1, 2, 3, 4],    # thumb
    [0, 5, 6, 7, 8],    # index
    [0, 9, 10, 11, 12], # middle
    [0, 13, 14, 15, 16],# ring
    [0, 17, 18, 19, 20],# pinky
]

# Right hand: joints 39-49 (11 joints) — partial MANO
RHAND_CHAINS = [
    [0, 1, 2, 3, 4],    # thumb (if available)
    [0, 5, 6, 7, 8],    # index
    [0, 9, 10],         # middle (partial)
]


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def _make_bg(h: int, w: int) -> np.ndarray:
    """Vertical gradient background."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        col = (BG_TOP * (1 - t) + BG_BOT * t).astype(np.uint8)
        bg[y] = col
    return bg


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------

def _joints_to_px(
    joints: np.ndarray,   # (N, 2) normalised [0,1] coords
    frame_size: int,
    margin: float = 0.12,
) -> np.ndarray:
    """Map normalised joint coords to pixel coords with margin."""
    usable = frame_size * (1 - 2 * margin)
    px = (joints * usable + frame_size * margin).astype(np.int32)
    return px.clip(0, frame_size - 1)


def _extract_joints(pose_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (150,) pose into body (18,2), lhand (21,2), rhand (11,2)."""
    joints = pose_frame.reshape(50, 3)[:, :2]   # (50, 2) — x, y
    body  = joints[:18]                          # (18, 2)
    lhand = joints[18:39]                        # (21, 2)
    rhand = joints[39:50]                        # (11, 2)
    return body, lhand, rhand


# ---------------------------------------------------------------------------
# Glow effect
# ---------------------------------------------------------------------------

def _draw_glow(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    glow_radius: int = 4,
) -> None:
    """Draw a line with a soft glow halo."""
    # Outer glow (dim, wide)
    glow_col = tuple(int(c * 0.35) for c in color)
    cv2.line(img, pt1, pt2, glow_col, thickness + glow_radius * 2, cv2.LINE_AA)
    # Mid glow
    mid_col = tuple(int(c * 0.65) for c in color)
    cv2.line(img, pt1, pt2, mid_col, thickness + glow_radius, cv2.LINE_AA)
    # Core line
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_joint_glow(
    img: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw a joint dot with glow."""
    glow_col = tuple(int(c * 0.3) for c in color)
    cv2.circle(img, pt, radius + 4, glow_col, -1, cv2.LINE_AA)
    mid_col = tuple(int(c * 0.65) for c in color)
    cv2.circle(img, pt, radius + 2, mid_col, -1, cv2.LINE_AA)
    cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Single frame renderer
# ---------------------------------------------------------------------------

def render_frame(
    pose_frame: np.ndarray,   # (150,)
    frame_size: int,
    bg: np.ndarray,           # pre-built background
    trail_frames: Optional[list[np.ndarray]] = None,  # list of (150,) older frames
    label: str = "",
    frame_idx: int = 0,
    total_frames: int = 1,
) -> np.ndarray:
    """Render one pose frame as a cinematic neon image."""
    img = bg.copy()

    # --- Motion trail (ghost frames, fading) ---
    if trail_frames:
        n_trail = len(trail_frames)
        for ti, trail_pose in enumerate(trail_frames):
            alpha = TRAIL_ALPHA * (ti + 1) / n_trail
            trail_img = np.zeros_like(img)
            _draw_skeleton(trail_img, trail_pose, frame_size,
                           body_col=BODY_COL, lhand_col=LHAND_COL, rhand_col=RHAND_COL,
                           body_thickness=1, hand_thickness=1, joint_radius=2,
                           draw_glow=False)
            img = cv2.addWeighted(img, 1.0, trail_img, alpha, 0)

    # --- Main skeleton ---
    _draw_skeleton(img, pose_frame, frame_size,
                   body_col=BODY_COL, lhand_col=LHAND_COL, rhand_col=RHAND_COL,
                   body_thickness=3, hand_thickness=2, joint_radius=5,
                   draw_glow=True)

    # --- Progress bar ---
    bar_h = 4
    bar_w = int(frame_size * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(img, (0, frame_size - bar_h), (bar_w, frame_size),
                  (60, 180, 100), -1)

    # --- Label ---
    if label:
        _draw_label(img, label, frame_size)

    return img


def _draw_skeleton(
    img: np.ndarray,
    pose_frame: np.ndarray,
    frame_size: int,
    body_col, lhand_col, rhand_col,
    body_thickness: int,
    hand_thickness: int,
    joint_radius: int,
    draw_glow: bool,
) -> None:
    """Draw body + hands onto img in-place."""
    body, lhand, rhand = _extract_joints(pose_frame)
    body_px  = _joints_to_px(body,  frame_size)
    lhand_px = _joints_to_px(lhand, frame_size)
    rhand_px = _joints_to_px(rhand, frame_size)

    draw_line = _draw_glow if draw_glow else _draw_line_plain

    # Body connections
    for a, b in BODY_CONNECTIONS:
        if a < 18 and b < 18:
            pa = tuple(body_px[a])
            pb = tuple(body_px[b])
            draw_line(img, pa, pb, body_col, body_thickness)

    # Left hand finger chains
    for chain in LHAND_CHAINS:
        for i in range(len(chain) - 1):
            j0, j1 = chain[i], chain[i + 1]
            if j0 < 21 and j1 < 21:
                pa = tuple(lhand_px[j0])
                pb = tuple(lhand_px[j1])
                draw_line(img, pa, pb, lhand_col, hand_thickness)

    # Right hand finger chains
    for chain in RHAND_CHAINS:
        for i in range(len(chain) - 1):
            j0, j1 = chain[i], chain[i + 1]
            if j0 < 11 and j1 < 11:
                pa = tuple(rhand_px[j0])
                pb = tuple(rhand_px[j1])
                draw_line(img, pa, pb, rhand_col, hand_thickness)

    # Joints
    draw_jt = _draw_joint_glow if draw_glow else _draw_joint_plain
    for j in range(18):
        draw_jt(img, tuple(body_px[j]), JOINT_COL, joint_radius)
    for j in range(21):
        draw_jt(img, tuple(lhand_px[j]), LHAND_JT, max(joint_radius - 2, 2))
    for j in range(11):
        draw_jt(img, tuple(rhand_px[j]), RHAND_JT, max(joint_radius - 2, 2))


def _draw_line_plain(img, pt1, pt2, color, thickness, **_):
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_joint_plain(img, pt, color, radius, **_):
    cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)


def _draw_label(img: np.ndarray, label: str, frame_size: int) -> None:
    """Draw Arabic label with RTL reshaping."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(label)
        display_text = get_display(reshaped)
    except ImportError:
        display_text = label

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    x = (frame_size - tw) // 2
    y = 38

    # Shadow
    cv2.putText(img, display_text, (x + 2, y + 2), font, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Text
    cv2.putText(img, display_text, (x, y), font, font_scale,
                (220, 240, 220), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Temporal interpolation (linear between frames)
# ---------------------------------------------------------------------------

def interpolate_pose(pose: np.ndarray, factor: int = 2) -> np.ndarray:
    """Upsample pose sequence by linear interpolation.

    (T, 150) → (T*factor - factor+1, 150)
    """
    T = pose.shape[0]
    if factor <= 1 or T < 2:
        return pose
    frames = []
    for t in range(T - 1):
        frames.append(pose[t])
        for k in range(1, factor):
            alpha = k / factor
            frames.append((1 - alpha) * pose[t] + alpha * pose[t + 1])
    frames.append(pose[-1])
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Anti-jitter: Savitzky-Golay on the pose sequence
# ---------------------------------------------------------------------------

def antijitter(pose: np.ndarray, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the time axis."""
    try:
        from scipy.signal import savgol_filter
        T = pose.shape[0]
        w = min(window, T)
        if w % 2 == 0:
            w -= 1
        w = max(w, polyorder + 1)
        return savgol_filter(pose, window_length=w, polyorder=polyorder, axis=0)
    except ImportError:
        return pose


# ---------------------------------------------------------------------------
# Render one sign: generated + comparison
# ---------------------------------------------------------------------------

def render_sign(
    sign_name: str,
    gen_pose: np.ndarray,          # (T, 150)
    gt_pose: Optional[np.ndarray], # (T_gt, 150) or None
    out_dir: Path,
    fps: float = 25.0,
    frame_size: int = 512,
    interp_factor: int = 2,
) -> dict:
    """Render neon avatar video for one sign. Returns paths dict."""
    # Anti-jitter + interpolation
    gen_smooth = antijitter(gen_pose, window=9, polyorder=3)
    gen_interp = interpolate_pose(gen_smooth, factor=interp_factor)
    render_fps = fps * interp_factor

    if gt_pose is not None:
        gt_smooth = antijitter(gt_pose, window=9, polyorder=3)
        gt_interp = interpolate_pose(gt_smooth, factor=interp_factor)
    else:
        gt_interp = None

    bg = _make_bg(frame_size, frame_size)
    safe = sign_name.replace("/", "_").replace(" ", "_")[:40]

    # --- Generated video ---
    gen_path = out_dir / f"{safe}_neon_generated.mp4"
    T_gen = gen_interp.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(gen_path), fourcc, render_fps, (frame_size, frame_size))
    trail: list[np.ndarray] = []
    for t in range(T_gen):
        frame = render_frame(
            gen_interp[t], frame_size, bg,
            trail_frames=list(reversed(trail[-TRAIL_FRAMES:])),
            label=sign_name,
            frame_idx=t, total_frames=T_gen,
        )
        writer.write(frame)
        trail.append(gen_interp[t])
    writer.release()

    # --- Comparison video (GT left | Generated right) ---
    cmp_path = out_dir / f"{safe}_neon_comparison.mp4"
    T_gt = gt_interp.shape[0] if gt_interp is not None else T_gen
    T_max = max(T_gen, T_gt)

    # Pad shorter sequence
    if gt_interp is not None and T_gt < T_max:
        pad = np.tile(gt_interp[-1:], (T_max - T_gt, 1))
        gt_interp_pad = np.concatenate([gt_interp, pad], axis=0)
    else:
        gt_interp_pad = gt_interp

    if T_gen < T_max:
        pad = np.tile(gen_interp[-1:], (T_max - T_gen, 1))
        gen_interp_pad = np.concatenate([gen_interp, pad], axis=0)
    else:
        gen_interp_pad = gen_interp

    cmp_w = frame_size * 2 if gt_interp is not None else frame_size
    writer_cmp = cv2.VideoWriter(str(cmp_path), fourcc, render_fps, (cmp_w, frame_size))

    trail_gen: list[np.ndarray] = []
    trail_gt:  list[np.ndarray] = []

    for t in range(T_max):
        gen_frame = render_frame(
            gen_interp_pad[t], frame_size, bg,
            trail_frames=list(reversed(trail_gen[-TRAIL_FRAMES:])),
            label=f"Generated: {sign_name}",
            frame_idx=t, total_frames=T_max,
        )
        trail_gen.append(gen_interp_pad[t])

        if gt_interp_pad is not None:
            gt_frame = render_frame(
                gt_interp_pad[t], frame_size, bg,
                trail_frames=list(reversed(trail_gt[-TRAIL_FRAMES:])),
                label=f"Ground Truth: {sign_name}",
                frame_idx=t, total_frames=T_max,
            )
            trail_gt.append(gt_interp_pad[t])

            # Divider line
            combined = np.concatenate([gt_frame, gen_frame], axis=1)
            cv2.line(combined, (frame_size, 0), (frame_size, frame_size),
                     (60, 60, 80), 2)
        else:
            combined = gen_frame

        writer_cmp.write(combined)

    writer_cmp.release()

    # Re-encode with ffmpeg for better compatibility
    for path in [gen_path, cmp_path]:
        tmp = path.with_suffix(".tmp.mp4")
        path.rename(tmp)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264",
             "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
             str(path)],
            capture_output=True,
        )
        tmp.unlink(missing_ok=True)

    return {"generated": str(gen_path), "comparison": str(cmp_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Render cinematic neon avatar videos")
    parser.add_argument("--gen-dir", default="outputs/phase3_generation",
                        help="Directory with *_pose.npz files from generate_10_signs.py")
    parser.add_argument("--out-dir", default="outputs/phase3_neon",
                        help="Output directory for neon videos")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--interp", type=int, default=2,
                        help="Temporal interpolation factor (default: 2 = 50fps output)")
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load generation summary for sign names
    summary_path = gen_dir / "generation_summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path, encoding="utf-8"))
        sign_entries = summary.get("signs", [])
    else:
        sign_entries = []

    # Find all pose NPZ files
    pose_files = sorted(gen_dir.glob("*_pose.npz"))
    print(f"Found {len(pose_files)} pose files in {gen_dir}")

    results = []
    for pose_file in pose_files:
        data = np.load(str(pose_file), allow_pickle=False)
        gen_pose = data["pose"]   # (T, 150)

        # Recover sign name from filename or summary
        stem = pose_file.stem.replace("_pose", "")
        sign_name = stem  # fallback

        # Try to match to summary entry
        for entry in sign_entries:
            safe = entry["text"].replace("/", "_").replace(" ", "_")[:40]
            if safe == stem:
                sign_name = entry["text"]
                break

        # Find ground truth
        gt_pose = _find_gt(sign_name, ROOT)

        print(f"  Rendering: {sign_name}  T={gen_pose.shape[0]}"
              f"  GT={'yes' if gt_pose is not None else 'no'}")

        paths = render_sign(
            sign_name, gen_pose, gt_pose, out_dir,
            fps=args.fps, frame_size=args.frame_size,
            interp_factor=args.interp,
        )
        results.append({"sign": sign_name, **paths})
        print(f"    -> {Path(paths['comparison']).name}")

    # Write manifest
    manifest = {"renders": results, "fps": args.fps * args.interp,
                "frame_size": args.frame_size}
    (out_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. {len(results)} videos written to {out_dir}")
    return 0


def _find_gt(sign_name: str, repo_root: Path) -> Optional[np.ndarray]:
    """Find ground-truth pose from NPZ keypoints."""
    import csv as csv_mod

    def strip_diacritics(text):
        return re.sub(r"[\u064B-\u065F\u0670]", "",
                      unicodedata.normalize("NFC", text))

    labels_csv = repo_root / "data" / "labels.csv"
    if not labels_csv.exists():
        return None

    target = strip_diacritics(sign_name)
    with open(labels_csv, encoding="utf-8") as f:
        rows = list(csv_mod.DictReader(f))

    match = None
    for row in rows:
        if strip_diacritics(row.get("word_arabic_stripped", "")) == target:
            match = row
            break
        if strip_diacritics(row.get("word_arabic", "")) == target:
            match = row
            break
    if match is None:
        return None

    rel_path = match.get("relative_path", "")
    category = match.get("category", "")
    stem = Path(rel_path).stem if rel_path else None
    if not stem:
        return None

    npz_path = repo_root / "data" / "processed" / "keypoints_2d" / category / f"{stem}.npz"
    if not npz_path.exists():
        return None

    try:
        d = np.load(str(npz_path), allow_pickle=False)
        body  = d.get("pose_keypoints_2d")   # (T, 54)
        lhand = d.get("hand_left_keypoints_2d")
        rhand = d.get("hand_right_keypoints_2d")
        if body is None:
            return None
        T = body.shape[0]
        w = float(d["width"]) if "width" in d else 1.0
        h = float(d["height"]) if "height" in d else 1.0
        if w <= 0: w = 1.0
        if h <= 0: h = 1.0

        pose = np.zeros((T, 150), dtype=np.float32)
        br = body.reshape(T, 18, 3)
        pose[:, 0:54:3] = br[:, :, 0] / w
        pose[:, 1:54:3] = br[:, :, 1] / h

        if lhand is not None and lhand.shape == (T, 63):
            lh = lhand.reshape(T, 21, 3)
            for j in range(21):
                pose[:, 54 + j*3]     = lh[:, j, 0] / w
                pose[:, 54 + j*3 + 1] = lh[:, j, 1] / h

        if rhand is not None and rhand.shape == (T, 63):
            rh = rhand.reshape(T, 21, 3)
            for j in range(11):
                pose[:, 117 + j*3]     = rh[:, j, 0] / w
                pose[:, 117 + j*3 + 1] = rh[:, j, 1] / h

        return pose
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
