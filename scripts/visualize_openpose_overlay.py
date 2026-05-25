"""Render OpenPose keypoints overlaid on the original MoSL videos.

For each requested clip, reads the raw .mp4 and the corresponding NPZ keypoint
file, draws the body + hand skeletons on every frame, and writes:

  - <out_dir>/<category>__<stem>.mp4          full clip with overlay
  - <out_dir>/<category>__<stem>_frames.png   contact-sheet of 8 keyframes

Joint topologies match the upstream OpenPose conventions:
  * 18 COCO body keypoints + 17 body bones
  * 21 hand keypoints + 20 finger bones per hand

Usage (inside container):
    docker/run.sh python scripts/visualize_openpose_overlay.py \\
        --clip-stem 'أَنَا' --category Pronouns \\
        [--out-dir openpose_overlay]

    # batch over a category:
    docker/run.sh python scripts/visualize_openpose_overlay.py \\
        --category Pronouns --limit 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Skeleton topology (matches OpenPose conventions, identical to the ones
# our extractor produces).
# ---------------------------------------------------------------------------
BODY_EDGES = [
    (0, 1),   # nose - neck
    (1, 2), (2, 3), (3, 4),                # neck - R shoulder - R elbow - R wrist
    (1, 5), (5, 6), (6, 7),                # neck - L shoulder - L elbow - L wrist
    (1, 8), (8, 9),  (9, 10),              # neck - R hip - R knee - R ankle
    (1, 11), (11, 12), (12, 13),           # neck - L hip - L knee - L ankle
    (0, 14), (14, 16), (0, 15), (15, 17),  # nose - eyes/ears
]
_FINGERS = [
    [0, 1, 2, 3, 4],     # thumb
    [0, 5, 6, 7, 8],     # index
    [0, 9, 10, 11, 12],  # middle
    [0, 13, 14, 15, 16], # ring
    [0, 17, 18, 19, 20], # pinky
]
def _hand_edges() -> list[tuple[int, int]]:
    out = []
    for finger in _FINGERS:
        for a, b in zip(finger, finger[1:]):
            out.append((a, b))
    return out
HAND_EDGES = _hand_edges()


# OpenPose-style colour scheme (BGR for OpenCV).
BODY_JOINT_COLOR = (0, 100, 255)     # orange-ish dot
BODY_BONE_COLOR  = (0, 200, 0)       # green
LEFT_HAND_COLOR  = (255, 100, 0)     # blue
RIGHT_HAND_COLOR = (0, 0, 255)       # red


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_keypoints(
    frame: np.ndarray,
    flat: np.ndarray,
    edges: list[tuple[int, int]],
    joint_color: tuple[int, int, int],
    bone_color: tuple[int, int, int],
    joint_radius: int,
    bone_thickness: int,
    confidence_threshold: float = 0.0,
) -> None:
    """flat is a (N*3,) numpy array of (x, y, c) triples.  Draws on `frame` in place."""
    n_kp = flat.size // 3
    pts = []
    for k in range(n_kp):
        x = float(flat[3 * k])
        y = float(flat[3 * k + 1])
        c = float(flat[3 * k + 2])
        if c <= confidence_threshold or (x == 0.0 and y == 0.0):
            pts.append(None)
        else:
            pts.append((int(round(x)), int(round(y))))
    for a, b in edges:
        if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], bone_color, bone_thickness, cv2.LINE_AA)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, joint_radius, joint_color, -1, cv2.LINE_AA)


def draw_openpose_overlay(
    frame: np.ndarray,
    pose_flat: np.ndarray,
    hand_left_flat: np.ndarray,
    hand_right_flat: np.ndarray,
) -> None:
    """Mutate `frame` in place, drawing the body + hand skeletons."""
    h, w = frame.shape[:2]
    # Scale stroke widths with frame size; 460×460 → r=3, thickness=2.
    scale = max(1, min(h, w) // 200)
    _draw_keypoints(frame, pose_flat, BODY_EDGES,
                    BODY_JOINT_COLOR, BODY_BONE_COLOR,
                    joint_radius=2 + scale, bone_thickness=1 + scale)
    _draw_keypoints(frame, hand_left_flat, HAND_EDGES,
                    LEFT_HAND_COLOR, LEFT_HAND_COLOR,
                    joint_radius=1 + scale, bone_thickness=1 + scale // 2)
    _draw_keypoints(frame, hand_right_flat, HAND_EDGES,
                    RIGHT_HAND_COLOR, RIGHT_HAND_COLOR,
                    joint_radius=1 + scale, bone_thickness=1 + scale // 2)


# ---------------------------------------------------------------------------
# Per-clip pipeline
# ---------------------------------------------------------------------------

def process_clip(video_path: Path, npz_path: Path, out_mp4: Path,
                 out_contact_png: Path | None = None,
                 contact_columns: int = 8) -> int:
    """Process one clip end-to-end.  Returns the number of frames written."""
    z = np.load(npz_path)
    pose = z["pose_keypoints_2d"]              # (T, 54)
    hl = z["hand_left_keypoints_2d"]           # (T, 63)
    hr = z["hand_right_keypoints_2d"]          # (T, 63)
    T_npz = pose.shape[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {out_mp4}")

    # Collect snapshots for the contact sheet.
    snapshots = []
    if out_contact_png is not None:
        sample_indices = np.linspace(0, T_npz - 1, contact_columns).round().astype(int).tolist()
    else:
        sample_indices = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= T_npz:
            break
        draw_openpose_overlay(frame, pose[i], hl[i], hr[i])
        writer.write(frame)
        if i in sample_indices:
            snapshots.append((i, frame.copy()))
        i += 1
    cap.release()
    writer.release()

    if out_contact_png is not None and snapshots:
        n = len(snapshots)
        contact = np.zeros((h + 24, w * n + (n - 1) * 8, 3), dtype=np.uint8) + 30
        for col, (idx, snap) in enumerate(snapshots):
            x0 = col * (w + 8)
            contact[24 : 24 + h, x0 : x0 + w] = snap
            cv2.putText(contact, f"frame {idx}", (x0 + 6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        out_contact_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_contact_png), contact)

    return i


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def find_clip(category: str, stem: str) -> Path | None:
    labels_csv = ROOT / "data" / "labels.csv"
    with open(labels_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] == category and Path(row["relative_path"]).stem == stem:
                return ROOT / row["relative_path"]
    return None


def sample_clips_for_category(category: str, n: int) -> list[tuple[str, str]]:
    """Return [(category, stem)] biased toward multi-clip signs."""
    from collections import Counter
    rows = list(csv.DictReader(open(ROOT / "data" / "labels.csv", encoding="utf-8")))
    sign_count = Counter((r["category"], r["word_arabic"]) for r in rows)
    cat_rows = [r for r in rows if r["category"] == category]
    cat_rows.sort(key=lambda r: (-sign_count[(r["category"], r["word_arabic"])],
                                 r["relative_path"]))
    return [(r["category"], Path(r["relative_path"]).stem) for r in cat_rows[:n]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip-stem", help="single clip stem (use with --category)")
    ap.add_argument("--category", help="MoSL category (Pronouns, Letters, Numbers, days_months_seasons, Diverse)")
    ap.add_argument("--limit", type=int, default=0,
                    help="when --category is given without --clip-stem, render up to N clips from that category")
    ap.add_argument("--out-dir", default=str(ROOT / "openpose_overlay"),
                    help="output directory")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    if args.clip_stem and args.category:
        targets.append((args.category, args.clip_stem))
    elif args.category:
        targets.extend(sample_clips_for_category(args.category, max(args.limit, 1)))
    else:
        ap.error("must provide --clip-stem (with --category) or --category alone")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for category, stem in targets:
        video_path = find_clip(category, stem)
        if video_path is None or not video_path.exists():
            print(f"[skip] {category}/{stem}: video not found")
            continue
        npz_path = ROOT / "data" / "processed" / "keypoints_2d" / category / f"{stem}.npz"
        if not npz_path.exists():
            print(f"[skip] {category}/{stem}: NPZ not found ({npz_path})")
            continue

        safe = f"{category}__{stem}".replace("/", "_").replace(" ", "_")
        out_mp4 = out_dir / f"{safe}.mp4"
        out_png = out_dir / f"{safe}_frames.png"
        print(f"[ok]   {category}/{stem}")
        n = process_clip(video_path, npz_path, out_mp4, out_png)
        print(f"       wrote {out_mp4.name}  ({n} frames)")
        print(f"       wrote {out_png.name}")
        ok += 1

    print(f"\nDone — {ok} clip(s) rendered to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
