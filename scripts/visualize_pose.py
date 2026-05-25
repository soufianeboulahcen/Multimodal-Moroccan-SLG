"""Render a SignLLM pose sequence as a 2D skeleton animation.

Reads a .npz file produced by `scripts/predict.py` (or any 50-joint pose
array of shape (T, 150)) and writes an animated stick figure to GIF.

Joint indexing (50 total joints, 150 = 50 * (x, y, z) floats per frame):
    0..7   body, COCO upper-body subset:
            0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
            5=LShoulder, 6=LElbow, 7=LWrist
    8..28  left hand (21 OpenPose hand keypoints)
    29..49 right hand (21 OpenPose hand keypoints)

We render the (x, y) projection — the z dimension from the SignLLM 2D-to-3D
pipeline is a derived hallucination from the back-prop optimizer, not a
real depth measurement.

Usage (inside container):
    docker/run.sh python scripts/visualize_pose.py predictions/<file>.npz \\
        [--out /path/to/<name>.gif]
        [--side-by-side]   # if the npz contains target_pose, render both
        [--fps 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless inside the container
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# Arabic typography setup: matplotlib's default text path does not
# (a) join cursive Arabic letters or (b) apply right-to-left ordering.
# We use arabic_reshaper + python-bidi to fix this before passing strings
# to matplotlib.  If either is unavailable we fall back to raw text (still
# legible character-by-character, just ugly).
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _HAS_ARABIC_TOOLING = True
except ImportError:
    _HAS_ARABIC_TOOLING = False

# Pick a font that contains Arabic glyphs (with diacritic support).  Noto
# Naskh Arabic ships in the project's Docker image; we fall back to the
# matplotlib default if it is not installed.
import matplotlib.font_manager as fm

def _pick_arabic_font() -> str | None:
    for candidate in ("Amiri", "Noto Naskh Arabic", "Scheherazade", "DejaVu Sans"):
        try:
            fm.findfont(candidate, fallback_to_default=False)
            return candidate
        except (ValueError, Exception):  # findfont raises a generic exception variant
            continue
    return None

_ARABIC_FONT = _pick_arabic_font()


def shape_arabic(text: str) -> str:
    """Reshape + BiDi-reorder Arabic so matplotlib draws it correctly.
    Non-Arabic input is returned unchanged."""
    if not _HAS_ARABIC_TOOLING:
        return text
    # arabic_reshaper handles non-Arabic strings transparently; cheap.
    return get_display(arabic_reshaper.reshape(text))


# Skeleton edges (joint pairs) — body upper-body + hand topology --------------

BODY_EDGES = [
    (0, 1),   # nose - neck
    (1, 2), (2, 3), (3, 4),                # neck - R shoulder - R elbow - R wrist
    (1, 5), (5, 6), (6, 7),                # neck - L shoulder - L elbow - L wrist
]

# OpenPose hand keypoint topology: 0=palm, 1-4=thumb, 5-8=index, 9-12=middle,
# 13-16=ring, 17-20=pinky.  Each finger is a chain from palm.
_HAND_FINGERS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

def _hand_edges(offset: int) -> list[tuple[int, int]]:
    edges = []
    for finger in _HAND_FINGERS:
        for a, b in zip(finger, finger[1:]):
            edges.append((offset + a, offset + b))
    return edges

LEFT_HAND_EDGES = _hand_edges(8)
RIGHT_HAND_EDGES = _hand_edges(8 + 21)


def reshape_to_xyz(pose_flat: np.ndarray) -> np.ndarray:
    """(T, 150) -> (T, 50, 3)"""
    T, D = pose_flat.shape
    if D != 150:
        raise ValueError(f"expected 150 coords per frame, got {D}")
    return pose_flat.reshape(T, 50, 3)


def render(
    pose_seq: np.ndarray,                 # (T, 50, 3)
    out_path: Path,
    fps: int,
    title: str,
    second_pose: np.ndarray | None = None,
    second_title: str = "target",
) -> None:
    """Save a GIF of the skeleton sequence (with optional side-by-side panel)."""
    n_panels = 1 if second_pose is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
    panel_data: list[tuple[np.ndarray, str, plt.Axes]] = [
        (pose_seq, title, axes[0, 0]),
    ]
    if second_pose is not None:
        panel_data.append((second_pose, second_title, axes[0, 1]))

    # Common axis bounds across all panels and frames.
    all_xy = np.concatenate(
        [seq[..., :2].reshape(-1, 2) for seq, _, _ in panel_data], axis=0
    )
    # Filter out the 0.01600 placeholder used in .skels for missing keypoints,
    # and the 0.0 from naively-zero outputs (model only — placeholders are
    # specific to ground-truth from .skels).
    keep_mask = ~np.isclose(all_xy[:, 0], 0.01600) & (np.abs(all_xy).sum(axis=1) > 1e-6)
    visible = all_xy[keep_mask]
    if visible.size == 0:
        print("[visualize] warning: no visible keypoints in any frame", file=sys.stderr)
        visible = all_xy
    pad = 0.1
    x_min, x_max = visible[:, 0].min() - pad, visible[:, 0].max() + pad
    y_min, y_max = visible[:, 1].min() - pad, visible[:, 1].max() + pad

    # Set up each axis once (image y-axis flipped to match natural pose orientation).
    artists: list[dict] = []
    all_edges = BODY_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES
    title_font_kwargs = {"fontsize": 10}
    if _ARABIC_FONT:
        title_font_kwargs["fontname"] = _ARABIC_FONT
    for seq, sub_title, ax in panel_data:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(shape_arabic(sub_title), **title_font_kwargs)
        # draw initial frame artists; we will update .set_data() in animate()
        scat = ax.scatter([], [], s=8, c="C0", zorder=3)
        lines = []
        for _ in all_edges:
            (line,) = ax.plot([], [], color="C0", lw=1.2, zorder=2)
            lines.append(line)
        time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                            fontsize=8, color="0.3")
        artists.append({"seq": seq, "scat": scat, "lines": lines, "time": time_text})

    suptitle_text = (title if second_pose is None
                     else f"{title}  (predicted left  /  target right)")
    suptitle_kwargs = {"fontsize": 11}
    if _ARABIC_FONT:
        suptitle_kwargs["fontname"] = _ARABIC_FONT
    fig.suptitle(shape_arabic(suptitle_text), **suptitle_kwargs)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    n_frames = max(seq.shape[0] for seq, _, _ in panel_data)

    def _draw_one(panel: dict, t: int) -> None:
        seq = panel["seq"]
        if t >= seq.shape[0]:
            t = seq.shape[0] - 1
        frame = seq[t]                        # (50, 3)
        # Mask out placeholders / origin points
        valid = ~(np.isclose(frame[:, 0], 0.01600) & np.isclose(frame[:, 1], 0.01600))
        valid &= ~((np.abs(frame[:, 0]) < 1e-6) & (np.abs(frame[:, 1]) < 1e-6))
        panel["scat"].set_offsets(frame[valid, :2])
        for line, (a, b) in zip(panel["lines"], all_edges):
            if valid[a] and valid[b]:
                line.set_data([frame[a, 0], frame[b, 0]],
                              [frame[a, 1], frame[b, 1]])
            else:
                line.set_data([], [])
        panel["time"].set_text(f"frame {t+1}/{seq.shape[0]}")

    def animate(t: int):
        for panel in artists:
            _draw_one(panel, t)
        return [a for p in artists for a in
                ([p["scat"], p["time"]] + p["lines"])]

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 / fps, blit=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"[visualize] wrote {out_path}  ({n_frames} frames @ {fps} fps)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("npz", help="Input .npz produced by scripts/predict.py")
    p.add_argument("--out", default=None,
                   help="Output GIF path (default: <input>.gif)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--side-by-side", action="store_true",
                   help="If the npz contains target_pose, render predicted vs target")
    args = p.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"[visualize] error: {npz_path} not found", file=sys.stderr)
        return 1
    data = np.load(npz_path, allow_pickle=False)
    pose = reshape_to_xyz(data["pose"])
    target = None
    if args.side_by_side and "target_pose" in data.files:
        target = reshape_to_xyz(data["target_pose"])
    text = str(data["text"]) if "text" in data.files else npz_path.stem

    out = Path(args.out) if args.out else npz_path.with_suffix(".gif")
    title = f"{text}  ({data['run']})" if "run" in data.files else text
    render(pose, out, fps=args.fps, title=title, second_pose=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
