"""Convert our per-video NPZ keypoints into per-frame OpenPose JSON.

The 2024 Prompt2Sign 2D→3D pipeline (`tools/2D_to_3D/pipeline_demo_01_json2h5.py`)
ingests CMU-style per-frame OpenPose JSON files. Our extraction step writes one
NPZ per clip for compactness; this script unpacks the NPZ into the on-disk
layout the pipeline expects:

    <out-root>/<category>/<video_stem>/<frame_idx_000000>_keypoints.json

Each JSON file follows CMU OpenPose schema. We populate exactly the three
fields the pipeline reads (pose_keypoints_2d, hand_left_keypoints_2d,
hand_right_keypoints_2d) and leave the rest empty for compatibility.

Note on body keypoint format: Hzzone outputs COCO-18, CMU's default is BODY_25.
The pipeline only uses idxsPose = [0..7] — these indices match between the two
formats (nose, neck, R/L shoulder/elbow/wrist), so our 18-element array is fully
compatible. See docs/DECISIONS.md for details.

Usage (inside container):
    docker/run.sh python -m mosl.pose.export_openpose_json
        [--in-root data/processed/keypoints_2d]
        [--out-root data/processed/openpose_json]
        [--filter-category Pronouns Letters ...]
        [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


# Categories used by the dataset — anything else (e.g. leftover sample/ dirs)
# is skipped to avoid picking up stale test artifacts.
KNOWN_CATEGORIES = {
    "Diverse",
    "Letters",
    "Numbers",
    "Pronouns",
    "days_months_seasons",
}


def emit_json_for_clip(npz_path: Path, out_dir: Path) -> int:
    z = np.load(npz_path)
    pose = z["pose_keypoints_2d"]            # (T, 54) flat [x, y, c, ...]
    hl = z["hand_left_keypoints_2d"]         # (T, 63)
    hr = z["hand_right_keypoints_2d"]        # (T, 63)
    n_frames = pose.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    # File-naming convention required by Prompt2Sign's pipeline_demo_01_json2h5.py:
    # the regex r"([^\\/]+)_(\d+)_keypoints\.json$" expects <stem>_<frame>_keypoints.json
    # where group(1) = stem (clip identifier) and group(2) = frame index.
    stem = npz_path.stem  # the original video filename without .mp4
    for i in range(n_frames):
        record = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": pose[i].astype(float).tolist(),
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": hl[i].astype(float).tolist(),
                    "hand_right_keypoints_2d": hr[i].astype(float).tolist(),
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": [],
                }
            ],
        }
        out_name = f"{stem}_{i:012d}_keypoints.json"
        with open(out_dir / out_name, "w", encoding="utf-8") as f:
            json.dump(record, f)
    return n_frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default=str(ROOT / "data" / "processed" / "keypoints_2d"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "processed" / "openpose_json"))
    ap.add_argument("--filter-category", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    # Collect (category, npz_path) pairs from the known categories only
    work: list[tuple[str, Path]] = []
    for cat_dir in sorted(in_root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name not in KNOWN_CATEGORIES:
            continue
        if args.filter_category and cat_dir.name not in set(args.filter_category):
            continue
        for npz in sorted(cat_dir.glob("*.npz")):
            work.append((cat_dir.name, npz))

    if args.limit > 0:
        work = work[: args.limit]

    n_done = n_skipped = total_frames = 0
    pbar = tqdm(work, unit="clip")
    for cat, npz in pbar:
        clip_out = out_root / cat / npz.stem
        # idempotent: if directory already has the right number of files, skip
        if clip_out.exists():
            existing = len(list(clip_out.glob("*_keypoints.json")))
            expected = int(np.load(npz)["pose_keypoints_2d"].shape[0])
            if existing == expected:
                n_skipped += 1
                pbar.set_postfix(done=n_done, skip=n_skipped, frames=total_frames)
                continue

        n_frames = emit_json_for_clip(npz, clip_out)
        n_done += 1
        total_frames += n_frames
        pbar.set_postfix(done=n_done, skip=n_skipped, frames=total_frames)

    print(f"\nfinished — exported {n_done} clips, skipped {n_skipped}, total {total_frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
