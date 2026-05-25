"""Batch pose extraction over the MoSL dataset.

Walks every clip in data/labels.csv, runs Hzzone pytorch-openpose on each frame,
and saves one .npz per video at:

    data/processed/keypoints_2d/<category>/<filename_stem>.npz

Each .npz contains:
    pose_keypoints_2d   float32 (T, 54)   — 18 COCO keypoints * (x, y, c)
    hand_left_keypoints_2d   float32 (T, 63)   — 21 hand pts * (x, y, c)
    hand_right_keypoints_2d  float32 (T, 63)
    fps    float32 scalar
    width  int32    scalar
    height int32    scalar

Resume: if the .npz already exists we skip that clip. Delete it to re-extract.
Errors in one clip are logged and the clip is skipped — extraction never aborts.

Usage:
    docker/run.sh python -m mosl.pose.extract_dataset
        [--limit N]                 # process at most N clips (debugging)
        [--filter-category Pronouns Letters ...]
        [--out-root data/processed/keypoints_2d]

The script is idempotent and safe to re-run after Ctrl-C: it just resumes.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
HZZONE = ROOT / "third_party" / "pytorch-openpose"
sys.path.insert(0, str(HZZONE))

from src import util          # type: ignore  (Hzzone pkg, not ours)
from src.body import Body     # type: ignore
from src.hand import Hand     # type: ignore


def _person_to_flat(candidate: np.ndarray, subset: np.ndarray) -> list[float]:
    out = [0.0] * (18 * 3)
    if len(subset) == 0:
        return out
    person = subset[int(np.argmax(subset[:, 18]))]
    for kp in range(18):
        i = int(person[kp])
        if i < 0:
            continue
        x, y, score, _ = candidate[i]
        out[3 * kp] = float(x)
        out[3 * kp + 1] = float(y)
        out[3 * kp + 2] = float(score)
    return out


def _hand_to_flat(peaks: np.ndarray | None) -> list[float]:
    out = [0.0] * (21 * 3)
    if peaks is None:
        return out
    for k in range(21):
        x = float(peaks[k, 0])
        y = float(peaks[k, 1])
        out[3 * k] = x
        out[3 * k + 1] = y
        out[3 * k + 2] = 0.0 if (x == 0.0 and y == 0.0) else 1.0
    return out


def extract_video(video_path: Path, body: Body, hand: Hand) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose, hl, hr = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        candidate, subset = body(frame)
        pose.append(_person_to_flat(candidate, subset))

        hands = util.handDetect(candidate, subset, frame)
        left = [0.0] * 63
        right = [0.0] * 63
        for x, y, w, is_left in hands:
            crop = frame[y : y + w, x : x + w, :]
            if crop.size == 0:
                continue
            peaks = hand(crop).astype(np.float32, copy=True)
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            flat = _hand_to_flat(peaks)
            if is_left:
                left = flat
            else:
                right = flat
        hl.append(left)
        hr.append(right)

    cap.release()
    return {
        "pose_keypoints_2d": np.asarray(pose, dtype=np.float32),
        "hand_left_keypoints_2d": np.asarray(hl, dtype=np.float32),
        "hand_right_keypoints_2d": np.asarray(hr, dtype=np.float32),
        "fps": np.float32(fps),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data" / "labels.csv"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "processed" / "keypoints_2d"))
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N clips total (0 = all)")
    ap.add_argument("--filter-category", nargs="*", default=None,
                    help="only process these categories (e.g. Pronouns Letters)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    if args.filter_category:
        keep = set(args.filter_category)
        rows = [r for r in rows if r["category"] in keep]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"loading models…")
    body = Body(str(HZZONE / "model" / "body_pose_model.pth"))
    hand = Hand(str(HZZONE / "model" / "hand_pose_model.pth"))

    n_done, n_skipped, n_failed = 0, 0, 0
    t0 = time.perf_counter()
    pbar = tqdm(rows, unit="clip")
    for r in pbar:
        video = ROOT / r["relative_path"]
        out_dir = out_root / r["category"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / (video.stem + ".npz")

        if out_npz.exists():
            n_skipped += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)
            continue

        try:
            data = extract_video(video, body, hand)
            np.savez_compressed(out_npz, **data)
            n_done += 1
        except Exception as e:
            n_failed += 1
            print(f"\n[FAIL] {r['relative_path']}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)

    dt = time.perf_counter() - t0
    print(f"\nfinished — extracted {n_done}, skipped {n_skipped}, failed {n_failed} in {dt / 60:.1f} min")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
