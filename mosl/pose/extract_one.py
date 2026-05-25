"""Run pytorch-openpose (Hzzone) on one video and emit per-frame OpenPose JSON.

Output schema follows the CMU OpenPose --write_json convention so downstream
tooling that expects OpenPose JSON (incl. Prompt2Sign) can ingest it directly.

  <out_dir>/<frame_idx_000000>_keypoints.json

with body_keypoints_2d (54 = 18*3, COCO format) and hand_left/right_keypoints_2d
(63 = 21*3 each). face is omitted for now.

Usage:
    python -m src.pose.extract_one <video_path> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# pytorch-openpose lives under third_party/; add to path
ROOT = Path(__file__).resolve().parents[2]
HZZONE = ROOT / "third_party" / "pytorch-openpose"
sys.path.insert(0, str(HZZONE))

from src import util  # type: ignore
from src.body import Body  # type: ignore
from src.hand import Hand  # type: ignore


def coco_to_flat(candidate: np.ndarray, subset: np.ndarray) -> list[float]:
    """Pick the highest-scoring person and emit a flat [x1,y1,c1,...] of length 54.

    Hzzone returns:
      candidate: (N, 4) array — N detected keypoint instances; cols are [x, y, score, id]
      subset:    (P, 20)      — P people; cols 0..17 index into `candidate` (-1 if missing),
                                col 18 = total score, col 19 = parts count.
    """
    out = [0.0] * (18 * 3)
    if len(subset) == 0:
        return out
    # pick person with highest total score
    best = int(np.argmax(subset[:, 18]))
    person = subset[best]
    for kp in range(18):
        idx = int(person[kp])
        if idx < 0:
            continue
        x, y, score, _ = candidate[idx]
        out[3 * kp] = float(x)
        out[3 * kp + 1] = float(y)
        out[3 * kp + 2] = float(score)
    return out


def hand_to_flat(peaks: np.ndarray | None) -> list[float]:
    """Hzzone hand returns (21, 2) of (x, y) — score is implicit. We emit 63 = 21*3
    with a constant confidence of 1.0 when (x, y) != (0, 0), else 0.0."""
    out = [0.0] * (21 * 3)
    if peaks is None:
        return out
    for k in range(21):
        x, y = float(peaks[k, 0]), float(peaks[k, 1])
        out[3 * k] = x
        out[3 * k + 1] = y
        out[3 * k + 2] = 0.0 if (x == 0.0 and y == 0.0) else 1.0
    return out


def main(video_path: str, out_dir: str) -> int:
    body = Body(str(HZZONE / "model" / "body_pose_model.pth"))
    hand = Hand(str(HZZONE / "model" / "hand_pose_model.pth"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: could not open {video_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {video_path}  fps={fps:.2f}  frames={n_frames}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        candidate, subset = body(frame)  # frame is BGR — Hzzone expects BGR
        body_kpts = coco_to_flat(candidate, subset)

        hands_meta = util.handDetect(candidate, subset, frame)
        left_kpts = [0.0] * 63
        right_kpts = [0.0] * 63
        for x, y, w, is_left in hands_meta:
            crop = frame[y : y + w, x : x + w, :]
            if crop.size == 0:
                continue
            peaks = hand(crop)
            # offset peaks back to full-frame coordinates (mirroring demo.py)
            peaks = peaks.astype(np.float32, copy=True)
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            flat = hand_to_flat(peaks)
            if is_left:
                left_kpts = flat
            else:
                right_kpts = flat

        # OpenPose JSON: one file per frame, single "person" entry (highest-scoring).
        record = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": body_kpts,  # 18*3, COCO order
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": left_kpts,
                    "hand_right_keypoints_2d": right_kpts,
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": [],
                }
            ],
        }
        with open(out / f"{idx:012d}_keypoints.json", "w") as f:
            json.dump(record, f)
        idx += 1
        if idx % 25 == 0:
            print(f"  frame {idx}/{n_frames}")

    cap.release()
    print(f"done — wrote {idx} frames to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
