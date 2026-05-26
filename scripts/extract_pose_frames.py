"""Extract OpenPose skeleton frames from MoSL dataset videos.

Reads real dataset MP4 videos, runs pose detection, and saves OpenPose-style
skeleton PNG frames into outputs/pose_control/<sign>_keypoints/.

Usage:
    python scripts/extract_pose_frames.py --sign أَنْتِ
    python scripts/extract_pose_frames.py --batch
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class PoseDetector:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self._detector = None
        self._backend = None
        self._init()

    def _init(self):
        try:
            from controlnet_aux import DWposeDetector
            det = DWposeDetector()
            det.to("cpu")
            self._detector = det
            self._backend = "dwpose"
            print("[PoseDetector] Using DWPose")
            return
        except Exception as e:
            print(f"[PoseDetector] DWPose unavailable: {e}")

        try:
            from controlnet_aux import OpenposeDetector
            det = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self._detector = det
            self._backend = "openpose"
            print("[PoseDetector] Using OpenposeDetector")
            return
        except Exception as e:
            print(f"[PoseDetector] OpenposeDetector unavailable: {e}")

        self._backend = "passthrough"
        print("[PoseDetector] WARNING: No pose detector — using raw frames as conditioning")

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray:
        res = self.resolution
        if self._backend == "dwpose":
            pil = Image.fromarray(frame_rgb).resize((res, res))
            result = self._detector(pil, include_body=True, include_hand=True, include_face=False)
            return np.array(result.resize((res, res)))
        elif self._backend == "openpose":
            pil = Image.fromarray(frame_rgb).resize((res, res))
            result = self._detector(pil, hand_and_face=True)
            return np.array(result.resize((res, res)))
        else:
            return cv2.resize(frame_rgb, (res, res), interpolation=cv2.INTER_LANCZOS4)


def extract_video_frames(video_path: Path, resize: int = 512) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame_pattern = str(tmp_path / "frame_%04d.png")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vf", f"scale={resize}:{resize}", frame_pattern],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")
        frames = []
        for p in sorted(tmp_path.glob("frame_*.png")):
            bgr = cv2.imread(str(p))
            if bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def extract_pose_for_video(video_path, out_dir, resolution=512, detector=None, force=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("pose_*.png"))
    if existing and not force:
        print(f"  [SKIP] {out_dir.name}: {len(existing)} frames already exist")
        return len(existing)
    if detector is None:
        detector = PoseDetector(resolution=resolution)
    print(f"  Extracting: {Path(video_path).name}")
    frames = extract_video_frames(Path(video_path), resize=resolution)
    print(f"  {len(frames)} frames, running pose detection...")
    count = 0
    for i, frame_rgb in enumerate(frames):
        skeleton = detector.detect(frame_rgb)
        out_path = out_dir / f"pose_{i:04d}.png"
        bgr = cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        count += 1
    print(f"  {count} pose frames -> {out_dir}")
    return count


def find_dataset_video(sign_name, dataset_dir):
    dataset_dir = Path(dataset_dir)
    for subdir in dataset_dir.iterdir():
        if not subdir.is_dir():
            continue
        for mp4 in subdir.glob("*.mp4"):
            if sign_name in mp4.stem:
                return mp4
    return None


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sign", metavar="ARABIC_SIGN")
    g.add_argument("--video", type=Path)
    g.add_argument("--batch", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pose_control"))
    p.add_argument("--dataset-dir", type=Path, default=Path(".devcontainer/Dataset"))
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    detector = PoseDetector(resolution=args.resolution)

    if args.video:
        out_dir = args.out_dir / f"{args.video.stem}_keypoints"
        n = extract_pose_for_video(args.video, out_dir, args.resolution, detector, args.force)
        print(f"Done: {n} frames")
    elif args.sign:
        video = find_dataset_video(args.sign, args.dataset_dir)
        if video is None:
            print(f"ERROR: No video for '{args.sign}'")
            sys.exit(1)
        out_dir = args.out_dir / f"{args.sign}_keypoints"
        n = extract_pose_for_video(video, out_dir, args.resolution, detector, args.force)
        print(f"Done: {n} frames")
    elif args.batch:
        pronouns_dir = args.dataset_dir / "mosl_videos_dataset_Pronouns"
        total = 0
        for mp4 in sorted(pronouns_dir.glob("*.mp4")):
            out_dir = args.out_dir / f"{mp4.stem}_keypoints"
            n = extract_pose_for_video(mp4, out_dir, args.resolution, detector, args.force)
            total += n
        print(f"\nBatch done: {total} total frames")


if __name__ == "__main__":
    main()
