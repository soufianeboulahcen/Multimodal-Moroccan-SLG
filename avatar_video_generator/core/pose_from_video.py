"""Extract OpenPose-style conditioning frames from a real dataset video.

Detector priority:
  1. MediaPipe Pose Tasks API (>=0.10) — downloads model on first run
  2. controlnet-aux DWPose
  3. Synthetic static skeleton (always works, no deps)

Output: 512x512 RGB PIL Images — coloured skeleton on black background,
matching ControlNet OpenPose conditioning format.
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# COCO-18 connections
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17),
]
JOINT_COLORS_RGB = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
    (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
    (255,0,255),(255,0,170),(255,0,85),
]
# MediaPipe 33-landmark index -> COCO-18 index
MP_TO_COCO = {
    0:0, 2:14, 5:15, 7:17, 8:16,
    11:5, 12:2, 13:6, 14:3, 15:7, 16:4,
    23:11, 24:8, 25:12, 26:9, 27:13, 28:10,
}


def read_video_frames(video_path: Path, resolution: int = 512) -> List[np.ndarray]:
    """Read all frames from an MP4 -> list of (H,W,3) uint8 RGB."""
    tmp = Path(tempfile.mktemp(suffix=".mp4"))
    shutil.copy2(video_path, tmp)
    try:
        cap = cv2.VideoCapture(str(tmp))
        if not cap.isOpened():
            raise IOError(f"Cannot open: {video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        tmp.unlink(missing_ok=True)
    return frames


def extract_openpose_frames(
    video_path: Path,
    resolution: int = 512,
    out_dir: Path | None = None,
) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """Extract OpenPose conditioning frames from a dataset video.

    Returns:
        raw_frames  - original video frames (H,W,3) uint8 RGB
        pose_images - ControlNet-ready PIL Images
    """
    raw_frames = read_video_frames(video_path, resolution)
    if not raw_frames:
        raise ValueError(f"No frames extracted from {video_path}")

    pose_images = _detect_and_render(raw_frames, resolution)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(pose_images):
            img.save(out_dir / f"pose_{i:06d}.png")

    return raw_frames, pose_images


def _detect_and_render(frames: List[np.ndarray], resolution: int) -> List[Image.Image]:
    for fn in (_try_mediapipe_tasks, _try_mediapipe_legacy, _try_controlnet_aux):
        try:
            result = fn(frames, resolution)
            if result:
                print(f"  Pose detector: {fn.__name__}")
                return result
        except Exception as e:
            print(f"  {fn.__name__} failed: {e}")
    print("  Pose detector: synthetic fallback")
    return _synthetic_render(frames, resolution)


def _try_mediapipe_tasks(frames: List[np.ndarray], resolution: int) -> List[Image.Image]:
    """MediaPipe Tasks API (mediapipe >= 0.10)."""
    import mediapipe as mp

    model_path = Path("/tmp/pose_landmarker_full.task")
    if not model_path.exists():
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_full/float16/latest/"
               "pose_landmarker_full.task")
        print("  Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(url, model_path)

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python import vision as mpv

    options = mpv.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mpv.RunningMode.IMAGE,
    )
    out = []
    with mpv.PoseLandmarker.create_from_options(options) as detector:
        for frame in frames:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect(mp_img)
            kp = np.zeros((18, 3), dtype=np.float32)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ls, rs = lm[11], lm[12]
                kp[1] = [(ls.x+rs.x)/2*resolution, (ls.y+rs.y)/2*resolution, 1.0]
                for mp_i, coco_i in MP_TO_COCO.items():
                    if mp_i < len(lm):
                        l = lm[mp_i]
                        kp[coco_i] = [l.x*resolution, l.y*resolution,
                                      getattr(l, "visibility", 1.0)]
            out.append(Image.fromarray(_render_skeleton(kp, resolution)))
    return out


def _try_mediapipe_legacy(frames: List[np.ndarray], resolution: int) -> List[Image.Image]:
    """MediaPipe legacy solutions API (mediapipe < 0.10)."""
    import mediapipe as mp
    if not (hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")):
        raise ImportError("legacy solutions API not available")

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      smooth_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        results_list = [pose.process(f) for f in frames]

    out = []
    for res in results_list:
        kp = np.zeros((18, 3), dtype=np.float32)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ls, rs = lm[11], lm[12]
            kp[1] = [(ls.x+rs.x)/2*resolution, (ls.y+rs.y)/2*resolution,
                     (ls.visibility+rs.visibility)/2]
            for mp_i, coco_i in MP_TO_COCO.items():
                if mp_i < len(lm):
                    l = lm[mp_i]
                    kp[coco_i] = [l.x*resolution, l.y*resolution, l.visibility]
        out.append(Image.fromarray(_render_skeleton(kp, resolution)))
    return out


def _try_controlnet_aux(frames: List[np.ndarray], resolution: int) -> List[Image.Image]:
    from controlnet_aux import DWposeDetector  # type: ignore
    det = DWposeDetector()
    return [det(Image.fromarray(f), detect_resolution=resolution,
                image_resolution=resolution) for f in frames]


def _synthetic_render(frames: List[np.ndarray], resolution: int) -> List[Image.Image]:
    """Static upper-body skeleton — no dependencies, always works."""
    W = H = resolution
    kp_norm = np.array([
        [.50,.18,1],[.50,.26,1],[.62,.29,1],[.72,.43,1],[.80,.56,1],
        [.38,.29,1],[.28,.43,1],[.20,.56,1],[.56,.56,.8],[.56,.76,.5],
        [.56,.93,.3],[.44,.56,.8],[.44,.76,.5],[.44,.93,.3],
        [.53,.16,.9],[.47,.16,.9],[.56,.17,.8],[.44,.17,.8],
    ], dtype=np.float32)
    kp = kp_norm.copy()
    kp[:, 0] *= W
    kp[:, 1] *= H
    pil = Image.fromarray(_render_skeleton(kp, resolution))
    return [pil] * len(frames)


def _render_skeleton(kp: np.ndarray, resolution: int) -> np.ndarray:
    """Render COCO-18 keypoints on black canvas -> (H,W,3) uint8 RGB."""
    canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    for idx, (i, j) in enumerate(CONNECTIONS):
        if i >= len(kp) or j >= len(kp):
            continue
        ci = float(kp[i, 2]) if kp.shape[1] > 2 else 1.0
        cj = float(kp[j, 2]) if kp.shape[1] > 2 else 1.0
        if ci < 0.1 or cj < 0.1:
            continue
        pt1 = (int(kp[i, 0]), int(kp[i, 1]))
        pt2 = (int(kp[j, 0]), int(kp[j, 1]))
        r, g, b = JOINT_COLORS_RGB[idx % len(JOINT_COLORS_RGB)]
        cv2.line(canvas, pt1, pt2, (b, g, r), 3, cv2.LINE_AA)
    for k in range(min(len(kp), 18)):
        conf = float(kp[k, 2]) if kp.shape[1] > 2 else 1.0
        if conf < 0.1:
            continue
        x, y = int(kp[k, 0]), int(kp[k, 1])
        r, g, b = JOINT_COLORS_RGB[k % len(JOINT_COLORS_RGB)]
        cv2.circle(canvas, (x, y), 5, (b, g, r), -1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
