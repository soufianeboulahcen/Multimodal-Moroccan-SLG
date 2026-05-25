"""Extract face landmarks from MoSL video clips using MediaPipe FaceMesh.

Augments the existing NPZ files in data/processed/keypoints_2d/ with a
`face_keypoints` array. Does NOT replace or modify the existing
`pose_keypoints_2d`, `hand_left_keypoints_2d`, or `hand_right_keypoints_2d`
arrays — OpenPose body + hand data is preserved unchanged.

Output format added to each NPZ:
    face_keypoints  (T, 478, 3)  — MediaPipe FaceMesh 478 landmarks × (x, y, z)
                                   x, y normalised to [0,1] by frame size
                                   z is relative depth (MediaPipe convention)
    face_confidence (T,)         — 1.0 if face detected, 0.0 if not

Non-manual markers captured:
    - Eyebrows (landmarks 46-55, 276-285)
    - Eyes (landmarks 33-133, 362-263)
    - Mouth (landmarks 0-17, 61-91, 178-308)
    - Facial expressions (full 478-point mesh)

If a frame has no detected face, its face_keypoints row is zeros and
face_confidence is 0.0.

Usage:
    python -m mosl.pose.extract_face_keypoints
    python -m mosl.pose.extract_face_keypoints --category Diverse
    python -m mosl.pose.extract_face_keypoints --dry-run
    python -m mosl.pose.extract_face_keypoints --force
    python -m mosl.pose.extract_face_keypoints --workers 4

Requires: pip install mediapipe opencv-python-headless
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark region indices (for documentation / downstream use)
# ---------------------------------------------------------------------------

FACE_REGIONS = {
    "left_eyebrow":  list(range(46, 56)),
    "right_eyebrow": list(range(276, 286)),
    "left_eye":      [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye":     [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "mouth_outer":   [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "mouth_inner":   [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    "nose":          [1, 2, 98, 327, 168, 6, 197, 195, 5],
    "jaw":           list(range(0, 18)),
}


def _check_mediapipe() -> None:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        raise ImportError(
            "mediapipe is required for face extraction.\n"
            "Install with: pip install mediapipe"
        )


def extract_face_from_video(
    video_path: Path,
    max_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe FaceMesh landmarks from every frame of a video.

    Returns:
        face_keypoints  (T, 478, 3)  float32 — normalised (x, y, z) per landmark
        face_confidence (T,)         float32 — 1.0 if face detected, else 0.0
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    n_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_alloc = max(n_frames_reported, 1)
    n_landmarks = 478  # MediaPipe FaceMesh with refine_landmarks=True

    face_keypoints = np.zeros((n_alloc, n_landmarks, 3), dtype=np.float32)
    face_confidence = np.zeros(n_alloc, dtype=np.float32)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Grow buffer if video is longer than reported
            if frame_idx >= len(face_keypoints):
                extra = np.zeros((64, n_landmarks, 3), dtype=np.float32)
                face_keypoints = np.concatenate([face_keypoints, extra], axis=0)
                face_confidence = np.concatenate([face_confidence, np.zeros(64, dtype=np.float32)])

            # BGR → RGB (in-place slice is faster than cv2.cvtColor)
            rgb = frame[:, :, ::-1]
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                for i, lm in enumerate(lms):
                    face_keypoints[frame_idx, i, 0] = lm.x
                    face_keypoints[frame_idx, i, 1] = lm.y
                    face_keypoints[frame_idx, i, 2] = lm.z
                face_confidence[frame_idx] = 1.0

            frame_idx += 1
    finally:
        cap.release()
        face_mesh.close()

    return face_keypoints[:frame_idx], face_confidence[:frame_idx]


def augment_npz(
    npz_path: Path,
    face_keypoints: np.ndarray,   # (T_face, 478, 3)
    face_confidence: np.ndarray,  # (T_face,)
    force: bool = False,
) -> bool:
    """Add face arrays to an existing NPZ file.

    Existing arrays (pose_keypoints_2d, hand_left_keypoints_2d,
    hand_right_keypoints_2d) are preserved unchanged.

    Returns True if the file was written, False if skipped.
    """
    data = dict(np.load(npz_path, allow_pickle=False))

    if "face_keypoints" in data and not force:
        return False

    # Align frame counts: OpenPose may have a different T than the video
    T_pose = data["pose_keypoints_2d"].shape[0]
    T_face = face_keypoints.shape[0]

    if T_face < T_pose:
        pad_kps = np.zeros((T_pose - T_face, 478, 3), dtype=np.float32)
        face_keypoints = np.concatenate([face_keypoints, pad_kps], axis=0)
        pad_conf = np.zeros(T_pose - T_face, dtype=np.float32)
        face_confidence = np.concatenate([face_confidence, pad_conf], axis=0)
    elif T_face > T_pose:
        face_keypoints = face_keypoints[:T_pose]
        face_confidence = face_confidence[:T_pose]

    data["face_keypoints"] = face_keypoints
    data["face_confidence"] = face_confidence

    np.savez_compressed(npz_path, **data)
    return True


# ---------------------------------------------------------------------------
# Per-clip worker (runs in subprocess for parallel processing)
# ---------------------------------------------------------------------------

def _process_clip(args: tuple) -> dict:
    """Process a single clip. Designed to run in a subprocess."""
    npz_path_str, video_path_str, force = args
    npz_path = Path(npz_path_str)
    video_path = Path(video_path_str) if video_path_str else None

    result = {
        "npz": npz_path.name,
        "status": "skipped",
        "det_rate": 0.0,
        "n_frames": 0,
        "error": None,
    }

    if video_path is None:
        result["status"] = "no_video"
        return result

    if not force:
        try:
            existing = np.load(npz_path, allow_pickle=False)
            if "face_keypoints" in existing:
                result["status"] = "already_done"
                return result
        except Exception:
            pass

    try:
        face_kps, face_conf = extract_face_from_video(video_path)
        written = augment_npz(npz_path, face_kps, face_conf, force=force)
        result["status"] = "ok" if written else "skipped"
        result["det_rate"] = float(face_conf.mean()) if len(face_conf) > 0 else 0.0
        result["n_frames"] = len(face_conf)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def _find_video(stem: str, video_dir: Path, category: str) -> Optional[Path]:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        for cat_variant in (category, f"mosl_videos_dataset_{category}"):
            candidate = video_dir / cat_variant / (stem + ext)
            if candidate.exists():
                return candidate
    for match in video_dir.rglob(f"{stem}.mp4"):
        return match
    return None


def _tally(stats: dict, result: dict, verbose: bool) -> None:
    status = result["status"]
    if status == "ok":
        stats["processed"] += 1
        if verbose:
            print(f"  [OK] {result['npz']}  T={result['n_frames']}  "
                  f"det={result['det_rate']*100:.0f}%")
    elif status in ("skipped", "already_done"):
        stats["skipped"] += 1
    elif status == "no_video":
        stats["no_video"] = stats.get("no_video", 0) + 1
        if verbose:
            print(f"  [SKIP] {result['npz']} — no video found")
    elif status == "failed":
        stats["failed"] += 1
        if verbose:
            print(f"  [FAIL] {result['npz']}: {result['error']}")


def process_category(
    category: str,
    keypoints_dir: Path,
    video_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 1,
    verbose: bool = True,
) -> dict:
    """Process all clips in one category. Returns stats dict."""
    npz_dir = keypoints_dir / category
    if not npz_dir.exists():
        return {"processed": 0, "skipped": 0, "failed": 0, "no_video": 0, "total": 0}

    npz_files = sorted(npz_dir.glob("*.npz"))
    stats = {"processed": 0, "skipped": 0, "failed": 0, "no_video": 0, "total": len(npz_files)}

    if dry_run:
        for npz_path in npz_files:
            video_path = _find_video(npz_path.stem, video_dir, category)
            status = "would_process" if video_path else "no_video"
            if verbose:
                print(f"  [DRY] {npz_path.name} → {status}")
        stats["processed"] = len(npz_files)
        return stats

    work_items = [
        (str(npz_path), str(_find_video(npz_path.stem, video_dir, category) or ""), force)
        for npz_path in npz_files
    ]
    # Replace empty string with None sentinel
    work_items = [
        (a, b if b else None, c) for a, b, c in work_items
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_clip, item): item for item in work_items}
            for future in as_completed(futures):
                _tally(stats, future.result(), verbose)
    else:
        for item in work_items:
            _tally(stats, _process_clip(item), verbose)

    return stats


def main(
    categories: Optional[list[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    workers: int = 1,
    repo_root: Optional[Path] = None,
) -> None:
    _check_mediapipe()

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    keypoints_dir = repo_root / "data" / "processed" / "keypoints_2d"
    video_dir = repo_root / "data" / "raw" / "vedios-dataset"

    if not keypoints_dir.exists():
        raise FileNotFoundError(f"keypoints directory not found: {keypoints_dir}")

    available = sorted(d.name for d in keypoints_dir.iterdir() if d.is_dir())
    target = categories if categories else available

    unknown = set(target) - set(available)
    if unknown:
        raise ValueError(f"unknown categories: {unknown}. Available: {available}")

    print("Face keypoint extraction (MediaPipe FaceMesh)")
    print(f"  keypoints_dir: {keypoints_dir}")
    print(f"  video_dir:     {video_dir}")
    print(f"  categories:    {target}")
    print(f"  force: {force}  dry_run: {dry_run}  workers: {workers}")
    print()

    total = {"processed": 0, "skipped": 0, "failed": 0, "no_video": 0, "total": 0}
    t0 = time.time()

    for cat in target:
        print(f"[{cat}]")
        stats = process_category(cat, keypoints_dir, video_dir,
                                 force=force, dry_run=dry_run, workers=workers)
        for k in total:
            total[k] += stats.get(k, 0)
        print(f"  → processed={stats['processed']}  skipped={stats['skipped']}  "
              f"failed={stats['failed']}  no_video={stats.get('no_video', 0)}  "
              f"total={stats['total']}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")
    print(f"Total: processed={total['processed']}  skipped={total['skipped']}  "
          f"failed={total['failed']}  no_video={total['no_video']}  "
          f"total={total['total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe FaceMesh landmarks and append to existing NPZ files"
    )
    parser.add_argument("--category", nargs="+", default=None,
                        help="Process only these categories (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if face_keypoints already present")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    args = parser.parse_args()

    main(
        categories=args.category,
        force=args.force,
        dry_run=args.dry_run,
        workers=args.workers,
    )
