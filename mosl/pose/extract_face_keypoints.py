"""Extract face landmarks from MoSL video clips using MediaPipe FaceMesh.

Augments the existing NPZ files in data/processed/keypoints_2d/ with a
`face_keypoints` array — it does NOT replace or modify the existing
`pose_keypoints_2d`, `hand_left`, or `hand_right` arrays.

OpenPose body + hand extraction is unchanged and remains the primary motion
source. Face landmarks are an additive modality for expression animation.

Output format added to each NPZ:
    face_keypoints  (T, 478, 3)  — MediaPipe FaceMesh 478 landmarks × (x, y, z)
                                   x, y are normalised to [0,1] by frame size
                                   z is relative depth (MediaPipe convention)
    face_confidence (T,)         — 1.0 if face detected, 0.0 if not

If a frame has no detected face, its face_keypoints row is filled with zeros
and face_confidence is 0.0.

Usage:
    # Process all clips (appends face_keypoints to existing NPZ files)
    python -m mosl.pose.extract_face_keypoints

    # Process a single category
    python -m mosl.pose.extract_face_keypoints --category Diverse

    # Dry run (no writes)
    python -m mosl.pose.extract_face_keypoints --dry-run

    # Re-extract even if face_keypoints already present
    python -m mosl.pose.extract_face_keypoints --force

Requires: pip install mediapipe
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe FaceMesh landmarks from every frame of a video.

    Returns:
        face_keypoints  (T, 478, 3)  float32 — normalised (x, y, z) per landmark
        face_confidence (T,)         float32 — 1.0 if face detected, else 0.0
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_landmarks = 478  # MediaPipe FaceMesh with refine_landmarks=True

    face_keypoints = np.zeros((n_frames, n_landmarks, 3), dtype=np.float32)
    face_confidence = np.zeros(n_frames, dtype=np.float32)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= n_frames:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            for i, lm in enumerate(lms):
                face_keypoints[frame_idx, i, 0] = lm.x
                face_keypoints[frame_idx, i, 1] = lm.y
                face_keypoints[frame_idx, i, 2] = lm.z
            face_confidence[frame_idx] = 1.0

        frame_idx += 1

    cap.release()
    face_mesh.close()

    # Trim to actual frame count (some videos report wrong CAP_PROP_FRAME_COUNT)
    actual = frame_idx
    return face_keypoints[:actual], face_confidence[:actual]


def augment_npz(
    npz_path: Path,
    face_keypoints: np.ndarray,   # (T_face, 478, 3)
    face_confidence: np.ndarray,  # (T_face,)
    force: bool = False,
) -> bool:
    """Add face arrays to an existing NPZ file.

    The existing arrays (pose_keypoints_2d, hand_left, hand_right) are
    preserved unchanged. Only face_keypoints and face_confidence are added.

    If the NPZ already contains face_keypoints and force=False, skips.
    Returns True if the file was written, False if skipped.
    """
    data = dict(np.load(npz_path, allow_pickle=False))

    if "face_keypoints" in data and not force:
        return False

    # Align frame counts: the NPZ pose arrays may have a different T than the
    # video (e.g. if OpenPose skipped frames). Pad or trim face arrays to match.
    T_pose = data["pose_keypoints_2d"].shape[0]
    T_face = face_keypoints.shape[0]

    if T_face < T_pose:
        pad = np.zeros((T_pose - T_face, 478, 3), dtype=np.float32)
        face_keypoints = np.concatenate([face_keypoints, pad], axis=0)
        conf_pad = np.zeros(T_pose - T_face, dtype=np.float32)
        face_confidence = np.concatenate([face_confidence, conf_pad], axis=0)
    elif T_face > T_pose:
        face_keypoints = face_keypoints[:T_pose]
        face_confidence = face_confidence[:T_pose]

    data["face_keypoints"] = face_keypoints
    data["face_confidence"] = face_confidence

    np.savez_compressed(npz_path, **data)
    return True


def process_category(
    category: str,
    keypoints_dir: Path,
    video_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Process all clips in one category directory.

    Returns stats dict: {processed, skipped, failed, total}.
    """
    npz_dir = keypoints_dir / category
    if not npz_dir.exists():
        return {"processed": 0, "skipped": 0, "failed": 0, "total": 0}

    npz_files = sorted(npz_dir.glob("*.npz"))
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total": len(npz_files)}

    for npz_path in npz_files:
        # Find the corresponding video file
        # NPZ stem matches the video filename stem (without extension)
        stem = npz_path.stem
        video_path = None
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = video_dir / category / (stem + ext)
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            # Try searching recursively — some categories have subdirectories
            matches = list(video_dir.rglob(f"{stem}.*"))
            video_matches = [m for m in matches if m.suffix.lower() in (".mp4", ".avi", ".mov")]
            if video_matches:
                video_path = video_matches[0]

        if video_path is None:
            if verbose:
                print(f"  [SKIP] no video found for {npz_path.name}")
            stats["skipped"] += 1
            continue

        # Check if already processed
        if not force:
            existing = np.load(npz_path, allow_pickle=False)
            if "face_keypoints" in existing:
                stats["skipped"] += 1
                continue

        if dry_run:
            if verbose:
                print(f"  [DRY] would process {npz_path.name}")
            stats["processed"] += 1
            continue

        try:
            t0 = time.time()
            face_kps, face_conf = extract_face_from_video(video_path)
            written = augment_npz(npz_path, face_kps, face_conf, force=force)
            elapsed = time.time() - t0

            if written:
                det_rate = face_conf.mean() * 100
                if verbose:
                    print(f"  [OK] {npz_path.name}  T={len(face_conf)}  "
                          f"det={det_rate:.0f}%  {elapsed:.1f}s")
                stats["processed"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            if verbose:
                print(f"  [FAIL] {npz_path.name}: {e}")
            stats["failed"] += 1

    return stats


def main(
    categories: Optional[list[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> None:
    _check_mediapipe()

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    keypoints_dir = repo_root / "data" / "processed" / "keypoints_2d"
    video_dir = repo_root / "data" / "raw" / "vedios-dataset"

    if not keypoints_dir.exists():
        raise FileNotFoundError(f"keypoints directory not found: {keypoints_dir}")

    available_categories = [d.name for d in keypoints_dir.iterdir() if d.is_dir()]
    if categories:
        # Validate requested categories
        unknown = set(categories) - set(available_categories)
        if unknown:
            raise ValueError(f"unknown categories: {unknown}. Available: {available_categories}")
        target_categories = categories
    else:
        target_categories = sorted(available_categories)

    print(f"Face keypoint extraction")
    print(f"  keypoints_dir: {keypoints_dir}")
    print(f"  video_dir:     {video_dir}")
    print(f"  categories:    {target_categories}")
    print(f"  force:         {force}")
    print(f"  dry_run:       {dry_run}")
    print()

    total_stats = {"processed": 0, "skipped": 0, "failed": 0, "total": 0}
    t_start = time.time()

    for cat in target_categories:
        print(f"[{cat}]")
        stats = process_category(cat, keypoints_dir, video_dir,
                                 force=force, dry_run=dry_run)
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"  → processed={stats['processed']}  skipped={stats['skipped']}  "
              f"failed={stats['failed']}  total={stats['total']}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Total: processed={total_stats['processed']}  "
          f"skipped={total_stats['skipped']}  failed={total_stats['failed']}  "
          f"total={total_stats['total']}")


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
    args = parser.parse_args()

    main(
        categories=args.category,
        force=args.force,
        dry_run=args.dry_run,
    )
