"""Build .skels / .text / .files from existing NPZ keypoints.

Reads data/processed/keypoints_2d/**/*.npz and data/splits.csv, then writes:
  third_party/Prompt2Sign/tools/2D_to_3D/final_data/
    train.skels  train.text  train.files
    dev.skels    dev.text    dev.files
    test.skels   test.text   test.files

Skels format (per line = one clip):
  151 floats per frame: 150 pose coords + 1 normalised time marker
  Frames separated by spaces, all on one line.

Joint layout (50 joints × xyz = 150 coords):
  Joints  0-17  body (COCO-18, from pose_keypoints_2d)
  Joints 18-38  left hand (21 joints, from hand_left_keypoints_2d)
  Joints 39-49  right hand (first 11 of 21, from hand_right_keypoints_2d)

Normalisation:
  - x, y divided by frame width/height → [0, 1]
  - z set to 0 (2D keypoints have no depth)
  - confidence values dropped (only xyz kept)
  - time marker = frame_index / (T - 1), clamped to (0, 1]
"""
from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
KP_DIR = ROOT / "data" / "processed" / "keypoints_2d"
SPLITS_CSV = ROOT / "data" / "splits.csv"
OUT_DIR = ROOT / "third_party" / "Prompt2Sign" / "tools" / "2D_to_3D" / "final_data"

# Joint counts
N_BODY = 18       # COCO-18
N_LHAND = 21      # MANO left
N_RHAND_USE = 11  # first 11 of 21 right-hand joints (matches Prompt2Sign convention)
N_JOINTS = N_BODY + N_LHAND + N_RHAND_USE  # = 50
POSE_DIM = N_JOINTS * 3  # = 150


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def npz_to_pose(npz_path: Path) -> np.ndarray | None:
    """Load NPZ and return (T, 150) float32 pose array, or None on failure."""
    try:
        d = np.load(str(npz_path), allow_pickle=False)
    except Exception as e:
        print(f"  [WARN] failed to load {npz_path.name}: {e}", file=sys.stderr)
        return None

    body = d.get("pose_keypoints_2d")    # (T, 54)  — 18 joints × (x,y,conf)
    lhand = d.get("hand_left_keypoints_2d")   # (T, 63)
    rhand = d.get("hand_right_keypoints_2d")  # (T, 63)

    if body is None or body.ndim != 2 or body.shape[1] != 54:
        return None

    T = body.shape[0]
    if T < 2:
        return None

    # Normalise by frame dimensions
    w = float(d["width"]) if "width" in d else 1.0
    h = float(d["height"]) if "height" in d else 1.0
    if w <= 0:
        w = 1.0
    if h <= 0:
        h = 1.0

    pose = np.zeros((T, POSE_DIM), dtype=np.float32)

    # Body joints 0-17: extract x,y from (x,y,conf) triplets, set z=0
    body_r = body.reshape(T, N_BODY, 3)
    pose[:, 0:N_BODY*3:3] = body_r[:, :, 0] / w   # x
    pose[:, 1:N_BODY*3:3] = body_r[:, :, 1] / h   # y
    # z stays 0

    # Left hand joints 18-38
    lh_start = N_BODY * 3  # 54
    if lhand is not None and lhand.ndim == 2 and lhand.shape == (T, 63):
        lh_r = lhand.reshape(T, N_LHAND, 3)
        for j in range(N_LHAND):
            base = lh_start + j * 3
            pose[:, base]     = lh_r[:, j, 0] / w
            pose[:, base + 1] = lh_r[:, j, 1] / h
            # z stays 0

    # Right hand joints 39-49 (first 11 joints)
    rh_start = (N_BODY + N_LHAND) * 3  # 117
    if rhand is not None and rhand.ndim == 2 and rhand.shape == (T, 63):
        rh_r = rhand.reshape(T, 21, 3)
        for j in range(N_RHAND_USE):
            base = rh_start + j * 3
            pose[:, base]     = rh_r[:, j, 0] / w
            pose[:, base + 1] = rh_r[:, j, 1] / h

    return pose


def pose_to_skels_line(pose: np.ndarray) -> str:
    """Convert (T, 150) pose to a single skels line with time markers."""
    T = pose.shape[0]
    # Time marker: frame_index / max(T-1, 1), so last frame = 1.0
    time_markers = np.linspace(0.0, 1.0, T, dtype=np.float32)
    # Clamp to avoid exact 0 at first frame (convention: time ∈ (0,1])
    time_markers = np.clip(time_markers, 1.0 / max(T, 1), 1.0)

    # Interleave: for each frame, 150 pose coords then 1 time marker
    frames_with_time = np.concatenate(
        [pose, time_markers.reshape(T, 1)], axis=1
    )  # (T, 151)

    # Flatten to one line
    flat = frames_with_time.flatten()
    return " ".join(f"{v:.6f}" for v in flat)


def build_split(
    split_name: str,
    rows: list[dict],
    out_dir: Path,
    verbose: bool = True,
) -> int:
    """Build .skels/.text/.files for one split. Returns number of clips written."""
    skels_lines = []
    text_lines = []
    file_lines = []
    n_skipped = 0

    for row in tqdm(rows, desc=f"  {split_name}", unit="clip"):
        rel_path = row["relative_path"]
        word = _nfc(row["word_arabic"])
        category = row["category"]

        # Resolve NPZ path from relative_path stem
        stem = Path(rel_path).stem
        npz_path = KP_DIR / category / f"{stem}.npz"

        if not npz_path.exists():
            # Try searching all categories
            found = None
            for cat_dir in KP_DIR.iterdir():
                candidate = cat_dir / f"{stem}.npz"
                if candidate.exists():
                    found = candidate
                    break
            if found is None:
                n_skipped += 1
                continue
            npz_path = found

        pose = npz_to_pose(npz_path)
        if pose is None:
            n_skipped += 1
            continue

        skels_lines.append(pose_to_skels_line(pose))
        text_lines.append(word)
        file_lines.append(stem)

    # Write files
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split_name}.skels").write_text("\n".join(skels_lines) + "\n", encoding="utf-8")
    (out_dir / f"{split_name}.text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (out_dir / f"{split_name}.files").write_text("\n".join(file_lines) + "\n", encoding="utf-8")

    n_written = len(skels_lines)
    if verbose:
        print(f"  {split_name}: {n_written} clips written, {n_skipped} skipped")

    return n_written


def main() -> int:
    print("Building .skels/.text/.files from NPZ keypoints...")
    print(f"  Source: {KP_DIR}")
    print(f"  Output: {OUT_DIR}")

    # Load splits
    with open(SPLITS_CSV, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Group by split — splits.csv uses 'val' but dataset.py expects 'dev'
    split_map = {"train": "train", "val": "dev", "test": "test"}
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for row in all_rows:
        s = split_map.get(row["split"], row["split"])
        if s in splits:
            splits[s].append(row)

    print(f"  Rows: train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")

    total = 0
    for split_name, rows in splits.items():
        n = build_split(split_name, rows, OUT_DIR)
        total += n

    print(f"\nDone. Total clips written: {total}")

    # Verify
    for split_name in ("train", "dev", "test"):
        for ext in ("skels", "text", "files"):
            p = OUT_DIR / f"{split_name}.{ext}"
            n_lines = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
            print(f"  {split_name}.{ext}: {n_lines} lines")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
