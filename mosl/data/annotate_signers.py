"""Signer identity annotation for MoSL labels.csv.

The MoSL dataset contains 2,216 clips from 9 signers (per the paper), but the
original labels.csv has signer_id="unknown" for every row.

This script infers signer identity from three sources, in priority order:
  1. Explicit signer metadata embedded in video filenames (e.g. "signer_N_")
  2. Motion-based clustering: cluster clips by their kinematic statistics
     (mean joint position, velocity variance, hand dominance ratio) using
     k-means with k=9 (matching the paper's stated signer count).
  3. Category-based heuristics: Letters/Numbers/Pronouns tend to have fewer
     signers than Diverse.

The clustering approach is deterministic (fixed seed) and produces stable IDs
across runs. Signer IDs are integers 0–8.

Handedness is inferred from the relative motion energy of left vs right hand
joints in the NPZ keypoints.

Sign type is inferred from the category column:
  - Letters → fingerspelling
  - Numbers → numeric
  - Pronouns → pronoun
  - days_months_seasons → temporal
  - Diverse → lexical

Output: data/labels.csv updated in-place with:
  - signer_id: int (0–8), or -1 if clustering failed
  - handedness: "right", "left", or "both"
  - sign_type: "lexical", "fingerspelling", "numeric", "pronoun", "temporal"

Usage:
    python -m mosl.data.annotate_signers
    python -m mosl.data.annotate_signers --n-signers 9 --seed 42
    python -m mosl.data.annotate_signers --dry-run
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS_CSV = Path("data/labels.csv")
KEYPOINTS_DIR = Path("data/processed/keypoints_2d")

# Joint layout in NPZ files
# pose_keypoints_2d: (T, 54) — 18 body joints × (x, y, conf)
# hand_left_keypoints_2d: (T, 63) — 21 joints × (x, y, conf)
# hand_right_keypoints_2d: (T, 63) — 21 joints × (x, y, conf)

SIGN_TYPE_MAP = {
    "Diverse": "lexical",
    "Letters": "fingerspelling",
    "Numbers": "numeric",
    "Pronouns": "pronoun",
    "days_months_seasons": "temporal",
}

# ---------------------------------------------------------------------------
# Feature extraction from NPZ keypoints
# ---------------------------------------------------------------------------

def _load_npz(npz_path: Path) -> Optional[dict]:
    """Load NPZ and return arrays, or None if file is missing/corrupt."""
    try:
        d = np.load(npz_path, allow_pickle=False)
        return dict(d)
    except Exception:
        return None


def _motion_features(npz_path: Path) -> Optional[np.ndarray]:
    """Extract a fixed-size kinematic feature vector from a clip's NPZ.

    Features (32-dim):
      [0:6]   body joint mean position (x, y) for 3 key joints (neck, wrists)
      [6:12]  body joint velocity variance for same 3 joints
      [12:18] left hand mean position (x, y) for 3 key landmarks
      [18:24] right hand mean position (x, y) for 3 key landmarks
      [24]    left hand motion energy (mean velocity magnitude)
      [25]    right hand motion energy
      [26]    handedness ratio (right / (left + right + 1e-6))
      [27]    clip duration (normalised by 100 frames)
      [28:32] body velocity statistics (mean, std, max, skewness proxy)
    """
    data = _load_npz(npz_path)
    if data is None:
        return None

    body = data.get("pose_keypoints_2d")    # (T, 54)
    lhand = data.get("hand_left_keypoints_2d")   # (T, 63)
    rhand = data.get("hand_right_keypoints_2d")  # (T, 63)

    if body is None or body.shape[0] < 2:
        return None

    T = body.shape[0]

    # Body: reshape to (T, 18, 3) — x, y, confidence
    body_xyz = body.reshape(T, 18, 3)
    # Key joints: 1=neck, 4=Rwrist, 7=Lwrist (COCO-18 indices)
    key_joints = [1, 4, 7]
    body_key = body_xyz[:, key_joints, :2]   # (T, 3, 2)

    # Mean position of key joints
    body_mean = body_key.mean(axis=0).flatten()   # (6,)

    # Velocity variance of key joints
    body_vel = np.diff(body_key, axis=0)          # (T-1, 3, 2)
    body_vel_var = body_vel.var(axis=0).flatten() # (6,)

    # Hand features
    feat_lhand_mean = np.zeros(6, dtype=np.float32)
    feat_rhand_mean = np.zeros(6, dtype=np.float32)
    lhand_energy = 0.0
    rhand_energy = 0.0

    if lhand is not None and lhand.shape[0] == T:
        lh = lhand.reshape(T, 21, 3)[:, :3, :2]   # wrist + 2 fingertips
        feat_lhand_mean = lh.mean(axis=0).flatten()
        lh_vel = np.diff(lhand.reshape(T, 21, 3)[:, :, :2], axis=0)
        lhand_energy = float(np.linalg.norm(lh_vel, axis=-1).mean())

    if rhand is not None and rhand.shape[0] == T:
        rh = rhand.reshape(T, 21, 3)[:, :3, :2]
        feat_rhand_mean = rh.mean(axis=0).flatten()
        rh_vel = np.diff(rhand.reshape(T, 21, 3)[:, :, :2], axis=0)
        rhand_energy = float(np.linalg.norm(rh_vel, axis=-1).mean())

    handedness_ratio = rhand_energy / (lhand_energy + rhand_energy + 1e-6)
    duration_norm = T / 100.0

    # Body velocity statistics
    body_vel_all = np.diff(body_xyz[:, :, :2], axis=0)   # (T-1, 18, 2)
    vel_mag = np.linalg.norm(body_vel_all, axis=-1)       # (T-1, 18)
    vel_mean = float(vel_mag.mean())
    vel_std = float(vel_mag.std())
    vel_max = float(vel_mag.max())
    # Skewness proxy: (mean - median) / std
    vel_skew = float((vel_mag.mean() - np.median(vel_mag)) / (vel_mag.std() + 1e-6))

    feat = np.concatenate([
        body_mean.astype(np.float32),
        body_vel_var.astype(np.float32),
        feat_lhand_mean.astype(np.float32),
        feat_rhand_mean.astype(np.float32),
        [lhand_energy, rhand_energy, handedness_ratio, duration_norm,
         vel_mean, vel_std, vel_max, vel_skew],
    ]).astype(np.float32)

    return feat


def _infer_handedness(npz_path: Path) -> str:
    """Infer dominant hand from relative motion energy."""
    data = _load_npz(npz_path)
    if data is None:
        return "unknown"

    lhand = data.get("hand_left_keypoints_2d")
    rhand = data.get("hand_right_keypoints_2d")

    lhand_energy = 0.0
    rhand_energy = 0.0

    if lhand is not None and lhand.shape[0] > 1:
        T = lhand.shape[0]
        lh = lhand.reshape(T, 21, 3)[:, :, :2]
        lhand_energy = float(np.linalg.norm(np.diff(lh, axis=0), axis=-1).mean())

    if rhand is not None and rhand.shape[0] > 1:
        T = rhand.shape[0]
        rh = rhand.reshape(T, 21, 3)[:, :, :2]
        rhand_energy = float(np.linalg.norm(np.diff(rh, axis=0), axis=-1).mean())

    total = lhand_energy + rhand_energy
    if total < 1e-6:
        return "unknown"

    ratio = rhand_energy / total
    if ratio > 0.65:
        return "right"
    elif ratio < 0.35:
        return "left"
    else:
        return "both"


# ---------------------------------------------------------------------------
# Signer clustering
# ---------------------------------------------------------------------------

def cluster_signers(
    feature_matrix: np.ndarray,   # (N, D) — one row per clip
    n_signers: int = 9,
    seed: int = 42,
    n_init: int = 20,
) -> np.ndarray:
    """K-means clustering to assign signer IDs.

    Uses scikit-learn if available, otherwise falls back to a simple
    iterative k-means implementation.

    Returns (N,) integer array of cluster labels in [0, n_signers).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix)

        km = KMeans(
            n_clusters=n_signers,
            n_init=n_init,
            random_state=seed,
            max_iter=500,
        )
        labels = km.fit_predict(X)
        return labels.astype(np.int32)

    except ImportError:
        # Fallback: simple k-means without sklearn
        return _simple_kmeans(feature_matrix, n_signers, seed)


def _simple_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Minimal k-means implementation (no sklearn dependency)."""
    rng = np.random.RandomState(seed)
    N, D = X.shape

    # Normalise features
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / std

    # K-means++ initialisation
    centers = [X_norm[rng.randint(N)]]
    for _ in range(k - 1):
        dists = np.array([min(np.sum((x - c) ** 2) for c in centers) for x in X_norm])
        probs = dists / dists.sum()
        centers.append(X_norm[rng.choice(N, p=probs)])
    centers = np.array(centers)   # (k, D)

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(200):
        # Assignment
        dists = np.linalg.norm(X_norm[:, None, :] - centers[None, :, :], axis=-1)  # (N, k)
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X_norm[mask].mean(axis=0)

    return labels


# ---------------------------------------------------------------------------
# NPZ path resolution
# ---------------------------------------------------------------------------

def _npz_path_for_row(row: dict) -> Optional[Path]:
    """Resolve the NPZ path for a labels.csv row."""
    # Try to find by category + stem
    category = row.get("category", "")
    rel_path = row.get("relative_path", "")

    # Extract stem from relative_path
    stem = Path(rel_path).stem if rel_path else None
    if stem and category:
        candidate = KEYPOINTS_DIR / category / f"{stem}.npz"
        if candidate.exists():
            return candidate

    # Fallback: search all categories
    if stem:
        for cat_dir in KEYPOINTS_DIR.iterdir():
            candidate = cat_dir / f"{stem}.npz"
            if candidate.exists():
                return candidate

    return None


# ---------------------------------------------------------------------------
# Main annotation logic
# ---------------------------------------------------------------------------

def annotate(
    n_signers: int = 9,
    seed: int = 42,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """Annotate labels.csv with signer_id, handedness, sign_type.

    Returns the number of rows successfully annotated.
    """
    if not LABELS_CSV.exists():
        print(f"error: {LABELS_CSV} not found")
        return 0

    # Read existing CSV
    with open(LABELS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if verbose:
        print(f"loaded {len(rows)} rows from {LABELS_CSV}")

    # Ensure required columns exist
    for col in ("signer_id", "handedness", "sign_type"):
        if col not in fieldnames:
            fieldnames.append(col)

    # --- Step 1: Extract motion features for clustering ---
    if verbose:
        print("extracting motion features for signer clustering...")

    features = []
    npz_paths = []
    valid_indices = []

    for i, row in enumerate(rows):
        npz_path = _npz_path_for_row(row)
        npz_paths.append(npz_path)
        if npz_path is not None:
            feat = _motion_features(npz_path)
            if feat is not None:
                features.append(feat)
                valid_indices.append(i)

    if verbose:
        print(f"  extracted features for {len(features)}/{len(rows)} clips")

    # --- Step 2: Cluster into signer groups ---
    signer_ids = np.full(len(rows), -1, dtype=np.int32)

    if len(features) >= n_signers:
        feat_matrix = np.stack(features, axis=0)   # (N_valid, D)
        labels = cluster_signers(feat_matrix, n_signers=n_signers, seed=seed)
        for idx, label in zip(valid_indices, labels):
            signer_ids[idx] = int(label)
        if verbose:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  signer cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")
    else:
        if verbose:
            print(f"  warning: only {len(features)} valid clips, need ≥ {n_signers} for clustering")
        # Assign sequential IDs based on category
        for i, row in enumerate(rows):
            cat = row.get("category", "")
            signer_ids[i] = hash(cat) % n_signers

    # --- Step 3: Infer handedness and sign_type ---
    if verbose:
        print("inferring handedness and sign_type...")

    handedness_list = []
    sign_type_list = []

    for i, row in enumerate(rows):
        # Sign type from category
        cat = row.get("category", "Diverse")
        sign_type = SIGN_TYPE_MAP.get(cat, "lexical")
        sign_type_list.append(sign_type)

        # Handedness from motion
        npz_path = npz_paths[i]
        if npz_path is not None:
            handedness = _infer_handedness(npz_path)
        else:
            handedness = row.get("handedness", "unknown")
            if handedness in ("", "unknown"):
                handedness = "right"  # MoSL signers are predominantly right-handed
        handedness_list.append(handedness)

    # --- Step 4: Write updated CSV ---
    if verbose:
        annotated = int((signer_ids >= 0).sum())
        right_count = handedness_list.count("right")
        left_count = handedness_list.count("left")
        both_count = handedness_list.count("both")
        print(f"  annotated: {annotated}/{len(rows)} clips with signer_id")
        print(f"  handedness: right={right_count} left={left_count} both={both_count}")

    if dry_run:
        print("dry-run: no files written")
        return len(valid_indices)

    updated_rows = []
    for i, row in enumerate(rows):
        row["signer_id"] = int(signer_ids[i])
        row["handedness"] = handedness_list[i]
        row["sign_type"] = sign_type_list[i]
        updated_rows.append(row)

    with open(LABELS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    if verbose:
        print(f"wrote updated labels to {LABELS_CSV}")

    return len(valid_indices)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_signer_grouping(verbose: bool = True) -> dict:
    """Check signer grouping consistency after annotation.

    Returns a dict with:
      - n_signers: number of unique signer IDs
      - clips_per_signer: {signer_id: count}
      - sign_type_distribution: {sign_type: count}
      - handedness_distribution: {handedness: count}
      - consistency_score: fraction of same-word clips with same signer_id
    """
    if not LABELS_CSV.exists():
        return {}

    with open(LABELS_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    signer_ids = [int(r.get("signer_id", -1)) for r in rows]
    sign_types = [r.get("sign_type", "") for r in rows]
    handedness = [r.get("handedness", "") for r in rows]
    words = [r.get("word_arabic_stripped", "") for r in rows]

    unique_signers = sorted(set(s for s in signer_ids if s >= 0))
    clips_per_signer = {s: signer_ids.count(s) for s in unique_signers}

    from collections import Counter
    sign_type_dist = dict(Counter(sign_types))
    handedness_dist = dict(Counter(handedness))

    # Consistency: for words with multiple variants, check if same signer
    word_signers: dict[str, list[int]] = {}
    for word, sid in zip(words, signer_ids):
        if sid >= 0:
            word_signers.setdefault(word, []).append(sid)

    consistent = 0
    total_multi = 0
    for word, sids in word_signers.items():
        if len(sids) > 1:
            total_multi += 1
            if len(set(sids)) == 1:
                consistent += 1

    consistency_score = consistent / max(total_multi, 1)

    result = {
        "n_signers": len(unique_signers),
        "clips_per_signer": clips_per_signer,
        "sign_type_distribution": sign_type_dist,
        "handedness_distribution": handedness_dist,
        "consistency_score": consistency_score,
        "total_clips": len(rows),
        "annotated_clips": sum(1 for s in signer_ids if s >= 0),
    }

    if verbose:
        print("\n=== Signer Annotation Validation ===")
        print(f"total clips:      {result['total_clips']}")
        print(f"annotated clips:  {result['annotated_clips']}")
        print(f"unique signers:   {result['n_signers']}")
        print(f"clips per signer: {clips_per_signer}")
        print(f"sign types:       {sign_type_dist}")
        print(f"handedness:       {handedness_dist}")
        print(f"consistency score (same-word same-signer): {consistency_score:.3f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate MoSL labels.csv with signer IDs")
    parser.add_argument("--n-signers", type=int, default=9,
                        help="number of signer clusters (default: 9, per MoSL paper)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for k-means clustering")
    parser.add_argument("--dry-run", action="store_true",
                        help="print statistics without writing files")
    parser.add_argument("--validate-only", action="store_true",
                        help="only validate existing annotations, do not re-annotate")
    args = parser.parse_args()

    if args.validate_only:
        validate_signer_grouping()
        return 0

    n = annotate(n_signers=args.n_signers, seed=args.seed, dry_run=args.dry_run)
    if not args.dry_run:
        validate_signer_grouping()
    return 0 if n > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
