"""Quick sanity check on extracted keypoint NPZ files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(category: str) -> int:
    d = ROOT / "data" / "processed" / "keypoints_2d" / category
    files = sorted(d.glob("*.npz"))
    print(f"category={category}  files={len(files)}")
    for f in files[:5]:
        z = np.load(f)
        pose = z["pose_keypoints_2d"]
        hl = z["hand_left_keypoints_2d"]
        hr = z["hand_right_keypoints_2d"]
        nz_pose = (pose[..., 2::3] > 0).sum() / max(pose.shape[0] * 18, 1)
        nz_hl = (hl[..., 2::3] > 0).sum() / max(hl.shape[0] * 21, 1)
        nz_hr = (hr[..., 2::3] > 0).sum() / max(hr.shape[0] * 21, 1)
        print(
            f"  {f.stem[:30]:<32} "
            f"T={pose.shape[0]:>3d} "
            f"body_nz={nz_pose:.0%} "
            f"Lhand_nz={nz_hl:.0%} "
            f"Rhand_nz={nz_hr:.0%} "
            f"fps={float(z['fps']):.1f}"
        )
    return 0


if __name__ == "__main__":
    cat = sys.argv[1] if len(sys.argv) > 1 else "Pronouns"
    raise SystemExit(main(cat))
