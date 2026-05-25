"""Fit SMPL-X parameters to OpenPose-derived 3D joint sequences.

Converts the existing (T, 150) pose format — 50 joints × xyz produced by the
Prompt2Sign kinematic optimizer — into SMPL-X body/hand/face parameters
suitable for avatar animation and neural rendering.

Pipeline per clip:
  (T, 150) OpenPose 3D joints  [from .skels files or NPZ]
      → joint-to-SMPL-X correspondence mapping
      → Stage 1: shape fitting (betas) on mean pose (100 iters)
      → Stage 2: per-frame pose fitting (theta) with fixed betas (50 iters)
      → MANO hand pose fitting (left + right)
      → optional FLAME face fitting (requires face_keypoints in NPZ)
      → output: (T, {theta, beta, transl, left_hand_pose, right_hand_pose, expression})

Output NPZ format (saved to data/processed/smplx_params/<category>/<clip>.npz):
    body_pose       (T, 63)   — 21 body joints × axis-angle (3 each)
    global_orient   (T, 3)    — root orientation axis-angle
    betas           (10,)     — shape parameters (per-clip, not per-frame)
    transl          (T, 3)    — root translation
    left_hand_pose  (T, 45)   — MANO left hand (15 joints × 3)
    right_hand_pose (T, 45)   — MANO right hand (15 joints × 3)
    expression      (T, 100)  — FLAME expression (zeros if no face data)
    jaw_pose        (T, 3)    — jaw rotation

Requires: pip install smplx torch
SMPL-X model files must be downloaded from https://smpl-x.is.tue.mpg.de/
and placed at: data/smplx_models/smplx/SMPLX_NEUTRAL.npz

Joint correspondence (OpenPose COCO-18 -> SMPL-X body joints):
  The Prompt2Sign 3D optimizer outputs 50 joints. The first 18 correspond
  to COCO-18 body keypoints. We map these to the nearest SMPL-X joints
  using the standard COCO->SMPL-X correspondence table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# COCO-18 -> SMPL-X joint correspondence
# ---------------------------------------------------------------------------

COCO18_TO_SMPLX = {
    0: 15,   # nose -> head
    1: 12,   # neck -> neck
    2: 17,   # R shoulder -> R_shoulder
    3: 19,   # R elbow -> R_elbow
    4: 21,   # R wrist -> R_wrist
    5: 16,   # L shoulder -> L_shoulder
    6: 18,   # L elbow -> L_elbow
    7: 20,   # L wrist -> L_wrist
    8: 2,    # R hip -> R_hip
    9: 5,    # R knee -> R_knee
    10: 8,   # R ankle -> R_ankle
    11: 1,   # L hip -> L_hip
    12: 4,   # L knee -> L_knee
    13: 7,   # L ankle -> L_ankle
}

# SMPL-X joint layout:
#   body: 0-21, jaw: 22, leye: 23, reye: 24
#   left_hand: 25-45 (21 joints), right_hand: 46-66 (21 joints)
SMPLX_LHAND_START = 25
SMPLX_RHAND_START = 46


# ---------------------------------------------------------------------------
# Input format helpers
# ---------------------------------------------------------------------------

def load_joints_from_skels_line(line: str) -> np.ndarray:
    """Parse one line from a .skels file -> (T, 150) float32.

    Each line is T x 151 floats: 150 pose coords + 1 time marker per frame.
    """
    vals = np.fromstring(line.strip(), dtype=np.float32, sep=" ")
    if vals.size == 0:
        return np.zeros((0, 150), dtype=np.float32)
    n_per_frame = 151
    T = vals.size // n_per_frame
    vals = vals[: T * n_per_frame].reshape(T, n_per_frame)
    return vals[:, :150]   # drop time marker


def load_joints_from_npz(npz_path: Path) -> Optional[np.ndarray]:
    """Load 3D joints from an NPZ file.

    Returns (T, 150) if the NPZ has 3D data, or builds a (T, 150) array
    from 2D keypoints with z=0 padding as fallback.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None

    if "joints_3d" in data:
        return data["joints_3d"].astype(np.float32)

    body = data.get("pose_keypoints_2d")
    if body is None:
        return None

    T = body.shape[0]
    body_xy = body.reshape(T, 18, 3)[:, :, :2]   # (T, 18, 2)

    joints = np.zeros((T, 150), dtype=np.float32)
    for j in range(18):
        joints[:, j * 3] = body_xy[:, j, 0]
        joints[:, j * 3 + 1] = body_xy[:, j, 1]

    lhand = data.get("hand_left_keypoints_2d")
    rhand = data.get("hand_right_keypoints_2d")
    if lhand is not None and lhand.shape[0] == T:
        lh = lhand.reshape(T, 21, 3)[:, :, :2]
        for j in range(21):
            base = (18 + j) * 3
            joints[:, base] = lh[:, j, 0]
            joints[:, base + 1] = lh[:, j, 1]
    if rhand is not None and rhand.shape[0] == T:
        rh = rhand.reshape(T, 21, 3)[:, :, :2]
        for j in range(min(11, 21)):
            base = (39 + j) * 3
            joints[:, base] = rh[:, j, 0]
            joints[:, base + 1] = rh[:, j, 1]

    return joints


# ---------------------------------------------------------------------------
# SMPL-X fitter
# ---------------------------------------------------------------------------

class SMPLXFitter:
    """Fits SMPL-X parameters to a sequence of 3D joint positions."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        gender: str = "neutral",
        n_betas: int = 10,
        n_expression: int = 100,
        use_pca: bool = False,
    ) -> None:
        try:
            import smplx
        except ImportError:
            raise ImportError(
                "smplx is required for SMPL-X fitting.\n"
                "Install with: pip install smplx\n"
                "Download model files from: https://smpl-x.is.tue.mpg.de/"
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = smplx.create(
            model_path,
            model_type="smplx",
            gender=gender,
            num_betas=n_betas,
            use_pca=use_pca,
            num_expression_coeffs=n_expression,
            flat_hand_mean=False,
        ).to(self.device)

        self.n_betas = n_betas
        self.n_expression = n_expression

    def fit_sequence(
        self,
        joints_3d: np.ndarray,                          # (T, 150)
        face_keypoints: Optional[np.ndarray] = None,    # (T, 478, 3) MediaPipe
        n_iters_shape: int = 100,
        n_iters_pose: int = 50,
        lr: float = 0.01,
        chunk_size: int = 16,
        verbose: bool = False,
    ) -> dict:
        """Fit SMPL-X to a full clip. Returns dict of numpy arrays."""
        T = joints_3d.shape[0]
        joints_t = torch.from_numpy(joints_3d).float().to(self.device)
        joints_reshaped = joints_t.view(T, 50, 3)

        body_joints_obs = joints_reshaped[:, :18, :]
        lhand_joints_obs = joints_reshaped[:, 18:39, :]
        rhand_joints_obs = joints_reshaped[:, 39:50, :]

        # Stage 1: shape fitting
        betas = torch.zeros(1, self.n_betas, device=self.device, requires_grad=True)
        mean_body = body_joints_obs.mean(dim=0)

        opt_shape = torch.optim.Adam([betas], lr=lr)
        for _ in range(n_iters_shape):
            opt_shape.zero_grad()
            output = self.model(betas=betas, return_verts=False)
            smplx_j = output.joints[0]
            loss = self._body_joint_loss(smplx_j, mean_body)
            loss = loss + 0.01 * betas.pow(2).sum()
            loss.backward()
            opt_shape.step()

        betas_fit = betas.detach()

        # Stage 2: per-frame pose fitting
        body_pose = torch.zeros(T, 63, device=self.device, requires_grad=True)
        global_orient = torch.zeros(T, 3, device=self.device, requires_grad=True)
        transl = torch.zeros(T, 3, device=self.device, requires_grad=True)
        lhand_pose = torch.zeros(T, 45, device=self.device, requires_grad=True)
        rhand_pose = torch.zeros(T, 45, device=self.device, requires_grad=True)
        expression = torch.zeros(T, self.n_expression, device=self.device, requires_grad=True)
        jaw_pose = torch.zeros(T, 3, device=self.device, requires_grad=True)

        params = [body_pose, global_orient, transl, lhand_pose, rhand_pose]
        if face_keypoints is not None:
            params += [expression, jaw_pose]

        opt_pose = torch.optim.Adam(params, lr=lr * 0.5)

        for it in range(n_iters_pose):
            opt_pose.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                B = end - start
                chunk_betas = betas_fit.expand(B, -1)

                kw = dict(
                    betas=chunk_betas,
                    body_pose=body_pose[start:end],
                    global_orient=global_orient[start:end],
                    transl=transl[start:end],
                    left_hand_pose=lhand_pose[start:end],
                    right_hand_pose=rhand_pose[start:end],
                    return_verts=False,
                )
                if face_keypoints is not None:
                    kw["expression"] = expression[start:end]
                    kw["jaw_pose"] = jaw_pose[start:end]

                output = self.model(**kw)
                smplx_j = output.joints

                chunk_loss = self._body_joint_loss_batch(smplx_j, body_joints_obs[start:end])
                chunk_loss = chunk_loss + self._hand_joint_loss(
                    smplx_j, lhand_joints_obs[start:end], hand="left"
                )
                chunk_loss = chunk_loss + self._hand_joint_loss(
                    smplx_j, rhand_joints_obs[start:end], hand="right"
                )
                chunk_loss = chunk_loss + 0.001 * (
                    body_pose[start:end].pow(2).sum()
                    + lhand_pose[start:end].pow(2).sum()
                    + rhand_pose[start:end].pow(2).sum()
                )
                total_loss = total_loss + chunk_loss

            total_loss.backward()
            opt_pose.step()

            if verbose and it % 10 == 0:
                print(f"  pose iter {it:3d}  loss={total_loss.item():.4f}")

        return {
            "body_pose": body_pose.detach().cpu().numpy().astype(np.float32),
            "global_orient": global_orient.detach().cpu().numpy().astype(np.float32),
            "betas": betas_fit.squeeze(0).cpu().numpy().astype(np.float32),
            "transl": transl.detach().cpu().numpy().astype(np.float32),
            "left_hand_pose": lhand_pose.detach().cpu().numpy().astype(np.float32),
            "right_hand_pose": rhand_pose.detach().cpu().numpy().astype(np.float32),
            "expression": expression.detach().cpu().numpy().astype(np.float32),
            "jaw_pose": jaw_pose.detach().cpu().numpy().astype(np.float32),
        }

    def _body_joint_loss(self, smplx_joints, obs_joints):
        loss = torch.tensor(0.0, device=self.device)
        for coco_idx, smplx_idx in COCO18_TO_SMPLX.items():
            if smplx_idx < smplx_joints.shape[0]:
                loss = loss + F.mse_loss(smplx_joints[smplx_idx], obs_joints[coco_idx])
        return loss

    def _body_joint_loss_batch(self, smplx_joints, obs_joints):
        loss = torch.tensor(0.0, device=self.device)
        for coco_idx, smplx_idx in COCO18_TO_SMPLX.items():
            if smplx_idx < smplx_joints.shape[1]:
                loss = loss + F.mse_loss(
                    smplx_joints[:, smplx_idx, :], obs_joints[:, coco_idx, :]
                )
        return loss

    def _hand_joint_loss(self, smplx_joints, obs_hand, hand="left"):
        start = SMPLX_LHAND_START if hand == "left" else SMPLX_RHAND_START
        n_obs = obs_hand.shape[1]
        n_smplx = min(n_obs, 21)
        if start + n_smplx > smplx_joints.shape[1]:
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(
            smplx_joints[:, start:start + n_smplx, :],
            obs_hand[:, :n_smplx, :],
        )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_clip_from_npz(
    npz_path: Path,
    output_dir: Path,
    fitter: SMPLXFitter,
    force: bool = False,
    verbose: bool = False,
) -> str:
    """Fit SMPL-X to one clip from its NPZ file. Returns status string."""
    out_path = output_dir / npz_path.parent.name / npz_path.name
    if out_path.exists() and not force:
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    joints_3d = load_joints_from_npz(npz_path)
    if joints_3d is None or joints_3d.shape[0] < 2:
        return "no_data"

    try:
        data = np.load(npz_path, allow_pickle=False)
        face_kps = data.get("face_keypoints", None)
        params = fitter.fit_sequence(joints_3d, face_keypoints=face_kps, verbose=verbose)
        np.savez_compressed(out_path, **params)
        return "ok"
    except Exception as e:
        if verbose:
            print(f"  [FAIL] {npz_path.name}: {e}")
        return f"failed"


def batch_fit_from_npz(
    model_path: str,
    categories: Optional[list[str]] = None,
    force: bool = False,
    device: str = "cuda",
    repo_root: Optional[Path] = None,
) -> None:
    """Fit SMPL-X to all clips using NPZ keypoints as input."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    keypoints_dir = repo_root / "data" / "processed" / "keypoints_2d"
    output_dir = repo_root / "data" / "processed" / "smplx_params"

    fitter = SMPLXFitter(model_path, device=device)

    available = sorted(d.name for d in keypoints_dir.iterdir() if d.is_dir())
    targets = categories or available

    total: dict[str, int] = {}
    t0 = time.time()

    for cat in targets:
        npz_files = sorted((keypoints_dir / cat).glob("*.npz"))
        print(f"[{cat}] {len(npz_files)} clips")
        for npz_path in npz_files:
            status = process_clip_from_npz(npz_path, output_dir, fitter, force=force)
            total[status] = total.get(status, 0) + 1
            if status == "ok":
                print(f"  [OK] {npz_path.name}")
            elif status == "failed":
                print(f"  [FAIL] {npz_path.name}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit SMPL-X parameters to MoSL OpenPose sequences"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to SMPL-X model directory (containing SMPLX_NEUTRAL.npz). "
             "Download from https://smpl-x.is.tue.mpg.de/"
    )
    parser.add_argument("--category", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_fit_from_npz(
        model_path=args.model_path,
        categories=args.category,
        force=args.force,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
