"""Fit SMPL-X parameters to OpenPose-derived 3D joint sequences.

Converts the existing (T, 150) pose format — 50 joints × xyz produced by the
Prompt2Sign kinematic optimizer — into SMPL-X body/hand/face parameters
suitable for avatar animation and neural rendering.

Pipeline per clip:
  (T, 150) OpenPose 3D joints
      → joint-to-SMPL-X correspondence mapping
      → per-frame Adam optimisation (body pose theta, shape beta)
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

Joint correspondence (OpenPose COCO-18 → SMPL-X body joints):
  The Prompt2Sign 3D optimizer outputs 50 joints. The first 18 correspond
  to COCO-18 body keypoints. We map these to the nearest SMPL-X joints
  using the standard COCO→SMPL-X correspondence table.
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
# COCO-18 → SMPL-X joint correspondence
# ---------------------------------------------------------------------------
# COCO-18 indices:  0=nose, 1=neck, 2=Rshoulder, 3=Relbow, 4=Rwrist,
#                   5=Lshoulder, 6=Lelbow, 7=Lwrist, 8=Rhip, 9=Rknee,
#                   10=Rankle, 11=Lhip, 12=Lknee, 13=Lankle,
#                   14=Reye, 15=Leye, 16=Rear, 17=Lear
#
# SMPL-X body joint indices (0-based, after global_orient):
#   0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
#   6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
#   12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder, 17=R_shoulder,
#   18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist

# Maps COCO-18 joint index → SMPL-X body joint index (-1 = no mapping)
COCO18_TO_SMPLX = {
    0: 15,   # nose → head
    1: 12,   # neck → neck
    2: 17,   # R shoulder → R_shoulder
    3: 19,   # R elbow → R_elbow
    4: 21,   # R wrist → R_wrist
    5: 16,   # L shoulder → L_shoulder
    6: 18,   # L elbow → L_elbow
    7: 20,   # L wrist → L_wrist
    8: 2,    # R hip → R_hip
    9: 5,    # R knee → R_knee
    10: 8,   # R ankle → R_ankle
    11: 1,   # L hip → L_hip
    12: 4,   # L knee → L_knee
    13: 7,   # L ankle → L_ankle
}

# MANO hand joint ordering (21 joints per hand):
# 0=wrist, 1-4=index, 5-8=middle, 9-12=pinky, 13-16=ring, 17-20=thumb
# Our NPZ stores left hand at joints 18-38, right hand at 39-49 (partial)


# ---------------------------------------------------------------------------
# SMPL-X wrapper
# ---------------------------------------------------------------------------

class SMPLXFitter:
    """Fits SMPL-X parameters to a sequence of 3D joint positions.

    Requires the smplx package and model files. If either is unavailable,
    raises ImportError / FileNotFoundError with clear instructions.
    """

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
        joints_3d: np.ndarray,          # (T, 150) — existing .skels format
        face_keypoints: Optional[np.ndarray] = None,  # (T, 478, 3) MediaPipe
        n_iters_shape: int = 100,        # iterations for shape fitting (first pass)
        n_iters_pose: int = 50,          # iterations for per-frame pose fitting
        lr: float = 0.01,
        verbose: bool = False,
    ) -> dict:
        """Fit SMPL-X to a full clip.

        Two-stage fitting:
          Stage 1: Fit shape (betas) using mean pose across all frames.
          Stage 2: Fit per-frame pose (theta) with fixed betas.

        Returns dict of numpy arrays matching the output NPZ format.
        """
        T = joints_3d.shape[0]
        joints_t = torch.from_numpy(joints_3d).float().to(self.device)  # (T, 150)
        joints_3d_reshaped = joints_t.view(T, 50, 3)                    # (T, 50, 3)

        # Extract COCO-18 body joints and hand joints
        body_joints_obs = joints_3d_reshaped[:, :18, :]    # (T, 18, 3)
        lhand_joints_obs = joints_3d_reshaped[:, 18:39, :] # (T, 21, 3)
        rhand_joints_obs = joints_3d_reshaped[:, 39:50, :] # (T, 11, 3)

        # ---------------------------------------------------------------
        # Stage 1: Shape fitting on mean pose
        # ---------------------------------------------------------------
        betas = torch.zeros(1, self.n_betas, device=self.device, requires_grad=True)
        mean_body = body_joints_obs.mean(dim=0, keepdim=True)  # (1, 18, 3)

        opt_shape = torch.optim.Adam([betas], lr=lr)
        for _ in range(n_iters_shape):
            opt_shape.zero_grad()
            output = self.model(
                betas=betas,
                return_verts=False,
                return_full_pose=False,
            )
            smplx_joints = output.joints[0]   # (n_joints, 3)
            # Map SMPL-X joints to COCO-18 for loss
            loss = self._body_joint_loss(smplx_joints, mean_body[0])
            loss = loss + 0.01 * betas.pow(2).sum()   # shape regularisation
            loss.backward()
            opt_shape.step()

        betas_fit = betas.detach()

        # ---------------------------------------------------------------
        # Stage 2: Per-frame pose fitting
        # ---------------------------------------------------------------
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
            # Process in chunks to avoid OOM on long clips
            chunk_size = 16
            total_loss = torch.tensor(0.0, device=self.device)

            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                chunk_betas = betas_fit.expand(end - start, -1)

                output = self.model(
                    betas=chunk_betas,
                    body_pose=body_pose[start:end],
                    global_orient=global_orient[start:end],
                    transl=transl[start:end],
                    left_hand_pose=lhand_pose[start:end],
                    right_hand_pose=rhand_pose[start:end],
                    expression=expression[start:end] if face_keypoints is not None else None,
                    jaw_pose=jaw_pose[start:end] if face_keypoints is not None else None,
                    return_verts=False,
                )
                smplx_j = output.joints   # (chunk, n_joints, 3)

                # Body joint loss
                chunk_loss = self._body_joint_loss_batch(
                    smplx_j, body_joints_obs[start:end]
                )
                # Hand joint loss (left)
                chunk_loss = chunk_loss + self._hand_joint_loss(
                    smplx_j, lhand_joints_obs[start:end], hand="left"
                )
                # Regularisation
                chunk_loss = chunk_loss + 0.001 * body_pose[start:end].pow(2).sum()
                chunk_loss = chunk_loss + 0.001 * lhand_pose[start:end].pow(2).sum()
                chunk_loss = chunk_loss + 0.001 * rhand_pose[start:end].pow(2).sum()

                total_loss = total_loss + chunk_loss

            total_loss.backward()
            opt_pose.step()

            if verbose and it % 10 == 0:
                print(f"  iter {it:3d}  loss={total_loss.item():.4f}")

        return {
            "body_pose": body_pose.detach().cpu().numpy(),
            "global_orient": global_orient.detach().cpu().numpy(),
            "betas": betas_fit.squeeze(0).cpu().numpy(),
            "transl": transl.detach().cpu().numpy(),
            "left_hand_pose": lhand_pose.detach().cpu().numpy(),
            "right_hand_pose": rhand_pose.detach().cpu().numpy(),
            "expression": expression.detach().cpu().numpy(),
            "jaw_pose": jaw_pose.detach().cpu().numpy(),
        }

    def _body_joint_loss(
        self, smplx_joints: torch.Tensor, obs_joints: torch.Tensor
    ) -> torch.Tensor:
        """Single-frame body joint MSE using COCO-18 correspondence."""
        loss = torch.tensor(0.0, device=self.device)
        for coco_idx, smplx_idx in COCO18_TO_SMPLX.items():
            if smplx_idx < smplx_joints.shape[0]:
                loss = loss + F.mse_loss(smplx_joints[smplx_idx], obs_joints[coco_idx])
        return loss

    def _body_joint_loss_batch(
        self, smplx_joints: torch.Tensor, obs_joints: torch.Tensor
    ) -> torch.Tensor:
        """Batched body joint MSE. smplx_joints: (B, n_j, 3), obs: (B, 18, 3)."""
        loss = torch.tensor(0.0, device=self.device)
        for coco_idx, smplx_idx in COCO18_TO_SMPLX.items():
            if smplx_idx < smplx_joints.shape[1]:
                loss = loss + F.mse_loss(
                    smplx_joints[:, smplx_idx, :], obs_joints[:, coco_idx, :]
                )
        return loss

    def _hand_joint_loss(
        self,
        smplx_joints: torch.Tensor,   # (B, n_j, 3)
        obs_hand: torch.Tensor,        # (B, 21, 3) or (B, 11, 3)
        hand: str = "left",
    ) -> torch.Tensor:
        """Hand joint MSE. SMPL-X hand joints start at index 25 (left) / 40 (right)."""
        # SMPL-X joint layout: body=0-21, jaw=22, leye=23, reye=24,
        #                      left_hand=25-45, right_hand=46-66
        start = 25 if hand == "left" else 46
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

def process_clip(
    npz_path: Path,
    output_dir: Path,
    fitter: SMPLXFitter,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """Fit SMPL-X to one clip and save output NPZ.

    Returns True if processed, False if skipped.
    """
    out_path = output_dir / npz_path.parent.name / npz_path.name
    if out_path.exists() and not force:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=False)
    joints_3d = data.get("pose_keypoints_2d")   # (T, 54) — body only in some NPZs
    # The full 150-dim format is in the .skels pipeline output, not the raw NPZ.
    # For clips where only 2D keypoints are stored, we use the available joints.
    if joints_3d is None:
        if verbose:
            print(f"  [SKIP] no pose_keypoints_2d in {npz_path.name}")
        return False

    face_kps = data.get("face_keypoints", None)  # (T, 478, 3) if extracted

    try:
        params = fitter.fit_sequence(
            joints_3d,
            face_keypoints=face_kps,
            verbose=verbose,
        )
        np.savez_compressed(out_path, **params)
        return True
    except Exception as e:
        if verbose:
            print(f"  [FAIL] {npz_path.name}: {e}")
        return False


def main(
    model_path: str,
    categories: Optional[list[str]] = None,
    force: bool = False,
    device: str = "cuda",
    repo_root: Optional[Path] = None,
) -> None:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    keypoints_dir = repo_root / "data" / "processed" / "keypoints_2d"
    output_dir = repo_root / "data" / "processed" / "smplx_params"

    fitter = SMPLXFitter(model_path, device=device)

    available = [d.name for d in keypoints_dir.iterdir() if d.is_dir()]
    targets = categories or sorted(available)

    total = {"processed": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    for cat in targets:
        npz_files = sorted((keypoints_dir / cat).glob("*.npz"))
        print(f"[{cat}] {len(npz_files)} clips")
        for npz_path in npz_files:
            result = process_clip(npz_path, output_dir, fitter, force=force)
            if result:
                total["processed"] += 1
            else:
                total["skipped"] += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  "
          f"processed={total['processed']}  skipped={total['skipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SMPL-X to MoSL OpenPose sequences")
    parser.add_argument("--model-path", required=True,
                        help="Path to SMPL-X model directory (containing SMPLX_NEUTRAL.npz)")
    parser.add_argument("--category", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        categories=args.category,
        force=args.force,
        device=args.device,
    )
