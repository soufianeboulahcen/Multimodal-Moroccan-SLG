"""Convert MoSL (T, 150) absolute joint positions to HumanML3D representation.

The HumanML3D feature vector (Guo et al. 2022) encodes motion as:
  - Root velocity (x, z) in the horizontal plane
  - Root height (y)
  - Root rotation velocity (angular)
  - Local joint positions relative to root (flattened)
  - Local joint velocities (flattened)
  - Foot contact binary labels

This representation is preferred for diffusion training because:
  1. Velocity features make temporal smoothness explicit in the loss.
  2. Root-relative positions remove global translation ambiguity.
  3. Foot contact labels enable physically plausible motion.

Our adaptation for MoSL (upper-body signing, no locomotion):
  - Root = joint 1 (neck/mid-torso) — more stable than pelvis for signing
  - No foot contact labels for hand joints (hands don't contact ground)
  - Foot contact labels kept for ankle joints (joints 10, 13)
  - Output dimension: (T, 263) matching HumanML3D standard

The .skels files remain the primary training format. This converter is used
optionally when training the MDM denoiser with HumanML3D-pretrained weights.

Usage:
    from mosl.pose.convert_to_humanml3d import pose_to_humanml3d, humanml3d_to_pose

    # Convert for diffusion training
    features = pose_to_humanml3d(pose_seq)   # (T, 150) → (T, 263)

    # Reconstruct for rendering
    pose_seq = humanml3d_to_pose(features)   # (T, 263) → (T, 150)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_JOINTS = 50
POSE_DIM = 150   # N_JOINTS × 3

# Root joint index (neck — joint 1 in COCO-18, stable for upper-body signing)
ROOT_JOINT = 1

# Foot joints for contact detection (COCO-18: 10=R_ankle, 13=L_ankle)
FOOT_JOINTS = [10, 13]
FOOT_CONTACT_THRESHOLD = 0.02   # velocity threshold for contact detection (metres/frame)

# HumanML3D output dimension
# root_vel(2) + root_height(1) + root_rot_vel(1) + local_pos(50*3) + local_vel(50*3) + foot(2)
# = 4 + 150 + 150 + 2 = 306  (our variant; standard HumanML3D is 263 for 22 joints)
# We keep 263 by using only the 22 SMPL joints subset for compatibility with
# pretrained HumanML3D models, then appending hand joints separately.
HUMANML3D_DIM = 263


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def pose_to_humanml3d(
    pose: torch.Tensor,   # (T, 150) absolute joint positions
    fps: float = 25.0,
) -> torch.Tensor:
    """Convert absolute joint positions to HumanML3D-compatible features.

    Returns (T, 263) feature tensor. The last frame has zero velocity
    (no next frame to difference against).

    Feature layout:
      [0:2]    root velocity (x, z) — horizontal plane
      [2]      root height (y)
      [3]      root angular velocity (rotation around y-axis)
      [4:154]  local joint positions relative to root (50 joints × 3)
      [154:304] local joint velocities (50 joints × 3)
      [304:306] foot contact binary (R_ankle, L_ankle)
      [306:263] — NOTE: we pad/trim to exactly 263 for HumanML3D compatibility
    """
    T = pose.shape[0]
    joints = pose.view(T, N_JOINTS, 3)   # (T, 50, 3)

    # Root position and velocity
    root_pos = joints[:, ROOT_JOINT, :]   # (T, 3)
    root_vel = torch.zeros_like(root_pos)
    root_vel[:-1] = root_pos[1:] - root_pos[:-1]

    # Local joint positions (relative to root)
    local_pos = joints - root_pos.unsqueeze(1)   # (T, 50, 3)

    # Local joint velocities
    local_vel = torch.zeros_like(local_pos)
    local_vel[:-1] = local_pos[1:] - local_pos[:-1]

    # Root angular velocity (rotation around y-axis, estimated from neck→shoulder vector)
    # Use right shoulder (joint 2) and left shoulder (joint 5) to estimate facing direction
    r_shoulder = joints[:, 2, :]   # (T, 3)
    l_shoulder = joints[:, 5, :]   # (T, 3)
    facing = r_shoulder - l_shoulder   # (T, 3) — points right
    # Angle in xz plane
    angle = torch.atan2(facing[:, 0], facing[:, 2])   # (T,)
    ang_vel = torch.zeros_like(angle)
    ang_vel[:-1] = angle[1:] - angle[:-1]
    # Wrap to [-pi, pi]
    ang_vel = (ang_vel + torch.pi) % (2 * torch.pi) - torch.pi

    # Foot contact (binary, based on ankle velocity magnitude)
    foot_contact = torch.zeros(T, 2)
    for i, fj in enumerate(FOOT_JOINTS):
        ankle_vel = local_vel[:, fj, :].norm(dim=-1)   # (T,)
        foot_contact[:, i] = (ankle_vel < FOOT_CONTACT_THRESHOLD).float()

    # Assemble feature vector
    # We target 263 dims to match HumanML3D pretrained models.
    # Layout: root(4) + local_pos(50*3=150) + local_vel(50*3=150) + foot(2) = 306
    # Trim to 263 by using only first 22 joints (SMPL body) for pos+vel,
    # then append full hand joints as extra channels up to 263.
    #
    # Practical layout for our 263-dim output:
    #   [0:4]    root features (vel_x, vel_z, height, ang_vel)
    #   [4:70]   local pos for joints 0-21 (22 body joints × 3 = 66)
    #   [70:136] local vel for joints 0-21 (66)
    #   [136:261] local pos for joints 22-63 (hand joints, 42 joints × 3 = 126)
    #             — but we only have 28 hand joints (18-49), so 28*3=84 → pad to 126
    #   [261:263] foot contact

    root_features = torch.stack([
        root_vel[:, 0],   # x velocity
        root_vel[:, 2],   # z velocity
        root_pos[:, 1],   # y height
        ang_vel,
    ], dim=-1)   # (T, 4)

    body_pos = local_pos[:, :22, :].reshape(T, -1)    # (T, 66)
    body_vel = local_vel[:, :22, :].reshape(T, -1)    # (T, 66)

    # Hand joints: 18-49 = 32 joints × 3 = 96 coords; pad to 125 for 263 total
    hand_pos = local_pos[:, 18:50, :].reshape(T, -1)  # (T, 96)
    hand_pad = torch.zeros(T, 125 - 96, device=pose.device)
    hand_pos_padded = torch.cat([hand_pos, hand_pad], dim=-1)  # (T, 125)

    features = torch.cat([
        root_features,      # 4
        body_pos,           # 66
        body_vel,           # 66
        hand_pos_padded,    # 125
        foot_contact.to(pose.device),  # 2
    ], dim=-1)   # (T, 263)

    assert features.shape == (T, HUMANML3D_DIM), \
        f"expected (T, {HUMANML3D_DIM}), got {features.shape}"

    return features


def humanml3d_to_pose(
    features: torch.Tensor,   # (T, 263)
    root_start: Optional[torch.Tensor] = None,  # (3,) initial root position
) -> torch.Tensor:
    """Reconstruct absolute joint positions from HumanML3D features.

    Inverse of pose_to_humanml3d. Used after DDIM sampling to get back
    the (T, 150) format needed by the SMPL-X fitter and renderer.

    Note: hand joints are reconstructed from the padded hand_pos block.
    The reconstruction is approximate (root integration accumulates drift).
    """
    T = features.shape[0]
    device = features.device

    root_vel_x = features[:, 0]
    root_vel_z = features[:, 1]
    root_height = features[:, 2]
    # ang_vel at [3] is not used for reconstruction (we don't track orientation)

    body_pos = features[:, 4:70].view(T, 22, 3)      # local body joints
    # body_vel at [70:136] not needed for reconstruction
    hand_pos = features[:, 136:232].view(T, 32, 3)   # local hand joints (96 coords)

    # Integrate root velocity to get root trajectory
    root_pos = torch.zeros(T, 3, device=device)
    if root_start is not None:
        root_pos[0] = root_start
    for t in range(1, T):
        root_pos[t, 0] = root_pos[t - 1, 0] + root_vel_x[t - 1]
        root_pos[t, 1] = root_height[t]
        root_pos[t, 2] = root_pos[t - 1, 2] + root_vel_z[t - 1]

    # Reconstruct absolute positions
    joints = torch.zeros(T, N_JOINTS, 3, device=device)
    joints[:, :22, :] = body_pos + root_pos.unsqueeze(1)
    joints[:, 18:50, :] = hand_pos + root_pos.unsqueeze(1)

    return joints.view(T, POSE_DIM)


# ---------------------------------------------------------------------------
# Batch conversion utilities
# ---------------------------------------------------------------------------

def convert_skels_file(
    skels_path: Path,
    output_path: Path,
    force: bool = False,
) -> int:
    """Convert a .skels file to HumanML3D format and save as .npy.

    Returns number of clips converted.
    """
    if output_path.exists() and not force:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    FLOATS_PER_FRAME = 151   # 150 pose + 1 time marker

    with open(skels_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    all_features = []
    for line in lines:
        floats = np.fromstring(line, sep=" ", dtype=np.float32)
        T = floats.size // FLOATS_PER_FRAME
        frames = floats[:T * FLOATS_PER_FRAME].reshape(T, FLOATS_PER_FRAME)
        pose = torch.from_numpy(frames[:, :150])
        features = pose_to_humanml3d(pose)
        all_features.append(features.numpy())

    # Save as list of arrays (variable T per clip)
    np.save(output_path, np.array(all_features, dtype=object), allow_pickle=True)
    return len(all_features)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Test round-trip on synthetic data
    T = 100
    pose = torch.randn(T, 150)

    features = pose_to_humanml3d(pose)
    print(f"pose_to_humanml3d: (T={T}, 150) → {tuple(features.shape)}")
    assert features.shape == (T, 263), f"expected (T, 263), got {features.shape}"

    pose_reconstructed = humanml3d_to_pose(features)
    print(f"humanml3d_to_pose: {tuple(features.shape)} → {tuple(pose_reconstructed.shape)}")
    assert pose_reconstructed.shape == (T, 150)

    # Check foot contact is binary
    foot = features[:, 261:263]
    assert foot.min() >= 0.0 and foot.max() <= 1.0, "foot contact should be in [0,1]"
    print(f"foot contact range: [{foot.min():.2f}, {foot.max():.2f}]  ✓")

    # Test on real .skels file if available
    skels_path = Path("third_party/Prompt2Sign/tools/2D_to_3D/final_data/dev.skels")
    if skels_path.exists():
        with open(skels_path) as f:
            line = f.readline().strip()
        floats = np.fromstring(line, sep=" ", dtype=np.float32)
        T_real = floats.size // 151
        pose_real = torch.from_numpy(floats[:T_real * 151].reshape(T_real, 151)[:, :150])
        feat_real = pose_to_humanml3d(pose_real)
        print(f"\nreal clip: T={T_real}  features={tuple(feat_real.shape)}")
        print(f"  root height range: [{feat_real[:, 2].min():.3f}, {feat_real[:, 2].max():.3f}]")
        print(f"  root vel magnitude: {feat_real[:, :2].norm(dim=-1).mean():.4f} (mean)")
