"""Diffusion-specific losses for MoSL motion generation.

Extends the existing mosl/train/losses.py (which handles SignLLM MSE/RL/PLC)
with motion-quality losses needed for the MDM denoiser.

All losses operate on (B, T, 150) pose tensors in the same coordinate system
as the existing .skels files — no format conversion required.

Joint layout (50 joints × xyz = 150 coords):
  Joints  0-17  → body (COCO-18 subset, coords  0-53)
  Joints 18-38  → left hand (MANO 21 joints, coords 54-116)
  Joints 39-49  → right hand (partial MANO, coords 117-149)

Hand joints receive a configurable upweight (default 3×) because hands carry
the primary lexical content of sign language.

Bone connectivity is defined for the COCO-18 body subset only; hand bones
use a chain topology from the Prompt2Sign joint ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Joint / coord index constants
# ---------------------------------------------------------------------------

POSE_DIM = 150
N_JOINTS = 50

# Body joints: 0-17 (COCO-18 subset used by Prompt2Sign)
BODY_JOINT_END = 18
BODY_COORD_END = BODY_JOINT_END * 3   # 54

# Left hand: joints 18-38 (MANO 21 joints)
LHAND_JOINT_START = 18
LHAND_JOINT_END = 39
LHAND_COORD_START = LHAND_JOINT_START * 3   # 54
LHAND_COORD_END = LHAND_JOINT_END * 3       # 117

# Right hand: joints 39-49 (partial MANO from OpenPose 21→11 mapping)
RHAND_JOINT_START = 39
RHAND_JOINT_END = 50
RHAND_COORD_START = RHAND_JOINT_START * 3   # 117
RHAND_COORD_END = RHAND_JOINT_END * 3       # 150

# All hand coords combined
HAND_COORD_START = LHAND_COORD_START   # 54
HAND_COORD_END = RHAND_COORD_END       # 150


def build_hand_weight_vector(pose_dim: int = POSE_DIM, hand_weight: float = 3.0) -> torch.Tensor:
    """(pose_dim,) weight tensor: 1.0 for body coords, hand_weight for hand coords."""
    w = torch.ones(pose_dim)
    w[HAND_COORD_START:HAND_COORD_END] = hand_weight
    return w


# ---------------------------------------------------------------------------
# COCO-18 body bone pairs (joint index pairs that form bones)
# Used for bone-length consistency loss.
# ---------------------------------------------------------------------------

BODY_BONES = [
    (0, 1),   # nose → neck
    (1, 2),   # neck → right shoulder
    (2, 3),   # right shoulder → right elbow
    (3, 4),   # right elbow → right wrist
    (1, 5),   # neck → left shoulder
    (5, 6),   # left shoulder → left elbow
    (6, 7),   # left elbow → left wrist
    (1, 8),   # neck → mid hip
    (8, 9),   # mid hip → right hip
    (9, 10),  # right hip → right knee
    (10, 11), # right knee → right ankle
    (8, 12),  # mid hip → left hip
    (12, 13), # left hip → left knee
    (13, 14), # left knee → left ankle
    (0, 15),  # nose → right eye
    (0, 16),  # nose → left eye
    (15, 17), # right eye → right ear
    (16, 18), # left eye → left ear  (joint 18 is first hand joint — reuse as ear proxy)
]

# Left hand chain: wrist (18) → finger tips via MCP→PIP→DIP→TIP
# Simplified to palm-to-fingertip chains (5 fingers × 3 segments)
LHAND_BONES = [(18 + i, 18 + i + 1) for i in range(20)]   # 20 bones for 21 joints

# Right hand chain (joints 39-49, 11 joints → 10 bones)
RHAND_BONES = [(39 + i, 39 + i + 1) for i in range(10)]

ALL_BONES = BODY_BONES + LHAND_BONES + RHAND_BONES


def _bone_lengths(pose: torch.Tensor, bones: list[tuple[int, int]]) -> torch.Tensor:
    """Compute bone lengths for a pose sequence.

    Args:
        pose: (B, T, 150) or (T, 150)
        bones: list of (joint_a, joint_b) index pairs

    Returns:
        (B, T, n_bones) or (T, n_bones) bone lengths
    """
    batched = pose.dim() == 3
    if not batched:
        pose = pose.unsqueeze(0)
    B, T, _ = pose.shape
    joints = pose.view(B, T, N_JOINTS, 3)   # (B, T, 50, 3)

    lengths = []
    for a, b in bones:
        if a >= N_JOINTS or b >= N_JOINTS:
            continue
        diff = joints[:, :, a, :] - joints[:, :, b, :]   # (B, T, 3)
        lengths.append(diff.norm(dim=-1, keepdim=True))   # (B, T, 1)
    result = torch.cat(lengths, dim=-1)   # (B, T, n_bones)
    return result if batched else result.squeeze(0)


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def diffusion_mse_loss(
    x0_pred: torch.Tensor,     # (B, T, 150)
    x0_target: torch.Tensor,   # (B, T, 150)
    pose_mask: torch.Tensor,   # (B, T) True=real frame
    hand_weight: torch.Tensor, # (150,) per-coord weights
) -> torch.Tensor:
    """Weighted MSE between predicted and target clean pose, masked to real frames."""
    diff = (x0_pred - x0_target) ** 2                          # (B, T, 150)
    diff = diff * hand_weight.unsqueeze(0).unsqueeze(0)        # broadcast
    per_frame = diff.mean(dim=-1)                              # (B, T)
    mask = pose_mask.to(per_frame.dtype)
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return ((per_frame * mask).sum(dim=1) / n_real).mean()


def velocity_loss(
    x0_pred: torch.Tensor,     # (B, T, 150)
    x0_target: torch.Tensor,   # (B, T, 150)
    pose_mask: torch.Tensor,   # (B, T)
    hand_weight: torch.Tensor, # (150,)
) -> torch.Tensor:
    """MSE on first-order finite differences (joint velocities).

    Penalises velocity mismatch between predicted and target motion,
    which reduces temporal jitter in generated sequences.
    """
    pred_vel = x0_pred[:, 1:] - x0_pred[:, :-1]       # (B, T-1, 150)
    tgt_vel = x0_target[:, 1:] - x0_target[:, :-1]
    # Mask: a velocity frame is valid if both adjacent pose frames are real
    vel_mask = pose_mask[:, 1:] & pose_mask[:, :-1]    # (B, T-1)

    diff = (pred_vel - tgt_vel) ** 2
    diff = diff * hand_weight.unsqueeze(0).unsqueeze(0)
    per_frame = diff.mean(dim=-1)
    mask = vel_mask.to(per_frame.dtype)
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return ((per_frame * mask).sum(dim=1) / n_real).mean()


def acceleration_loss(
    x0_pred: torch.Tensor,     # (B, T, 150)
    pose_mask: torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """Penalise high acceleration in the predicted sequence.

    Unlike velocity_loss (which matches target velocities), this is a
    regularisation term that penalises any high-frequency motion in the
    prediction, regardless of the target. Keeps motion smooth even when
    the target itself has some jitter.
    """
    vel = x0_pred[:, 1:] - x0_pred[:, :-1]            # (B, T-1, 150)
    acc = vel[:, 1:] - vel[:, :-1]                    # (B, T-2, 150)
    acc_mask = pose_mask[:, 2:] & pose_mask[:, 1:-1] & pose_mask[:, :-2]

    per_frame = acc.pow(2).mean(dim=-1)                # (B, T-2)
    mask = acc_mask.to(per_frame.dtype)
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return ((per_frame * mask).sum(dim=1) / n_real).mean()


def bone_length_loss(
    x0_pred: torch.Tensor,     # (B, T, 150)
    x0_target: torch.Tensor,   # (B, T, 150)
    pose_mask: torch.Tensor,   # (B, T)
    bones: list[tuple[int, int]] = ALL_BONES,
) -> torch.Tensor:
    """Penalise bone length deviation between predicted and target.

    Prevents limb stretching / compression artifacts in generated motion.
    Operates on all body + hand bones.
    """
    pred_lens = _bone_lengths(x0_pred, bones)      # (B, T, n_bones)
    tgt_lens = _bone_lengths(x0_target, bones)

    diff = (pred_lens - tgt_lens) ** 2             # (B, T, n_bones)
    per_frame = diff.mean(dim=-1)                  # (B, T)
    mask = pose_mask.to(per_frame.dtype)
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return ((per_frame * mask).sum(dim=1) / n_real).mean()


def face_expression_loss(
    face_pred: torch.Tensor,   # (B, T, F) predicted face params
    face_target: torch.Tensor, # (B, T, F)
    pose_mask: torch.Tensor,   # (B, T)
) -> torch.Tensor:
    """MSE on face expression parameters (FLAME psi or MediaPipe landmarks).

    Only active once face data is integrated. Called from diffusion_step_loss
    when face tensors are provided.
    """
    diff = (face_pred - face_target) ** 2
    per_frame = diff.mean(dim=-1)
    mask = pose_mask.to(per_frame.dtype)
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return ((per_frame * mask).sum(dim=1) / n_real).mean()


# ---------------------------------------------------------------------------
# Combined loss config + step function
# ---------------------------------------------------------------------------

@dataclass
class DiffusionLossConfig:
    """Weights for each loss component.

    Start with lambda_vel=1.0, lambda_acc=0.1, lambda_bone=0.5 and tune
    based on the velocity/acceleration curves in the training log.
    """
    lambda_vel: float = 1.0       # velocity matching loss weight
    lambda_acc: float = 0.1       # acceleration regularisation weight
    lambda_bone: float = 0.5      # bone length consistency weight
    lambda_face: float = 1.0      # face expression loss weight (when active)
    hand_weight: float = 3.0      # per-coord upweight for hand joints


def diffusion_step_loss(
    x0_pred: torch.Tensor,         # (B, T, 150)
    x0_target: torch.Tensor,       # (B, T, 150)
    pose_mask: torch.Tensor,       # (B, T)
    cfg: DiffusionLossConfig,
    face_pred: Optional[torch.Tensor] = None,    # (B, T, F) optional
    face_target: Optional[torch.Tensor] = None,  # (B, T, F) optional
) -> dict:
    """Compute all diffusion losses for one training step.

    Returns a dict with scalar tensors:
        "loss"             — total weighted loss (call .backward() on this)
        "diffusion_loss"   — weighted MSE component
        "velocity_loss"    — velocity matching component
        "acceleration_loss"— acceleration regularisation component
        "bone_loss"        — bone length consistency component
        "face_loss"        — face expression component (0 if no face data)
    """
    hw = build_hand_weight_vector(POSE_DIM, cfg.hand_weight).to(x0_pred.device)

    d_loss = diffusion_mse_loss(x0_pred, x0_target, pose_mask, hw)
    v_loss = velocity_loss(x0_pred, x0_target, pose_mask, hw)
    a_loss = acceleration_loss(x0_pred, pose_mask)
    b_loss = bone_length_loss(x0_pred, x0_target, pose_mask)

    total = d_loss + cfg.lambda_vel * v_loss + cfg.lambda_acc * a_loss + cfg.lambda_bone * b_loss

    f_loss = torch.tensor(0.0, device=x0_pred.device)
    if face_pred is not None and face_target is not None:
        f_loss = face_expression_loss(face_pred, face_target, pose_mask)
        total = total + cfg.lambda_face * f_loss

    return {
        "loss": total,
        "diffusion_loss": d_loss.detach(),
        "velocity_loss": v_loss.detach(),
        "acceleration_loss": a_loss.detach(),
        "bone_loss": b_loss.detach(),
        "face_loss": f_loss.detach() if isinstance(f_loss, torch.Tensor) else f_loss,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, D = 4, 100, 150
    x0_pred = torch.randn(B, T, D, requires_grad=True)
    x0_target = torch.randn(B, T, D)
    pose_mask = torch.ones(B, T, dtype=torch.bool)
    # Simulate variable-length clips
    pose_mask[0, 80:] = False
    pose_mask[2, 60:] = False

    cfg = DiffusionLossConfig()
    losses = diffusion_step_loss(x0_pred, x0_target, pose_mask, cfg)
    losses["loss"].backward()

    print("diffusion losses:")
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k:>20}: {val:.6f}")
    print(f"  grad norm: {x0_pred.grad.norm().item():.4f}")

    # Verify hand upweighting is active
    hw = build_hand_weight_vector()
    print(f"\nhand weight vector: body={hw[:54].mean():.1f}  hands={hw[54:].mean():.1f}")
