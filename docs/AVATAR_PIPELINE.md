# Avatar Pipeline

Migration from the OpenPose research prototype to realistic avatar video generation.
All existing pipeline stages are preserved; new components are additive.

---

## Architecture Overview

```
Arabic text
  → SignLLMTextEncoder (frozen, reused from existing training)
  → MDMDenoiser (new — transformer denoiser, DDIM sampling)
  → (T, 150) smooth pose sequence
  → Savitzky-Golay temporal smoothing
  ─────────────────────────────────────────────────────────
  Stage 2 (requires smplx):
  → SMPLXFitter (OpenPose joints → body/hand/face params)
  → PyTorch3D mesh renderer
  ─────────────────────────────────────────────────────────
  Stage 3 (requires Blender ≥ 3.6):
  → Blender Cycles photorealistic render
  ─────────────────────────────────────────────────────────
  Stage 4 (future — requires Champ/AnimateAnyone):
  → Neural video renderer (SMPL-X render + reference signer image)
  → Photorealistic signer video
```

---

## Preserved Components

Every existing file and pipeline stage is unchanged:

| Component | Status | Location |
|---|---|---|
| OpenPose extraction | ✅ unchanged | `mosl/pose/extract_dataset.py` |
| JSON keypoint export | ✅ unchanged | `mosl/pose/export_openpose_json.py` |
| Prompt2Sign 2D→3D | ✅ unchanged | `third_party/Prompt2Sign/` |
| `.skels` file format | ✅ unchanged | `final_data/{train,dev,test}.skels` |
| SignLLM model | ✅ unchanged | `mosl/model/signllm.py` |
| SignLLM training loop | ✅ unchanged | `mosl/train/train.py` |
| MSE/RL/PLC losses | ✅ unchanged | `mosl/train/losses.py` |
| MoSLSkelsDataset | ✅ extended (backward-compatible) | `mosl/data/dataset.py` |
| Existing visualisers | ✅ unchanged | `scripts/visualize_*.py` |

---

## New Components

### Motion Diffusion

| File | Purpose |
|---|---|
| `mosl/model/mdm_denoiser.py` | MDM-style transformer denoiser. Reuses `SignLLM.encode_text` as frozen text conditioning backbone. AdaLN injection of timestep + signer style. |
| `mosl/train/noise_schedule.py` | Cosine beta schedule, DDPM forward process, DDIM deterministic reverse sampling, Savitzky-Golay smoother. |
| `mosl/train/diffusion_train.py` | Full diffusion training loop. Reads existing `.skels` files unchanged. Freezes text encoder, trains only the denoiser. |
| `mosl/train/losses_diffusion.py` | Velocity loss, acceleration loss, bone-length consistency, hand-weighted MSE (3× upweight on joints 18–49). |

### Pose Processing

| File | Purpose |
|---|---|
| `mosl/pose/extract_face_keypoints.py` | MediaPipe FaceMesh extraction. Appends `face_keypoints (T,478,3)` to existing NPZ files without modifying body/hand arrays. |
| `mosl/pose/fit_smplx.py` | Two-stage SMPL-X fitting: shape fitting on mean pose, then per-frame pose optimisation. Maps COCO-18 → SMPL-X joints via `COCO18_TO_SMPLX` table. |
| `mosl/pose/convert_to_humanml3d.py` | Converts `(T,150)` absolute positions to `(T,263)` velocity+contact representation for HumanML3D-pretrained diffusion models. |

### Signer Style

| File | Purpose |
|---|---|
| `mosl/model/signer_encoder.py` | Pose style encoder (transformer over reference clip) + optional DINO-ViT appearance encoder. Outputs `(B, cond_dim)` injected into MDMDenoiser via AdaLN. Contrastive consistency loss for same-signer pairs. |

### Rendering

| File | Purpose |
|---|---|
| `scripts/render_smplx_video.py` | Three backends: `overlay` (OpenCV skeleton, no deps), `pytorch3d` (mesh render), `blender` (Cycles photorealistic). |
| `scripts/generate_avatar_video.py` | End-to-end inference: Arabic text → DDIM sampling → optional SMPL-X → video. Supports reference clip for signer style conditioning. |

### Data Extensions

| File | Change |
|---|---|
| `data/labels.csv` | Added columns: `signer_id` (placeholder `unknown`), `handedness` (placeholder `unknown`), `sign_type` (placeholder `lexical`). |
| `mosl/data/dataset.py` | Extended `MoSLSkelsDataset` with `load_face` flag and `signer_id` loading. Extended `mosl_collate` with `signer_ids`, `face_kps`, `face_conf` fields. All changes are backward-compatible — existing code using the dataset is unaffected. |
| `mosl/model/signllm.py` | Added `SignLLMTextEncoder` standalone class. Loads only the encoder half of a SignLLM checkpoint. Used by `MDMDenoiser` to avoid instantiating the full decoder. |

---

## Joint Layout Reference

The `(T, 150)` pose format is 50 joints × xyz, produced by the Prompt2Sign kinematic optimizer:

```
Joints  0-17   body (COCO-18 subset)          coords   0-53
Joints 18-38   left hand (MANO 21 joints)     coords  54-116
Joints 39-49   right hand (partial MANO)      coords 117-149
```

Hand joints (coords 54–149) receive **3× loss weight** in all diffusion losses because hands carry the primary lexical content of sign language.

---

## Training Sequence

### Step 0 — Prerequisites (data preparation)

```bash
# 1. Extract face keypoints (adds face_keypoints to existing NPZ files)
python -m mosl.pose.extract_face_keypoints

# 2. Annotate signer_id in data/labels.csv
#    Open the file and replace "unknown" with integer IDs per signer.
#    This is a manual step — review the 2,216 clips to identify unique signers.

# 3. (Optional) Fit SMPL-X to all clips for Stage 2/3 rendering
python -m mosl.pose.fit_smplx --model-path data/smplx_models/
```

### Step 1 — Train SignLLM (existing, unchanged)

```bash
bash scripts/run_ablation.sh
# Best checkpoint: runs/<run_name>/best.pt
```

### Step 2 — Train MDM diffusion denoiser

```bash
python -m mosl.train.diffusion_train \
  --run-name mdm_mosl_v1 \
  --signllm-checkpoint runs/rl_plc/best.pt \
  --batch-size 32 \
  --max-epochs 300
```

The text encoder is frozen automatically. Only the denoiser (~15M params) is trained.

For best results, pretrain on HumanML3D first:
```bash
# Clone MDM and pretrain on HumanML3D (3-5 days on A100)
# Then use --resume-from to fine-tune on MoSL
python -m mosl.train.diffusion_train \
  --run-name mdm_mosl_finetune \
  --resume-from runs_diffusion/mdm_humanml3d/best.pt \
  --signllm-checkpoint runs/rl_plc/best.pt
```

### Step 3 — Generate avatar videos

```bash
# Stage 1: skeleton overlay (always available)
python scripts/generate_avatar_video.py \
  --text "الأذان" \
  --diffusion-checkpoint runs_diffusion/mdm_mosl_v1/best.pt \
  --signllm-checkpoint runs/rl_plc/best.pt \
  --stage 1

# Stage 2: SMPL-X mesh (requires smplx + pytorch3d)
python scripts/generate_avatar_video.py \
  --text "الأذان" \
  --stage 2 \
  --smplx-model-path data/smplx_models/

# With signer style conditioning (requires annotated signer_id)
python scripts/generate_avatar_video.py \
  --text "الأذان" \
  --ref-clip data/processed/keypoints_2d/Diverse/الأذان.npz \
  --stage 1
```

---

## Loss Function Summary

### Diffusion losses (`mosl/train/losses_diffusion.py`)

| Loss | Weight | Purpose |
|---|---|---|
| `diffusion_mse_loss` | 1.0 | Weighted MSE between predicted and target clean pose. Hand coords upweighted 3×. |
| `velocity_loss` | 1.0 (`lambda_vel`) | MSE on first-order finite differences. Reduces temporal jitter. |
| `acceleration_loss` | 0.1 (`lambda_acc`) | Penalises high acceleration in predicted sequence. Smoothness regularisation. |
| `bone_length_loss` | 0.5 (`lambda_bone`) | Penalises bone length deviation. Prevents limb stretching. |
| `face_expression_loss` | 1.0 (`lambda_face`) | MSE on face params. Active only when face data is loaded. |

All weights are configurable via `DiffusionLossConfig`.

---

## Signer Style Preservation

The `SignerEncoder` in `mosl/model/signer_encoder.py` captures signer identity from a reference clip:

1. **Pose style encoder**: transformer over reference pose sequence → captures rhythm, amplitude, hand dynamics
2. **Motion statistics**: mean/std of joint positions and velocities → captures signing tempo and range
3. **Appearance encoder** (optional, requires `timm`): frozen DINO-ViT features from reference frames → captures visual identity

The style embedding is injected into every AdaLN layer of the MDMDenoiser, conditioning all generated motion on the reference signer's style.

**Prerequisite**: `signer_id` must be annotated in `data/labels.csv` before training the style encoder. The current placeholder value is `unknown`.

---

## Hardware Requirements

| Task | Min VRAM | Recommended |
|---|---|---|
| SignLLM training | 4 GB | 8 GB |
| MDM diffusion training (batch=32) | 18 GB | 24 GB (RTX 3090) |
| SMPL-X fitting (batch=8) | 8 GB | 16 GB |
| PyTorch3D rendering (512×512) | 12 GB | 16 GB |
| Full pipeline inference | 20 GB | 24 GB |

Optimisation flags for the DGX Spark GB10 Blackwell:
```python
DiffusionTrainConfig(bf16=True, grad_checkpoint=True)
# + torch.compile() in diffusion_train.py for ~20% speedup
```

---

## Rendering Backend Comparison

| Backend | Quality | Speed | Dependencies | Use case |
|---|---|---|---|---|
| `overlay` | Skeleton only | Fast | OpenCV (already installed) | Sanity checks, quick demos |
| `pytorch3d` | Textured mesh | ~0.1s/frame GPU | `pip install pytorch3d smplx` | Research evaluation |
| `blender` | Photorealistic | ~10-30s/frame | Blender ≥ 3.6 + SMPL-X add-on | Demo videos |
| Champ/AnimateAnyone | Photorealistic | ~1s/frame GPU | Separate model download | Production output |

---

## File Dependency Graph

```
data/labels.csv  ←── signer_id annotation (manual)
       │
       ▼
mosl/data/dataset.py  ←── extended (backward-compatible)
       │
       ├── mosl/model/signllm.py  ←── SignLLMTextEncoder added
       │         │
       │         ▼
       │   mosl/model/mdm_denoiser.py  (new)
       │         │
       │   mosl/train/noise_schedule.py  (new)
       │         │
       │   mosl/train/diffusion_train.py  (new)
       │         │
       │   mosl/train/losses_diffusion.py  (new)
       │
       ├── mosl/pose/extract_face_keypoints.py  (new)
       │         │
       │         ▼
       │   data/processed/keypoints_2d/*.npz  ←── face_keypoints added
       │
       ├── mosl/pose/fit_smplx.py  (new)
       │         │
       │         ▼
       │   data/processed/smplx_params/*.npz  (new)
       │
       ├── mosl/pose/convert_to_humanml3d.py  (new)
       │
       └── mosl/model/signer_encoder.py  (new)
                 │
                 ▼
         scripts/generate_avatar_video.py  (new)
                 │
                 ▼
         scripts/render_smplx_video.py  (new)
                 │
                 ▼
         outputs/generated_videos/*.mp4
```
