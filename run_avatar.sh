#!/usr/bin/env bash
# =============================================================================
# run_avatar.sh — Photorealistic MoSL Avatar Generation
# =============================================================================
# Converts MoSL dataset videos into photorealistic avatar videos using
# ControlNet-OpenPose + AnimateDiff + SD1.5 + InsightFace identity locking.
#
# Usage:
#   bash run_avatar.sh                          # أَنْتِ, default settings
#   bash run_avatar.sh --sign أَنَا             # different sign
#   bash run_avatar.sh --batch-pronouns         # all Pronouns category
#   bash run_avatar.sh --batch-all              # entire dataset
#   bash run_avatar.sh --pose-only              # fast test, no diffusion
#   bash run_avatar.sh --no-animatediff         # ControlNet frame-by-frame
#   bash run_avatar.sh --steps 40 --res 768     # higher quality
#   bash run_avatar.sh --dgx                    # DGX preset (768px, 40 steps)
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Defaults ──────────────────────────────────────────────────────────────────
SIGN="أَنْتِ"
BATCH_DIR=""
EXTRA=()
POSE_ONLY=false
DGX=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sign)              SIGN="$2";                         shift 2 ;;
    --batch-pronouns)    BATCH_DIR=".devcontainer/Dataset/mosl_videos_dataset_Pronouns"; shift ;;
    --batch-letters)     BATCH_DIR=".devcontainer/Dataset/mosl_videos_dataset_Letters";  shift ;;
    --batch-numbers)     BATCH_DIR=".devcontainer/Dataset/mosl_videos_dataset_Numbers";  shift ;;
    --batch-all)         BATCH_DIR=".devcontainer/Dataset";                               shift ;;
    --pose-only)         POSE_ONLY=true;                    shift ;;
    --dgx)               DGX=true;                          shift ;;
    --no-animatediff)    EXTRA+=(--no-animatediff);         shift ;;
    --no-rife)           EXTRA+=(--rife-multiplier 1);      shift ;;
    --steps)             EXTRA+=(--steps "$2");             shift 2 ;;
    --res|--resolution)  EXTRA+=(--resolution "$2");        shift 2 ;;
    --seed)              EXTRA+=(--seed "$2");              shift 2 ;;
    --cpu-offload)       EXTRA+=(--cpu-offload);            shift ;;
    --fp32)              EXTRA+=(--fp32);                   shift ;;
    --device)            EXTRA+=(--device "$2");            shift 2 ;;
    --help|-h)           grep "^#" "$0" | head -20 | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

# ── DGX preset ────────────────────────────────────────────────────────────────
if $DGX; then
  EXTRA+=(--resolution 768 --steps 40 --rife-multiplier 4)
  echo "[DGX] 768px, 40 steps, 4x RIFE"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  echo "[GPU] $GPU"
else
  echo "[WARN] No GPU detected — adding --device cpu --no-animatediff --cpu-offload"
  EXTRA+=(--device cpu --no-animatediff --cpu-offload --rife-multiplier 1)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
POSE_FLAG=""
$POSE_ONLY && POSE_FLAG="--pose-only"

if [[ -n "$BATCH_DIR" ]]; then
  echo "[BATCH] $BATCH_DIR"
  python3 avatar_video_generator/run.py \
    --batch-dir "$BATCH_DIR" \
    --output-dir outputs/avatar_photorealistic \
    $POSE_FLAG \
    "${EXTRA[@]}"
else
  echo "[SIGN] $SIGN"
  python3 avatar_video_generator/run.py \
    --sign "$SIGN" \
    --output-dir outputs/avatar_photorealistic \
    $POSE_FLAG \
    "${EXTRA[@]}"
fi

echo ""
echo "[DONE] outputs/avatar_photorealistic/"
ls -lh outputs/avatar_photorealistic/*.mp4 2>/dev/null || true
