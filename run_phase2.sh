#!/usr/bin/env bash
# Phase 2 — Real Diffusion Training + First Realistic Avatar Generation
#
# Master orchestration script. Runs all Phase 2 tasks in order inside the
# project Docker container (docker/run.sh wraps all commands with GPU access).
#
# Usage:
#   ./run_phase2.sh                          # full pipeline
#   ./run_phase2.sh --task annotate          # single task
#   ./run_phase2.sh --task train             # train diffusion only
#   ./run_phase2.sh --task generate          # generate 10 signs (needs trained model)
#   ./run_phase2.sh --task validate          # temporal stability validation
#   ./run_phase2.sh --skip-face              # skip face extraction (no raw videos)
#   ./run_phase2.sh --skip-smplx             # skip SMPL-X fitting (no model files)
#   ./run_phase2.sh --stage 2                # use SMPL-X mesh rendering
#   ./run_phase2.sh --stage 3                # use Blender Cycles rendering
#
# Prerequisites:
#   1. Docker image built:  docker build -t pfe-pose:latest docker/
#   2. SignLLM checkpoint:  runs/baseline_mse/best.pt  (from Phase 1)
#   3. .skels files:        third_party/Prompt2Sign/tools/2D_to_3D/final_data/
#   4. (optional) SMPL-X model files: data/smplx_models/smplx/SMPLX_NEUTRAL.npz
#   5. (optional) Raw videos: data/raw/vedios-dataset/ (for face extraction)
#
# Outputs:
#   runs_diffusion/mdm_mosl/best.pt          — trained MDM checkpoint
#   outputs/phase2_generation/               — generated avatar videos
#   outputs/phase2_generation/generation_summary.json — metrics

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNLLM_CHECKPOINT="${SIGNLLM_CHECKPOINT:-runs/baseline_mse/best.pt}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-runs_diffusion/mdm_mosl/best.pt}"
SMPLX_MODEL_PATH="${SMPLX_MODEL_PATH:-data/smplx_models}"
VOCAB_PATH="${VOCAB_PATH:-data/processed/vocab.json}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
LR="${LR:-1e-4}"
N_SAMPLE_STEPS="${N_SAMPLE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-2.5}"
STAGE="${STAGE:-1}"

# Feature flags
SKIP_ANNOTATE="${SKIP_ANNOTATE:-0}"
SKIP_FACE="${SKIP_FACE:-0}"
SKIP_SMPLX="${SKIP_SMPLX:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
TASK="${TASK:-all}"

# Docker wrapper
RUN="docker/run.sh"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)         TASK="$2"; shift 2 ;;
        --stage)        STAGE="$2"; shift 2 ;;
        --skip-face)    SKIP_FACE=1; shift ;;
        --skip-smplx)   SKIP_SMPLX=1; shift ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --max-epochs)   MAX_EPOCHS="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --cfg-scale)    CFG_SCALE="$2"; shift 2 ;;
        --no-docker)    RUN=""; shift ;;
        -h|--help)
            head -40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() {
    if [[ -n "$RUN" ]]; then
        "$RUN" "$@"
    else
        "$@"
    fi
}

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "WARNING: $1 not found — $2"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Task 1: Signer identity annotation
# ---------------------------------------------------------------------------

task_annotate() {
    log "Task 1: Signer identity annotation"
    run python -m mosl.data.annotate_signers \
        --n-signers 9 \
        --seed 42
    log "Task 1 complete"
}

# ---------------------------------------------------------------------------
# Task 2: Face extraction
# ---------------------------------------------------------------------------

task_face() {
    log "Task 2: Face keypoint extraction (MediaPipe FaceMesh)"
    if [[ ! -d "data/raw/vedios-dataset" ]]; then
        log "WARNING: data/raw/vedios-dataset not found — skipping face extraction"
        log "  Face extraction requires the original MoSL video files."
        log "  The diffusion model will train without face conditioning."
        return 0
    fi
    run python -m mosl.pose.extract_face_keypoints \
        --workers 4
    log "Task 2 complete"
}

# ---------------------------------------------------------------------------
# Task 3: SMPL-X fitting
# ---------------------------------------------------------------------------

task_smplx() {
    log "Task 3: SMPL-X fitting"
    if [[ ! -d "$SMPLX_MODEL_PATH" ]]; then
        log "WARNING: SMPL-X model files not found at $SMPLX_MODEL_PATH"
        log "  Download from https://smpl-x.is.tue.mpg.de/"
        log "  Place SMPLX_NEUTRAL.npz at $SMPLX_MODEL_PATH/smplx/"
        log "  Skipping SMPL-X fitting — Stage 1 (skeleton overlay) will still work."
        return 0
    fi
    run python -m mosl.pose.fit_smplx \
        --model-path "$SMPLX_MODEL_PATH" \
        --device cuda
    log "Task 3 complete"
}

# ---------------------------------------------------------------------------
# Task 4: Train diffusion model
# ---------------------------------------------------------------------------

task_train() {
    log "Task 4: Training MDM diffusion model"

    TRAIN_ARGS=(
        --run-name mdm_mosl
        --out-dir runs_diffusion
        --batch-size "$BATCH_SIZE"
        --max-epochs "$MAX_EPOCHS"
        --lr "$LR"
        --cfg-scale "$CFG_SCALE"
        --cfg-dropout 0.1
        --grad-checkpoint
    )

    if check_file "$SIGNLLM_CHECKPOINT" "training without frozen text encoder"; then
        TRAIN_ARGS+=(--signllm-checkpoint "$SIGNLLM_CHECKPOINT")
    fi

    # Resume from checkpoint if it exists
    if [[ -f "runs_diffusion/mdm_mosl/last.pt" ]]; then
        log "  Resuming from runs_diffusion/mdm_mosl/last.pt"
        TRAIN_ARGS+=(--resume-from runs_diffusion/mdm_mosl/last.pt)
    fi

    run python -m mosl.train.diffusion_train "${TRAIN_ARGS[@]}"
    log "Task 4 complete"
}

# ---------------------------------------------------------------------------
# Task 5 + 6: Generate 10 signs with temporal validation
# ---------------------------------------------------------------------------

task_generate() {
    log "Task 6: Generating 10 Arabic signs"

    GEN_ARGS=(
        --out-dir outputs/phase2_generation
        --stage "$STAGE"
        --n-sample-steps "$N_SAMPLE_STEPS"
        --cfg-scale "$CFG_SCALE"
        --fps 25.0
        --frame-size 512
        --device cuda
    )

    if check_file "$SIGNLLM_CHECKPOINT" "generating without text encoder"; then
        GEN_ARGS+=(--signllm-checkpoint "$SIGNLLM_CHECKPOINT")
    fi

    if check_file "$DIFFUSION_CHECKPOINT" "generating with random weights"; then
        GEN_ARGS+=(--diffusion-checkpoint "$DIFFUSION_CHECKPOINT")
    fi

    if [[ "$STAGE" -ge 2 ]] && [[ -d "$SMPLX_MODEL_PATH" ]]; then
        GEN_ARGS+=(--smplx-model-path "$SMPLX_MODEL_PATH")
    fi

    run python scripts/generate_10_signs.py "${GEN_ARGS[@]}"
    log "Task 6 complete"
}

# ---------------------------------------------------------------------------
# Task 5: Temporal stability validation
# ---------------------------------------------------------------------------

task_validate() {
    log "Task 5: Temporal stability validation"

    if ! check_file "$DIFFUSION_CHECKPOINT" "cannot validate without checkpoint"; then
        log "  Skipping validation — no trained checkpoint found"
        return 0
    fi

    run python -m mosl.train.temporal_stability \
        "$DIFFUSION_CHECKPOINT" \
        --n-samples 20 \
        --device cuda \
        --savgol-window 7

    log "Task 5 complete"
}

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

log "Phase 2 — Real Diffusion Training + First Realistic Avatar Generation"
log "Task: $TASK  Stage: $STAGE"
log ""

case "$TASK" in
    all)
        [[ "$SKIP_ANNOTATE" -eq 0 ]] && task_annotate
        [[ "$SKIP_FACE" -eq 0 ]]     && task_face
        [[ "$SKIP_SMPLX" -eq 0 ]]    && task_smplx
        [[ "$SKIP_TRAIN" -eq 0 ]]    && task_train
        [[ "$SKIP_VALIDATE" -eq 0 ]] && task_validate
        [[ "$SKIP_GENERATE" -eq 0 ]] && task_generate
        ;;
    annotate)   task_annotate ;;
    face)       task_face ;;
    smplx)      task_smplx ;;
    train)      task_train ;;
    validate)   task_validate ;;
    generate)   task_generate ;;
    *)
        echo "Unknown task: $TASK" >&2
        echo "Valid tasks: all, annotate, face, smplx, train, validate, generate" >&2
        exit 1
        ;;
esac

log ""
log "Phase 2 complete."
log ""
log "Outputs:"
log "  Trained model:  runs_diffusion/mdm_mosl/best.pt"
log "  Generated videos: outputs/phase2_generation/"
log "  Summary:        outputs/phase2_generation/generation_summary.json"
log ""
log "Next steps:"
log "  Stage 2 (SMPL-X mesh):  ./run_phase2.sh --task generate --stage 2"
log "  Stage 3 (Blender):      ./run_phase2.sh --task generate --stage 3"
log "  Neural render (Champ):  docker/run.sh python scripts/neural_render.py --setup"
