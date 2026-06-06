#!/usr/bin/env bash
# =============================================================================
# run_avatar_photorealistic.sh — Photorealistic Avatar Generation for MoSL
# =============================================================================
# Converts existing skeleton/OpenPose motion into photorealistic avatar video
# while preserving the identity of the real signer from the MoSL dataset.
#
# The SignLLM pipeline is NOT modified. This script only reads its outputs.
#
# Usage:
#   bash run_avatar_photorealistic.sh                        # single sign (أَنْتِ)
#   bash run_avatar_photorealistic.sh --sign أَنْتِ           # explicit sign
#   bash run_avatar_photorealistic.sh --batch                # all signs
#   bash run_avatar_photorealistic.sh --dgx                  # DGX high-quality
#   bash run_avatar_photorealistic.sh --official-hd-openpose # HD OpenPose prototype
#   bash run_avatar_photorealistic.sh --no-sdxl              # SD1.5 fallback
#   bash run_avatar_photorealistic.sh --no-rife              # skip interpolation
#   bash run_avatar_photorealistic.sh --steps 40 --res 768   # quality override
#
# Prerequisites:
#   pip install -r requirements_avatar.txt
#   (GPU with ≥8 GB VRAM recommended; ≥16 GB for SDXL)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
SIGN="أَنْتِ"
BATCH=false
DGX=false
OFFICIAL_HD_OPENPOSE=false
CONFIG=""
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sign)           SIGN="$2";                    shift 2 ;;
        --batch)          BATCH=true;                   shift   ;;
        --dgx)            DGX=true;                     shift   ;;
        --official-hd-openpose) OFFICIAL_HD_OPENPOSE=true; shift ;;
        --config)         CONFIG="$2";                  shift 2 ;;
        --reference-video) EXTRA_ARGS+=(--reference-video "$2"); shift 2 ;;
        --reference-image) EXTRA_ARGS+=(--reference-image "$2"); shift 2 ;;
        --reference-frames-dir) EXTRA_ARGS+=(--reference-frames-dir "$2"); shift 2 ;;
        --allow-no-reference) EXTRA_ARGS+=(--allow-no-reference); shift ;;
        --frames-dir)     EXTRA_ARGS+=(--frames-dir "$2"); shift 2 ;;
        --output)         EXTRA_ARGS+=(--output "$2"); shift 2 ;;
        --no-sdxl)        EXTRA_ARGS+=(--no-sdxl);      shift   ;;
        --no-animatediff) EXTRA_ARGS+=(--no-animatediff); shift ;;
        --no-rife)        EXTRA_ARGS+=(--no-rife);      shift   ;;
        --cpu-offload)    EXTRA_ARGS+=(--cpu-offload);  shift   ;;
        --steps)          EXTRA_ARGS+=(--steps "$2");   shift 2 ;;
        --res|--resolution) EXTRA_ARGS+=(--resolution "$2"); shift 2 ;;
        --seed)           EXTRA_ARGS+=(--seed "$2");    shift 2 ;;
        --device)         EXTRA_ARGS+=(--device "$2");  shift 2 ;;
        --quiet)          EXTRA_ARGS+=(--quiet);        shift   ;;
        --help|-h)
            grep "^#" "$0" | head -25 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── GPU check ─────────────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        ok "GPU: $GPU"
    else
        warn "No GPU detected. Diffusion rendering will be very slow on CPU."
        EXTRA_ARGS+=(--device cpu --no-animatediff --no-rife --cpu-offload)
    fi
}

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
    log "Checking dependencies..."
    if ! python3 -c "import diffusers" 2>/dev/null; then
        warn "diffusers not installed. Installing requirements_avatar.txt..."
        pip install -r requirements_avatar.txt -q
    else
        ok "Core dependencies present."
    fi

    if ! python3 -c "import insightface" 2>/dev/null; then
        warn "insightface not installed (identity locking disabled)."
        warn "Install with: pip install insightface onnxruntime-gpu"
    fi
}

# ── Config selection ──────────────────────────────────────────────────────────
select_config() {
    if [[ -n "$CONFIG" ]]; then
        EXTRA_ARGS+=(--config "$CONFIG")
    elif [[ "$DGX" == "true" ]]; then
        EXTRA_ARGS+=(--config avatar_video_generator/configs/dgx.yaml)
        log "Using DGX high-quality config (768px, SDXL, 4× RIFE)"
    else
        EXTRA_ARGS+=(--config avatar_video_generator/configs/default.yaml)
    fi
}

# ── Print banner ──────────────────────────────────────────────────────────────
print_banner() {
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  MoSL Photorealistic Avatar Generation${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
    if [[ "$BATCH" == "true" ]]; then
        echo -e "  Mode     : batch (all signs in outputs/pose_control/)"
    elif [[ "$OFFICIAL_HD_OPENPOSE" == "true" ]]; then
        echo -e "  Mode     : official HD OpenPose prototype"
        echo -e "  Input    : outputs/avatar_from_video_hd/alsbt_ishara_2_pose/"
    else
        echo -e "  Sign     : $SIGN"
    fi
    echo -e "  DGX mode : $DGX"
    echo -e "  Output   : outputs/avatar_photorealistic/"
    echo -e "${BOLD}══════════════════════════════════════════════════════${NC}"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    print_banner
    check_gpu
    check_deps
    select_config

    if [[ "$BATCH" == "true" ]]; then
        log "Starting batch generation..."
        python3 scripts/generate_photorealistic_avatar.py \
            --batch \
            --output-dir outputs/avatar_photorealistic \
            "${EXTRA_ARGS[@]}"
    elif [[ "$OFFICIAL_HD_OPENPOSE" == "true" ]]; then
        log "Generating avatar from official HD OpenPose frames..."
        python3 scripts/generate_photorealistic_avatar.py \
            --official-hd-openpose \
            --allow-no-reference \
            --output-dir outputs/avatar_from_video_hd \
            "${EXTRA_ARGS[@]}"
    else
        log "Generating avatar for: $SIGN"
        python3 scripts/generate_photorealistic_avatar.py \
            --sign "$SIGN" \
            --output-dir outputs/avatar_photorealistic \
            "${EXTRA_ARGS[@]}"
    fi

    echo ""
    ok "Generation complete!"
    echo -e "  Output: ${BOLD}outputs/avatar_photorealistic/${NC}"
    ls -lh outputs/avatar_photorealistic/*.mp4 2>/dev/null || true
}

main "$@"
