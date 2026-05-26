#!/usr/bin/env bash
# =============================================================================
# run_pose2video.sh — Pipeline Pose-to-Video ControlNet pour MoSL
# =============================================================================
# Exécute le pipeline complet :
#   1. Installation des dépendances
#   2. Extraction des frames de pose (si nécessaire)
#   3. Génération des vidéos avatar via ControlNet + Stable Diffusion
#
# Usage :
#   bash run_pose2video.sh                        # batch complet
#   bash run_pose2video.sh --sign أَنْتِ           # un seul signe
#   bash run_pose2video.sh --from-skeleton        # depuis vidéos MP4
#   bash run_pose2video.sh --steps 30 --res 768   # haute qualité
#
# Prérequis :
#   - GPU NVIDIA avec CUDA 11.8+ et ≥ 8 GB VRAM
#   - Python 3.10+
#   - Connexion internet (téléchargement modèles HuggingFace ~5 GB)
# =============================================================================

set -euo pipefail

# ── Répertoire racine du projet ───────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Paramètres par défaut ─────────────────────────────────────────────────────
SIGN=""                          # signe spécifique (vide = batch)
FROM_SKELETON=false              # utiliser vidéos skeleton au lieu de PNG
RESOLUTION=512                   # 512 ou 768
STEPS=20                         # étapes de débruitage (20=rapide, 50=qualité)
GUIDANCE=7.5                     # classifier-free guidance scale
CONTROLNET_SCALE=1.0             # force du conditionnement ControlNet
TEMPORAL_SIGMA=1.0               # lissage temporel (0=désactivé)
SEED=42                          # seed pour reproductibilité
DEVICE="cuda"                    # cuda ou cpu
SKIP_INSTALL=false               # passer l'installation des dépendances
FP32=false                       # float32 (plus lent, plus précis)
CPU_OFFLOAD=false                # CPU offload pour VRAM < 6 GB

# Chemins
POSE_CONTROL_DIR="outputs/pose_control"
SKELETON_DIR="outputs/videos/skeleton"
OUTPUT_DIR="outputs/avatar"

# ── Parsing des arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sign)         SIGN="$2";           shift 2 ;;
        --from-skeleton) FROM_SKELETON=true; shift   ;;
        --res|--resolution) RESOLUTION="$2"; shift 2 ;;
        --steps)        STEPS="$2";          shift 2 ;;
        --guidance)     GUIDANCE="$2";       shift 2 ;;
        --controlnet-scale) CONTROLNET_SCALE="$2"; shift 2 ;;
        --temporal-sigma) TEMPORAL_SIGMA="$2"; shift 2 ;;
        --seed)         SEED="$2";           shift 2 ;;
        --device)       DEVICE="$2";         shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --skip-install) SKIP_INSTALL=true;   shift   ;;
        --fp32)         FP32=true;           shift   ;;
        --cpu-offload)  CPU_OFFLOAD=true;    shift   ;;
        --help|-h)
            grep "^#" "$0" | head -20 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
done

# ── Couleurs terminal ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERR]${NC}  $*"; }

# ── Vérification GPU ──────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        log_success "GPU détecté : $GPU_NAME ($VRAM)"
    else
        log_warn "nvidia-smi non trouvé. Vérifier l'installation CUDA."
        if [[ "$DEVICE" == "cuda" ]]; then
            log_warn "Basculement sur CPU (génération très lente)"
            DEVICE="cpu"
        fi
    fi
}

# ── Installation des dépendances ──────────────────────────────────────────────
install_deps() {
    if [[ "$SKIP_INSTALL" == "true" ]]; then
        log_info "Installation ignorée (--skip-install)"
        return
    fi

    log_info "Installation des dépendances diffusion..."

    # Vérifier si pip est disponible
    if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
        log_error "pip non trouvé. Installer Python 3.10+ avec pip."
        exit 1
    fi

    PIP_CMD=$(command -v pip3 || command -v pip)

    # Installer PyTorch avec CUDA si pas déjà présent
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_info "Installation PyTorch + CUDA 11.8..."
        $PIP_CMD install torch==2.2.0+cu118 torchvision==0.17.0+cu118 \
            --extra-index-url https://download.pytorch.org/whl/cu118 -q
    else
        log_success "PyTorch CUDA déjà installé"
    fi

    # Installer les autres dépendances
    log_info "Installation diffusers, controlnet-aux, imageio..."
    $PIP_CMD install \
        diffusers==0.27.2 \
        transformers==4.40.1 \
        accelerate==0.29.3 \
        huggingface-hub==0.22.2 \
        safetensors==0.4.3 \
        controlnet-aux==0.0.7 \
        opencv-python==4.9.0.80 \
        Pillow==10.3.0 \
        imageio==2.34.1 \
        imageio-ffmpeg==0.4.9 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        tqdm==4.66.2 \
        einops==0.7.0 \
        -q

    # xformers (optionnel, améliore la vitesse)
    log_info "Installation xformers (optionnel)..."
    $PIP_CMD install xformers==0.0.25 \
        --index-url https://download.pytorch.org/whl/cu118 -q 2>/dev/null \
        && log_success "xformers installé" \
        || log_warn "xformers non installé (génération plus lente mais fonctionnelle)"

    log_success "Dépendances installées"
}

# ── Extraction des frames de pose ─────────────────────────────────────────────
extract_pose_frames() {
    if [[ "$FROM_SKELETON" == "true" ]]; then
        log_info "Extraction des frames depuis les vidéos skeleton..."
        python3 scripts/extract_pose_frames.py \
            --batch-skeleton-dir "$SKELETON_DIR" \
            --output-dir "$POSE_CONTROL_DIR" \
            --resolution "$RESOLUTION"
        log_success "Frames extraites dans $POSE_CONTROL_DIR"
    else
        # Vérifier que des PNG existent déjà
        PNG_COUNT=$(find "$POSE_CONTROL_DIR" -name "*.png" 2>/dev/null | wc -l)
        if [[ "$PNG_COUNT" -eq 0 ]]; then
            log_warn "Aucun PNG dans $POSE_CONTROL_DIR, extraction depuis skeleton..."
            python3 scripts/extract_pose_frames.py \
                --batch-skeleton-dir "$SKELETON_DIR" \
                --output-dir "$POSE_CONTROL_DIR" \
                --resolution "$RESOLUTION"
        else
            log_success "$PNG_COUNT frames PNG déjà disponibles dans $POSE_CONTROL_DIR"
        fi
    fi
}

# ── Construction des arguments Python ────────────────────────────────────────
build_python_args() {
    ARGS=(
        "--resolution" "$RESOLUTION"
        "--fps" "25"
        "--steps" "$STEPS"
        "--guidance-scale" "$GUIDANCE"
        "--controlnet-scale" "$CONTROLNET_SCALE"
        "--temporal-sigma" "$TEMPORAL_SIGMA"
        "--seed" "$SEED"
        "--device" "$DEVICE"
        "--output-dir" "$OUTPUT_DIR"
    )

    [[ "$FP32" == "true" ]]        && ARGS+=("--fp32")
    [[ "$CPU_OFFLOAD" == "true" ]] && ARGS+=("--cpu-offload")

    echo "${ARGS[@]}"
}

# ── Génération d'un signe unique ──────────────────────────────────────────────
generate_single_sign() {
    local sign="$1"
    log_info "Génération du signe : $sign"

    # Chercher le dossier de pose correspondant
    POSE_DIR=$(find "$POSE_CONTROL_DIR" -maxdepth 1 -type d -name "*${sign}*" 2>/dev/null | head -1)

    if [[ -z "$POSE_DIR" ]]; then
        # Chercher la vidéo skeleton
        SKEL_VIDEO=$(find "$SKELETON_DIR" -name "*${sign}*_skeleton.mp4" 2>/dev/null | head -1)
        if [[ -z "$SKEL_VIDEO" ]]; then
            log_error "Source introuvable pour le signe : $sign"
            log_error "Vérifier dans $POSE_CONTROL_DIR ou $SKELETON_DIR"
            exit 1
        fi
        log_info "Utilisation de la vidéo skeleton : $SKEL_VIDEO"
        SOURCE_ARG="--skeleton-video"
        SOURCE_PATH="$SKEL_VIDEO"
    else
        log_info "Utilisation du dossier pose : $POSE_DIR"
        SOURCE_ARG="--pose-dir"
        SOURCE_PATH="$POSE_DIR"
    fi

    OUTPUT_PATH="$OUTPUT_DIR/${sign}_avatar.mp4"
    read -ra ARGS <<< "$(build_python_args)"

    python3 scripts/pose2video_controlnet.py \
        "$SOURCE_ARG" "$SOURCE_PATH" \
        --output "$OUTPUT_PATH" \
        "${ARGS[@]}"

    log_success "Vidéo générée : $OUTPUT_PATH"
}

# ── Génération batch ──────────────────────────────────────────────────────────
generate_batch() {
    log_info "Génération batch depuis $POSE_CONTROL_DIR"
    read -ra ARGS <<< "$(build_python_args)"

    python3 scripts/pose2video_controlnet.py \
        --batch-dir "$POSE_CONTROL_DIR" \
        --skeleton-dir "$SKELETON_DIR" \
        "${ARGS[@]}"

    log_success "Batch terminé. Vidéos dans $OUTPUT_DIR"
}

# ── Résumé de la configuration ────────────────────────────────────────────────
print_config() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Pipeline Pose-to-Video ControlNet — MoSL${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "  Modèle SD   : runwayml/stable-diffusion-v1-5"
    echo -e "  ControlNet  : lllyasviel/control_v11p_sd15_openpose"
    echo -e "  Résolution  : ${RESOLUTION}×${RESOLUTION}"
    echo -e "  FPS         : 25"
    echo -e "  Étapes      : $STEPS"
    echo -e "  Guidance    : $GUIDANCE"
    echo -e "  CN Scale    : $CONTROLNET_SCALE"
    echo -e "  Lissage σ   : $TEMPORAL_SIGMA"
    echo -e "  Device      : $DEVICE"
    echo -e "  Sortie      : $OUTPUT_DIR"
    if [[ -n "$SIGN" ]]; then
        echo -e "  Mode        : signe unique → $SIGN"
    else
        echo -e "  Mode        : batch"
    fi
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
}

# ── Point d'entrée principal ──────────────────────────────────────────────────
main() {
    print_config
    check_gpu
    install_deps
    extract_pose_frames

    mkdir -p "$OUTPUT_DIR"

    if [[ -n "$SIGN" ]]; then
        generate_single_sign "$SIGN"
    else
        generate_batch
    fi

    echo ""
    log_success "Pipeline terminé !"
    echo -e "  Vidéos disponibles dans : ${BOLD}$OUTPUT_DIR${NC}"
    ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || true
}

main "$@"
