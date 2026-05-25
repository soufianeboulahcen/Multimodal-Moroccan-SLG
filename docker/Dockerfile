# Extension of NGC PyTorch for the MoSL × SignLLM Phase 2 avatar pipeline.
#
# Base image notes:
#   - nvcr.io/nvidia/pytorch:26.04-py3 ships with PyTorch built against CUDA 13
#     and includes native kernels for Blackwell SM_120 (DGX Spark GB10),
#     which solves the SM_120 PTX-JIT slowdown we saw with the cu128 wheel.
#   - It already contains numpy, scipy, opencv, ffmpeg, matplotlib, tqdm, and
#     most other ML utilities, so the extra layer here is small.
#
# Phase 2 additions over Phase 1:
#   - mediapipe:          face landmark extraction (Task 2)
#   - smplx:              SMPL-X body model fitting (Task 3)
#   - pytorch3d:          differentiable mesh rendering (Task 6 Stage 2)
#   - scikit-learn:       k-means signer clustering (Task 1)
#   - timm:               DINO-ViT appearance encoder (SignerEncoder)
#   - einops:             tensor rearrangement (used by pytorch3d / diffusion)
#   - accelerate:         HuggingFace training utilities
#   - diffusers:          HuggingFace diffusion pipeline (neural rendering)
#   - transformers:       HuggingFace model hub
#   - blender:            installed separately — see docker/install_blender.sh
#
# All project code is bind-mounted at runtime via docker/run.sh.

FROM nvcr.io/nvidia/pytorch:26.04-py3

# System packages
#   - fonts-hosny-amiri:  Arabic typography for matplotlib
#   - libgl1:             OpenCV headless runtime dependency
#   - ffmpeg:             video encoding (may already be present in NGC image)
#   - blender:            Cycles renderer (optional, for Stage 3 rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-hosny-amiri \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv >/dev/null

# Phase 1 dependencies (unchanged)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    scikit-image \
    tqdm \
    arabic_reshaper \
    python-bidi \
    h5py

# Phase 2 dependencies — motion diffusion + avatar pipeline
#
# mediapipe:    face landmark extraction (Task 2)
# smplx:        SMPL-X body model (Task 3, 6)
# scikit-learn: k-means signer clustering (Task 1)
# timm:         DINO-ViT appearance encoder (SignerEncoder)
# einops:       tensor rearrangement (pytorch3d, diffusion models)
# scipy:        Savitzky-Golay smoothing (Task 5) — may already be in NGC
RUN pip install --no-cache-dir \
    mediapipe \
    smplx \
    scikit-learn \
    timm \
    einops \
    scipy

# HuggingFace ecosystem for neural rendering (Task 8)
# accelerate + diffusers + transformers are needed by Champ / AnimateAnyone
RUN pip install --no-cache-dir \
    accelerate \
    diffusers \
    transformers \
    huggingface_hub

# PyTorch3D — differentiable mesh renderer (Task 6 Stage 2)
# Built from source against the NGC PyTorch version to avoid CUDA mismatch.
# If the pre-built wheel is available for this CUDA version, use it instead.
# The --no-build-isolation flag avoids re-downloading torch during build.
RUN pip install --no-cache-dir \
    fvcore \
    iopath \
    && pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || echo "pytorch3d build failed — overlay backend will be used as fallback"

# Default cwd matches the bind-mount target used by docker/run.sh
WORKDIR /workspace/PFE-SOUFIAN
