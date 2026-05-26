#!/usr/bin/env bash
# Install all dependencies for the photorealistic avatar pipeline.
# Run once before first use.
#
# Usage:
#   bash avatar_video_generator/install.sh          # CUDA 11.8 (default)
#   bash avatar_video_generator/install.sh --cu121  # CUDA 12.1
#   bash avatar_video_generator/install.sh --cpu    # CPU only (testing)
set -euo pipefail

CUDA_TAG="cu118"
for arg in "$@"; do
  case "$arg" in
    --cu121) CUDA_TAG="cu121" ;;
    --cu124) CUDA_TAG="cu124" ;;
    --cpu)   CUDA_TAG="cpu"   ;;
  esac
done

echo "=== MoSL Avatar Pipeline — Dependency Install (${CUDA_TAG}) ==="

# System packages
if command -v apt-get &>/dev/null; then
  sudo apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgles2 libegl1 2>/dev/null || true
fi

# PyTorch
echo "--- Installing PyTorch (${CUDA_TAG})..."
pip install torch torchvision \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" -q

# Core diffusion stack
echo "--- Installing diffusion stack..."
pip install -q \
  "diffusers>=0.27.0" \
  "transformers>=4.38.0" \
  "accelerate>=0.28.0" \
  "safetensors>=0.4.2" \
  "huggingface-hub>=0.22.0" \
  einops omegaconf

# ControlNet auxiliary (pose detection)
echo "--- Installing controlnet-aux..."
pip install -q "controlnet-aux>=0.0.7" || true

# Identity
echo "--- Installing InsightFace..."
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install -q insightface onnxruntime || true
else
  pip install -q insightface "onnxruntime-gpu>=1.17.0" || true
fi

# Vision / post-processing
echo "--- Installing vision stack..."
pip install -q \
  "opencv-python-headless>=4.9.0" \
  "Pillow>=10.0.0" \
  "numpy>=1.24.0" \
  "scipy>=1.11.0" \
  "imageio>=2.34.0" \
  "imageio-ffmpeg>=0.4.9" \
  mediapipe tqdm pyyaml

# xformers (optional, GPU only)
if [ "$CUDA_TAG" != "cpu" ]; then
  echo "--- Installing xformers..."
  pip install -q xformers \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" || true
fi

echo ""
echo "=== Install complete. Test with:"
echo "    python avatar_video_generator/run.py --sign أَنْتِ --pose-only"
echo "    python avatar_video_generator/run.py --sign أَنْتِ  # full GPU run"
