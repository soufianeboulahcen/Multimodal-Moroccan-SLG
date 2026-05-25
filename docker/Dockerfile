# Minimal extension of NGC PyTorch for the PFE Soufian / MoSL × SignLLM project.
#
# Base image notes:
#   - nvcr.io/nvidia/pytorch:26.04-py3 ships with PyTorch built against CUDA 13
#     and includes native kernels for Blackwell SM_120 (DGX Spark GB10),
#     which solves the SM_120 PTX-JIT slowdown we saw with the cu128 wheel.
#   - It already contains numpy, scipy, opencv, ffmpeg, matplotlib, tqdm, and
#     most other ML utilities, so the extra layer here is small.
#
# We layer Hzzone's pytorch-openpose runtime deps. NGC PyTorch already provides
# numpy, scipy, and matplotlib, but NOT opencv or scikit-image — we add those.
# opencv-python-headless avoids GUI deps we don't need on this server.
# tqdm is small enough to always install in case the NGC base drops it.
#
# All project code is bind-mounted at runtime via docker/run.sh, so this image
# stays a thin layer over NGC and rebuilds are seconds.

FROM nvcr.io/nvidia/pytorch:26.04-py3

# Arabic typography for matplotlib figures (visualize_pose.py).  Without
# these, Arabic glyphs render disconnected and in the wrong order because
# matplotlib's default text engine does not perform Arabic shaping or
# BiDi reordering.
#   - fonts-noto-naskh-arabic: high-quality Arabic font with diacritic support
#   - arabic_reshaper: converts Unicode Arabic to presentation forms (letter joining)
#   - python-bidi:     handles right-to-left ordering
RUN apt-get update && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-hosny-amiri \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv >/dev/null

RUN pip install --no-cache-dir \
    opencv-python-headless \
    scikit-image \
    tqdm \
    arabic_reshaper \
    python-bidi \
    h5py

# Default cwd matches the bind-mount target used by docker/run.sh
WORKDIR /workspace/PFE-SOUFIAN
