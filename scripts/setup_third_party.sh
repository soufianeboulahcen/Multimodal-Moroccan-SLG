#!/usr/bin/env bash
# Fetch upstream third-party dependencies and apply our patches.
# Idempotent: re-running is a no-op if everything is already in place.
#
# Usage:
#   ./scripts/setup_third_party.sh
#
# Requires:
#   * git
#   * curl + unzip  (for downloading the pytorch-openpose model weights)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
PATCHES="$ROOT/patches"
mkdir -p "$TP"

# ---------------------------------------------------------------------------
# 1. Hzzone's pytorch-openpose — PyTorch reimplementation of CMU OpenPose
# ---------------------------------------------------------------------------
HZ="$TP/pytorch-openpose"
if [ ! -d "$HZ" ]; then
    echo "[setup] cloning pytorch-openpose..."
    git clone --depth=1 https://github.com/Hzzone/pytorch-openpose.git "$HZ"
else
    echo "[setup] pytorch-openpose already present, skipping clone."
fi

# Body + hand model weights (~350 MB combined; not redistributed).
if [ ! -f "$HZ/model/body_pose_model.pth" ] || [ ! -f "$HZ/model/hand_pose_model.pth" ]; then
    echo "[setup] downloading pytorch-openpose model weights..."
    TMP_ZIP=$(mktemp -t pytorch-openpose-weights-XXXXXX.zip)
    # Dropbox direct-download URL (dl=1) — the upstream README's canonical source.
    curl -L -o "$TMP_ZIP" \
        "https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=1"
    unzip -o "$TMP_ZIP" 'body_pose_model.pth' 'hand_pose_model.pth' -d "$HZ/model/"
    rm -f "$TMP_ZIP"
else
    echo "[setup] pytorch-openpose weights already present, skipping download."
fi

# ---------------------------------------------------------------------------
# 2. SignLLM / Prompt2Sign — data-preprocessing pipeline
# ---------------------------------------------------------------------------
P2S="$TP/Prompt2Sign"
if [ ! -d "$P2S" ]; then
    echo "[setup] cloning Prompt2Sign..."
    git clone --depth=1 https://github.com/SignLLM/Prompt2Sign.git "$P2S"
else
    echo "[setup] Prompt2Sign already present, skipping clone."
fi

# Apply our two patches (PyTorch port of pose3D + TF stub in pipeline_demo_02).
echo "[setup] applying patches..."
cp "$PATCHES/pose3D.py" "$P2S/tools/2D_to_3D/pose3D.py"
cp "$PATCHES/pipeline_demo_02_h5totxt.py" "$P2S/tools/2D_to_3D/pipeline_demo_02_h5totxt.py"

echo "[setup] done."
echo
echo "Third-party tree:"
ls -la "$TP"
