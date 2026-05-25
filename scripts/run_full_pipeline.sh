#!/usr/bin/env bash
# Phase 2h: Run the full SignLLM data-prep pipeline on all 2,216 MoSL clips.
#
# Stages (per split mode in {train, dev, test}):
#   0. Export NPZ keypoints to per-frame OpenPose JSON (idempotent, all categories)
#   1. setup_p2s_pipeline.py            — hardlink JSONs + write .files / .text
#   2. pipeline_demo_01_json2h5.py      — pack to H5
#   3. pipeline_demo_02_h5totxt.py      — Step I/II/III + PyTorch backprop refinement
#   4. pipeline_demo_03_txt2skels.py    — flatten to compressed pose format
#
# Output: third_party/Prompt2Sign/tools/2D_to_3D/final_data/{train,dev,test}.{skels,files,text}
#
# Designed to run inside a detached docker container.  Exit code reflects
# overall success — set -e aborts on any stage failure.
set -euo pipefail

cd /workspace/PFE-SOUFIAN
P2S=third_party/Prompt2Sign/tools/2D_to_3D

echo "=========================================="
echo "Phase 2h: Full pipeline run started: $(date)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 0: Export NPZ → OpenPose JSON for every clip we have.
# Idempotent — clips already exported (e.g. Pronouns from yesterday) are skipped.
# ---------------------------------------------------------------------------
echo
echo "--- step 0: NPZ → OpenPose JSON (all categories) ---"
date
python -m mosl.pose.export_openpose_json
echo

# ---------------------------------------------------------------------------
# Wipe stale state from earlier dev/Pronouns smoke run so stage 03 doesn't
# append to old files.  out_data/ is left intact: stage 02 has resume logic
# (skips clips with existing demo5.txt), saving recompute on already-processed
# clips, while still re-processing every other clip.
# ---------------------------------------------------------------------------
rm -rf "$P2S/final_data"
mkdir -p "$P2S/final_data"

# ---------------------------------------------------------------------------
# Run all four pipeline stages for each mode.
# ---------------------------------------------------------------------------
for MODE in train dev test; do
    echo "=========================================="
    echo "MODE: $MODE  (started $(date))"
    echo "=========================================="

    echo "--- setup ---"
    python scripts/setup_p2s_pipeline.py --mode "$MODE"

    pushd "$P2S" >/dev/null
    echo
    echo "--- stage 01: json2h5 ---"
    python pipeline_demo_01_json2h5.py --data_subset "$MODE"

    echo
    echo "--- stage 02: h5totxt (Step I/II/III + PyTorch backprop refinement) ---"
    python pipeline_demo_02_h5totxt.py --data_subset "$MODE"

    echo
    echo "--- stage 03: txt2skels ---"
    python pipeline_demo_03_txt2skels.py --data_subset "$MODE"

    popd >/dev/null
    echo "MODE $MODE done at $(date)"
    echo
done

echo "=========================================="
echo "All modes complete: $(date)"
echo "=========================================="
echo
echo "final_data/ contents:"
ls -la "$P2S/final_data/"
echo
echo "per-mode line counts:"
for f in "$P2S/final_data/"*.{skels,files,text}; do
    if [ -f "$f" ]; then
        printf "  %-30s  %s lines  (%s bytes)\n" \
            "$(basename "$f")" "$(wc -l < "$f")" "$(stat -c %s "$f")"
    fi
done
