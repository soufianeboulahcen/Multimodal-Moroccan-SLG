#!/usr/bin/env bash
# Full SignLLM ablation matrix on MoSL.  Three sequential 200-epoch runs
# (max), early-stopped on dev pose MSE with patience=20.  ~75 min total
# expected on GB10.
#
# Reproduces the structure of the paper's Table 5 ablation rows for our
# single-language MoSL setup:
#   baseline_mse  — Base + Normal MSE Loss
#   rl            — Base + RL Loss
#   rl_plc        — Base + RL Loss + PLC
set -euo pipefail
cd /workspace/PFE-SOUFIAN

run_one () {
    local MODE=$1 NAME=$2
    echo
    echo "=========================================="
    echo "Run: $NAME (mode=$MODE)  $(date)"
    echo "=========================================="
    python scripts/train_signllm.py --mode "$MODE" \
        --run-name "$NAME" --max-epochs 200 --batch-size 32 \
        --warmup-steps 4000 --num-workers 4 --early-stop-patience 20
}

run_one mse    baseline_mse
run_one rl     rl
run_one rl_plc rl_plc

echo
echo "=========================================="
echo "All runs complete: $(date)"
echo "=========================================="
for n in baseline_mse rl rl_plc; do
    f=runs/${n}/summary.json
    if [ -f "$f" ]; then
        printf "  %-15s  " "$n"
        cat "$f" | tr -d '\n'
        echo
    fi
done
