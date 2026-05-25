#!/usr/bin/env bash
# 10-epoch verification of all three loss modes.
# Used as a sanity check before committing to the full ablation run.
# ~3 minutes total inside the NGC container.
set -euo pipefail
cd /workspace/PFE-SOUFIAN

echo "=== verifying mse ==="
date
python scripts/train_signllm.py --mode mse \
    --run-name verify_mse --max-epochs 10 --batch-size 32 \
    --warmup-steps 200 --num-workers 0 --early-stop-patience 1000

echo
echo "=== verifying rl ==="
date
python scripts/train_signllm.py --mode rl \
    --run-name verify_rl --max-epochs 10 --batch-size 32 \
    --warmup-steps 200 --num-workers 0 --early-stop-patience 1000

echo
echo "=== verifying rl_plc ==="
date
python scripts/train_signllm.py --mode rl_plc \
    --run-name verify_rl_plc --max-epochs 10 --batch-size 32 \
    --warmup-steps 200 --num-workers 0 --early-stop-patience 1000

echo
echo "=== verify summary ==="
date
for mode in mse rl rl_plc; do
    f=runs/verify_${mode}/summary.json
    if [ -f "$f" ]; then
        printf "  %-7s  " "$mode"
        cat "$f" | tr -d '\n'
        echo
    else
        echo "  $mode: missing summary.json"
    fi
done
