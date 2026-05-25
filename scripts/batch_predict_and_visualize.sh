#!/usr/bin/env bash
# Generate predictions + side-by-side GIFs for a diverse sample of MoSL
# signs, one per category, drawn directly from data/labels.csv to avoid
# Arabic-encoding mismatches between hand-typed strings and the dataset's
# canonical NFC forms.
#
# Usage:
#   ./scripts/batch_predict_and_visualize.sh                 # uses baseline_mse
#   ./scripts/batch_predict_and_visualize.sh rl_plc          # different checkpoint
#   N_PER_CAT=5 ./scripts/batch_predict_and_visualize.sh     # more samples per category
set -euo pipefail
cd /workspace/PFE-SOUFIAN

RUN="${1:-baseline_mse}"
OUT_DIR="predictions"
N_PER_CAT="${N_PER_CAT:-4}"
mkdir -p "$OUT_DIR"

# Sample N_PER_CAT clip-stems per category from labels.csv, biased toward
# multi-clip signs (the ones the model has actually seen multiple variants
# of).  Stems are written as raw bytes so encoding stays byte-identical to
# the CSV.
python <<PY > /tmp/clips_to_run.txt
import csv, collections, random
random.seed(0)
rows = list(csv.DictReader(open('data/labels.csv', encoding='utf-8')))
by_cat = collections.defaultdict(list)
sign_count = collections.Counter((r['category'], r['word_arabic']) for r in rows)
for r in rows:
    n_for_sign = sign_count[(r['category'], r['word_arabic'])]
    by_cat[r['category']].append((n_for_sign, r))
for cat, items in by_cat.items():
    items.sort(key=lambda x: (-x[0], x[1]['relative_path']))
    picked = items[:${N_PER_CAT}]
    for _, r in picked:
        from pathlib import Path
        stem = Path(r['relative_path']).stem
        print(f"{r['category']}\t{stem}")
PY

cat /tmp/clips_to_run.txt | awk -F'\t' '{printf "  %s / %s\n", $1, $2}'
total=$(wc -l < /tmp/clips_to_run.txt)
echo
echo "Sampled ${total} clips total (N_PER_CAT=${N_PER_CAT})"
echo

ok=0
fail=0
while IFS=$'\t' read -r cat stem; do
    safe=$(echo "${cat}_${stem}" | tr ' ()/' '____')
    out_npz="$OUT_DIR/${RUN}_${safe}.npz"

    echo "=========================================="
    echo "  $cat / $stem"
    echo "=========================================="
    if python scripts/predict.py --run "$RUN" --clip-stem "$stem" \
        --category "$cat" --out "$out_npz" 2>&1 | tail -4; then
        if python scripts/visualize_pose.py "$out_npz" --side-by-side --fps 12 \
            2>&1 | tail -1; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
            echo "  WARN: visualize failed for $cat/$stem"
        fi
    else
        fail=$((fail+1))
        echo "  WARN: predict failed for $cat/$stem"
    fi
done < /tmp/clips_to_run.txt

echo
echo "=========================================="
echo "  Done: $ok succeeded, $fail failed"
echo "=========================================="
ls -lh "$OUT_DIR/"*.gif 2>/dev/null | tail -30
