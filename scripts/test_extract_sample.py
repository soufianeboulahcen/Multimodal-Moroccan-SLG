"""Pick a sample MoSL video from labels.csv and run pose extraction on it.

Usage:
    docker/run.sh python scripts/test_extract_sample.py [out_dir]

Reports timing of the extraction loop only (excluding container startup and
model load). All paths stay UTF-8 (Python handles them natively); no shell
quoting in the loop. Output is written to data/processed/keypoints_2d/sample/
by default.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# extract_one is module-style — call its main() directly to avoid subprocess
# overhead and shell-quoting issues with Arabic filenames.
# (Our package is named 'mosl' — 'src' is Hzzone's, kept distinct on purpose.)
from mosl.pose import extract_one as ex


def main(out_dir: str) -> int:
    labels = ROOT / "data" / "labels.csv"
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))

    # Pick a short pronoun video for a fast sanity check
    sample = next(
        r for r in rows
        if r["category"] == "Pronouns" and r["variant"] == "" and r["word_arabic"].startswith("أ")
    )
    video = ROOT / sample["relative_path"]
    print(f"sample: {sample['word_arabic']!r}  ({sample['relative_path']})")

    t0 = time.perf_counter()
    rc = ex.main(str(video), out_dir)
    dt = time.perf_counter() - t0

    if rc == 0:
        # report fps once we know how many frames were written
        n = len(list(Path(out_dir).glob("*_keypoints.json")))
        print(f"\nextraction: {n} frames in {dt:.2f}s  =>  {n / dt:.1f} fps")
    return rc


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/processed/keypoints_2d/sample"
    raise SystemExit(main(out))
