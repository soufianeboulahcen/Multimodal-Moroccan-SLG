"""Set up everything Prompt2Sign's tools/2D_to_3D/ pipeline needs to run on
our MoSL data: split-aware hardlinks + <mode>.files + <mode>.text inputs.

Layout produced (under third_party/Prompt2Sign/tools/2D_to_3D/):
    output_of_openpose/<mode>/json/<mode>__<category>__<clip_stem>/
                                      <clip_stem>_<frame>_keypoints.json
    input_data/<mode>.files     # one line per clip: "<mode>/<category>__<clip>"
    input_data/<mode>.text      # one line per clip: the Arabic word

Mode mapping: our splits use {train, val, test}, but the upstream pipeline
hardcodes {train, dev, test}.  We map val→dev so the upstream scripts run
unchanged.

Files inside each clip directory are hardlinks to our exporter's output (no disk
duplication).  `walkDir` uses `os.walk(followlinks=False)` — symlinks at the
directory level wouldn't be entered, but ordinary directories with hardlinked
files inside work transparently.

Usage (inside container):
    python scripts/setup_p2s_pipeline.py --mode {train|dev|test}
        [--filter-category Pronouns Letters ...]
        [--limit N]

Idempotent per mode: the destination tree for that mode is rebuilt from scratch.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P2S = ROOT / "third_party" / "Prompt2Sign" / "tools" / "2D_to_3D"
EXPORT_ROOT = ROOT / "data" / "processed" / "openpose_json"
SPLITS_CSV = ROOT / "data" / "splits.csv"

# splits.csv uses 'val'; pipeline expects 'dev'.
MODE_FROM_SPLIT = {"train": "train", "val": "dev", "test": "test"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--filter-category", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    json_root = P2S / "output_of_openpose" / args.mode / "json"
    if json_root.exists():
        shutil.rmtree(json_root)
    json_root.mkdir(parents=True, exist_ok=True)
    (P2S / "input_data").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    target_split = next(k for k, v in MODE_FROM_SPLIT.items() if v == args.mode)

    # Build the list of clips assigned to this split, for which we have an
    # exported JSON directory available.
    work: list[dict] = []
    for r in rows:
        if r["split"] != target_split:
            continue
        if args.filter_category and r["category"] not in set(args.filter_category):
            continue
        clip_stem = Path(r["relative_path"]).stem
        json_dir = EXPORT_ROOT / r["category"] / clip_stem
        if not json_dir.exists():
            continue
        work.append({**r, "clip_stem": clip_stem, "json_dir": json_dir})
    if args.limit > 0:
        work = work[: args.limit]

    files_lines: list[str] = []
    text_lines: list[str] = []
    n_clips = n_files = 0
    for r in work:
        # Folder name as the pipeline sees it (after stripping the "<mode>/" prefix).
        clip_id = f"{r['category']}__{r['clip_stem']}"
        target_dir = json_root / clip_id
        target_dir.mkdir()
        for src_json in r["json_dir"].glob("*_keypoints.json"):
            os.link(src_json, target_dir / src_json.name)
            n_files += 1
        n_clips += 1
        files_lines.append(f"{args.mode}/{clip_id}")
        text_lines.append(r["word_arabic"])

    # Write <mode>.files and <mode>.text — overwrite (not append) for idempotency.
    files_path = P2S / "input_data" / f"{args.mode}.files"
    text_path = P2S / "input_data" / f"{args.mode}.text"
    files_path.write_text("\n".join(files_lines) + "\n", encoding="utf-8")
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print(f"prepared {n_clips} clips, {n_files} hardlinked JSONs under {json_root}")
    print(f"wrote {files_path} ({len(files_lines)} lines)")
    print(f"wrote {text_path}  ({len(text_lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
