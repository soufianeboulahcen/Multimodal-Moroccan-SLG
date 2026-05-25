"""Compute dataset statistics from data/labels.csv.

Reports per-category counts, unique-word distributions, variants-per-word histogram,
and (via ffprobe) duration/frame-count statistics. Caches ffprobe results in
data/video_meta.csv so re-runs are fast.
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

LABELS_CSV = Path("data/labels.csv")
META_CSV = Path("data/video_meta.csv")


def load_labels() -> list[dict]:
    with open(LABELS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ffprobe_meta(path: str) -> tuple[float, int, float]:
    """Return (duration_sec, n_frames, fps) via ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json",
            path,
        ],
        text=True,
    )
    s = json.loads(out)["streams"][0]
    duration = float(s.get("duration", 0.0) or 0.0)
    nb_frames = int(s.get("nb_frames", 0) or 0)
    rate_num, rate_den = (int(x) for x in s["r_frame_rate"].split("/"))
    fps = rate_num / rate_den if rate_den else 0.0
    return duration, nb_frames, fps


def build_or_load_meta(rows: list[dict]) -> dict[str, dict]:
    if META_CSV.exists():
        with open(META_CSV, encoding="utf-8") as f:
            return {r["relative_path"]: r for r in csv.DictReader(f)}

    print(f"running ffprobe on {len(rows)} videos…")
    meta = {}
    for i, row in enumerate(rows):
        path = row["relative_path"]
        try:
            dur, nf, fps = ffprobe_meta(path)
        except Exception as e:
            print(f"  ffprobe failed: {path} — {e}")
            dur, nf, fps = 0.0, 0, 0.0
        meta[path] = {
            "relative_path": path,
            "duration_sec": f"{dur:.3f}",
            "n_frames": str(nf),
            "fps": f"{fps:.3f}",
        }
        if (i + 1) % 200 == 0 or i == len(rows) - 1:
            print(f"  [{i + 1}/{len(rows)}] {path[-60:]}")

    with open(META_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["relative_path", "duration_sec", "n_frames", "fps"]
        )
        writer.writeheader()
        writer.writerows(meta.values())
    print(f"wrote {META_CSV}")
    return meta


def fmt_dist(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)}  min={min(values):.2f}  "
        f"p25={sorted(values)[len(values) // 4]:.2f}  "
        f"median={median(values):.2f}  "
        f"mean={mean(values):.2f}±{stdev(values):.2f}  "
        f"p75={sorted(values)[3 * len(values) // 4]:.2f}  "
        f"max={max(values):.2f}"
    )


def main() -> int:
    rows = load_labels()
    meta = build_or_load_meta(rows)

    print("\n" + "=" * 70)
    print("PER-CATEGORY COUNTS")
    print("=" * 70)
    cat_counter = Counter(r["category"] for r in rows)
    for cat, n in sorted(cat_counter.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {n:>5}")
    print(f"  {'TOTAL':<30} {len(rows):>5}")

    print("\n" + "=" * 70)
    print("UNIQUE WORDS (by stripped Arabic, per category)")
    print("=" * 70)
    by_cat_words: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_cat_words[r["category"]].add(r["word_arabic_stripped"])
    total_unique = set()
    for cat, words in sorted(by_cat_words.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<30} {len(words):>5} unique words")
        total_unique |= words
    print(f"  {'TOTAL UNIQUE (cross-cat)':<30} {len(total_unique):>5}")

    print("\n" + "=" * 70)
    print("CLIPS PER WORD (variants/repetitions per unique word)")
    print("=" * 70)
    word_counter: Counter[tuple[str, str]] = Counter()
    for r in rows:
        word_counter[(r["category"], r["word_arabic_stripped"])] += 1
    counts = list(word_counter.values())
    print(f"  total (category, word) pairs: {len(counts)}")
    hist = Counter(counts)
    for k in sorted(hist):
        bar = "█" * min(50, hist[k])
        print(f"  {k:>2} clip(s):  {hist[k]:>4} words   {bar}")

    print("\n" + "=" * 70)
    print("VIDEO DURATIONS (seconds)")
    print("=" * 70)
    durations = [float(meta[r["relative_path"]]["duration_sec"]) for r in rows
                 if r["relative_path"] in meta]
    print("  ALL:        " + fmt_dist(durations))
    for cat in sorted(cat_counter):
        cat_durs = [float(meta[r["relative_path"]]["duration_sec"]) for r in rows
                    if r["category"] == cat and r["relative_path"] in meta]
        print(f"  {cat:<10} " + fmt_dist(cat_durs))

    print("\n" + "=" * 70)
    print("FRAME COUNTS")
    print("=" * 70)
    frames = [int(meta[r["relative_path"]]["n_frames"]) for r in rows
              if r["relative_path"] in meta]
    print("  ALL:        " + fmt_dist([float(x) for x in frames]))

    print("\n" + "=" * 70)
    print("FPS")
    print("=" * 70)
    fpss = Counter(meta[r["relative_path"]]["fps"] for r in rows
                   if r["relative_path"] in meta)
    for fps, n in fpss.most_common():
        print(f"  {fps:>8} fps:  {n:>5}")

    print("\n" + "=" * 70)
    print("SPLIT FEASIBILITY")
    print("=" * 70)
    n_singletons = sum(1 for c in counts if c == 1)
    n_atleast2 = sum(1 for c in counts if c >= 2)
    n_atleast3 = sum(1 for c in counts if c >= 3)
    print(f"  words with exactly 1 clip (cannot be split): {n_singletons}")
    print(f"  words with >=2 clips (can split train/val):   {n_atleast2}")
    print(f"  words with >=3 clips (can split train/val/test): {n_atleast3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
