"""Build deterministic train/val/test splits at the clip level.

Strategy (driven by the long-tail distribution: 74% of words have only 1 clip):

  - Group clips by (category, word_arabic_NFC).
  - 1 clip in group   -> all to TRAIN.
  - 2 clips in group  -> 1 to TRAIN, 1 to VAL.
  - 3+ clips in group -> hold out 1 for VAL, 1 for TEST, the rest to TRAIN.

This guarantees every multi-clip word has at least one held-out instance, so
val/test measure "produce a known sign in a new variant" — the realistic eval
question for an isolated SLP model. Single-clip words are unavoidable but stay
in train so the model still learns to emit them.

Held-out clips are picked deterministically: the lexicographically last clip
goes to VAL, the second-to-last (when present) goes to TEST. Seeded shuffle would
be equivalent but lex-sort makes the split reproducible without storing a seed.

Output: data/splits.csv with one column added: split in {train,val,test}.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

LABELS_CSV = Path("data/labels.csv")
OUT_CSV = Path("data/splits.csv")


def main() -> int:
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["word_arabic"])].append(r)

    train, val, test = [], [], []
    for key, clips in groups.items():
        clips_sorted = sorted(clips, key=lambda r: r["relative_path"])
        n = len(clips_sorted)
        if n == 1:
            train.extend(clips_sorted)
        elif n == 2:
            train.append(clips_sorted[0])
            val.append(clips_sorted[1])
        else:  # n >= 3
            train.extend(clips_sorted[:-2])
            test.append(clips_sorted[-2])
            val.append(clips_sorted[-1])

    out_rows = []
    for split, items in (("train", train), ("val", val), ("test", test)):
        for r in items:
            out_rows.append({**r, "split": split})

    fieldnames = list(rows[0].keys()) + ["split"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} rows to {OUT_CSV}")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")

    # sanity: every val/test word should also appear in train (so the model has seen the label)
    train_words = {(r["category"], r["word_arabic"]) for r in train}
    val_orphans = [r for r in val if (r["category"], r["word_arabic"]) not in train_words]
    test_orphans = [r for r in test if (r["category"], r["word_arabic"]) not in train_words]
    print(f"  val labels not in train:  {len(val_orphans)}")
    print(f"  test labels not in train: {len(test_orphans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
