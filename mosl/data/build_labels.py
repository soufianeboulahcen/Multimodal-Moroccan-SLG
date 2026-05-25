"""Build the label index CSV from extracted MoSL videos.

For each .mp4 under data/raw/vedios-dataset/<category>/, parse:
  - category: one of {Diverse, Letters, Numbers, Pronouns, days_months_seasons}
  - word_arabic: the base Arabic word, with diacritics preserved
  - word_arabic_stripped: same word with diacritics removed (for grouping/dedup)
  - variant: integer N if filename ends with " (إِشَارَة N)", else None
  - relative_path: data/raw/... path for downstream tools

Output: data/labels.csv
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

RAW_ROOT = Path("data/raw/vedios-dataset")
OUT_CSV = Path("data/labels.csv")

CATEGORY_PREFIX = "mosl_videos_dataset_"

# "إِشَارَة" possibly with diacritics; we match the base letters ا ش ا ر ة
# allowing diacritic chars between them. Followed by whitespace + integer.
ISHARA_PATTERN = re.compile(
    r"\s*\(\s*"
    r"[ء-ي][ً-ٰٟ]*"
    r"(?:[ء-ي][ً-ٰٟ]*){4,}"
    r"\s+(\d+)\s*\)\s*$"
)

ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟ]")


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text)


def parse_filename(stem: str) -> tuple[str, int | None]:
    """Return (base_word, variant). variant is None if no '(إِشَارَة N)' suffix."""
    m = ISHARA_PATTERN.search(stem)
    if m:
        variant = int(m.group(1))
        base = stem[: m.start()].rstrip()
        return base, variant
    return stem.strip(), None


def category_from_dir(name: str) -> str:
    return name.removeprefix(CATEGORY_PREFIX)


def main() -> int:
    if not RAW_ROOT.exists():
        print(f"error: {RAW_ROOT} not found — run extract_dataset.py first")
        return 1

    rows = []
    for cat_dir in sorted(RAW_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = category_from_dir(cat_dir.name)
        for video in sorted(cat_dir.glob("*.mp4")):
            base, variant = parse_filename(video.stem)
            base_nfc = unicodedata.normalize("NFC", base)
            rows.append({
                "relative_path": str(video.relative_to(Path("data/raw").parent.parent)) if False else str(video),
                "category": category,
                "word_arabic": base_nfc,
                "word_arabic_stripped": strip_diacritics(base_nfc),
                "variant": variant if variant is not None else "",
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relative_path", "category", "word_arabic", "word_arabic_stripped", "variant"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
