"""Extract vedios-dataset.zip with proper UTF-8 decoding of Arabic filenames.

The MoSL zip stores filenames as raw bytes flagged neither as UTF-8 nor as a known
codepage. Python's zipfile decodes them as cp437 by default, which mangles Arabic.
We re-encode each filename as cp437 (round-trip to original bytes) and decode as UTF-8.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path("vedios-dataset.zip")
OUT_DIR = Path("data/raw")


def fix_name(raw_name: str) -> str:
    """zipfile gives us cp437-decoded text; re-encode and decode as UTF-8."""
    try:
        return raw_name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw_name


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"error: {ZIP_PATH} not found in cwd ({Path.cwd()})", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        infos = zf.infolist()
        n_files = sum(1 for i in infos if not i.is_dir())
        print(f"opening {ZIP_PATH} — {len(infos)} entries, {n_files} files")

        for i, info in enumerate(infos):
            if info.is_dir():
                continue
            corrected = fix_name(info.filename)
            target = OUT_DIR / corrected
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 200 == 0 or i == len(infos) - 1:
                print(f"  [{i + 1}/{len(infos)}] {corrected[:80]}")

    print(f"done — extracted to {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
