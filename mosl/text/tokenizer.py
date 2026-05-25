"""Word-level tokenizer for the SignLLM-on-MoSL text input.

Design choices:

  * Each `word_arabic` value from `data/labels.csv` is **one atomic token**,
    even when it contains internal spaces (e.g., "الأَمْنُ الوَطَنِيُّ" —
    "national security").  377 of the 1,631 unique signs are multi-word
    phrases; treating them as composite tokens would let the model see
    "national" and "security" separately, but the *signs* themselves are
    atomic, so we keep one token per sign.  This matches how the SignLLM
    paper treats gloss tokens.

  * NFC normalisation on input and vocab.  Two distinct NFC strings can map
    to the same diacritic-stripped form (e.g., رَجُلٌ "man" vs رِجْلٌ "leg") —
    we keep them as separate tokens because they are separate signs.  The
    full strict-fidelity reasoning is in docs/STATS.md.

  * Vocab includes 4 specials in fixed positions: `<pad>`=0, `<bos>`=1,
    `<eos>`=2, `<unk>`=3.  Sign tokens start at id 4.

  * The tokenizer is built once from `data/labels.csv` (covering ALL splits)
    so train and val/test share the same vocabulary, and saved as JSON to
    `data/processed/vocab.json`.

Usage:
    from mosl.text.tokenizer import WordTokenizer
    tok = WordTokenizer.from_labels_csv("data/labels.csv")
    tok.save("data/processed/vocab.json")
    tok = WordTokenizer.load("data/processed/vocab.json")
    ids = tok.encode("أَنَا")            # → [1, 4+i, 2]   ([bos, sign, eos])
    text = tok.decode(ids)              # → "أَنَا"
"""
from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


class WordTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        # vocab MUST start with the four specials in the canonical order so
        # PAD=0, BOS=1, EOS=2, UNK=3 is invariant across save/load round-trips.
        if vocab[: len(SPECIALS)] != list(SPECIALS):
            raise ValueError(
                f"vocab must start with specials {SPECIALS} in order, "
                f"got {vocab[: len(SPECIALS)]}"
            )
        self.itos: list[str] = list(vocab)
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def n_signs(self) -> int:
        return self.vocab_size - len(SPECIALS)

    def encode(self, text: str, add_specials: bool = True) -> list[int]:
        """Encode a single MoSL label string to token ids.

        The input is treated as ONE atomic sign label (multi-word labels are
        not split on whitespace).  Returns [BOS, sign_id, EOS] when
        add_specials=True, else [sign_id].  Unknown labels map to [UNK].
        """
        sign = _nfc(text.strip())
        sign_id = self.stoi.get(sign, self.unk_id)
        if add_specials:
            return [self.bos_id, sign_id, self.eos_id]
        return [sign_id]

    def decode(self, ids: list[int], strip_specials: bool = True) -> str:
        """Decode token ids back to a string.  Joins remaining signs with ' | '
        as a separator since real labels can themselves contain spaces — using
        ' ' as a join would be ambiguous."""
        words: list[str] = []
        for i in ids:
            if not 0 <= i < self.vocab_size:
                continue
            tok = self.itos[i]
            if strip_specials and tok in SPECIALS:
                continue
            words.append(tok)
        return " | ".join(words)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])

    @classmethod
    def from_labels_csv(cls, csv_path: str | Path) -> "WordTokenizer":
        """Build a vocabulary covering every NFC-normalised sign label in
        the CSV (across all splits).  Sign tokens are added in sorted order
        for reproducibility."""
        signs: set[str] = set()
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                signs.add(_nfc(row["word_arabic"]))
        vocab = list(SPECIALS) + sorted(signs)
        return cls(vocab)


if __name__ == "__main__":
    # Quick smoke test if run as a script.
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/labels.csv"
    tok = WordTokenizer.from_labels_csv(csv_path)
    print(f"vocab size: {tok.vocab_size}  (= 4 specials + {tok.n_signs} signs)")
    sample = "أَنَا"
    enc = tok.encode(sample)
    print(f"encode({sample!r}) -> {enc}")
    print(f"decode({enc}) -> {tok.decode(enc)!r}")
    multi = "الأَمْنُ الوَطَنِيُّ"
    enc2 = tok.encode(multi)
    print(f"encode({multi!r}) -> {enc2}  (multi-word stays as one token)")
    unk = tok.encode("xyz_not_in_vocab")
    print(f"encode(unknown) -> {unk}  (should map to unk={tok.unk_id})")
