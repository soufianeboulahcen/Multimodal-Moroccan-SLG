"""PyTorch dataset + collate for the SignLLM-on-MoSL training pipeline.

Reads the three line-aligned files our Phase 2 pipeline produces:

    <mode>.skels    one space-separated sequence per clip:
                    151 floats per frame (= 150 pose coords + 1 time marker),
                    repeated T times for a clip with T frames.
    <mode>.text     one MoSL label per clip (Arabic, NFC-normalised).
    <mode>.files    line-aligned clip identifiers (kept for diagnostics).

Returns one sample as a dict:

    {
        "text_ids":  LongTensor (L_text,)     # [bos, sign_id, eos]
        "pose":      FloatTensor (T, 150)     # 150 pose coords per frame
        "time":      FloatTensor (T,)         # frame-time markers, ∈ (0,1]
        "n_frames":  int                       # T
        "clip_id":   str                       # for logging
        # Extended fields (present when available):
        "signer_id": int                       # -1 if not annotated
        "face_keypoints": FloatTensor (T, 478, 3) or None
        "face_confidence": FloatTensor (T,) or None
    }

The default `collate_fn` pads variable-T pose sequences and emits attention masks.
Text sequences are length-3 in our setting (always [bos, sign, eos]) so they
don't need padding, but we pad for generality.

Extended fields (face_keypoints, signer_id) are loaded opportunistically:
  - face_keypoints: loaded from data/processed/keypoints_2d/<cat>/<clip>.npz
    if the file contains a "face_keypoints" array (added by extract_face_keypoints.py)
  - signer_id: loaded from data/labels.csv if the "signer_id" column exists

Usage:
    from torch.utils.data import DataLoader
    from mosl.data.dataset import MoSLSkelsDataset, mosl_collate
    from mosl.text.tokenizer import WordTokenizer

    tok = WordTokenizer.load("data/processed/vocab.json")
    ds = MoSLSkelsDataset("train", tokenizer=tok)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=mosl_collate)
    for batch in dl:
        # batch["pose"]:        (B, T_max, 150)
        # batch["pose_mask"]:   (B, T_max)        bool — True for real frames
        # batch["text_ids"]:    (B, L_text_max)
        # batch["text_mask"]:   (B, L_text_max)
        # batch["time"]:        (B, T_max)
        # batch["n_frames"]:    (B,)
        # batch["signer_ids"]:  (B,)              int64, -1 if unknown
        # batch["has_face"]:    bool               True if any sample has face data
        # batch["face_kps"]:    (B, T_max, 478, 3) or None
        ...
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from mosl.text.tokenizer import WordTokenizer

# Per-frame layout of <mode>.skels: 150 pose coords + 1 time marker.
COORDS_PER_FRAME = 150
FLOATS_PER_FRAME = COORDS_PER_FRAME + 1   # = 151

N_FACE_LANDMARKS = 478   # MediaPipe FaceMesh with refine_landmarks=True


def _default_p2s_dir(repo_root: Path) -> Path:
    return repo_root / "third_party" / "Prompt2Sign" / "tools" / "2D_to_3D" / "final_data"


def _load_signer_ids(labels_csv: Path) -> dict[str, int]:
    """Load signer_id mapping from labels.csv.

    Returns a dict mapping clip stem → signer_id (int).
    Returns empty dict if the signer_id column does not exist yet.
    """
    if not labels_csv.exists():
        return {}
    with open(labels_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "signer_id" not in (reader.fieldnames or []):
            return {}
        mapping = {}
        for row in reader:
            path = row.get("relative_path", "")
            stem = Path(path).stem if path else ""
            sid = row.get("signer_id", "").strip()
            if stem and sid and sid not in ("", "unknown", "-1"):
                try:
                    mapping[stem] = int(sid)
                except ValueError:
                    pass
    return mapping


class MoSLSkelsDataset(Dataset):
    """One training sample = one MoSL clip.  Text is a single Arabic label
    encoded as `[bos, sign_id, eos]`; target is the full pose-frame sequence.

    Parameters
    ----------
    mode
        One of {"train", "dev", "test"}.  Mirrors the upstream pipeline.
    tokenizer
        A loaded `WordTokenizer`.  Must be the same vocab used at inference.
    final_data_dir
        Directory containing `<mode>.{skels,text,files}`.  Defaults to the
        Prompt2Sign tree under `third_party/`.
    repo_root
        Project root.  Defaults to two parents up from this file (works inside
        the container's `/workspace/PFE-SOUFIAN` and on the host alike).
    load_face
        If True, attempt to load face_keypoints from the NPZ files.
        Silently skips clips where face data is not yet extracted.
    """

    def __init__(
        self,
        mode: str,
        tokenizer: WordTokenizer,
        final_data_dir: Optional[Path] = None,
        repo_root: Optional[Path] = None,
        load_face: bool = False,
    ) -> None:
        if mode not in {"train", "dev", "test"}:
            raise ValueError(f"mode must be one of train/dev/test, got {mode!r}")
        self.mode = mode
        self.tokenizer = tokenizer
        self.load_face = load_face

        repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._repo_root = repo_root
        final_data_dir = final_data_dir or _default_p2s_dir(repo_root)

        skels_path = final_data_dir / f"{mode}.skels"
        text_path = final_data_dir / f"{mode}.text"
        files_path = final_data_dir / f"{mode}.files"
        for p in (skels_path, text_path, files_path):
            if not p.exists():
                raise FileNotFoundError(f"missing pipeline output: {p}")

        # Load the three line-aligned files into memory.  The skels file is
        # ~210 MB for train (1,674 clips × ~125 KB) — fine to hold in RAM
        # given the DGX Spark has 121 GiB.  If we ever hit memory pressure we
        # can lazy-load per __getitem__ via line offsets.
        with open(skels_path, encoding="utf-8") as f:
            self._skels_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        with open(text_path, encoding="utf-8") as f:
            self._text_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        with open(files_path, encoding="utf-8") as f:
            self._file_ids = [ln.rstrip("\n") for ln in f if ln.strip()]

        if not (len(self._skels_lines) == len(self._text_lines) == len(self._file_ids)):
            raise ValueError(
                f"length mismatch in {mode}: "
                f"skels={len(self._skels_lines)} text={len(self._text_lines)} "
                f"files={len(self._file_ids)}"
            )

        # Load signer_id mapping (empty dict if column not yet annotated)
        self._signer_ids = _load_signer_ids(repo_root / "data" / "labels.csv")

        # Index NPZ paths for face loading (lazy — only opened in __getitem__)
        self._keypoints_dir = repo_root / "data" / "processed" / "keypoints_2d"

    def __len__(self) -> int:
        return len(self._skels_lines)

    def __getitem__(self, idx: int) -> dict:
        # Parse the skels line into (T, 151) and split coords / time.
        floats = np.fromstring(self._skels_lines[idx], sep=" ", dtype=np.float32)
        if floats.size % FLOATS_PER_FRAME != 0:
            raise ValueError(
                f"clip {self._file_ids[idx]}: skels line has {floats.size} "
                f"floats, not divisible by {FLOATS_PER_FRAME}"
            )
        frames = floats.reshape(-1, FLOATS_PER_FRAME)
        pose = torch.from_numpy(frames[:, :COORDS_PER_FRAME])     # (T, 150)
        time = torch.from_numpy(frames[:, COORDS_PER_FRAME])      # (T,)

        text_ids = torch.tensor(
            self.tokenizer.encode(self._text_lines[idx], add_specials=True),
            dtype=torch.long,
        )

        clip_id = self._file_ids[idx]
        clip_stem = Path(clip_id).stem if "/" in clip_id or "\\" in clip_id else clip_id

        # Signer ID (-1 = not yet annotated)
        signer_id = self._signer_ids.get(clip_stem, -1)

        # Face keypoints (optional — loaded only if load_face=True and data exists)
        face_keypoints = None
        face_confidence = None
        if self.load_face:
            face_keypoints, face_confidence = self._load_face(clip_stem, pose.shape[0])

        sample = {
            "text_ids": text_ids,
            "pose": pose,
            "time": time,
            "n_frames": int(pose.shape[0]),
            "clip_id": clip_id,
            "signer_id": signer_id,
        }
        if face_keypoints is not None:
            sample["face_keypoints"] = face_keypoints
            sample["face_confidence"] = face_confidence
        return sample

    def _load_face(
        self, clip_stem: str, T: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load face_keypoints from the NPZ file for this clip, if available."""
        # Search across all category subdirectories
        for cat_dir in self._keypoints_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            npz_path = cat_dir / f"{clip_stem}.npz"
            if npz_path.exists():
                try:
                    data = np.load(npz_path, allow_pickle=False)
                    if "face_keypoints" in data:
                        fkps = data["face_keypoints"]   # (T_npz, 478, 3)
                        fconf = data.get("face_confidence",
                                         np.ones(fkps.shape[0], dtype=np.float32))
                        # Align to pose T
                        T_npz = fkps.shape[0]
                        if T_npz < T:
                            pad = np.zeros((T - T_npz, N_FACE_LANDMARKS, 3), dtype=np.float32)
                            fkps = np.concatenate([fkps, pad], axis=0)
                            fconf = np.concatenate([fconf, np.zeros(T - T_npz, dtype=np.float32)])
                        elif T_npz > T:
                            fkps = fkps[:T]
                            fconf = fconf[:T]
                        return torch.from_numpy(fkps), torch.from_numpy(fconf)
                except Exception:
                    pass
        return None, None


def mosl_collate(batch: list[dict]) -> dict:
    """Pad variable-length pose and text sequences and produce attention masks.

    Pose padding uses zeros — `pose_mask` distinguishes pad frames from real
    ones.  Text padding uses `<pad>` from the tokenizer (id 0); we expose
    `text_mask` for symmetry.

    Extended fields:
      - signer_ids: (B,) int64 — -1 if not annotated
      - face_kps:   (B, T_max, 478, 3) float32 — None if no sample has face data
      - face_conf:  (B, T_max) float32 — None if no sample has face data
    """
    B = len(batch)
    T_max = max(s["n_frames"] for s in batch)
    L_max = max(s["text_ids"].size(0) for s in batch)

    pose = torch.zeros(B, T_max, COORDS_PER_FRAME, dtype=torch.float32)
    time = torch.zeros(B, T_max, dtype=torch.float32)
    pose_mask = torch.zeros(B, T_max, dtype=torch.bool)

    text_ids = torch.zeros(B, L_max, dtype=torch.long)            # 0 = <pad>
    text_mask = torch.zeros(B, L_max, dtype=torch.bool)

    n_frames = torch.empty(B, dtype=torch.long)
    signer_ids = torch.full((B,), -1, dtype=torch.long)
    clip_ids: list[str] = []

    # Face data — only allocate if at least one sample has it
    has_face = any("face_keypoints" in s for s in batch)
    face_kps = torch.zeros(B, T_max, N_FACE_LANDMARKS, 3, dtype=torch.float32) if has_face else None
    face_conf = torch.zeros(B, T_max, dtype=torch.float32) if has_face else None

    for b, s in enumerate(batch):
        T = s["n_frames"]
        L = s["text_ids"].size(0)
        pose[b, :T] = s["pose"]
        time[b, :T] = s["time"]
        pose_mask[b, :T] = True
        text_ids[b, :L] = s["text_ids"]
        text_mask[b, :L] = True
        n_frames[b] = T
        signer_ids[b] = s.get("signer_id", -1)
        clip_ids.append(s["clip_id"])

        if has_face and "face_keypoints" in s:
            fkps = s["face_keypoints"]   # (T, 478, 3)
            fconf = s["face_confidence"] # (T,)
            face_kps[b, :T] = fkps
            face_conf[b, :T] = fconf

    result = {
        "pose": pose,
        "pose_mask": pose_mask,
        "time": time,
        "text_ids": text_ids,
        "text_mask": text_mask,
        "n_frames": n_frames,
        "signer_ids": signer_ids,
        "clip_ids": clip_ids,
        "has_face": has_face,
    }
    if has_face:
        result["face_kps"] = face_kps
        result["face_conf"] = face_conf
    return result


if __name__ == "__main__":
    # Smoke test: load tokenizer + dev set, build a small batch, print shapes.
    import sys
    from torch.utils.data import DataLoader

    tok = WordTokenizer.load("data/processed/vocab.json")
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    ds = MoSLSkelsDataset(mode, tokenizer=tok)
    print(f"{mode} set: {len(ds)} clips")
    one = ds[0]
    print(f"  sample[0]: clip={one['clip_id']!r}")
    print(f"             text_ids={one['text_ids'].tolist()}  -> {tok.decode(one['text_ids'].tolist())!r}")
    print(f"             pose={tuple(one['pose'].shape)}  time={tuple(one['time'].shape)}  T={one['n_frames']}")
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=mosl_collate)
    batch = next(iter(dl))
    print()
    print(f"first batch (B=4):")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:>12}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:>12}  {type(v).__name__}={v if not isinstance(v, list) else f'[{len(v)} items]'}")
