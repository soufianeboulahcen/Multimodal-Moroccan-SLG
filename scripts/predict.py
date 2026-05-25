"""Run inference on a single text input through a trained SignLLM checkpoint.

Loads the chosen checkpoint, encodes the input Arabic word with the project
tokenizer, runs autoregressive generation, and saves the predicted pose
sequence as a NumPy NPZ.

Usage (inside container):

    # by typing the Arabic word directly (UTF-8 in your shell)
    docker/run.sh python scripts/predict.py --run baseline_mse \\
        --text 'أَنَا' --out predictions/ana.npz

    # or by referencing a clip in labels.csv (handles Arabic encoding for you,
    # also loads the ground-truth pose for side-by-side comparison)
    docker/run.sh python scripts/predict.py --run baseline_mse \\
        --clip-stem 'أَنَا' --category Pronouns

The output NPZ contains:
    pose       (T_pred, 150)  predicted pose coords
    time       (T_pred,)      predicted frame-time markers
    T_pred     int             predicted clip length
    log_T_pred float          length-head raw output
    text       str             input text (UTF-8)
    run        str             which checkpoint was used
    is_oov     bool            whether the input was out-of-vocabulary
    target_pose (T_true, 150)  ground-truth pose if --clip-stem given (else absent)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosl.model.signllm import SignLLM, SignLLMConfig
from mosl.text.tokenizer import WordTokenizer


def lookup_clip(category: str, stem: str) -> tuple[str, str | None]:
    """Resolve --clip-stem against labels.csv → (text, relative_path or None)."""
    labels = ROOT / "data" / "labels.csv"
    if not labels.exists():
        return stem, None
    with open(labels, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["category"] == category and Path(row["relative_path"]).stem == stem:
                return row["word_arabic"], row["relative_path"]
    return stem, None


def load_target_pose_from_skels(stem: str, category: str) -> np.ndarray | None:
    """Find the clip in any of train/dev/test.skels and return its (T, 150) array."""
    base = ROOT / "third_party" / "Prompt2Sign" / "tools" / "2D_to_3D" / "final_data"
    target_id = f"{category}__{stem}"
    for mode in ("train", "dev", "test"):
        files_path = base / f"{mode}.files"
        skels_path = base / f"{mode}.skels"
        if not files_path.exists() or not skels_path.exists():
            continue
        with open(files_path, encoding="utf-8") as f:
            file_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        try:
            idx = next(i for i, ln in enumerate(file_lines)
                       if ln.endswith(target_id) or ln == f"{mode}/{target_id}")
        except StopIteration:
            continue
        with open(skels_path, encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j == idx:
                    floats = np.fromstring(line, sep=" ", dtype=np.float32)
                    frames = floats.reshape(-1, 151)
                    return frames[:, :150]
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True,
                   help="Run name under runs/ (e.g., baseline_mse, rl, rl_plc)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Arabic word/sign to generate")
    g.add_argument("--clip-stem", help="A clip filename stem from labels.csv "
                                       "(implies looking up the canonical Arabic form)")
    p.add_argument("--category", default="Pronouns",
                   help="Category for --clip-stem lookup (Pronouns, Diverse, ...)")
    p.add_argument("--out", default=None,
                   help="Output .npz path (default: predictions/<run>_<text>.npz)")
    p.add_argument("--vocab", default=str(ROOT / "data" / "processed" / "vocab.json"))
    p.add_argument("--max-T", type=int, default=None,
                   help="Cap predicted length (default: model's max_pose_len)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] device: {device}")

    # Resolve text
    target_pose = None
    if args.clip_stem:
        text, _ = lookup_clip(args.category, args.clip_stem)
        target_pose = load_target_pose_from_skels(args.clip_stem, args.category)
        print(f"[predict] resolved clip-stem={args.clip_stem!r} -> text={text!r} "
              f"(target_pose: {'found' if target_pose is not None else 'NOT found'})")
    else:
        text = args.text

    # Load tokenizer + checkpoint
    tok = WordTokenizer.load(args.vocab)
    ckpt_path = ROOT / "runs" / args.run / "best.pt"
    if not ckpt_path.exists():
        print(f"[predict] error: no checkpoint at {ckpt_path}", file=sys.stderr)
        return 1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = SignLLMConfig(**ckpt["model_cfg"])
    model = SignLLM(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[predict] loaded {args.run!r}  epoch={ckpt['epoch']}  "
          f"best_dev_pose_mse={ckpt['metric']:.5f}")

    # Encode + check OOV
    ids = tok.encode(text)
    sign_id = ids[1]                       # [bos, sign_id, eos]
    is_oov = sign_id == tok.unk_id
    print(f"[predict] text={text!r}  token_id={sign_id}"
          f"{'  (OUT-OF-VOCAB → predictions will be unreliable)' if is_oov else ''}")

    # Generate
    text_ids = torch.tensor([ids], dtype=torch.long, device=device)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    with torch.no_grad():
        gen = model.generate(text_ids, text_mask, max_T=args.max_T)
    T_pred = int(gen["lengths"][0].item())
    log_T = float(gen["log_T_pred"][0].item())
    pose = gen["pose"][0, :T_pred].cpu().numpy()
    time_marks = np.linspace(1.0 / max(T_pred, 1), 1.0, T_pred, dtype=np.float32)
    print(f"[predict] generated T={T_pred} frames (log_T={log_T:.3f})  "
          f"pose shape={pose.shape}")

    # Save
    if args.out is None:
        safe_text = "".join(c for c in text if c.isalnum() or c in "-_")
        if not safe_text:
            safe_text = "text"
        out_path = ROOT / "predictions" / f"{args.run}_{safe_text}.npz"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "pose": pose,
        "time": time_marks,
        "T_pred": np.int32(T_pred),
        "log_T_pred": np.float32(log_T),
        "text": text,
        "run": args.run,
        "is_oov": np.bool_(is_oov),
    }
    if target_pose is not None:
        save_kwargs["target_pose"] = target_pose
        # quick frame-by-frame MSE for sanity
        T_min = min(target_pose.shape[0], pose.shape[0])
        if T_min > 0:
            naive_mse = float(((pose[:T_min] - target_pose[:T_min]) ** 2).mean())
            print(f"[predict] aligned-frame MSE (first {T_min} frames vs target): "
                  f"{naive_mse:.5f}  (T_target={target_pose.shape[0]})")

    np.savez(out_path, **save_kwargs)
    print(f"[predict] saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
