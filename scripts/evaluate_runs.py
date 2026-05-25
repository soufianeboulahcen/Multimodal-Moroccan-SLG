"""Evaluate trained SignLLM checkpoints on dev + test.

For each requested run directory:

  * Reload the model from `best.pt` and the saved `model_cfg`.
  * Run teacher-forced eval (cheap; reports pose MSE + log-T error).
  * Run autoregressive eval (paper-comparable; reports DTW mean / median).

All metrics are computed for both dev and test splits.  The combined results
are written to `runs/evaluation.json` for use by the report writeup, and a
table is printed to stdout.

Usage (inside container):
    docker/run.sh python scripts/evaluate_runs.py
        [--runs baseline_mse rl rl_plc]
        [--runs-dir runs]
        [--splits dev test]
        [--max-clips N]            # cap for fast smoke test of this script
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosl.data.dataset import MoSLSkelsDataset, mosl_collate
from mosl.model.signllm import SignLLM, SignLLMConfig
from mosl.text.tokenizer import WordTokenizer
from mosl.train.eval import evaluate_autoregressive, evaluate_teacher_forced


def load_run(run_dir: Path, device: torch.device) -> tuple[SignLLM, dict]:
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing best.pt in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = SignLLMConfig(**ckpt["model_cfg"])
    model = SignLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline_mse", "rl", "rl_plc"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Cap autoregressive eval for fast development feedback.")
    ap.add_argument("--vocab", default="data/processed/vocab.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    tok = WordTokenizer.load(args.vocab)

    runs_dir = Path(args.runs_dir)
    results: dict[str, dict] = {}
    loaders: dict[str, DataLoader] = {}

    # Build loaders once and reuse across runs.
    for split in args.splits:
        ds = MoSLSkelsDataset(split, tokenizer=tok)
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=mosl_collate, num_workers=0, pin_memory=True,
        )
        print(f"[eval] {split}: {len(ds)} clips")

    for run_name in args.runs:
        run_dir = runs_dir / run_name
        print(f"\n[eval] === {run_name} ({run_dir}) ===")
        model, ckpt = load_run(run_dir, device)
        results[run_name] = {
            "ckpt_epoch": ckpt.get("epoch"),
            "ckpt_step": ckpt.get("step"),
            "ckpt_metric": ckpt.get("metric"),
            "splits": {},
        }

        for split, dl in loaders.items():
            print(f"[eval]   {split}: ", end="", flush=True)
            t0 = time.perf_counter()
            tf = evaluate_teacher_forced(model, dl, device)
            t_tf = time.perf_counter() - t0

            t0 = time.perf_counter()
            ar = evaluate_autoregressive(model, dl, device, max_clips=args.max_clips)
            t_ar = time.perf_counter() - t0

            results[run_name]["splits"][split] = {**tf, **ar,
                                                   "tf_time_s": t_tf, "ar_time_s": t_ar}
            print(
                f"tf_pose_mse={tf['tf_pose_mse']:.5f}  "
                f"tf_log_T_err={tf['tf_log_T_abs_err']:.3f}  "
                f"ar_dtw_mean={ar['ar_dtw_mean']:.4f}  "
                f"ar_dtw_median={ar['ar_dtw_median']:.4f}  "
                f"[{t_tf:.0f}s tf + {t_ar:.0f}s ar]"
            )
        del model
        torch.cuda.empty_cache()

    # Persist + print final comparison table.
    out_path = runs_dir / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] wrote {out_path}")

    print("\n" + "=" * 84)
    print("Final comparison")
    print("=" * 84)
    print(f"{'run':<15}  {'split':<5}  {'tf_pose_mse':>11}  {'tf_log_T_err':>12}  "
          f"{'ar_dtw_mean':>11}  {'ar_dtw_median':>13}  {'n':>5}")
    print("-" * 84)
    for run_name in args.runs:
        for split in args.splits:
            r = results[run_name]["splits"][split]
            print(
                f"{run_name:<15}  {split:<5}  "
                f"{r['tf_pose_mse']:>11.5f}  "
                f"{r['tf_log_T_abs_err']:>12.3f}  "
                f"{r['ar_dtw_mean']:>11.4f}  "
                f"{r['ar_dtw_median']:>13.4f}  "
                f"{r['n_clips']:>5}"
            )
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
