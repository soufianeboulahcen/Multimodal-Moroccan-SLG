"""CLI entry point for SignLLM training.

Usage (inside container):
    docker/run.sh python scripts/train_signllm.py --mode mse --run-name baseline
    docker/run.sh python scripts/train_signllm.py --mode rl_plc --run-name rl_plc_v1

Smoke options:
    --max-epochs 2          (sanity-check the loop completes)
    --batch-size 4          (use less memory)

The default config lives in mosl/train/train.py (TrainConfig); CLI args only
override the few knobs we expect to vary across runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `mosl` importable when running as a plain script (cwd may be anywhere).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosl.model.signllm import SignLLMConfig
from mosl.text.tokenizer import WordTokenizer
from mosl.train.losses import LossConfig, PLCConfig
from mosl.train.train import TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mse", "rl", "rl_plc"], required=True,
                   help="Loss configuration to use (paper Table 5 ablation row).")
    p.add_argument("--run-name", required=True,
                   help="Name of the output subdirectory under runs/.")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--length-weight", type=float, default=1.0)
    p.add_argument("--plc-eta", type=float, default=1.0)
    p.add_argument("--plc-skip-quantile", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--vocab", default="data/processed/vocab.json")
    args = p.parse_args()

    tok = WordTokenizer.load(args.vocab)
    model_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    train_cfg = TrainConfig(
        out_dir=args.out_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    loss_cfg = LossConfig(
        mode=args.mode,
        length_weight=args.length_weight,
        plc=PLCConfig(
            eta=args.plc_eta,
            skip_below_quantile=args.plc_skip_quantile,
            enabled=(args.mode == "rl_plc"),
        ),
    )

    print(f"[main] model_cfg={model_cfg}")
    print(f"[main] train_cfg={train_cfg}")
    print(f"[main] loss_cfg=mode={loss_cfg.mode} length_w={loss_cfg.length_weight} "
          f"plc_eta={loss_cfg.plc.eta} plc_q={loss_cfg.plc.skip_below_quantile}")

    summary = train(model_cfg=model_cfg, train_cfg=train_cfg, loss_cfg=loss_cfg, tokenizer=tok)
    print(f"[main] summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
