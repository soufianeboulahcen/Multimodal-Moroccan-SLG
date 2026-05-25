"""Training loop for SignLLM on MoSL.

Per docs/DECISIONS.md (locked 2026-05-10):
  * Optimizer: Adam (NOT AdamW — Vaswani 2017 default).
  * LR schedule: Noam (Vaswani §5.3) with warmup_steps=4000.
  * Batch size: 32.
  * Training budget: up to 200 epochs with early stopping on dev TF MSE.
  * Loss mode: chosen via LossConfig.mode ∈ {mse, rl, rl_plc}.

Per-epoch eval is teacher-forced (fast).  Autoregressive DTW is run only at
the end (or on demand) because per-clip generation is O(T) forward passes.

Outputs are written to a run directory:
    <out_dir>/<run_name>/
        config.json          — full TrainConfig + LossConfig serialised
        log.jsonl            — per-step + per-epoch metrics
        best.pt              — best-on-dev checkpoint
        last.pt              — most-recent checkpoint
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from mosl.data.dataset import MoSLSkelsDataset, mosl_collate
from mosl.model.signllm import SignLLM, SignLLMConfig
from mosl.text.tokenizer import WordTokenizer
from mosl.train.eval import evaluate_teacher_forced
from mosl.train.losses import LossConfig, signllm_step_loss
from mosl.train.scheduler import NoamLR


@dataclass
class TrainConfig:
    out_dir: str = "runs"
    run_name: str = "default"
    batch_size: int = 32
    max_epochs: int = 200
    early_stop_patience: int = 20         # epochs without improvement
    log_every_steps: int = 50
    warmup_steps: int = 4000              # Vaswani 2017 default
    grad_clip: float = 1.0                # gradient clipping (paper-silent; standard safety)
    seed: int = 0
    num_workers: int = 4
    device: str = "cuda"
    eval_max_clips: Optional[int] = None  # cap dev eval (debugging only)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def train(
    model_cfg: SignLLMConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    tokenizer: WordTokenizer,
    repo_root: Optional[Path] = None,
) -> dict:
    """Run end-to-end training.  Returns the best-dev metrics dict."""
    torch.manual_seed(train_cfg.seed)

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    run_dir = Path(train_cfg.out_dir) / train_cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the full config so re-runs and report writeups are unambiguous.
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "loss": {"mode": loss_cfg.mode, "length_weight": loss_cfg.length_weight,
                          "plc": asdict(loss_cfg.plc)},
            },
            f, indent=2, ensure_ascii=False,
        )

    log_path = run_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)

    def _log(record: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- Data ---------------------------------------------------------------
    train_ds = MoSLSkelsDataset("train", tokenizer=tokenizer, repo_root=repo_root)
    dev_ds = MoSLSkelsDataset("dev", tokenizer=tokenizer, repo_root=repo_root)
    train_dl = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=mosl_collate, num_workers=train_cfg.num_workers, pin_memory=True,
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=mosl_collate, num_workers=train_cfg.num_workers, pin_memory=True,
    )
    print(f"[train] train={len(train_ds)} dev={len(dev_ds)}  "
          f"steps/epoch={(len(train_ds) + train_cfg.batch_size - 1) // train_cfg.batch_size}")

    # --- Model + optim -----------------------------------------------------
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = SignLLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model: {n_params:,} params on {device}")

    # Vaswani 2017: Adam(beta1=0.9, beta2=0.98, eps=1e-9) + Noam schedule.  The
    # base LR in the optimizer is 1.0 — Noam scales it via lr_lambda.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=train_cfg.warmup_steps)

    # --- Train --------------------------------------------------------------
    best_metric = float("inf")
    best_epoch = -1
    no_improve = 0
    global_step = 0
    train_start = time.time()

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_start = time.time()
        running = {"loss": 0.0, "pose_loss": 0.0, "length_loss": 0.0, "n_kept": 0, "steps": 0}

        for batch in train_dl:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                text_ids=batch["text_ids"], text_mask=batch["text_mask"],
                pose_target=batch["pose"], time=batch["time"], pose_mask=batch["pose_mask"],
            )
            loss_info = signllm_step_loss(out, batch, loss_cfg)
            loss_info["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            running["loss"] += loss_info["loss"].item()
            running["pose_loss"] += loss_info["pose_loss"].item()
            running["length_loss"] += loss_info["length_loss"].item()
            running["n_kept"] += loss_info["n_kept"]
            running["steps"] += 1

            if global_step % train_cfg.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                _log({
                    "kind": "step", "epoch": epoch, "step": global_step, "lr": lr,
                    "loss": loss_info["loss"].item(),
                    "pose_loss": loss_info["pose_loss"].item(),
                    "length_loss": loss_info["length_loss"].item(),
                    "n_kept": loss_info["n_kept"],
                })

        # End of epoch: eval, log, checkpoint
        epoch_train_time = time.time() - epoch_start
        train_avg = {k: v / max(running["steps"], 1) for k, v in running.items() if k != "n_kept"}

        eval_metrics = evaluate_teacher_forced(model, dev_dl, device)
        epoch_metric = eval_metrics["tf_pose_mse"]   # the value used for early stopping

        is_best = epoch_metric < best_metric
        record = {
            "kind": "epoch", "epoch": epoch, "step": global_step,
            "train": train_avg, "dev": eval_metrics,
            "epoch_train_time_s": epoch_train_time,
            "is_best": is_best,
        }
        _log(record)

        # Checkpoint every epoch (last) + best
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "loss_cfg": {"mode": loss_cfg.mode},
            "epoch": epoch, "step": global_step,
            "metric": epoch_metric, "is_best": is_best,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")
            best_metric = epoch_metric
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[train] epoch {epoch:>3}  step {global_step:>6}  "
            f"train_loss={train_avg['loss']:.4f}  pose={train_avg['pose_loss']:.4f}  "
            f"len={train_avg['length_loss']:.4f}  "
            f"dev_pose_mse={eval_metrics['tf_pose_mse']:.4f}  "
            f"dev_log_T_err={eval_metrics['tf_log_T_abs_err']:.3f}  "
            f"{'(best)' if is_best else f'(no improve {no_improve})'}  "
            f"[{epoch_train_time:.1f}s]"
        )

        if no_improve >= train_cfg.early_stop_patience:
            print(f"[train] early stop at epoch {epoch} (no improvement for {no_improve} epochs)")
            break

    total_time = time.time() - train_start
    print(f"[train] done.  best epoch={best_epoch}  best dev_pose_mse={best_metric:.4f}  "
          f"total time={total_time / 60:.1f} min")

    summary = {"best_epoch": best_epoch, "best_metric": best_metric,
               "total_time_min": total_time / 60}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
