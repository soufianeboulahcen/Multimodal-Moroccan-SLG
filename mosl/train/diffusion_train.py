"""Diffusion training loop for MoSL motion generation.

Trains the MDMDenoiser on top of the existing MoSL .skels dataset.
The SignLLM text encoder is frozen; only the denoiser is updated.

Compatibility guarantees:
  - Reads the same final_data/{train,dev}.{skels,text,files} produced by the
    existing Prompt2Sign pipeline — no data format changes required.
  - Uses the same MoSLSkelsDataset / mosl_collate from mosl/data/dataset.py.
  - Checkpoint format mirrors the existing SignLLM runs/ layout so evaluation
    scripts can be reused.

Training objective (x0-prediction):
  1. Sample a clean pose x0 from the dataset.
  2. Sample a random timestep t ~ Uniform[0, T).
  3. Add noise: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps.
  4. Denoiser predicts x0_pred from (x_t, t, text).
  5. Loss = diffusion_loss(x0_pred, x0) + lambda_vel * velocity_loss
           + lambda_acc * acceleration_loss + lambda_bone * bone_length_loss.

Losses are defined in mosl/train/losses.py (extended version).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mosl.data.dataset import MoSLSkelsDataset, mosl_collate
from mosl.model.mdm_denoiser import MDMConfig, MDMDenoiser, _HAS_SDPA
from mosl.model.signllm import SignLLMConfig
from mosl.text.tokenizer import WordTokenizer
from mosl.train.noise_schedule import NoiseSchedule, savgol_smooth
from mosl.train.losses_diffusion import DiffusionLossConfig, diffusion_step_loss


@dataclass
class DiffusionTrainConfig:
    out_dir: str = "runs_diffusion"
    run_name: str = "mdm_mosl"

    # Data
    batch_size: int = 32
    num_workers: int = 4

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 300
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    early_stop_patience: int = 30

    # Diffusion
    n_diffusion_steps: int = 1000
    n_sample_steps: int = 50        # DDIM steps at eval time
    noise_schedule: str = "cosine"

    # Classifier-free guidance
    cfg_dropout: float = 0.1        # fraction of training steps with null text conditioning
    cfg_scale: float = 2.5          # guidance scale at inference

    # Logging
    log_every_steps: int = 50
    eval_every_epochs: int = 5      # run DDIM generation on dev subset

    # Hardware
    device: str = "cuda"
    bf16: bool = True               # use bfloat16 on Ampere/Blackwell
    grad_checkpoint: bool = False   # gradient checkpointing (saves ~40% VRAM)
    use_flash: bool = True          # Flash Attention via SDPA

    # Checkpoints
    signllm_checkpoint: Optional[str] = None   # path to SignLLM best.pt
    resume_from: Optional[str] = None          # resume diffusion training

    seed: int = 42


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _warmup_cosine_lr(optimizer: torch.optim.Optimizer,
                      step: int, warmup_steps: int, base_lr: float) -> None:
    """Linear warmup then cosine decay (no restart)."""
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        # Cosine decay to 1e-6 floor
        progress = (step - warmup_steps) / max(1, 1_000_000 - warmup_steps)
        lr = max(1e-6, base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _log(run_dir: Path, record: dict) -> None:
    with open(run_dir / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def train_diffusion(
    mdm_cfg: MDMConfig,
    signllm_cfg: SignLLMConfig,
    train_cfg: DiffusionTrainConfig,
    loss_cfg: DiffusionLossConfig,
    tokenizer: WordTokenizer,
    repo_root: Optional[Path] = None,
) -> dict:
    """Run the full diffusion training loop.

    Returns the best-dev metrics dict.
    """
    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    run_dir = Path(train_cfg.out_dir) / train_cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "mdm": asdict(mdm_cfg),
            "train": asdict(train_cfg),
            "loss": asdict(loss_cfg),
        }, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_ds = MoSLSkelsDataset("train", tokenizer=tokenizer, repo_root=repo_root)
    dev_ds = MoSLSkelsDataset("dev", tokenizer=tokenizer, repo_root=repo_root)
    train_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                          shuffle=True, num_workers=train_cfg.num_workers,
                          collate_fn=mosl_collate, pin_memory=True, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=train_cfg.batch_size,
                        shuffle=False, num_workers=train_cfg.num_workers,
                        collate_fn=mosl_collate, pin_memory=True)

    print(f"train: {len(train_ds)} clips  dev: {len(dev_ds)} clips")

    # -----------------------------------------------------------------------
    # Model + noise schedule
    # -----------------------------------------------------------------------
    # Propagate hardware flags into model config
    mdm_cfg.grad_checkpoint = train_cfg.grad_checkpoint
    mdm_cfg.use_flash = train_cfg.use_flash
    mdm_cfg.cfg_dropout = train_cfg.cfg_dropout
    mdm_cfg.cfg_scale = train_cfg.cfg_scale

    model = MDMDenoiser(mdm_cfg, signllm_cfg).to(device)

    if train_cfg.signllm_checkpoint:
        print(f"loading SignLLM text encoder from {train_cfg.signllm_checkpoint}")
        model.load_text_encoder(train_cfg.signllm_checkpoint)
    else:
        model.freeze_text_encoder()
        print("SignLLM text encoder frozen (random weights — load a checkpoint for best results)")

    flash_status = "enabled" if (train_cfg.use_flash and _HAS_SDPA) else "disabled"
    print(f"Flash Attention: {flash_status}")
    print(f"Gradient checkpointing: {'enabled' if train_cfg.grad_checkpoint else 'disabled'}")
    print(f"CFG dropout: {train_cfg.cfg_dropout}  CFG scale: {train_cfg.cfg_scale}")

    schedule = NoiseSchedule(
        n_steps=train_cfg.n_diffusion_steps,
        schedule=train_cfg.noise_schedule,
    ).to(device)

    # -----------------------------------------------------------------------
    # Optimiser (only denoiser params — text encoder is frozen)
    # -----------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay, betas=(0.9, 0.999))

    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg.bf16 and device.type == "cuda"))
    amp_dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_dev_loss = float("inf")
    patience_counter = 0

    if train_cfg.resume_from:
        ckpt = torch.load(train_cfg.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_dev_loss = ckpt.get("best_dev_loss", float("inf"))
        print(f"resumed from epoch {start_epoch}  step {global_step}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, train_cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in train_dl:
            batch = _move(batch, device)
            x0 = batch["pose"]                  # (B, T, 150)
            pose_mask = batch["pose_mask"]       # (B, T)
            text_ids = batch["text_ids"]
            text_mask = batch["text_mask"]

            # Sample random timesteps
            B = x0.size(0)
            t_int = torch.randint(0, train_cfg.n_diffusion_steps, (B,), device=device)

            # Forward diffusion: add noise
            x_t, _ = schedule.q_sample(x0, t_int)

            # Classifier-free guidance: randomly drop text conditioning
            drop_text = None
            if train_cfg.cfg_dropout > 0:
                drop_text = torch.rand(B, device=device) < train_cfg.cfg_dropout

            # Denoiser forward
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=(train_cfg.bf16 and device.type == "cuda")):
                x0_pred = model(x_t, t_int, text_ids, text_mask,
                                pose_mask=pose_mask, drop_text=drop_text)
                losses = diffusion_step_loss(x0_pred, x0, pose_mask, loss_cfg)
                loss = losses["loss"]

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            _warmup_cosine_lr(optimizer, global_step, train_cfg.warmup_steps, train_cfg.lr)

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % train_cfg.log_every_steps == 0:
                record = {
                    "step": global_step, "epoch": epoch,
                    "loss": losses["loss"].item(),
                    "diffusion_loss": losses["diffusion_loss"].item(),
                    "velocity_loss": losses["velocity_loss"].item(),
                    "acceleration_loss": losses["acceleration_loss"].item(),
                    "bone_loss": losses["bone_loss"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                _log(run_dir, record)

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0

        # -----------------------------------------------------------------------
        # Dev evaluation
        # -----------------------------------------------------------------------
        dev_loss = _eval_dev(model, schedule, dev_dl, loss_cfg, device,
                             train_cfg.bf16, amp_dtype)

        print(f"epoch {epoch:4d}  train={avg_train_loss:.4f}  dev={dev_loss:.4f}  "
              f"time={elapsed:.0f}s")
        _log(run_dir, {"epoch": epoch, "train_loss": avg_train_loss,
                       "dev_loss": dev_loss, "elapsed": elapsed})

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dev_loss": best_dev_loss,
            "mdm_config": asdict(mdm_cfg),
        }
        torch.save(ckpt_data, run_dir / "last.pt")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(ckpt_data, run_dir / "best.pt")
            print(f"  ✓ new best dev loss: {best_dev_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stop_patience:
                print(f"early stopping at epoch {epoch} (patience={train_cfg.early_stop_patience})")
                break

    return {"best_dev_loss": best_dev_loss, "epochs_trained": epoch + 1}


@torch.no_grad()
def _eval_dev(
    model: MDMDenoiser,
    schedule: NoiseSchedule,
    dev_dl: DataLoader,
    loss_cfg: DiffusionLossConfig,
    device: torch.device,
    bf16: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Teacher-forced dev loss: add noise at random t, predict x0, compute loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dev_dl:
        batch = _move(batch, device)
        x0 = batch["pose"]
        pose_mask = batch["pose_mask"]
        text_ids = batch["text_ids"]
        text_mask = batch["text_mask"]

        B = x0.size(0)
        t_int = torch.randint(0, schedule.n_steps, (B,), device=device)
        x_t, _ = schedule.q_sample(x0, t_int)

        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=(bf16 and device.type == "cuda")):
            x0_pred = model(x_t, t_int, text_ids, text_mask, pose_mask=pose_mask)
            losses = diffusion_step_loss(x0_pred, x0, pose_mask, loss_cfg)

        total_loss += losses["loss"].item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser(description="Train MDM diffusion model on MoSL")
    parser.add_argument("--run-name", default="mdm_mosl")
    parser.add_argument("--out-dir", default="runs_diffusion")
    parser.add_argument("--signllm-checkpoint", default=None,
                        help="Path to SignLLM best.pt (recommended)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing (~40%% VRAM savings)")
    parser.add_argument("--no-flash", action="store_true",
                        help="Disable Flash Attention (use standard MHA)")
    parser.add_argument("--cfg-dropout", type=float, default=0.1,
                        help="Classifier-free guidance dropout rate (default: 0.1)")
    parser.add_argument("--cfg-scale", type=float, default=2.5,
                        help="CFG guidance scale at inference (default: 2.5)")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    tok = WordTokenizer.load("data/processed/vocab.json")
    signllm_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    mdm_cfg = MDMConfig(signllm_checkpoint=args.signllm_checkpoint)
    train_cfg = DiffusionTrainConfig(
        run_name=args.run_name,
        out_dir=args.out_dir,
        signllm_checkpoint=args.signllm_checkpoint,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        bf16=not args.no_bf16,
        grad_checkpoint=args.grad_checkpoint,
        use_flash=not args.no_flash,
        cfg_dropout=args.cfg_dropout,
        cfg_scale=args.cfg_scale,
        resume_from=args.resume_from,
    )
    loss_cfg = DiffusionLossConfig()

    result = train_diffusion(mdm_cfg, signllm_cfg, train_cfg, loss_cfg, tok)
    print(f"\ntraining complete: {result}")
