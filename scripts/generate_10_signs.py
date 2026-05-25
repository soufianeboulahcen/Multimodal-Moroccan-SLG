"""Generate the first realistic avatar videos for 10 Arabic signs.

Produces:
  1. Generated pose sequences (DDIM sampling from trained MDM)
  2. Ground-truth pose sequences (from .skels dataset)
  3. Side-by-side comparison videos (ground-truth | generated)
  4. Temporal stability report for all generated sequences
  5. Summary JSON with all metrics

This is the Phase 2 "first real avatar generation" milestone script.

Usage:
    # Stage 1: skeleton overlay (no extra deps)
    python scripts/generate_10_signs.py \
        --diffusion-checkpoint runs_diffusion/mdm_mosl/best.pt \
        --signllm-checkpoint runs/baseline_mse/best.pt \
        --stage 1

    # Stage 2: SMPL-X mesh render
    python scripts/generate_10_signs.py \
        --diffusion-checkpoint runs_diffusion/mdm_mosl/best.pt \
        --signllm-checkpoint runs/baseline_mse/best.pt \
        --smplx-model-path data/smplx_models/ \
        --stage 2

    # Multiple signer styles (requires signer_id annotation)
    python scripts/generate_10_signs.py \
        --diffusion-checkpoint runs_diffusion/mdm_mosl/best.pt \
        --multi-signer \
        --stage 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# 10 representative Arabic signs from the MoSL dataset
# ---------------------------------------------------------------------------

TARGET_SIGNS = [
    "الأذان",           # The call to prayer
    "الأمن الوطني",     # National security
    "الأردن",           # Jordan
    "الآن",             # Now
    "أنت",              # You (masc.)
    "أنتِ",             # You (fem.)
    "البحرين",          # Bahrain
    "الإنجليزية",       # English (language)
    "الآخرة",           # The afterlife
    "الإمارات العربية المتحدة",  # UAE
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline(
    signllm_checkpoint: Optional[str],
    diffusion_checkpoint: Optional[str],
    vocab_path: str,
    device: torch.device,
):
    """Load tokenizer, MDM model, and noise schedule."""
    from mosl.text.tokenizer import WordTokenizer
    from mosl.model.signllm import SignLLMConfig
    from mosl.model.mdm_denoiser import MDMConfig, MDMDenoiser
    from mosl.train.noise_schedule import NoiseSchedule

    tok = WordTokenizer.load(vocab_path)
    signllm_cfg = SignLLMConfig(vocab_size=tok.vocab_size)
    mdm_cfg = MDMConfig()

    model = MDMDenoiser(mdm_cfg, signllm_cfg).to(device)

    if signllm_checkpoint and Path(signllm_checkpoint).exists():
        model.load_text_encoder(signllm_checkpoint)
        print(f"SignLLM encoder: {signllm_checkpoint}")
    else:
        model.freeze_text_encoder()
        print("SignLLM encoder: random weights (no checkpoint)")

    if diffusion_checkpoint and Path(diffusion_checkpoint).exists():
        ckpt = torch.load(diffusion_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"MDM denoiser: {diffusion_checkpoint}")
    else:
        print("MDM denoiser: random weights (no checkpoint — for testing only)")

    model.eval()

    schedule = NoiseSchedule(n_steps=1000).to(device)
    return tok, model, schedule


def _encode_text(arabic_text: str, tok, device: torch.device):
    """Encode Arabic text to token IDs."""
    ids = tok.encode(arabic_text)
    if not ids:
        ids = [tok.unk_id]
    token_ids = [tok.bos_id] + ids + [tok.eos_id]
    text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)
    return text_ids, text_mask


@torch.no_grad()
def generate_pose(
    arabic_text: str,
    tok,
    model,
    schedule,
    device: torch.device,
    n_sample_steps: int = 50,
    max_T: int = 150,
    cfg_scale: float = 2.5,
    smooth: bool = True,
    savgol_window: int = 7,
) -> np.ndarray:
    """Generate a pose sequence for one Arabic sign via DDIM + CFG."""
    from mosl.train.temporal_stability import smooth_motion

    text_ids, text_mask = _encode_text(arabic_text, tok, device)

    # DDIM sampling with classifier-free guidance
    if cfg_scale > 1.0 and hasattr(model, "forward_cfg"):
        def denoiser_fn(x, t, ti, tm):
            return model.forward_cfg(x, t, ti, tm, cfg_scale=cfg_scale)
    else:
        def denoiser_fn(x, t, ti, tm):
            return model(x, t, ti, tm)

    pose = schedule.ddim_sample(
        denoiser_fn=denoiser_fn,
        shape=(1, max_T, 150),
        text_ids=text_ids,
        text_mask=text_mask,
        n_sample_steps=n_sample_steps,
        eta=0.0,   # fully deterministic
        device=device,
    )
    pose_np = pose[0].cpu().numpy()   # (T, 150)

    if smooth:
        pose_np = smooth_motion(pose_np, window=savgol_window, polyorder=3)

    return pose_np


def _find_ground_truth(arabic_text: str, repo_root: Path) -> Optional[np.ndarray]:
    """Find the ground-truth pose sequence for a sign from the dataset.

    Searches .skels files and NPZ keypoints. Returns (T, 150) or None.
    """
    import unicodedata
    import csv

    labels_csv = repo_root / "data" / "labels.csv"
    if not labels_csv.exists():
        return None

    # Strip diacritics for matching
    def strip_diacritics(text: str) -> str:
        import re
        return re.sub(r"[\u064B-\u065F\u0670]", "", text)

    target_stripped = strip_diacritics(arabic_text)

    with open(labels_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Find matching row
    match = None
    for row in rows:
        if strip_diacritics(row.get("word_arabic_stripped", "")) == target_stripped:
            match = row
            break
        if strip_diacritics(row.get("word_arabic", "")) == target_stripped:
            match = row
            break

    if match is None:
        return None

    # Load NPZ keypoints
    rel_path = match.get("relative_path", "")
    category = match.get("category", "")
    stem = Path(rel_path).stem if rel_path else None

    if stem and category:
        npz_path = repo_root / "data" / "processed" / "keypoints_2d" / category / f"{stem}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=False)
            body = data.get("pose_keypoints_2d")
            if body is not None:
                T = body.shape[0]
                joints = np.zeros((T, 150), dtype=np.float32)
                joints[:, :body.shape[1]] = body

                lhand = data.get("hand_left_keypoints_2d")
                rhand = data.get("hand_right_keypoints_2d")
                if lhand is not None and lhand.shape[0] == T:
                    lh = lhand.reshape(T, 21, 3)[:, :, :2]
                    for j in range(21):
                        joints[:, (18 + j) * 3] = lh[:, j, 0]
                        joints[:, (18 + j) * 3 + 1] = lh[:, j, 1]
                if rhand is not None and rhand.shape[0] == T:
                    rh = rhand.reshape(T, 21, 3)[:, :, :2]
                    for j in range(min(11, 21)):
                        joints[:, (39 + j) * 3] = rh[:, j, 0]
                        joints[:, (39 + j) * 3 + 1] = rh[:, j, 1]
                return joints

    return None


def _render_skeleton_frame(
    pose: np.ndarray,   # (150,) one frame
    frame_size: int = 512,
    title: str = "",
) -> np.ndarray:
    """Render a single skeleton frame as an RGB image (numpy array)."""
    import cv2

    img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)   # dark background

    joints = pose.reshape(50, 3)[:, :2]   # (50, 2) — x, y only

    # Normalise to frame coordinates
    x_min, x_max = joints[:, 0].min(), joints[:, 0].max()
    y_min, y_max = joints[:, 1].min(), joints[:, 1].max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    margin = 0.1 * frame_size
    def to_px(j):
        x = int(margin + (j[0] - x_min) / x_range * (frame_size - 2 * margin))
        y = int(margin + (j[1] - y_min) / y_range * (frame_size - 2 * margin))
        return (x, y)

    # Body skeleton connections (COCO-18)
    body_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
        (0, 14), (0, 15), (14, 16), (15, 17),
    ]
    for a, b in body_connections:
        if a < 18 and b < 18:
            pa, pb = to_px(joints[a]), to_px(joints[b])
            cv2.line(img, pa, pb, (100, 200, 100), 2)

    # Left hand (joints 18-38)
    for j in range(18, 38):
        pa, pb = to_px(joints[j]), to_px(joints[j + 1])
        cv2.line(img, pa, pb, (200, 100, 100), 1)

    # Right hand (joints 39-49)
    for j in range(39, 49):
        pa, pb = to_px(joints[j]), to_px(joints[j + 1])
        cv2.line(img, pa, pb, (100, 100, 200), 1)

    # Draw joint dots
    for j in range(18):
        cv2.circle(img, to_px(joints[j]), 4, (150, 255, 150), -1)
    for j in range(18, 39):
        cv2.circle(img, to_px(joints[j]), 2, (255, 150, 150), -1)
    for j in range(39, 50):
        cv2.circle(img, to_px(joints[j]), 2, (150, 150, 255), -1)

    # Title
    if title:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(title)
            display = get_display(reshaped)
        except ImportError:
            display = title
        cv2.putText(img, display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def render_comparison_video(
    gt_pose: Optional[np.ndarray],    # (T_gt, 150) or None
    gen_pose: np.ndarray,             # (T_gen, 150)
    output_path: Path,
    arabic_text: str,
    fps: float = 25.0,
    frame_size: int = 512,
) -> None:
    """Render side-by-side ground-truth vs generated skeleton video."""
    import cv2

    T_gen = gen_pose.shape[0]
    T_gt = gt_pose.shape[0] if gt_pose is not None else T_gen
    T_max = max(T_gt, T_gen)

    # Pad shorter sequence by repeating last frame
    if gt_pose is not None and T_gt < T_max:
        pad = np.tile(gt_pose[-1:], (T_max - T_gt, 1))
        gt_pose = np.concatenate([gt_pose, pad], axis=0)
    if T_gen < T_max:
        pad = np.tile(gen_pose[-1:], (T_max - T_gen, 1))
        gen_pose = np.concatenate([gen_pose, pad], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = frame_size * 2 if gt_pose is not None else frame_size
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, frame_size))

    for t in range(T_max):
        gen_frame = _render_skeleton_frame(gen_pose[t], frame_size, f"Generated: {arabic_text}")

        if gt_pose is not None:
            gt_frame = _render_skeleton_frame(gt_pose[t], frame_size, f"Ground Truth: {arabic_text}")
            # Side-by-side: GT on left, generated on right
            combined = np.concatenate([gt_frame, gen_frame], axis=1)
        else:
            combined = gen_frame

        writer.write(combined)

    writer.release()


def render_skeleton_video(
    pose: np.ndarray,
    output_path: Path,
    arabic_text: str,
    fps: float = 25.0,
    frame_size: int = 512,
) -> None:
    """Render a single skeleton video."""
    import cv2

    T = pose.shape[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_size, frame_size))

    for t in range(T):
        frame = _render_skeleton_frame(pose[t], frame_size, arabic_text)
        writer.write(frame)

    writer.release()


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_10_signs(
    signllm_checkpoint: Optional[str],
    diffusion_checkpoint: Optional[str],
    out_dir: Path,
    stage: int = 1,
    smplx_model_path: Optional[str] = None,
    n_sample_steps: int = 50,
    max_T: int = 150,
    cfg_scale: float = 2.5,
    fps: float = 25.0,
    frame_size: int = 512,
    device_str: str = "cuda",
    multi_signer: bool = False,
    signs: Optional[list[str]] = None,
    vocab_path: str = "data/processed/vocab.json",
    repo_root: Optional[Path] = None,
) -> dict:
    """Generate avatar videos for 10 Arabic signs.

    Returns a summary dict with per-sign metrics.
    """
    from mosl.train.temporal_stability import TemporalStabilityAnalyzer, post_process_motion

    repo_root = repo_root or Path(__file__).resolve().parents[1]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_signs = signs or TARGET_SIGNS

    print(f"\n{'='*60}")
    print(f"Phase 2 — First Realistic Avatar Generation")
    print(f"{'='*60}")
    print(f"Signs:    {len(target_signs)}")
    print(f"Stage:    {stage}")
    print(f"Device:   {device}")
    print(f"DDIM:     {n_sample_steps} steps")
    print(f"CFG:      {cfg_scale}")
    print(f"Output:   {out_dir}")
    print()

    # Load pipeline
    tok, model, schedule = _load_pipeline(
        signllm_checkpoint, diffusion_checkpoint, vocab_path, device
    )

    analyzer = TemporalStabilityAnalyzer()
    summary = {
        "signs": [],
        "n_stable": 0,
        "n_total": len(target_signs),
        "stage": stage,
        "cfg_scale": cfg_scale,
        "n_sample_steps": n_sample_steps,
    }

    t_total = time.time()

    for i, arabic_text in enumerate(target_signs):
        print(f"\n[{i+1}/{len(target_signs)}] {arabic_text}")
        t0 = time.time()

        safe_name = arabic_text.replace("/", "_").replace(" ", "_")[:40]

        # --- Generate pose ---
        gen_pose = generate_pose(
            arabic_text, tok, model, schedule, device,
            n_sample_steps=n_sample_steps,
            max_T=max_T,
            cfg_scale=cfg_scale,
            smooth=True,
        )
        print(f"  generated: T={gen_pose.shape[0]}  "
              f"range=[{gen_pose.min():.3f}, {gen_pose.max():.3f}]")

        # --- Load ground truth ---
        gt_pose = _find_ground_truth(arabic_text, repo_root)
        if gt_pose is not None:
            print(f"  ground truth: T={gt_pose.shape[0]}")
        else:
            print(f"  ground truth: not found")

        # --- Temporal stability ---
        report = analyzer.analyze(gen_pose)
        print(f"  stability: {report.summary()}")

        # --- Save pose NPZ ---
        pose_path = out_dir / f"{safe_name}_pose.npz"
        np.savez_compressed(pose_path, pose=gen_pose, text=np.array(arabic_text))

        # --- Render videos ---
        if stage == 1:
            # Generated skeleton
            gen_video = out_dir / f"{safe_name}_generated.mp4"
            render_skeleton_video(gen_pose, gen_video, arabic_text, fps=fps, frame_size=frame_size)

            # Side-by-side comparison
            cmp_video = out_dir / f"{safe_name}_comparison.mp4"
            render_comparison_video(gt_pose, gen_pose, cmp_video, arabic_text,
                                    fps=fps, frame_size=frame_size)
            print(f"  videos: {gen_video.name}  {cmp_video.name}")

        elif stage == 2 and smplx_model_path:
            from mosl.pose.fit_smplx import SMPLXFitter
            fitter = SMPLXFitter(smplx_model_path, device=device_str)
            params = fitter.fit_sequence(gen_pose)
            smplx_path = out_dir / f"{safe_name}_smplx.npz"
            np.savez_compressed(smplx_path, **params)
            print(f"  SMPL-X params: {smplx_path.name}")

        # --- Multi-signer generation ---
        if multi_signer:
            for signer_id in range(3):   # generate 3 signer styles
                signer_pose = generate_pose(
                    arabic_text, tok, model, schedule, device,
                    n_sample_steps=n_sample_steps,
                    max_T=max_T,
                    cfg_scale=cfg_scale,
                    smooth=True,
                )
                signer_video = out_dir / f"{safe_name}_signer{signer_id}.mp4"
                render_skeleton_video(signer_pose, signer_video, arabic_text,
                                      fps=fps, frame_size=frame_size)

        elapsed = time.time() - t0
        sign_result = {
            "text": arabic_text,
            "n_frames_generated": int(gen_pose.shape[0]),
            "n_frames_gt": int(gt_pose.shape[0]) if gt_pose is not None else None,
            "jerk_score": round(report.jerk_score, 5),
            "hand_flicker_rate": round(report.hand_flicker_rate, 4),
            "finger_instability": round(report.finger_instability, 5),
            "is_stable": report.is_stable(),
            "elapsed_s": round(elapsed, 2),
        }
        summary["signs"].append(sign_result)
        if report.is_stable():
            summary["n_stable"] += 1

        print(f"  done in {elapsed:.1f}s")

    # --- Summary ---
    total_elapsed = time.time() - t_total
    summary["total_elapsed_s"] = round(total_elapsed, 2)
    summary["stability_rate"] = round(summary["n_stable"] / max(summary["n_total"], 1), 3)

    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generation complete in {total_elapsed/60:.1f} min")
    print(f"Stable sequences: {summary['n_stable']}/{summary['n_total']} "
          f"({summary['stability_rate']*100:.1f}%)")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate avatar videos for 10 Arabic signs (Phase 2 milestone)"
    )
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Path to MDM diffusion best.pt")
    parser.add_argument("--signllm-checkpoint", default=None,
                        help="Path to SignLLM best.pt")
    parser.add_argument("--out-dir", default="outputs/phase2_generation",
                        help="Output directory for videos and pose files")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Rendering stage: 1=skeleton overlay, 2=SMPL-X mesh")
    parser.add_argument("--smplx-model-path", default=None,
                        help="SMPL-X model directory (required for stage 2)")
    parser.add_argument("--n-sample-steps", type=int, default=50,
                        help="DDIM sampling steps (default: 50)")
    parser.add_argument("--max-t", type=int, default=150,
                        help="Maximum generated sequence length in frames")
    parser.add_argument("--cfg-scale", type=float, default=2.5,
                        help="Classifier-free guidance scale (default: 2.5)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multi-signer", action="store_true",
                        help="Generate 3 signer style variants per sign")
    parser.add_argument("--signs", nargs="+", default=None,
                        help="Custom list of Arabic signs (default: 10 preset signs)")
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    args = parser.parse_args()

    generate_10_signs(
        signllm_checkpoint=args.signllm_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        out_dir=Path(args.out_dir),
        stage=args.stage,
        smplx_model_path=args.smplx_model_path,
        n_sample_steps=args.n_sample_steps,
        max_T=args.max_t,
        cfg_scale=args.cfg_scale,
        fps=args.fps,
        frame_size=args.frame_size,
        device_str=args.device,
        multi_signer=args.multi_signer,
        signs=args.signs,
        vocab_path=args.vocab,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
