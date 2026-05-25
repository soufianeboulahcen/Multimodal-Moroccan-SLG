"""End-to-end avatar video generation from Arabic text.

Full pipeline:
  Arabic text
    → SignLLM text encoder (frozen)
    → MDM diffusion denoiser (DDIM sampling)
    → (T, 150) smooth pose sequence
    → Savitzky-Golay temporal smoothing
    → [optional] SMPL-X fitting
    → [optional] PyTorch3D / Blender rendering
    → Avatar video

The pipeline is designed to be run incrementally:
  Stage 1 (always available): overlay skeleton video from diffusion output
  Stage 2 (requires smplx):   SMPL-X mesh video via PyTorch3D
  Stage 3 (requires Blender): photorealistic Cycles render

Usage:
    # Stage 1: skeleton overlay (no extra deps beyond existing repo)
    python scripts/generate_avatar_video.py --text "الأذان" --stage 1

    # Stage 2: SMPL-X mesh render
    python scripts/generate_avatar_video.py --text "الأذان" --stage 2 \
        --smplx-model-path data/smplx_models/

    # Stage 3: Blender photorealistic render
    python scripts/generate_avatar_video.py --text "الأذان" --stage 3

    # Use a reference clip for signer style conditioning
    python scripts/generate_avatar_video.py --text "الأذان" \
        --ref-clip data/processed/keypoints_2d/Diverse/الأذان.npz

    # Batch generation from a text file (one Arabic word per line)
    python scripts/generate_avatar_video.py --text-file signs.txt --stage 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _load_models(
    signllm_checkpoint: Optional[str],
    diffusion_checkpoint: Optional[str],
    vocab_path: str,
    device: torch.device,
) -> tuple:
    """Load tokenizer, SignLLM config, and MDM denoiser."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        print(f"loaded SignLLM text encoder: {signllm_checkpoint}")
    else:
        model.freeze_text_encoder()
        print("SignLLM text encoder: random weights (no checkpoint provided)")

    if diffusion_checkpoint and Path(diffusion_checkpoint).exists():
        ckpt = torch.load(diffusion_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded diffusion denoiser: {diffusion_checkpoint}")
    else:
        print("diffusion denoiser: random weights (no checkpoint provided)")
        print("  → output will be noise; train the model first with:")
        print("    python -m mosl.train.diffusion_train")

    model.eval()
    schedule = NoiseSchedule(n_steps=1000, schedule="cosine").to(device)

    return tok, model, schedule


def generate_pose(
    arabic_text: str,
    tokenizer,
    model,
    schedule,
    device: torch.device,
    n_sample_steps: int = 50,
    max_T: int = 150,
    smooth: bool = True,
    ref_pose: Optional[torch.Tensor] = None,
    ref_mask: Optional[torch.Tensor] = None,
    signer_encoder=None,
) -> np.ndarray:
    """Generate a (T, 150) pose sequence for one Arabic sign.

    Returns numpy array (T, 150).
    """
    from mosl.train.noise_schedule import savgol_smooth

    # Tokenise
    token_ids = tokenizer.encode(arabic_text, add_specials=True)
    text_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    # Predict sequence length from text encoder
    with torch.no_grad():
        text_features = model.encode_text(text_ids, text_mask)
        log_T = model.text_encoder_model.predict_length(text_features, text_mask)
        T_pred = int(log_T.exp().round().clamp(min=10, max=max_T).item())

    # Optional signer style conditioning
    style_emb = None
    if signer_encoder is not None and ref_pose is not None:
        with torch.no_grad():
            style_emb = signer_encoder(
                ref_pose.unsqueeze(0).to(device),
                ref_mask.unsqueeze(0).to(device) if ref_mask is not None else None,
            )

    # DDIM sampling
    shape = (1, T_pred, 150)

    def denoiser_fn(x_noisy, t, text_ids, text_mask, **kw):
        return model(x_noisy, t, text_ids, text_mask,
                     style_emb=style_emb, pose_mask=kw.get("pose_mask"))

    with torch.no_grad():
        pose_tensor = schedule.ddim_sample(
            denoiser_fn=denoiser_fn,
            shape=shape,
            text_ids=text_ids,
            text_mask=text_mask,
            n_sample_steps=n_sample_steps,
            eta=0.0,   # fully deterministic
            device=device,
        )

    pose = pose_tensor.squeeze(0)   # (T, 150)

    if smooth:
        pose = savgol_smooth(pose, window=5, polyorder=2)

    return pose.cpu().numpy()


def save_pose_npz(pose: np.ndarray, output_path: Path, arabic_text: str) -> None:
    """Save generated pose as NPZ for downstream use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pose=pose,
        text=np.array(arabic_text),
    )


def stage1_overlay(
    pose: np.ndarray,
    output_path: Path,
    fps: float = 25.0,
    frame_size: int = 512,
) -> None:
    """Render skeleton overlay video (no extra dependencies)."""
    from scripts.render_smplx_video import render_overlay_video
    render_overlay_video(pose, output_path, fps=fps, frame_size=frame_size)


def stage2_smplx(
    pose: np.ndarray,
    output_path: Path,
    smplx_model_path: str,
    fps: float = 25.0,
    frame_size: int = 512,
    device: str = "cuda",
) -> None:
    """Fit SMPL-X and render mesh video via PyTorch3D."""
    import tempfile
    from mosl.pose.fit_smplx import SMPLXFitter
    from scripts.render_smplx_video import render_pytorch3d_video

    fitter = SMPLXFitter(smplx_model_path, device=device)
    params = fitter.fit_sequence(pose)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)
    np.savez_compressed(tmp_path, **params)

    render_pytorch3d_video(
        tmp_path, output_path,
        smplx_model_path=smplx_model_path,
        fps=fps, frame_size=frame_size, device=device,
    )
    tmp_path.unlink(missing_ok=True)


def stage3_blender(
    pose: np.ndarray,
    output_path: Path,
    smplx_model_path: str,
    fps: float = 25.0,
    frame_size: int = 512,
    device: str = "cuda",
) -> None:
    """Fit SMPL-X and render photorealistic video via Blender Cycles."""
    import tempfile
    from mosl.pose.fit_smplx import SMPLXFitter
    from scripts.render_smplx_video import render_blender_video

    fitter = SMPLXFitter(smplx_model_path, device=device)
    params = fitter.fit_sequence(pose)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = Path(f.name)
    np.savez_compressed(tmp_path, **params)

    render_blender_video(tmp_path, output_path, fps=fps, frame_size=frame_size)
    tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate avatar video from Arabic sign language text"
    )
    parser.add_argument("--text", default=None, help="Arabic sign text (single word/phrase)")
    parser.add_argument("--text-file", default=None,
                        help="Text file with one Arabic sign per line")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Rendering stage: 1=overlay, 2=SMPL-X mesh, 3=Blender")
    parser.add_argument("--out-dir", default="outputs/generated_videos")
    parser.add_argument("--signllm-checkpoint", default=None,
                        help="Path to SignLLM best.pt")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Path to MDM diffusion best.pt")
    parser.add_argument("--smplx-model-path", default=None,
                        help="Path to SMPL-X model directory (required for stage 2/3)")
    parser.add_argument("--ref-clip", default=None,
                        help="Reference clip NPZ for signer style conditioning")
    parser.add_argument("--n-sample-steps", type=int, default=50,
                        help="DDIM sampling steps (fewer = faster, more = better quality)")
    parser.add_argument("--max-t", type=int, default=150,
                        help="Maximum generated sequence length in frames")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable Savitzky-Golay post-processing")
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    args = parser.parse_args()

    if args.text is None and args.text_file is None:
        parser.print_help()
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    tok, model, schedule = _load_models(
        args.signllm_checkpoint,
        args.diffusion_checkpoint,
        args.vocab,
        device,
    )

    # Load signer style encoder if ref clip provided
    signer_encoder = None
    ref_pose = None
    ref_mask = None
    if args.ref_clip:
        ref_path = Path(args.ref_clip)
        if ref_path.exists():
            from mosl.model.signer_encoder import SignerEncoder, SignerEncoderConfig
            signer_encoder = SignerEncoder(SignerEncoderConfig()).to(device)
            signer_encoder.eval()

            data = np.load(ref_path, allow_pickle=False)
            if "pose_keypoints_2d" in data:
                ref_np = data["pose_keypoints_2d"]
                T_ref = ref_np.shape[0]
                full = np.zeros((T_ref, 150), dtype=np.float32)
                full[:, :ref_np.shape[1]] = ref_np
                ref_pose = torch.from_numpy(full)
            elif "pose" in data:
                ref_pose = torch.from_numpy(data["pose"].astype(np.float32))
            if ref_pose is not None:
                ref_mask = torch.ones(ref_pose.shape[0], dtype=torch.bool)
            print(f"loaded reference clip: {ref_path.name}  T={ref_pose.shape[0] if ref_pose is not None else 0}")

    # Collect texts to generate
    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            texts.extend(ln.strip() for ln in f if ln.strip())

    print(f"\ngenerating {len(texts)} sign(s)  stage={args.stage}  "
          f"device={device}  ddim_steps={args.n_sample_steps}")

    for arabic_text in texts:
        t0 = time.time()
        safe_name = arabic_text.replace("/", "_").replace("\\", "_")[:50]

        print(f"\n[{arabic_text}]")

        # Generate pose
        pose = generate_pose(
            arabic_text, tok, model, schedule, device,
            n_sample_steps=args.n_sample_steps,
            max_T=args.max_t,
            smooth=not args.no_smooth,
            ref_pose=ref_pose,
            ref_mask=ref_mask,
            signer_encoder=signer_encoder,
        )
        print(f"  pose: T={pose.shape[0]}  range=[{pose.min():.3f}, {pose.max():.3f}]")

        # Save pose NPZ
        pose_path = out_dir / f"{safe_name}_pose.npz"
        save_pose_npz(pose, pose_path, arabic_text)

        # Render
        video_path = out_dir / f"{safe_name}_stage{args.stage}.mp4"

        if args.stage == 1:
            stage1_overlay(pose, video_path, fps=args.fps, frame_size=args.frame_size)
        elif args.stage == 2:
            if not args.smplx_model_path:
                print("  [SKIP] --smplx-model-path required for stage 2")
                continue
            stage2_smplx(pose, video_path, args.smplx_model_path,
                         fps=args.fps, frame_size=args.frame_size, device=args.device)
        elif args.stage == 3:
            if not args.smplx_model_path:
                print("  [SKIP] --smplx-model-path required for stage 3")
                continue
            stage3_blender(pose, video_path, args.smplx_model_path,
                           fps=args.fps, frame_size=args.frame_size, device=args.device)

        elapsed = time.time() - t0
        print(f"  video: {video_path}  ({elapsed:.1f}s)")

    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    main()
