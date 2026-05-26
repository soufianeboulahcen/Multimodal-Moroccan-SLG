"""End-to-end photorealistic avatar video generation.

Usage:
    # Single sign — auto-discovers reference video
    python avatar_video_generator/run.py --sign أَنْتِ

    # Explicit paths
    python avatar_video_generator/run.py \\
        --video ".devcontainer/Dataset/mosl_videos_dataset_Pronouns/أَنْتِ.mp4" \\
        --output outputs/avatar_photorealistic/anti_avatar.mp4

    # Batch all Pronouns
    python avatar_video_generator/run.py --batch-dir ".devcontainer/Dataset/mosl_videos_dataset_Pronouns"

    # SD1.5 only (no AnimateDiff), lower VRAM
    python avatar_video_generator/run.py --sign أَنْتِ --no-animatediff

    # Skip diffusion entirely — just export pose frames as video (fast test)
    python avatar_video_generator/run.py --sign أَنْتِ --pose-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_video_generator.core.pose_from_video import extract_openpose_frames, read_video_frames
from avatar_video_generator.core.identity import extract_identity
from avatar_video_generator.core.postprocess import (
    smooth_temporal, interpolate_frames, write_mp4, write_comparison
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Dataset discovery ─────────────────────────────────────────────────────────

DATASET_ROOT = Path(".devcontainer/Dataset")

def find_video(sign_name: str) -> Path | None:
    """Search dataset for a video matching sign_name.

    Arabic text can have multiple Unicode representations for the same word
    (different hamza forms, diacritic encoding). We compare using three
    strategies in order:
      1. Exact NFC match
      2. Strip all diacritics (harakat) and compare base consonants only
      3. Strip diacritics + normalise alef variants (أ إ آ ا → ا)
    """
    import unicodedata, re

    def nfc(s: str) -> str:
        return unicodedata.normalize("NFC", s)

    # Arabic diacritics (harakat) Unicode range U+064B–U+065F
    DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")

    def strip(s: str) -> str:
        return DIACRITICS.sub("", nfc(s))

    def norm_alef(s: str) -> str:
        # Normalise all alef variants to plain alef ا
        return re.sub(r"[\u0622\u0623\u0625\u0671]", "\u0627", strip(s))

    sign_nfc   = nfc(sign_name)
    sign_strip = strip(sign_name)
    sign_alef  = norm_alef(sign_name)

    all_videos: list[Path] = []
    for subdir in DATASET_ROOT.iterdir():
        if subdir.is_dir():
            all_videos.extend(subdir.glob("*.mp4"))

    # Pass 1: exact NFC
    for mp4 in all_videos:
        if nfc(mp4.stem) == sign_nfc:
            return mp4

    # Pass 2: strip diacritics
    for mp4 in all_videos:
        if strip(mp4.stem) == sign_strip:
            return mp4

    # Pass 3: normalise alef variants
    for mp4 in all_videos:
        if norm_alef(mp4.stem) == sign_alef:
            return mp4

    return None


# ── Single sign pipeline ──────────────────────────────────────────────────────

def run_single(
    video_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict:
    """Full pipeline for one video. Returns timing/stats dict."""
    t0 = time.time()
    sign_name = video_path.stem
    logger.info(f"\n{'='*55}")
    logger.info(f"Sign: {sign_name}")
    logger.info(f"{'='*55}")

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract pose frames ───────────────────────────────────────────
    logger.info("Step 1/5 — Pose extraction")
    pose_out_dir = out_dir / f"{sign_name}_pose_frames" if args.save_pose else None
    raw_frames, pose_images = extract_openpose_frames(
        video_path, resolution=args.resolution, out_dir=pose_out_dir
    )
    logger.info(f"  {len(raw_frames)} frames @ {args.resolution}px")

    # ── Pose-only mode (fast test, no diffusion) ──────────────────────────────
    if args.pose_only:
        pose_np = [np.array(p) for p in pose_images]
        write_mp4(pose_np, output_path, fps=25.0)
        logger.info(f"  Pose-only output: {output_path}")
        return {"sign": sign_name, "frames": len(raw_frames),
                "elapsed": time.time()-t0, "mode": "pose_only"}

    # ── Step 2: Identity extraction ───────────────────────────────────────────
    logger.info("Step 2/5 — Identity extraction")
    identity = extract_identity(raw_frames, crop_size=224)
    logger.info(f"  backend={identity.backend}  desc='{identity.appearance_prompt}'")

    # ── Step 3: Diffusion rendering ───────────────────────────────────────────
    logger.info("Step 3/5 — Diffusion rendering")
    from avatar_video_generator.core.diffusion_render import DiffusionRenderer

    renderer = DiffusionRenderer(
        device=args.device,
        resolution=args.resolution,
        steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_scale=args.cn_scale,
        seed=args.seed,
        use_animatediff=not args.no_animatediff,
        use_fp16=not args.fp32,
        cpu_offload=args.cpu_offload,
    )
    renderer.load()

    # Inject IP-Adapter if available
    if identity.face_crop is not None and not args.no_ip_adapter:
        renderer.load_ip_adapter(identity.face_crop)

    raw_avatar = renderer.render(
        pose_images,
        face_image=identity.face_crop,
        extra_prompt=identity.appearance_prompt,
    )
    renderer.unload()
    logger.info(f"  {len(raw_avatar)} frames rendered via {renderer._backend}")

    # ── Step 4: Temporal smoothing ────────────────────────────────────────────
    logger.info("Step 4/5 — Temporal smoothing")
    smooth = smooth_temporal(raw_avatar, sigma=args.smooth_sigma)

    # ── Step 5: Interpolation + export ───────────────────────────────────────
    logger.info("Step 5/5 — Interpolation + export")
    if args.rife_multiplier > 1:
        final = interpolate_frames(smooth, multiplier=args.rife_multiplier)
        out_fps = 25.0 * args.rife_multiplier
    else:
        final = smooth
        out_fps = 25.0

    write_mp4(final, output_path, fps=out_fps, crf=args.crf)

    # Comparison video
    if args.comparison:
        comp_path = output_path.parent / (output_path.stem + "_comparison.mp4")
        pose_np = [np.array(p) for p in pose_images]
        write_comparison(pose_np, smooth, comp_path, fps=25.0)
        logger.info(f"  Comparison: {comp_path}")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s → {output_path}")
    return {
        "sign": sign_name,
        "frames_in": len(raw_frames),
        "frames_out": len(final),
        "fps": out_fps,
        "elapsed": elapsed,
        "backend": renderer._backend,
        "identity": identity.backend,
        "output": str(output_path),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoSL photorealistic avatar generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sign", help="Arabic sign name (auto-discovers video)")
    src.add_argument("--video", type=Path, help="Explicit dataset video path")
    src.add_argument("--batch-dir", type=Path, help="Process all MP4s in directory")

    # Output
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/avatar_photorealistic"))

    # Diffusion
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--resolution", type=int, default=512, choices=[512, 768])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--cn-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-animatediff", action="store_true")
    p.add_argument("--no-ip-adapter", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--cpu-offload", action="store_true")

    # Post-processing
    p.add_argument("--smooth-sigma", type=float, default=0.8)
    p.add_argument("--rife-multiplier", type=int, default=2, choices=[1, 2, 4])
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--comparison", action="store_true", default=True)
    p.add_argument("--no-comparison", dest="comparison", action="store_false")

    # Misc
    p.add_argument("--pose-only", action="store_true",
                   help="Export pose frames only (no diffusion — fast test)")
    p.add_argument("--save-pose", action="store_true",
                   help="Save extracted pose PNG frames to disk")

    return p.parse_args()


def main() -> int:
    import numpy as np  # needed in run_single scope
    args = parse_args()

    if args.sign is None and args.video is None and args.batch_dir is None:
        print("Specify --sign, --video, or --batch-dir")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch_dir:
        videos = sorted(args.batch_dir.glob("*.mp4"))
        if not videos:
            print(f"No MP4 files in {args.batch_dir}")
            return 1
        logger.info(f"Batch: {len(videos)} videos")
        results = []
        for vid in videos:
            safe = vid.stem.replace("/", "_")[:60]
            out = args.output_dir / f"{safe}_photorealistic.mp4"
            if out.exists():
                logger.info(f"  [SKIP] {out.name}")
                continue
            try:
                r = run_single(vid, out, args)
                results.append(r)
            except Exception as e:
                logger.error(f"  [FAIL] {vid.name}: {e}")

        summary_path = args.output_dir / "generation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary: {summary_path}")
        return 0

    # ── Single mode ───────────────────────────────────────────────────────────
    if args.sign:
        video_path = find_video(args.sign)
        if video_path is None:
            print(f"No video found for sign '{args.sign}' in {DATASET_ROOT}")
            return 1
    else:
        video_path = args.video

    safe = video_path.stem.replace("/", "_")[:60]
    output_path = args.output or (args.output_dir / f"{safe}_photorealistic.mp4")

    result = run_single(video_path, output_path, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import numpy as np
    raise SystemExit(main())
