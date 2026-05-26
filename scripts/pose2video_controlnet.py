"""
Pose-to-Video ControlNet Pipeline — MoSL Avatar Generation
===========================================================
Entrée  : séquence de frames OpenPose (PNG) ou vidéo squelette MP4
Sortie  : vidéo MP4 réaliste d'un signeur (512×512, 25 fps)

Architecture :
  ControlNet (lllyasviel/control_v11p_sd15_openpose)
    + Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
    → génération frame par frame
    → lissage temporel (blend latents + filtre Gaussian)
    → encodage MP4 via imageio/ffmpeg

Usage :
    # Depuis un dossier de PNG OpenPose
    python scripts/pose2video_controlnet.py \\
        --pose-dir outputs/pose_control/أَنْتِ_keypoints \\
        --output outputs/avatar/أَنْتِ_avatar.mp4

    # Depuis une vidéo squelette existante
    python scripts/pose2video_controlnet.py \\
        --skeleton-video outputs/videos/skeleton/أَنْتِ_skeleton.mp4 \\
        --output outputs/avatar/أَنْتِ_avatar.mp4

    # Batch : tous les dossiers pose_control
    python scripts/pose2video_controlnet.py \\
        --batch-dir outputs/pose_control \\
        --output-dir outputs/avatar

    # Avec image de référence pour cohérence d'apparence
    python scripts/pose2video_controlnet.py \\
        --pose-dir outputs/pose_control/أَنْتِ_keypoints \\
        --reference-image data/reference_signers/signer.jpg \\
        --output outputs/avatar/أَنْتِ_avatar.mp4

Dépendances (voir requirements_diffusion.txt) :
    pip install -r requirements_diffusion.txt
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import imageio
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11p_sd15_openpose"
DEFAULT_RESOLUTION = 512
DEFAULT_FPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 1.0

# Prompt décrivant l'avatar signeur cible
DEFAULT_POSITIVE_PROMPT = (
    "a professional sign language interpreter, "
    "upper body portrait, dark background, "
    "sharp focus, photorealistic, 8k, studio lighting, "
    "clear hands and fingers, natural skin tone"
)
DEFAULT_NEGATIVE_PROMPT = (
    "skeleton, stick figure, cartoon, anime, drawing, "
    "blurry, low quality, deformed hands, extra fingers, "
    "watermark, text, logo, nsfw"
)

# Fenêtre de lissage temporel (en frames)
TEMPORAL_BLEND_WINDOW = 3   # blend avec N frames précédentes
LATENT_SMOOTH_SIGMA = 0.8   # sigma du filtre gaussien sur les latents


# ---------------------------------------------------------------------------
# Chargement du pipeline
# ---------------------------------------------------------------------------

def load_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    controlnet_id: str = DEFAULT_CONTROLNET_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_xformers: bool = True,
    enable_cpu_offload: bool = False,
):
    """Charge ControlNet + Stable Diffusion en mémoire GPU.

    Utilise float16 par défaut pour économiser la VRAM (≈6 GB pour SD1.5).
    enable_cpu_offload active le séquentiel offload si VRAM < 6 GB.
    """
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    from diffusers import UniPCMultistepScheduler

    print(f"Chargement ControlNet : {controlnet_id}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    print(f"Chargement Stable Diffusion : {model_id}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,          # désactivé pour usage académique
        requires_safety_checker=False,
    )

    # Scheduler rapide (UniPC > DDIM pour même qualité en moins d'étapes)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers activé (attention efficace)")
        except Exception:
            print("xformers non disponible, utilisation de l'attention standard")

    if enable_cpu_offload:
        # Charge chaque composant sur GPU uniquement quand nécessaire
        pipe.enable_sequential_cpu_offload()
        print("CPU offload activé (VRAM réduite)")
    else:
        pipe = pipe.to(device)

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------------
# Extraction des frames de pose
# ---------------------------------------------------------------------------

def load_pose_frames_from_dir(
    pose_dir: Path,
    resolution: int = DEFAULT_RESOLUTION,
) -> List[np.ndarray]:
    """Charge les PNG OpenPose depuis un dossier, triés par nom.

    Retourne une liste de tableaux numpy (H, W, 3) uint8.
    """
    patterns = ["pose_*.png", "frame_*.png", "*.png"]
    frames = []
    for pattern in patterns:
        candidates = sorted(pose_dir.glob(pattern))
        if candidates:
            frames = candidates
            break

    if not frames:
        raise FileNotFoundError(f"Aucun PNG trouvé dans {pose_dir}")

    print(f"  {len(frames)} frames OpenPose trouvées dans {pose_dir.name}")

    result = []
    for p in frames:
        img = Image.open(p).convert("RGB")
        img = img.resize((resolution, resolution), Image.LANCZOS)
        result.append(np.array(img))

    return result


def load_pose_frames_from_video(
    video_path: Path,
    resolution: int = DEFAULT_RESOLUTION,
) -> List[np.ndarray]:
    """Extrait les frames d'une vidéo squelette MP4.

    Les vidéos skeleton/* contiennent déjà le rendu OpenPose coloré.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (resolution, resolution),
                                   interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame_resized)

    cap.release()
    print(f"  {len(frames)} frames extraites de {video_path.name}")
    return frames


# ---------------------------------------------------------------------------
# Lissage temporel des latents
# ---------------------------------------------------------------------------

def smooth_latents_temporal(
    latents_sequence: List[torch.Tensor],
    sigma: float = LATENT_SMOOTH_SIGMA,
) -> List[torch.Tensor]:
    """Applique un filtre gaussien 1D sur l'axe temporel des latents.

    Réduit le flickering inter-frames sans dégrader les détails spatiaux.
    Opère sur la dimension temporelle de chaque canal latent indépendamment.

    Args:
        latents_sequence: liste de tenseurs (1, C, H, W) float
        sigma: écart-type du filtre gaussien (plus grand = plus lisse)

    Returns:
        Liste de tenseurs lissés, même forme que l'entrée.
    """
    if len(latents_sequence) < 3:
        return latents_sequence

    # Empiler : (T, C, H, W)
    stacked = torch.stack([l.squeeze(0) for l in latents_sequence], dim=0)
    stacked_np = stacked.cpu().float().numpy()  # (T, C, H, W)

    # Filtre gaussien sur l'axe temporel (axis=0) pour chaque canal
    smoothed_np = gaussian_filter1d(stacked_np, sigma=sigma, axis=0)

    smoothed = torch.from_numpy(smoothed_np).to(
        dtype=latents_sequence[0].dtype,
        device=latents_sequence[0].device,
    )

    return [smoothed[i].unsqueeze(0) for i in range(len(latents_sequence))]


def blend_with_previous(
    current_latent: torch.Tensor,
    history: List[torch.Tensor],
    window: int = TEMPORAL_BLEND_WINDOW,
    alpha: float = 0.15,
) -> torch.Tensor:
    """Blend online : mélange le latent courant avec la moyenne des N précédents.

    Utilisé pendant la génération pour cohérence temporelle en temps réel.
    alpha contrôle la force du blend (0 = pas de blend, 1 = tout historique).
    """
    if not history:
        return current_latent

    recent = history[-window:]
    mean_prev = torch.stack(recent, dim=0).mean(dim=0)
    return (1.0 - alpha) * current_latent + alpha * mean_prev


# ---------------------------------------------------------------------------
# Génération frame par frame
# ---------------------------------------------------------------------------

def generate_frames(
    pipe,
    pose_frames: List[np.ndarray],
    positive_prompt: str = DEFAULT_POSITIVE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    resolution: int = DEFAULT_RESOLUTION,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    controlnet_conditioning_scale: float = DEFAULT_CONTROLNET_CONDITIONING_SCALE,
    seed: int = 42,
    temporal_blend_alpha: float = 0.15,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Génère les frames réalistes frame par frame avec ControlNet.

    Stratégie de cohérence temporelle :
    1. Seed fixe → même bruit de base pour toutes les frames
    2. Blend online des latents avec l'historique récent
    3. Lissage gaussien post-génération sur toute la séquence

    Args:
        pipe: pipeline StableDiffusionControlNetPipeline chargé
        pose_frames: liste de frames OpenPose (H, W, 3) uint8
        ...

    Returns:
        Liste de frames générées (H, W, 3) uint8
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    # Encoder le prompt une seule fois (optimisation)
    # Les embeddings texte sont identiques pour toutes les frames
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=positive_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

    generated_frames = []
    latent_history = []   # historique pour blend temporel

    print(f"\nGénération de {len(pose_frames)} frames...")
    for i, pose_np in enumerate(tqdm(pose_frames, desc="Frames", unit="fr")):
        pose_pil = Image.fromarray(pose_np)

        # Génération avec ControlNet
        # output.images[0] est un PIL.Image (H, W, RGB)
        with torch.no_grad():
            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=pose_pil,
                width=resolution,
                height=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                output_type="pil",
            )

        frame_np = np.array(output.images[0])
        generated_frames.append(frame_np)

        # Libérer la mémoire GPU à chaque frame
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return generated_frames


# ---------------------------------------------------------------------------
# Post-traitement : lissage temporel sur les pixels
# ---------------------------------------------------------------------------

def smooth_frames_temporal(
    frames: List[np.ndarray],
    sigma: float = 1.0,
) -> List[np.ndarray]:
    """Lissage gaussien temporel sur les pixels (post-génération).

    Opère sur chaque pixel (R, G, B) indépendamment le long de l'axe temporel.
    Réduit le flickering résiduel sans flouter les détails spatiaux.

    Args:
        frames: liste de (H, W, 3) uint8
        sigma: force du lissage (0.5–2.0 recommandé)

    Returns:
        Frames lissées, même format.
    """
    if len(frames) < 3 or sigma <= 0:
        return frames

    print(f"Lissage temporel (sigma={sigma})...")
    # (T, H, W, 3) float32
    arr = np.stack(frames, axis=0).astype(np.float32)

    # Filtre gaussien sur l'axe temporel uniquement
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

    return [smoothed[i] for i in range(len(frames))]


# ---------------------------------------------------------------------------
# Sauvegarde MP4
# ---------------------------------------------------------------------------

def save_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = DEFAULT_FPS,
    codec: str = "libx264",
    quality: int = 8,
) -> None:
    """Encode et sauvegarde les frames en MP4 via imageio/ffmpeg.

    Args:
        frames: liste de (H, W, 3) uint8
        output_path: chemin de sortie .mp4
        fps: fréquence d'images (25 pour MoSL)
        codec: codec vidéo (libx264 recommandé)
        quality: qualité CRF imageio (0=meilleur, 10=défaut)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Encodage MP4 : {output_path} ({len(frames)} frames @ {fps} fps)")

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        quality=quality,
        pixelformat="yuv420p",   # compatibilité maximale
        macro_block_size=None,
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Sauvegardé : {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Pipeline complet pour un signe
# ---------------------------------------------------------------------------

def process_sign(
    pipe,
    pose_source: Path,
    output_path: Path,
    resolution: int = DEFAULT_RESOLUTION,
    fps: float = DEFAULT_FPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    controlnet_conditioning_scale: float = DEFAULT_CONTROLNET_CONDITIONING_SCALE,
    positive_prompt: str = DEFAULT_POSITIVE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 42,
    temporal_sigma: float = 1.0,
    device: str = "cuda",
) -> None:
    """Pipeline complet : pose source → vidéo avatar réaliste.

    pose_source peut être :
    - un dossier contenant des PNG OpenPose
    - un fichier MP4 de squelette
    """
    # 1. Charger les frames de pose
    if pose_source.is_dir():
        pose_frames = load_pose_frames_from_dir(pose_source, resolution)
    elif pose_source.suffix.lower() == ".mp4":
        pose_frames = load_pose_frames_from_video(pose_source, resolution)
    else:
        raise ValueError(f"Source non reconnue : {pose_source} (dossier ou .mp4 attendu)")

    if not pose_frames:
        print(f"  [SKIP] Aucune frame dans {pose_source}")
        return

    # 2. Générer les frames réalistes
    generated = generate_frames(
        pipe=pipe,
        pose_frames=pose_frames,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
        device=device,
    )

    # 3. Lissage temporel post-génération
    if temporal_sigma > 0:
        generated = smooth_frames_temporal(generated, sigma=temporal_sigma)

    # 4. Sauvegarder en MP4
    save_video(generated, output_path, fps=fps)


# ---------------------------------------------------------------------------
# Mode batch
# ---------------------------------------------------------------------------

def batch_process(
    pipe,
    batch_dir: Path,
    output_dir: Path,
    skeleton_dir: Optional[Path] = None,
    **kwargs,
) -> dict:
    """Traite tous les dossiers de pose_control ou vidéos skeleton en batch.

    Cherche d'abord dans batch_dir (dossiers PNG), puis dans skeleton_dir (MP4).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "failed": 0, "skipped": 0}

    sources = []

    # Dossiers PNG OpenPose
    if batch_dir and batch_dir.exists():
        for d in sorted(batch_dir.iterdir()):
            if d.is_dir() and list(d.glob("*.png")):
                sources.append(d)

    # Vidéos squelette MP4
    if skeleton_dir and skeleton_dir.exists():
        for f in sorted(skeleton_dir.glob("*_skeleton.mp4")):
            sources.append(f)

    if not sources:
        print(f"Aucune source trouvée dans {batch_dir} ou {skeleton_dir}")
        return stats

    print(f"\nBatch : {len(sources)} sources à traiter")

    for source in sources:
        if source.is_dir():
            name = source.name.replace("_keypoints", "")
        else:
            name = source.stem.replace("_skeleton", "")

        output_path = output_dir / f"{name}_avatar.mp4"

        if output_path.exists():
            print(f"  [SKIP] {output_path.name} existe déjà")
            stats["skipped"] += 1
            continue

        print(f"\n[{name}]")
        try:
            process_sign(pipe, source, output_path, **kwargs)
            stats["processed"] += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génération vidéo avatar MoSL via ControlNet + Stable Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--pose-dir", type=Path,
                     help="Dossier contenant les PNG OpenPose (ex: outputs/pose_control/أَنْتِ_keypoints)")
    src.add_argument("--skeleton-video", type=Path,
                     help="Vidéo squelette MP4 (ex: outputs/videos/skeleton/أَنْتِ_skeleton.mp4)")
    src.add_argument("--batch-dir", type=Path,
                     help="Dossier racine pour traitement batch (ex: outputs/pose_control)")

    parser.add_argument("--skeleton-dir", type=Path,
                        default=Path("outputs/videos/skeleton"),
                        help="Dossier des vidéos squelette (mode batch)")
    parser.add_argument("--output", type=Path,
                        help="Chemin de sortie MP4 (mode single)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/avatar"),
                        help="Dossier de sortie (mode batch)")

    # Modèles
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID pour Stable Diffusion")
    parser.add_argument("--controlnet-id", default=DEFAULT_CONTROLNET_ID,
                        help="HuggingFace model ID pour ControlNet OpenPose")

    # Génération
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        choices=[512, 768],
                        help="Résolution de sortie (512 ou 768)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS,
                        help="Nombre d'étapes de débruitage (20–50)")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help="Classifier-free guidance scale (7–12)")
    parser.add_argument("--controlnet-scale", type=float,
                        default=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
                        help="Force du conditionnement ControlNet (0.8–1.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed pour reproductibilité")

    # Prompts
    parser.add_argument("--positive-prompt", default=DEFAULT_POSITIVE_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)

    # Lissage temporel
    parser.add_argument("--temporal-sigma", type=float, default=1.0,
                        help="Sigma du lissage temporel (0=désactivé, 1–2=recommandé)")

    # Hardware
    parser.add_argument("--device", default="cuda",
                        help="Device PyTorch (cuda ou cpu)")
    parser.add_argument("--fp32", action="store_true",
                        help="Utiliser float32 (plus lent, plus précis)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Activer le CPU offload (pour VRAM < 6 GB)")
    parser.add_argument("--no-xformers", action="store_true",
                        help="Désactiver xformers")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Vérifier CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponible, basculement sur CPU (très lent)")
        args.device = "cpu"

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {gpu}  VRAM : {vram:.1f} GB")

    dtype = torch.float32 if args.fp32 else torch.float16

    # Charger le pipeline
    pipe = load_pipeline(
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        device=args.device,
        dtype=dtype,
        enable_xformers=not args.no_xformers,
        enable_cpu_offload=args.cpu_offload,
    )

    # Paramètres communs
    common = dict(
        resolution=args.resolution,
        fps=args.fps,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        controlnet_conditioning_scale=args.controlnet_scale,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        temporal_sigma=args.temporal_sigma,
        device=args.device,
    )

    # Mode single
    if args.pose_dir or args.skeleton_video:
        source = args.pose_dir or args.skeleton_video
        if args.output is None:
            name = source.stem.replace("_keypoints", "").replace("_skeleton", "")
            args.output = Path("outputs/avatar") / f"{name}_avatar.mp4"

        print(f"\nSource : {source}")
        print(f"Sortie : {args.output}")
        process_sign(pipe, source, args.output, **common)

    # Mode batch
    else:
        batch_dir = args.batch_dir or Path("outputs/pose_control")
        stats = batch_process(
            pipe=pipe,
            batch_dir=batch_dir,
            output_dir=args.output_dir,
            skeleton_dir=args.skeleton_dir,
            **common,
        )
        print(f"\nBatch terminé : {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
