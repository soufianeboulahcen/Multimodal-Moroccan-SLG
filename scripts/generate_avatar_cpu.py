"""
Génération vidéo avatar MoSL — rendu CPU (sans GPU/diffusion)
=============================================================
Produit une vidéo MP4 réaliste d'un signeur à partir des squelettes OpenPose
en utilisant un rendu stylisé haute qualité + lissage temporel.

Deux modes :
  1. --mode skeleton  : rendu squelette coloré neon sur fond sombre (rapide)
  2. --mode studio    : rendu "studio" avec silhouette humaine synthétique

Usage :
    python scripts/generate_avatar_cpu.py \\
        --source outputs/videos/skeleton \\
        --output-dir outputs/avatar_cpu \\
        --mode studio

    python scripts/generate_avatar_cpu.py \\
        --source outputs/pose_control/أَنْتِ_keypoints \\
        --output outputs/avatar_cpu/أَنْتِ_avatar.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import imageio

# ── Constantes ────────────────────────────────────────────────────────────────

FPS = 25
RESOLUTION = 512

# Connexions squelette COCO 18 keypoints
SKELETON_CONNECTIONS = [
    (0, 1),   # nez → cou
    (1, 2), (2, 3), (3, 4),    # cou → épaule D → coude D → poignet D
    (1, 5), (5, 6), (6, 7),    # cou → épaule G → coude G → poignet G
    (1, 8), (8, 9), (9, 10),   # cou → hanche D → genou D → cheville D
    (1, 11), (11, 12), (12, 13), # cou → hanche G → genou G → cheville G
    (0, 14), (14, 16),          # nez → œil D → oreille D
    (0, 15), (15, 17),          # nez → œil G → oreille G
]

# Couleurs par segment (BGR)
SEGMENT_COLORS = {
    "head":       (220, 180, 140),   # peau
    "torso":      (40,  40,  80),    # vêtement sombre
    "arm_right":  (40,  40,  80),
    "arm_left":   (40,  40,  80),
    "leg_right":  (30,  30,  60),
    "leg_left":   (30,  30,  60),
}

# Couleurs neon pour mode skeleton
NEON_COLORS = [
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (0, 255, 0),     # vert
    (255, 255, 0),   # jaune
    (255, 128, 0),   # orange
    (128, 0, 255),   # violet
]


# ── Chargement des frames ─────────────────────────────────────────────────────

def load_frames_from_dir(pose_dir: Path, resolution: int = RESOLUTION) -> List[np.ndarray]:
    """Charge les PNG OpenPose depuis un dossier, triés par numéro."""
    # Utiliser os.listdir pour éviter les problèmes de normalisation Unicode
    import os
    dir_str = str(pose_dir)
    try:
        all_files = os.listdir(dir_str)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dossier introuvable : {pose_dir}")
    png_names = sorted([f for f in all_files if f.lower().endswith(".png")])
    files = [Path(os.path.join(dir_str, n)) for n in png_names]
    if not files:
        raise FileNotFoundError(f"Aucun PNG dans {pose_dir}")
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        frames.append(img)  # BGR
    print(f"  {len(frames)} frames chargées depuis {pose_dir.name}")
    return frames


def load_frames_from_video(video_path: Path, resolution: int = RESOLUTION) -> List[np.ndarray]:
    """Extrait les frames d'une vidéo MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir : {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()
    print(f"  {len(frames)} frames extraites de {video_path.name}")
    return frames


# ── Rendu mode "studio" ───────────────────────────────────────────────────────

def render_studio_frame(skeleton_frame_bgr: np.ndarray, resolution: int = RESOLUTION) -> np.ndarray:
    """
    Transforme une frame squelette OpenPose en rendu "studio" stylisé :
    - Fond dégradé sombre (studio photo)
    - Silhouette humaine synthétique autour du squelette
    - Squelette coloré visible par-dessus
    - Lissage et effets de lumière
    """
    H, W = resolution, resolution

    # 1. Fond studio : dégradé radial sombre
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    center_x, center_y = W // 2, H // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    gradient = (1.0 - dist / max_dist) * 0.3
    bg[:, :, 0] = (gradient * 25).astype(np.uint8)
    bg[:, :, 1] = (gradient * 25).astype(np.uint8)
    bg[:, :, 2] = (gradient * 35).astype(np.uint8)

    # 2. Extraire les pixels non-noirs du squelette (les os colorés)
    gray = cv2.cvtColor(skeleton_frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # 3. Dilater le masque pour créer une silhouette corporelle
    kernel_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    body_mask = cv2.dilate(mask, kernel_body, iterations=3)

    # Appliquer un flou gaussien pour adoucir les bords de la silhouette
    body_mask_blur = cv2.GaussianBlur(body_mask.astype(np.float32), (51, 51), 20)
    body_mask_blur = np.clip(body_mask_blur / 255.0, 0, 1)

    # 4. Couleur de la silhouette (vêtement sombre + peau)
    # Zone haute = peau (tête), zone basse = vêtement
    silhouette_color = np.zeros((H, W, 3), dtype=np.float32)

    # Vêtement (corps entier, sombre)
    silhouette_color[:, :, 0] = 35   # B
    silhouette_color[:, :, 1] = 35   # G
    silhouette_color[:, :, 2] = 45   # R

    # Zone tête (quart supérieur) : teinte peau
    head_zone = int(H * 0.28)
    head_mask_zone = np.zeros((H, W), dtype=np.float32)
    head_mask_zone[:head_zone, :] = 1.0
    head_mask_zone = cv2.GaussianBlur(head_mask_zone, (61, 61), 25)

    skin_color = np.array([140, 180, 210], dtype=np.float32)  # BGR peau
    for c in range(3):
        silhouette_color[:, :, c] = (
            silhouette_color[:, :, c] * (1 - head_mask_zone) +
            skin_color[c] * head_mask_zone
        )

    # 5. Lumière studio : source lumineuse en haut à gauche
    light_x, light_y = int(W * 0.3), int(H * 0.1)
    light_dist = np.sqrt((X - light_x)**2 + (Y - light_y)**2)
    light_intensity = np.clip(1.0 - light_dist / (W * 0.9), 0, 1) * 0.4
    for c in range(3):
        silhouette_color[:, :, c] = np.clip(
            silhouette_color[:, :, c] + light_intensity * 60, 0, 255
        )

    # 6. Composer : fond + silhouette + squelette
    result = bg.astype(np.float32)

    # Ajouter la silhouette
    alpha_body = body_mask_blur[:, :, np.newaxis]
    result = result * (1 - alpha_body) + silhouette_color * alpha_body

    # Superposer le squelette coloré avec glow
    skel_float = skeleton_frame_bgr.astype(np.float32)

    # Effet glow : flou du squelette
    skel_glow = cv2.GaussianBlur(skeleton_frame_bgr, (9, 9), 3).astype(np.float32)
    skel_mask = (gray > 15).astype(np.float32)[:, :, np.newaxis]

    result = result + skel_glow * skel_mask * 0.4   # glow
    result = result + skel_float * skel_mask * 0.8  # squelette net

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 7. Post-traitement : légère vignette
    vignette = 1.0 - (dist / max_dist) * 0.5
    vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]
    result = (result.astype(np.float32) * vignette).astype(np.uint8)

    return result


def render_neon_frame(skeleton_frame_bgr: np.ndarray, resolution: int = RESOLUTION) -> np.ndarray:
    """
    Rendu neon : squelette lumineux sur fond noir avec effet glow.
    Plus rapide que le mode studio.
    """
    H, W = resolution, resolution

    # Fond noir profond
    result = np.zeros((H, W, 3), dtype=np.float32)

    skel = skeleton_frame_bgr.astype(np.float32)
    gray = cv2.cvtColor(skeleton_frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.float32)[:, :, np.newaxis]

    # Glow multi-niveaux
    for sigma, strength in [(15, 0.3), (7, 0.5), (3, 0.7), (0, 1.0)]:
        if sigma > 0:
            blurred = cv2.GaussianBlur(skeleton_frame_bgr, (sigma*2+1, sigma*2+1), sigma)
            layer = blurred.astype(np.float32)
        else:
            layer = skel
        result += layer * mask * strength

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ── Lissage temporel ──────────────────────────────────────────────────────────

def smooth_frames(frames: List[np.ndarray], sigma: float = 1.0) -> List[np.ndarray]:
    """Filtre gaussien 1D sur l'axe temporel pour réduire le flickering."""
    if len(frames) < 3 or sigma <= 0:
        return frames
    arr = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, 3)
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    return [smoothed[i] for i in range(len(frames))]


# ── Sauvegarde MP4 ────────────────────────────────────────────────────────────

def save_mp4(frames_bgr: List[np.ndarray], output_path: Path, fps: float = FPS) -> None:
    """Encode les frames BGR en MP4 via imageio/ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir BGR → RGB pour imageio
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    for frame in frames_rgb:
        writer.append_data(frame)
    writer.close()

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Sauvegardé : {output_path.name}  ({len(frames_bgr)} frames, {size_mb:.1f} MB)")


# ── Pipeline complet pour un signe ───────────────────────────────────────────

def process_sign(
    source: Path,
    output_path: Path,
    mode: str = "studio",
    resolution: int = RESOLUTION,
    fps: float = FPS,
    temporal_sigma: float = 1.0,
) -> None:
    """Pipeline complet : source → vidéo avatar."""

    # 1. Charger les frames
    import os
    src_str = str(source)
    is_dir = os.path.isdir(src_str)
    is_mp4 = src_str.lower().endswith(".mp4") and os.path.isfile(src_str)

    if is_dir:
        frames = load_frames_from_dir(source, resolution)
    elif is_mp4:
        frames = load_frames_from_video(source, resolution)
    else:
        raise ValueError(f"Source non reconnue : {source}")

    if not frames:
        print(f"  [SKIP] Aucune frame dans {source}")
        return

    # 2. Rendre chaque frame
    render_fn = render_studio_frame if mode == "studio" else render_neon_frame
    rendered = []
    for frame in tqdm(frames, desc=f"  Rendu [{mode}]", unit="fr"):
        rendered.append(render_fn(frame, resolution))

    # 3. Lissage temporel
    if temporal_sigma > 0:
        rendered = smooth_frames(rendered, sigma=temporal_sigma)

    # 4. Sauvegarder
    save_mp4(rendered, output_path, fps=fps)


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_process(
    skeleton_dir: Path,
    pose_control_dir: Path,
    output_dir: Path,
    mode: str = "studio",
    resolution: int = RESOLUTION,
    fps: float = FPS,
    temporal_sigma: float = 1.0,
) -> None:
    """Traite toutes les sources disponibles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []

    # Priorité 1 : dossiers PNG dans pose_control
    import os
    if os.path.isdir(str(pose_control_dir)):
        for name in sorted(os.listdir(str(pose_control_dir))):
            full = Path(os.path.join(str(pose_control_dir), name))
            if os.path.isdir(str(full)) and os.listdir(str(full)):
                label = name.replace("_keypoints", "")
                sources.append(("dir", full, label))

    # Priorité 2 : vidéos skeleton MP4
    if skeleton_dir.exists():
        for v in sorted(skeleton_dir.glob("*_skeleton.mp4")):
            name = v.stem.replace("_skeleton", "")
            # Ne pas dupliquer si déjà dans pose_control
            if not any(s[2] == name for s in sources):
                sources.append(("video", v, name))

    if not sources:
        print("Aucune source trouvée.")
        return

    print(f"\nBatch : {len(sources)} signes à traiter")
    ok, fail = 0, 0

    for kind, src, name in sources:
        out = output_dir / f"{name}_avatar.mp4"
        if out.exists():
            print(f"  [SKIP] {out.name}")
            ok += 1
            continue
        print(f"\n[{name}]")
        try:
            process_sign(src, out, mode=mode, resolution=resolution,
                         fps=fps, temporal_sigma=temporal_sigma)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            fail += 1

    print(f"\nBatch terminé : {ok} OK, {fail} échecs")
    print(f"Vidéos dans : {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Génération vidéo avatar MoSL (CPU, sans GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--source", type=Path,
                     help="Dossier PNG ou vidéo MP4 source (mode single)")
    src.add_argument("--batch", action="store_true",
                     help="Traiter toutes les sources disponibles")

    parser.add_argument("--output", type=Path, default=None,
                        help="Chemin de sortie MP4 (mode single)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/avatar_cpu"),
                        help="Dossier de sortie (mode batch)")
    parser.add_argument("--skeleton-dir", type=Path,
                        default=Path("outputs/videos/skeleton"))
    parser.add_argument("--pose-control-dir", type=Path,
                        default=Path("outputs/pose_control"))
    parser.add_argument("--mode", choices=["studio", "neon"], default="studio",
                        help="Mode de rendu : studio (réaliste) ou neon (stylisé)")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--temporal-sigma", type=float, default=1.0,
                        help="Lissage temporel (0=désactivé)")

    args = parser.parse_args()

    if args.batch or args.source is None:
        batch_process(
            skeleton_dir=args.skeleton_dir,
            pose_control_dir=args.pose_control_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            resolution=args.resolution,
            fps=args.fps,
            temporal_sigma=args.temporal_sigma,
        )
    else:
        if args.output is None:
            name = args.source.stem.replace("_keypoints", "").replace("_skeleton", "")
            args.output = args.output_dir / f"{name}_avatar.mp4"
        print(f"Source : {args.source}")
        print(f"Sortie : {args.output}")
        process_sign(
            source=args.source,
            output_path=args.output,
            mode=args.mode,
            resolution=args.resolution,
            fps=args.fps,
            temporal_sigma=args.temporal_sigma,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
