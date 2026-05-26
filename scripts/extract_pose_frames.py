"""
Extraction et conversion des frames de pose OpenPose — MoSL
============================================================
Utilitaire pour préparer les frames de conditionnement ControlNet depuis :
  1. Vidéos squelette MP4 (outputs/videos/skeleton/)
  2. Keypoints JSON OpenPose (data/processed/keypoints_2d/)
  3. Dossiers PNG existants (outputs/pose_control/)

Sortie : dossiers de PNG OpenPose prêts pour pose2video_controlnet.py

Usage :
    # Extraire depuis une vidéo squelette
    python scripts/extract_pose_frames.py \\
        --source outputs/videos/skeleton/أَنْتِ_skeleton.mp4 \\
        --output-dir outputs/pose_control/أَنْتِ_keypoints

    # Extraire depuis JSON keypoints OpenPose
    python scripts/extract_pose_frames.py \\
        --source data/processed/keypoints_2d/sample \\
        --output-dir outputs/pose_control/sample_keypoints \\
        --from-json

    # Batch : toutes les vidéos skeleton
    python scripts/extract_pose_frames.py \\
        --batch-skeleton-dir outputs/videos/skeleton \\
        --output-dir outputs/pose_control
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constantes OpenPose (BODY_25 / COCO 18 keypoints)
# ---------------------------------------------------------------------------

# Connexions squelette COCO 18 points
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # bras droit
    (1, 5), (5, 6), (6, 7),             # bras gauche
    (1, 8), (8, 9), (9, 10),            # jambe droite
    (1, 11), (11, 12), (12, 13),        # jambe gauche
    (0, 14), (14, 16),                  # œil/oreille droite
    (0, 15), (15, 17),                  # œil/oreille gauche
]

# Couleurs par partie du corps (BGR pour OpenCV)
COCO_COLORS = [
    (255, 0, 0),    # 0: nez
    (255, 85, 0),   # 1: cou
    (255, 170, 0),  # 2: épaule droite
    (255, 255, 0),  # 3: coude droit
    (170, 255, 0),  # 4: poignet droit
    (85, 255, 0),   # 5: épaule gauche
    (0, 255, 0),    # 6: coude gauche
    (0, 255, 85),   # 7: poignet gauche
    (0, 255, 170),  # 8: hanche droite
    (0, 255, 255),  # 9: genou droit
    (0, 170, 255),  # 10: cheville droite
    (0, 85, 255),   # 11: hanche gauche
    (0, 0, 255),    # 12: genou gauche
    (85, 0, 255),   # 13: cheville gauche
    (170, 0, 255),  # 14: œil droit
    (255, 0, 255),  # 15: œil gauche
    (255, 0, 170),  # 16: oreille droite
    (255, 0, 85),   # 17: oreille gauche
]

# Couleurs des connexions (une par lien)
CONNECTION_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]


# ---------------------------------------------------------------------------
# Rendu OpenPose depuis keypoints
# ---------------------------------------------------------------------------

def render_openpose_frame(
    keypoints: np.ndarray,
    width: int = 512,
    height: int = 512,
    point_radius: int = 6,
    line_thickness: int = 3,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Rend une frame OpenPose colorée depuis un tableau de keypoints.

    Args:
        keypoints: tableau (N, 2) ou (N, 3) avec [x, y] ou [x, y, conf]
                   coordonnées normalisées [0, 1] ou pixels absolus
        width, height: dimensions de l'image de sortie
        point_radius: rayon des cercles de keypoints
        line_thickness: épaisseur des lignes de connexion
        background: couleur de fond (BGR)

    Returns:
        Image numpy (H, W, 3) uint8 au format RGB
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = background

    if keypoints is None or len(keypoints) == 0:
        return canvas

    kp = np.array(keypoints)

    # Normaliser si coordonnées [0, 1]
    if kp[:, :2].max() <= 1.0:
        kp[:, 0] = kp[:, 0] * width
        kp[:, 1] = kp[:, 1] * height

    kp = kp.astype(np.float32)
    n_kp = min(len(kp), 18)

    # Dessiner les connexions
    for idx, (i, j) in enumerate(COCO_SKELETON_CONNECTIONS):
        if i >= n_kp or j >= n_kp:
            continue

        # Vérifier la confiance si disponible
        conf_i = kp[i, 2] if kp.shape[1] > 2 else 1.0
        conf_j = kp[j, 2] if kp.shape[1] > 2 else 1.0

        if conf_i < 0.1 or conf_j < 0.1:
            continue

        pt1 = (int(kp[i, 0]), int(kp[i, 1]))
        pt2 = (int(kp[j, 0]), int(kp[j, 1]))

        color = CONNECTION_COLORS[idx % len(CONNECTION_COLORS)]
        cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    # Dessiner les keypoints
    for k in range(n_kp):
        conf = kp[k, 2] if kp.shape[1] > 2 else 1.0
        if conf < 0.1:
            continue

        x, y = int(kp[k, 0]), int(kp[k, 1])
        color = COCO_COLORS[k % len(COCO_COLORS)]
        cv2.circle(canvas, (x, y), point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Convertir BGR → RGB pour cohérence avec le reste du pipeline
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Chargement depuis JSON OpenPose
# ---------------------------------------------------------------------------

def load_keypoints_from_json(json_path: Path) -> Optional[np.ndarray]:
    """Charge les keypoints depuis un fichier JSON OpenPose standard.

    Supporte les formats :
    - OpenPose standard : {"people": [{"pose_keypoints_2d": [...]}]}
    - Format MoSL : {"keypoints": [...]} ou tableau direct
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Format OpenPose standard
    if isinstance(data, dict) and "people" in data:
        people = data["people"]
        if not people:
            return None
        kp_flat = people[0].get("pose_keypoints_2d", [])
        if not kp_flat:
            return None
        kp = np.array(kp_flat, dtype=np.float32).reshape(-1, 3)
        return kp

    # Format MoSL keypoints_2d
    if isinstance(data, dict) and "keypoints" in data:
        kp = np.array(data["keypoints"], dtype=np.float32)
        if kp.ndim == 1:
            kp = kp.reshape(-1, 3)
        return kp

    # Tableau direct
    if isinstance(data, list):
        kp = np.array(data, dtype=np.float32)
        if kp.ndim == 1:
            kp = kp.reshape(-1, 3)
        return kp

    return None


def extract_from_json_dir(
    json_dir: Path,
    output_dir: Path,
    resolution: int = 512,
) -> int:
    """Convertit un dossier de JSON OpenPose en PNG de pose.

    Returns:
        Nombre de frames générées.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"Aucun JSON dans {json_dir}")
        return 0

    count = 0
    for i, jf in enumerate(json_files):
        kp = load_keypoints_from_json(jf)
        if kp is None:
            continue

        frame = render_openpose_frame(kp, width=resolution, height=resolution)
        out_path = output_dir / f"pose_{i:06d}.png"
        Image.fromarray(frame).save(out_path)
        count += 1

    print(f"  {count} frames PNG générées dans {output_dir}")
    return count


# ---------------------------------------------------------------------------
# Extraction depuis vidéo MP4
# ---------------------------------------------------------------------------

def extract_from_video(
    video_path: Path,
    output_dir: Path,
    resolution: int = 512,
    max_frames: Optional[int] = None,
) -> int:
    """Extrait les frames d'une vidéo squelette MP4 en PNG.

    Les vidéos skeleton/* contiennent déjà le rendu OpenPose coloré,
    donc on extrait directement sans re-rendu.

    Returns:
        Nombre de frames extraites.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir : {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Vidéo : {video_path.name}  {total} frames @ {fps:.1f} fps")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and count >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (resolution, resolution),
                                   interpolation=cv2.INTER_LANCZOS4)

        out_path = output_dir / f"pose_{count:06d}.png"
        Image.fromarray(frame_resized).save(out_path)
        count += 1

    cap.release()
    print(f"  {count} frames extraites dans {output_dir}")
    return count


# ---------------------------------------------------------------------------
# Batch depuis dossier skeleton
# ---------------------------------------------------------------------------

def batch_extract_skeleton(
    skeleton_dir: Path,
    output_dir: Path,
    resolution: int = 512,
) -> None:
    """Extrait toutes les vidéos *_skeleton.mp4 en dossiers PNG."""
    videos = sorted(skeleton_dir.glob("*_skeleton.mp4"))
    if not videos:
        print(f"Aucune vidéo *_skeleton.mp4 dans {skeleton_dir}")
        return

    print(f"Batch extraction : {len(videos)} vidéos")
    for v in videos:
        name = v.stem.replace("_skeleton", "")
        out = output_dir / f"{name}_keypoints"
        if out.exists() and list(out.glob("*.png")):
            print(f"  [SKIP] {out.name} existe déjà")
            continue
        print(f"\n[{name}]")
        extract_from_video(v, out, resolution=resolution)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extraction frames OpenPose pour ControlNet MoSL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path,
                     help="Vidéo MP4 ou dossier JSON source")
    src.add_argument("--batch-skeleton-dir", type=Path,
                     help="Dossier de vidéos *_skeleton.mp4 (batch)")

    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/pose_control"),
                        help="Dossier de sortie pour les PNG")
    parser.add_argument("--from-json", action="store_true",
                        help="Source est un dossier de JSON OpenPose")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limite le nombre de frames extraites")

    args = parser.parse_args()

    if args.batch_skeleton_dir:
        batch_extract_skeleton(
            args.batch_skeleton_dir,
            args.output_dir,
            resolution=args.resolution,
        )
    elif args.from_json:
        extract_from_json_dir(args.source, args.output_dir, args.resolution)
    else:
        # Vidéo MP4 unique
        name = args.source.stem.replace("_skeleton", "")
        out = args.output_dir / f"{name}_keypoints"
        extract_from_video(args.source, out,
                           resolution=args.resolution,
                           max_frames=args.max_frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
