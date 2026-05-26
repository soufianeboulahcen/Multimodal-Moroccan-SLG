"""
Génération vidéo avatar MoSL — pipeline complet depuis JSON keypoints
=====================================================================
Lit les JSON OpenPose (data/processed/keypoints_2d/sample/),
rend chaque frame avec corps + mains détaillées,
applique un lissage temporel, et encode en MP4.

Usage :
    python scripts/run_avatar_generation.py
    python scripts/run_avatar_generation.py --mode studio --output-dir outputs/avatar_cpu
    python scripts/run_avatar_generation.py --mode neon
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import imageio

# ── Résolution et FPS ─────────────────────────────────────────────────────────
W, H = 512, 512
FPS = 25

# ── Connexions squelette BODY_25 / COCO 18 ───────────────────────────────────
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]

# Connexions main (21 keypoints MediaPipe/OpenPose hand)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # pouce
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # majeur
    (0,13),(13,14),(14,15),(15,16),  # annulaire
    (0,17),(17,18),(18,19),(19,20),  # auriculaire
    (5,9),(9,13),(13,17),            # paume
]

# Couleurs corps (BGR)
BODY_COLORS = [
    (0,255,255),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
    (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
    (255,0,255),(255,0,170),(255,0,85),
]

# Couleurs mains
HAND_R_COLOR = (0, 200, 255)    # cyan-orange pour main droite
HAND_L_COLOR = (255, 150, 0)    # orange pour main gauche
HAND_JOINT_R = (0, 255, 200)
HAND_JOINT_L = (255, 200, 0)


# ── Chargement JSON ───────────────────────────────────────────────────────────

def load_json_keypoints(json_path: str) -> dict:
    """Charge un JSON OpenPose et retourne les keypoints corps + mains."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "body": None,
        "hand_left": None,
        "hand_right": None,
        "face": None,
    }

    if "people" not in data or not data["people"]:
        return result

    person = data["people"][0]

    def parse_kp(flat, n_kp):
        if not flat:
            return None
        arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
        return arr[:n_kp]

    result["body"]       = parse_kp(person.get("pose_keypoints_2d", []), 18)
    result["hand_left"]  = parse_kp(person.get("hand_left_keypoints_2d", []), 21)
    result["hand_right"] = parse_kp(person.get("hand_right_keypoints_2d", []), 21)
    result["face"]       = parse_kp(person.get("face_keypoints_2d", []), 70)

    return result


# ── Rendu squelette ───────────────────────────────────────────────────────────

def draw_connections(canvas, kp, connections, color, thickness=3, conf_thresh=0.05):
    """Dessine les connexions entre keypoints sur le canvas."""
    if kp is None:
        return
    n = len(kp)
    for i, j in connections:
        if i >= n or j >= n:
            continue
        ci = kp[i, 2] if kp.shape[1] > 2 else 1.0
        cj = kp[j, 2] if kp.shape[1] > 2 else 1.0
        if ci < conf_thresh or cj < conf_thresh:
            continue
        pt1 = (int(kp[i, 0]), int(kp[i, 1]))
        pt2 = (int(kp[j, 0]), int(kp[j, 1]))
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_joints(canvas, kp, color, radius=5, conf_thresh=0.05):
    """Dessine les joints (cercles) sur le canvas."""
    if kp is None:
        return
    for k in range(len(kp)):
        c = kp[k, 2] if kp.shape[1] > 2 else 1.0
        if c < conf_thresh:
            continue
        x, y = int(kp[k, 0]), int(kp[k, 1])
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def render_skeleton_frame(kp_data: dict, w: int = W, h: int = H) -> np.ndarray:
    """Rend une frame squelette OpenPose colorée (fond noir)."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    body = kp_data.get("body")
    hl   = kp_data.get("hand_left")
    hr   = kp_data.get("hand_right")

    # Corps : connexions colorées par segment
    if body is not None:
        for idx, (i, j) in enumerate(BODY_CONNECTIONS):
            color = BODY_COLORS[idx % len(BODY_COLORS)]
            if i < len(body) and j < len(body):
                ci = body[i, 2] if body.shape[1] > 2 else 1.0
                cj = body[j, 2] if body.shape[1] > 2 else 1.0
                if ci > 0.05 and cj > 0.05:
                    pt1 = (int(body[i, 0]), int(body[i, 1]))
                    pt2 = (int(body[j, 0]), int(body[j, 1]))
                    cv2.line(canvas, pt1, pt2, color, 4, cv2.LINE_AA)
        draw_joints(canvas, body, (255, 255, 255), radius=6)

    # Mains : connexions fines
    draw_connections(canvas, hr, HAND_CONNECTIONS, HAND_R_COLOR, thickness=2)
    draw_connections(canvas, hl, HAND_CONNECTIONS, HAND_L_COLOR, thickness=2)
    draw_joints(canvas, hr, HAND_JOINT_R, radius=3)
    draw_joints(canvas, hl, HAND_JOINT_L, radius=3)

    return canvas


# ── Rendu mode "studio" ───────────────────────────────────────────────────────

def render_studio_frame(kp_data: dict, w: int = W, h: int = H) -> np.ndarray:
    """
    Rendu studio : silhouette humaine synthétique + squelette coloré.
    Simule un signeur devant un fond de studio sombre.
    """
    # 1. Fond dégradé radial sombre
    Y_idx, X_idx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((X_idx - cx)**2 + (Y_idx - cy)**2).astype(np.float32)
    max_d = float(np.sqrt(cx**2 + cy**2))
    grad = np.clip(1.0 - dist / max_d, 0, 1) * 0.35

    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = grad * 20   # B
    bg[:, :, 1] = grad * 20   # G
    bg[:, :, 2] = grad * 30   # R

    # 2. Rendre le squelette sur canvas noir
    skel = render_skeleton_frame(kp_data, w, h).astype(np.float32)
    gray = cv2.cvtColor(skel.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 3. Silhouette corporelle (dilatation du masque squelette)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    body_mask = cv2.dilate(mask, kernel, iterations=3).astype(np.float32)
    body_mask = cv2.GaussianBlur(body_mask, (61, 61), 22) / 255.0

    # 4. Couleur silhouette : vêtement sombre + zone tête peau
    silhouette = np.full((h, w, 3), [38, 38, 50], dtype=np.float32)

    # Zone tête : teinte peau naturelle
    head_h = int(h * 0.26)
    head_zone = np.zeros((h, w), dtype=np.float32)
    head_zone[:head_h, :] = 1.0
    head_zone = cv2.GaussianBlur(head_zone, (71, 71), 28)

    skin = np.array([145, 185, 215], dtype=np.float32)  # BGR peau
    for c in range(3):
        silhouette[:, :, c] = (
            silhouette[:, :, c] * (1 - head_zone) + skin[c] * head_zone
        )

    # 5. Lumière studio (source haut-gauche)
    lx, ly = int(w * 0.25), int(h * 0.05)
    ldist = np.sqrt((X_idx - lx)**2 + (Y_idx - ly)**2).astype(np.float32)
    light = np.clip(1.0 - ldist / (w * 0.85), 0, 1) * 0.45
    for c in range(3):
        silhouette[:, :, c] = np.clip(silhouette[:, :, c] + light * 55, 0, 255)

    # 6. Composer fond + silhouette
    alpha = body_mask[:, :, np.newaxis]
    result = bg * (1 - alpha) + silhouette * alpha

    # 7. Superposer squelette avec glow
    skel_mask = (gray > 10).astype(np.float32)[:, :, np.newaxis]
    glow = cv2.GaussianBlur(skel.astype(np.uint8), (11, 11), 4).astype(np.float32)
    result = result + glow * skel_mask * 0.45
    result = result + skel * skel_mask * 0.85

    # 8. Vignette
    vignette = np.clip(1.0 - dist / max_d * 0.55, 0, 1)[:, :, np.newaxis]
    result = result * vignette

    return np.clip(result, 0, 255).astype(np.uint8)


def render_neon_frame(kp_data: dict, w: int = W, h: int = H) -> np.ndarray:
    """Rendu neon : squelette lumineux avec glow sur fond noir."""
    skel = render_skeleton_frame(kp_data, w, h)
    gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.float32)[:, :, np.newaxis]

    result = np.zeros((h, w, 3), dtype=np.float32)
    for sigma, strength in [(17, 0.25), (9, 0.4), (5, 0.6), (3, 0.8), (0, 1.0)]:
        if sigma > 0:
            layer = cv2.GaussianBlur(skel, (sigma*2+1, sigma*2+1), sigma).astype(np.float32)
        else:
            layer = skel.astype(np.float32)
        result += layer * mask * strength

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Lissage temporel ──────────────────────────────────────────────────────────

def smooth_frames(frames: List[np.ndarray], sigma: float = 1.0) -> List[np.ndarray]:
    """Filtre gaussien 1D sur l'axe temporel (T, H, W, 3)."""
    if len(frames) < 3 or sigma <= 0:
        return frames
    arr = np.stack(frames, axis=0).astype(np.float32)
    smoothed = gaussian_filter1d(arr, sigma=sigma, axis=0)
    return [np.clip(smoothed[i], 0, 255).astype(np.uint8) for i in range(len(frames))]


# ── Sauvegarde MP4 ────────────────────────────────────────────────────────────

def save_mp4(frames_bgr: List[np.ndarray], output_path: str, fps: float = FPS) -> None:
    """Encode les frames BGR en MP4 via imageio/ffmpeg."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264",
        quality=8, pixelformat="yuv420p", macro_block_size=None,
    )
    for frame in frames_rgb:
        writer.append_data(frame)
    writer.close()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Sauvegardé : {os.path.basename(output_path)}  "
          f"({len(frames_bgr)} frames, {size_mb:.1f} MB)")


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate_from_json_dir(
    json_dir: str,
    output_path: str,
    mode: str = "studio",
    temporal_sigma: float = 1.0,
    fps: float = FPS,
    w: int = W,
    h: int = H,
) -> None:
    """Génère une vidéo avatar depuis un dossier de JSON OpenPose."""
    json_files = sorted(
        [f for f in os.listdir(json_dir) if f.endswith(".json")],
        key=lambda x: x
    )
    if not json_files:
        raise FileNotFoundError(f"Aucun JSON dans {json_dir}")

    print(f"  {len(json_files)} frames JSON")
    render_fn = render_studio_frame if mode == "studio" else render_neon_frame

    frames = []
    for jf in tqdm(json_files, desc=f"  Rendu [{mode}]", unit="fr"):
        fp = os.path.join(json_dir, jf)
        kp_data = load_json_keypoints(fp)
        frame = render_fn(kp_data, w, h)
        frames.append(frame)

    if temporal_sigma > 0:
        frames = smooth_frames(frames, sigma=temporal_sigma)

    save_mp4(frames, output_path, fps=fps)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Génération vidéo avatar MoSL depuis JSON keypoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-dir", default="data/processed/keypoints_2d/sample",
                        help="Dossier contenant les JSON OpenPose")
    parser.add_argument("--output-dir", default="outputs/avatar_cpu",
                        help="Dossier de sortie")
    parser.add_argument("--output-name", default="sample_avatar",
                        help="Nom de base du fichier MP4 de sortie")
    parser.add_argument("--mode", choices=["studio", "neon"], default="studio")
    parser.add_argument("--temporal-sigma", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--width", type=int, default=W)
    parser.add_argument("--height", type=int, default=H)
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, f"{args.output_name}_{args.mode}.mp4")
    print(f"\nSource JSON : {args.json_dir}")
    print(f"Mode        : {args.mode}")
    print(f"Sortie      : {output_path}")

    generate_from_json_dir(
        json_dir=args.json_dir,
        output_path=output_path,
        mode=args.mode,
        temporal_sigma=args.temporal_sigma,
        fps=args.fps,
        w=args.width,
        h=args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
