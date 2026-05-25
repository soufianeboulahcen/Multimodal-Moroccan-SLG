"""Render SMPL-X parameter sequences as avatar videos.

Three rendering backends, selectable via --backend:

  pytorch3d  (Phase 1 — default)
    Differentiable mesh renderer. Produces textured mesh videos at 512×512.
    Fast (~0.1s/frame on GPU). Not photorealistic but immediately runnable.
    Requires: pip install pytorch3d

  blender    (Phase 2)
    Photorealistic Cycles renderer driven via the bpy Python API.
    Requires Blender ≥ 3.6 installed and accessible as `blender` in PATH.
    Slow (~10-30s/frame on GPU with Cycles). Best for demo videos.

  overlay    (fallback — no extra dependencies)
    Draws the SMPL-X joint skeleton on a black background using OpenCV.
    Uses the same visual style as the existing OpenPose overlay scripts.
    Always available. Useful for quick sanity checks.

Input: data/processed/smplx_params/<category>/<clip>.npz
       (produced by mosl/pose/fit_smplx.py)

Output: outputs/avatar_videos/<category>/<clip>.mp4

Usage:
    # Render one clip with PyTorch3D
    python scripts/render_smplx_video.py --clip الأذان --backend pytorch3d

    # Render all clips with overlay backend (no extra deps)
    python scripts/render_smplx_video.py --all --backend overlay

    # Render from a raw (T, 150) NPZ (skips SMPL-X fitting)
    python scripts/render_smplx_video.py --pose-npz path/to/pose.npz --backend overlay
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Joint skeleton definition for overlay rendering
# ---------------------------------------------------------------------------

# COCO-18 body bones (joint index pairs)
BODY_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # nose-neck-Rshoulder-Relbow-Rwrist
    (1, 5), (5, 6), (6, 7),            # neck-Lshoulder-Lelbow-Lwrist
    (1, 8), (8, 9), (9, 10),           # neck-Rhip-Rknee-Rankle
    (8, 12), (12, 13), (13, 14),       # Rhip-Lhip-Lknee-Lankle
    (0, 15), (15, 17),                 # nose-Reye-Rear
    (0, 16), (16, 18),                 # nose-Leye-Lear
]

# Hand bones: left (18-38) and right (39-49)
LHAND_BONES = [(18 + i, 18 + i + 1) for i in range(20)]
RHAND_BONES = [(39 + i, 39 + i + 1) for i in range(10)]

# Colour scheme matching existing OpenPose overlay scripts
BODY_COLOUR = (0, 200, 0)     # green
LHAND_COLOUR = (200, 100, 0)  # blue-ish
RHAND_COLOUR = (0, 100, 200)  # red-ish
JOINT_COLOUR = (255, 165, 0)  # orange


# ---------------------------------------------------------------------------
# Overlay renderer (no extra dependencies)
# ---------------------------------------------------------------------------

def render_overlay_frame(
    joints: np.ndarray,   # (50, 3) — xyz in arbitrary units
    frame_size: int = 512,
    margin: float = 0.1,
) -> np.ndarray:
    """Render one frame as a skeleton on black background.

    Projects 3D joints to 2D by dropping the z-axis, then normalises
    to fit within the frame with a margin.

    Returns (H, W, 3) uint8 BGR image.
    """
    img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    # Normalise x, y to [margin, 1-margin] of frame_size
    xy = joints[:, :2].copy()
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = (xy_max - xy_min).clip(min=1e-6)
    xy_norm = (xy - xy_min) / xy_range   # [0, 1]
    xy_px = (xy_norm * (1 - 2 * margin) + margin) * frame_size
    xy_px = xy_px.astype(int)

    def draw_bones(bones, colour):
        for a, b in bones:
            if a < len(xy_px) and b < len(xy_px):
                pt_a = tuple(xy_px[a])
                pt_b = tuple(xy_px[b])
                cv2.line(img, pt_a, pt_b, colour, 2, cv2.LINE_AA)

    draw_bones(BODY_BONES, BODY_COLOUR)
    draw_bones(LHAND_BONES, LHAND_COLOUR)
    draw_bones(RHAND_BONES, RHAND_COLOUR)

    for pt in xy_px:
        cv2.circle(img, tuple(pt), 3, JOINT_COLOUR, -1, cv2.LINE_AA)

    return img


def render_overlay_video(
    pose_seq: np.ndarray,   # (T, 150)
    output_path: Path,
    fps: float = 25.0,
    frame_size: int = 512,
) -> None:
    """Render a full pose sequence as an overlay video."""
    T = pose_seq.shape[0]
    joints_seq = pose_seq.reshape(T, 50, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_size, frame_size))

    for t in range(T):
        frame = render_overlay_frame(joints_seq[t], frame_size=frame_size)
        writer.write(frame)

    writer.release()


# ---------------------------------------------------------------------------
# PyTorch3D renderer
# ---------------------------------------------------------------------------

def render_pytorch3d_video(
    smplx_params_path: Path,
    output_path: Path,
    smplx_model_path: str,
    fps: float = 25.0,
    frame_size: int = 512,
    device: str = "cuda",
) -> None:
    """Render SMPL-X mesh sequence using PyTorch3D.

    Requires: pip install pytorch3d smplx
    """
    try:
        import smplx
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
            SoftPhongShader,
            TexturesVertex,
        )
        from pytorch3d.structures import Meshes
    except ImportError as e:
        raise ImportError(
            f"pytorch3d and smplx are required for this backend.\n"
            f"Install: pip install pytorch3d smplx\n"
            f"Original error: {e}"
        )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load SMPL-X params
    data = np.load(smplx_params_path, allow_pickle=False)
    T = data["body_pose"].shape[0]

    body_pose = torch.from_numpy(data["body_pose"]).float().to(dev)
    global_orient = torch.from_numpy(data["global_orient"]).float().to(dev)
    betas = torch.from_numpy(data["betas"]).float().to(dev).unsqueeze(0).expand(T, -1)
    transl = torch.from_numpy(data["transl"]).float().to(dev)
    lhand = torch.from_numpy(data["left_hand_pose"]).float().to(dev)
    rhand = torch.from_numpy(data["right_hand_pose"]).float().to(dev)
    expression = torch.from_numpy(data["expression"]).float().to(dev)
    jaw_pose = torch.from_numpy(data["jaw_pose"]).float().to(dev)

    # Build SMPL-X model
    smplx_model = smplx.create(
        smplx_model_path, model_type="smplx",
        gender="neutral", num_betas=10,
        use_pca=False, num_expression_coeffs=100,
        flat_hand_mean=False,
    ).to(dev)

    # Set up PyTorch3D renderer
    cameras = FoVPerspectiveCameras(device=dev)
    raster_settings = RasterizationSettings(
        image_size=frame_size, blur_radius=0.0, faces_per_pixel=1
    )
    lights = PointLights(device=dev, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=dev, cameras=cameras, lights=lights),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_size, frame_size))

    chunk_size = 8
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        with torch.no_grad():
            output = smplx_model(
                betas=betas[start:end],
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                transl=transl[start:end],
                left_hand_pose=lhand[start:end],
                right_hand_pose=rhand[start:end],
                expression=expression[start:end],
                jaw_pose=jaw_pose[start:end],
                return_verts=True,
            )
        verts = output.vertices   # (chunk, 10475, 3)
        faces = torch.from_numpy(smplx_model.faces.astype(np.int64)).to(dev)
        faces = faces.unsqueeze(0).expand(end - start, -1, -1)

        # Neutral grey texture
        verts_rgb = torch.ones_like(verts) * 0.7
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = Meshes(verts=verts, faces=faces, textures=textures)

        images = renderer(meshes)   # (chunk, H, W, 4) RGBA float [0,1]
        images_np = (images[..., :3].cpu().numpy() * 255).astype(np.uint8)

        for i in range(end - start):
            frame_bgr = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    writer.release()


# ---------------------------------------------------------------------------
# Blender renderer
# ---------------------------------------------------------------------------

def render_blender_video(
    smplx_params_path: Path,
    output_path: Path,
    fps: float = 25.0,
    frame_size: int = 512,
    smplx_model_path: Optional[str] = None,
) -> None:
    """Render SMPL-X sequence using Blender Cycles via subprocess.

    Requires Blender >= 3.6 installed and accessible as `blender` in PATH.
    Drives the SMPL-X mesh via shape keys for per-frame vertex animation.

    Lighting: three-point rig (key/fill/rim) + HDRI environment.
    Material: Principled BSDF with subsurface scattering for realistic skin.
    Renderer: Cycles with 128 samples + OpenImageDenoise.
    """
    import subprocess
    import tempfile
    import json

    # Write params to a temp JSON for the Blender script to read
    data = np.load(smplx_params_path, allow_pickle=False)
    params_dict = {k: data[k].tolist() for k in data.files}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(params_dict, f)
        params_json = f.name

    blender_script = Path(__file__).parent / "blender_render_smplx.py"
    if not blender_script.exists():
        raise FileNotFoundError(
            f"Blender render script not found: {blender_script}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    smplx_path = smplx_model_path or "data/smplx_models"
    cmd = [
        "blender", "--background", "--python", str(blender_script),
        "--", str(params_json), str(output_path), str(fps), str(frame_size),
        str(smplx_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Blender render failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_from_pose_npz(
    pose_npz: Path,
    output_path: Path,
    backend: str = "overlay",
    fps: float = 25.0,
    frame_size: int = 512,
    smplx_model_path: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """Render directly from a (T, 150) pose NPZ (no SMPL-X fitting required)."""
    data = np.load(pose_npz, allow_pickle=False)
    # Support both raw keypoints_2d NPZ and .skels-derived NPZ
    if "pose_keypoints_2d" in data:
        pose = data["pose_keypoints_2d"]   # (T, 54) body only
        # Pad to 150 with zeros for hand joints
        T = pose.shape[0]
        full_pose = np.zeros((T, 150), dtype=np.float32)
        full_pose[:, :pose.shape[1]] = pose
    elif "pose" in data:
        full_pose = data["pose"]
    else:
        raise ValueError(f"No pose array found in {pose_npz}")

    if backend == "overlay":
        render_overlay_video(full_pose, output_path, fps=fps, frame_size=frame_size)
    else:
        raise ValueError(f"Backend {backend!r} requires SMPL-X params, not raw pose NPZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SMPL-X avatar videos from MoSL motion")
    parser.add_argument("--clip", default=None, help="Clip stem name (e.g. الأذان)")
    parser.add_argument("--all", action="store_true", help="Process all clips")
    parser.add_argument("--category", default=None, help="Limit to one category")
    parser.add_argument("--backend", default="overlay",
                        choices=["overlay", "pytorch3d", "blender"])
    parser.add_argument("--pose-npz", default=None,
                        help="Render directly from a (T,150) pose NPZ")
    parser.add_argument("--smplx-model-path", default=None,
                        help="Path to SMPL-X model directory (required for pytorch3d/blender)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="outputs/avatar_videos")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)

    if args.pose_npz:
        output_path = out_dir / (Path(args.pose_npz).stem + ".mp4")
        render_from_pose_npz(
            Path(args.pose_npz), output_path,
            backend=args.backend, fps=args.fps, frame_size=args.frame_size,
            smplx_model_path=args.smplx_model_path, device=args.device,
        )
        print(f"rendered: {output_path}")
        return

    smplx_dir = repo_root / "data" / "processed" / "smplx_params"

    if args.clip:
        # Find the clip across all categories
        matches = list(smplx_dir.rglob(f"{args.clip}.npz"))
        if not matches:
            # Fall back to overlay from keypoints_2d
            kp_matches = list((repo_root / "data" / "processed" / "keypoints_2d").rglob(
                f"{args.clip}.npz"
            ))
            if kp_matches:
                output_path = out_dir / f"{args.clip}.mp4"
                render_from_pose_npz(kp_matches[0], output_path,
                                     backend="overlay", fps=args.fps,
                                     frame_size=args.frame_size)
                print(f"rendered (overlay from keypoints): {output_path}")
            else:
                print(f"clip not found: {args.clip}", file=sys.stderr)
                sys.exit(1)
            return

        npz_path = matches[0]
        cat = npz_path.parent.name
        output_path = out_dir / cat / f"{args.clip}.mp4"

        if args.backend == "overlay":
            # Render from raw keypoints for overlay
            kp_path = repo_root / "data" / "processed" / "keypoints_2d" / cat / f"{args.clip}.npz"
            if kp_path.exists():
                render_from_pose_npz(kp_path, output_path, backend="overlay",
                                     fps=args.fps, frame_size=args.frame_size)
            else:
                print(f"keypoints NPZ not found: {kp_path}", file=sys.stderr)
                sys.exit(1)
        elif args.backend == "pytorch3d":
            render_pytorch3d_video(npz_path, output_path,
                                   smplx_model_path=args.smplx_model_path,
                                   fps=args.fps, frame_size=args.frame_size,
                                   device=args.device)
        elif args.backend == "blender":
            render_blender_video(npz_path, output_path, fps=args.fps,
                                 frame_size=args.frame_size)

        print(f"rendered: {output_path}")

    elif args.all:
        kp_dir = repo_root / "data" / "processed" / "keypoints_2d"
        categories = [d.name for d in kp_dir.iterdir() if d.is_dir()]
        if args.category:
            categories = [c for c in categories if c == args.category]

        total = 0
        for cat in sorted(categories):
            for npz_path in sorted((kp_dir / cat).glob("*.npz")):
                output_path = out_dir / cat / (npz_path.stem + ".mp4")
                if output_path.exists():
                    continue
                try:
                    render_from_pose_npz(npz_path, output_path, backend="overlay",
                                         fps=args.fps, frame_size=args.frame_size)
                    total += 1
                    if total % 50 == 0:
                        print(f"  rendered {total} clips...")
                except Exception as e:
                    print(f"  [FAIL] {npz_path.name}: {e}", file=sys.stderr)

        print(f"done: {total} clips rendered to {out_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
