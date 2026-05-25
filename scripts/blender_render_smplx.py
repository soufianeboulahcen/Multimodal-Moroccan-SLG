"""Blender Cycles photorealistic renderer for SMPL-X avatar sequences.

This script is invoked by render_smplx_video.py via subprocess:
    blender --background --python scripts/blender_render_smplx.py \
        -- params.json output.mp4 fps frame_size

It must be run inside Blender's Python environment (bpy is available).

Pipeline:
  1. Load SMPL-X body/hand/face parameters from JSON
  2. Build a SMPL-X mesh using the smplx Python package (inside Blender's Python)
  3. Import mesh into Blender scene
  4. Set up Cycles rendering:
     - HDRI environment lighting (studio setup)
     - Three-point lighting rig (key, fill, rim)
     - Realistic skin material (Principled BSDF)
     - Subsurface scattering for skin
     - Shadow catcher ground plane
  5. Animate the mesh by updating vertex positions per frame
  6. Render to image sequence, then encode to MP4 via ffmpeg

Requirements:
  - Blender >= 3.6 with Python 3.10+
  - smplx installed in Blender's Python: blender --background --python -c "import pip; pip.main(['install', 'smplx'])"
  - SMPL-X model files at data/smplx_models/

Usage (called by render_smplx_video.py, not directly):
    blender --background --python scripts/blender_render_smplx.py \
        -- /tmp/params.json outputs/avatar.mp4 25.0 512
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def setup_scene(frame_size: int = 512, fps: float = 25.0) -> None:
    """Configure Blender scene for Cycles rendering."""
    import bpy

    # Clear default scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if _has_gpu() else "CPU"
    scene.cycles.samples = 128          # balance quality vs speed
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"

    # Output settings
    scene.render.resolution_x = frame_size
    scene.render.resolution_y = frame_size
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    # Film
    scene.render.film_transparent = False
    scene.world.use_nodes = True


def _has_gpu() -> bool:
    """Check if a CUDA/OptiX GPU is available for Cycles."""
    try:
        import bpy
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.refresh_devices()
        for device in prefs.devices:
            if device.type in ("CUDA", "OPTIX", "HIP") and device.use:
                return True
    except Exception:
        pass
    return False


def setup_lighting() -> None:
    """Three-point lighting rig + HDRI environment."""
    import bpy
    import mathutils

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    # Background node with warm studio colour
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.05, 0.05, 0.08, 1.0)
    bg.inputs["Strength"].default_value = 0.5

    output = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    def add_area_light(name, location, rotation_euler, energy, colour=(1, 1, 1)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.active_object
        light.name = name
        light.rotation_euler = rotation_euler
        light.data.energy = energy
        light.data.color = colour
        light.data.size = 2.0
        light.data.use_shadow = True
        return light

    # Key light (warm, front-left)
    add_area_light(
        "KeyLight",
        location=(2.0, -2.0, 3.0),
        rotation_euler=(math.radians(60), 0, math.radians(45)),
        energy=800,
        colour=(1.0, 0.95, 0.85),
    )

    # Fill light (cool, front-right, softer)
    add_area_light(
        "FillLight",
        location=(-2.0, -2.0, 2.0),
        rotation_euler=(math.radians(50), 0, math.radians(-45)),
        energy=200,
        colour=(0.85, 0.90, 1.0),
    )

    # Rim light (back, creates edge separation)
    add_area_light(
        "RimLight",
        location=(0.0, 3.0, 2.5),
        rotation_euler=(math.radians(-30), 0, math.radians(180)),
        energy=400,
        colour=(1.0, 0.98, 0.95),
    )


def setup_camera(frame_size: int = 512) -> None:
    """Position camera for a full-body portrait view."""
    import bpy
    import mathutils

    bpy.ops.object.camera_add(location=(0, -4.0, 1.0))
    cam = bpy.context.active_object
    cam.name = "AvatarCamera"
    cam.rotation_euler = (math.radians(90), 0, 0)
    cam.data.lens = 50   # 50mm portrait lens
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0

    bpy.context.scene.camera = cam


def setup_ground_plane() -> None:
    """Add a shadow-catcher ground plane."""
    import bpy

    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -1.0))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"
    plane.cycles.is_shadow_catcher = True

    # Simple grey material
    mat = bpy.data.materials.new("GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.15, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(mat)


def create_skin_material() -> "bpy.types.Material":
    """Create a realistic skin material using Principled BSDF with SSS."""
    import bpy

    mat = bpy.data.materials.new("SkinMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]

    # Skin colour (medium tone)
    bsdf.inputs["Base Color"].default_value = (0.80, 0.60, 0.45, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular"].default_value = 0.3

    # Subsurface scattering for realistic skin
    bsdf.inputs["Subsurface"].default_value = 0.15
    bsdf.inputs["Subsurface Color"].default_value = (0.9, 0.5, 0.3, 1.0)
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.1)

    return mat


def build_smplx_mesh_sequence(params: dict, smplx_model_path: str) -> list[dict]:
    """Build SMPL-X vertex sequences for all frames.

    Returns list of dicts: [{"verts": (N, 3) np.ndarray, "faces": (F, 3) np.ndarray}]
    """
    import numpy as np
    import torch

    try:
        import smplx
    except ImportError:
        raise ImportError(
            "smplx must be installed in Blender's Python environment.\n"
            "Run: blender --background --python -c \"import pip; pip.main(['install', 'smplx'])\""
        )

    device = torch.device("cpu")   # Blender subprocess uses CPU

    T = len(params["body_pose"])
    body_pose = torch.tensor(params["body_pose"], dtype=torch.float32)
    global_orient = torch.tensor(params["global_orient"], dtype=torch.float32)
    betas = torch.tensor(params["betas"], dtype=torch.float32).unsqueeze(0).expand(T, -1)
    transl = torch.tensor(params["transl"], dtype=torch.float32)
    lhand = torch.tensor(params["left_hand_pose"], dtype=torch.float32)
    rhand = torch.tensor(params["right_hand_pose"], dtype=torch.float32)
    expression = torch.tensor(params["expression"], dtype=torch.float32)
    jaw_pose = torch.tensor(params["jaw_pose"], dtype=torch.float32)

    model = smplx.create(
        smplx_model_path, model_type="smplx",
        gender="neutral", num_betas=10,
        use_pca=False, num_expression_coeffs=100,
        flat_hand_mean=False,
    )

    frames = []
    chunk = 8
    faces = model.faces.astype(np.int32)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        with torch.no_grad():
            output = model(
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
        verts = output.vertices.numpy()   # (chunk, 10475, 3)
        for i in range(end - start):
            frames.append({"verts": verts[i], "faces": faces})

    return frames


def import_mesh_to_blender(verts: "np.ndarray", faces: "np.ndarray") -> "bpy.types.Object":
    """Import a mesh into Blender and return the object."""
    import bpy
    import bmesh

    mesh = bpy.data.meshes.new("SMPLXMesh")
    obj = bpy.data.objects.new("SMPLXAvatar", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    for v in verts:
        bm.verts.new(v.tolist())
    bm.verts.ensure_lookup_table()
    for f in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except Exception:
            pass
    bm.to_mesh(mesh)
    bm.free()

    mesh.calc_normals()
    return obj


def animate_mesh(obj: "bpy.types.Object", frames: list[dict]) -> None:
    """Animate the mesh by updating vertex positions per frame using shape keys."""
    import bpy
    import numpy as np

    mesh = obj.data

    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)

    for frame_idx, frame_data in enumerate(frames):
        verts = frame_data["verts"]   # (N, 3)
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        for i, v in enumerate(verts):
            sk.data[i].co = v.tolist()

        # Keyframe: this shape key is 1.0 at this frame, 0.0 elsewhere
        sk.value = 1.0
        sk.keyframe_insert("value", frame=frame_idx + 1)
        sk.value = 0.0
        if frame_idx > 0:
            sk.keyframe_insert("value", frame=frame_idx)
        if frame_idx < len(frames) - 1:
            sk.keyframe_insert("value", frame=frame_idx + 2)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frames)


def render_frames(output_dir: Path, n_frames: int) -> None:
    """Render all frames to PNG files."""
    import bpy

    bpy.context.scene.render.filepath = str(output_dir / "frame_")
    bpy.context.scene.render.image_settings.file_format = "PNG"

    for frame in range(1, n_frames + 1):
        bpy.context.scene.frame_set(frame)
        bpy.ops.render.render(write_still=True)


def encode_to_mp4(frames_dir: Path, output_path: Path, fps: float) -> None:
    """Encode PNG frame sequence to MP4 using ffmpeg."""
    pattern = str(frames_dir / "frame_%04d.png")
    # Blender names frames as frame_0001.png, frame_0002.png, etc.
    # The filepath template "frame_" + frame number produces "frame_0001.png"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed:\n{result.stderr}")


def main() -> None:
    """Entry point when called from Blender subprocess."""
    import bpy

    # Parse arguments passed after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python blender_render_smplx.py "
              "-- params.json output.mp4 fps frame_size")
        sys.exit(1)

    params_json = argv[0]
    output_mp4 = Path(argv[1])
    fps = float(argv[2]) if len(argv) > 2 else 25.0
    frame_size = int(argv[3]) if len(argv) > 3 else 512
    smplx_model_path = argv[4] if len(argv) > 4 else "data/smplx_models"

    print(f"Blender Cycles renderer")
    print(f"  params:     {params_json}")
    print(f"  output:     {output_mp4}")
    print(f"  fps:        {fps}")
    print(f"  frame_size: {frame_size}")
    print(f"  smplx:      {smplx_model_path}")

    # Load parameters
    with open(params_json, encoding="utf-8") as f:
        params = json.load(f)

    T = len(params["body_pose"])
    print(f"  frames:     {T}")

    # Build SMPL-X mesh sequence
    print("Building SMPL-X mesh sequence...")
    frames = build_smplx_mesh_sequence(params, smplx_model_path)

    # Set up Blender scene
    print("Setting up Blender scene...")
    setup_scene(frame_size=frame_size, fps=fps)
    setup_lighting()
    setup_camera(frame_size=frame_size)
    setup_ground_plane()

    # Import first frame mesh
    obj = import_mesh_to_blender(frames[0]["verts"], frames[0]["faces"])

    # Apply skin material
    skin_mat = create_skin_material()
    obj.data.materials.append(skin_mat)

    # Animate
    print("Animating mesh...")
    animate_mesh(obj, frames)

    # Render
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_dir = Path(tmp_dir)
        print(f"Rendering {T} frames to {frames_dir}...")
        render_frames(frames_dir, T)

        print(f"Encoding to {output_mp4}...")
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        encode_to_mp4(frames_dir, output_mp4, fps)

    print(f"Done: {output_mp4}")


if __name__ == "__main__":
    main()
