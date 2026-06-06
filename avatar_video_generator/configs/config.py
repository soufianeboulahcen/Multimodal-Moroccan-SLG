"""Configuration dataclasses for the avatar video generation pipeline.

All fields have sensible defaults tuned for the MoSL dataset (512×512, 25 fps,
Moroccan signer identity). Override via YAML or CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


# ---------------------------------------------------------------------------
# Identity preservation config
# ---------------------------------------------------------------------------

@dataclass
class IdentityConfig:
    """Controls how the signer's visual identity is extracted and injected.

    Two backends are supported:
      - "insightface"  : ArcFace embedding via InsightFace (recommended, fast)
      - "ip_adapter"   : IP-Adapter FaceID (richer, requires extra model download)

    The reference image/video is extracted from the MoSL dataset video for the
    same sign, ensuring the generated avatar matches the real signer exactly.
    """
    backend: str = "insightface"          # "insightface" | "ip_adapter" | "none"
    ip_adapter_model: str = "h94/IP-Adapter-FaceID"
    ip_adapter_scale: float = 0.8         # identity injection strength [0, 1]
    face_crop_size: int = 224             # face crop resolution for embedding
    reference_frame_idx: int = 0          # which frame to use as identity anchor
    # If True, average embeddings from multiple reference frames for robustness
    multi_frame_average: bool = True
    multi_frame_count: int = 5            # number of frames to average


# ---------------------------------------------------------------------------
# Diffusion rendering config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Controls the ControlNet + AnimateDiff + SDXL rendering pipeline.

    Model selection:
      - base_model: SDXL base (preferred) or SD1.5 fallback
      - controlnet_model: OpenPose ControlNet matching the base model
      - motion_module: AnimateDiff motion module for temporal consistency

    Prompt engineering:
      The positive prompt describes the target signer appearance.
      It is automatically augmented with identity tokens when using IP-Adapter.
    """
    # Base diffusion model
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # SD1.5 fallback (lower VRAM, faster)
    base_model_sd15: str = "runwayml/stable-diffusion-v1-5"

    # ControlNet — OpenPose conditioning
    controlnet_model_sdxl: str = "thibaud/controlnet-openpose-sdxl-1.0"
    controlnet_model_sd15: str = "lllyasviel/control_v11p_sd15_openpose"

    # AnimateDiff motion module (temporal consistency across frames)
    motion_module: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    # Use AnimateDiff for video generation (recommended for sign language)
    use_animatediff: bool = True

    # Generation parameters
    resolution: int = 512                 # output frame resolution
    num_inference_steps: int = 25         # denoising steps (20–50)
    guidance_scale: float = 7.5           # classifier-free guidance
    controlnet_conditioning_scale: float = 1.0  # pose adherence strength
    seed: int = 42                        # reproducibility

    # AnimateDiff-specific
    video_length: int = 16                # frames per AnimateDiff chunk
    video_stride: int = 8                 # overlap between chunks (temporal blend)

    # Prompts — describe the MoSL signer appearance
    positive_prompt: str = (
        "a Moroccan sign language interpreter, upper body portrait, "
        "natural skin tone, dark hair, professional appearance, "
        "studio lighting, sharp focus, photorealistic, 8k DSLR, "
        "cinematic, realistic skin texture, clear expressive hands, "
        "natural eye contact, neutral background"
    )
    negative_prompt: str = (
        "skeleton, stick figure, cartoon, anime, illustration, drawing, "
        "blurry, low quality, deformed hands, extra fingers, missing fingers, "
        "watermark, text, logo, nsfw, ugly, distorted face, "
        "ghosting, flickering, artifacts, CGI, 3D render, plastic skin"
    )

    # Memory optimisation
    enable_xformers: bool = True
    enable_cpu_offload: bool = False      # set True if VRAM < 8 GB
    use_fp16: bool = True                 # float16 (saves ~50% VRAM)
    use_sdxl: bool = True                 # False = fall back to SD1.5


# ---------------------------------------------------------------------------
# Temporal smoothing config
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Controls inter-frame consistency and smoothing."""
    # Gaussian filter on pixel values post-generation
    pixel_smooth_sigma: float = 0.8       # 0 = disabled
    # Latent-space blend with previous frames during generation
    latent_blend_alpha: float = 0.12      # 0 = no blend
    latent_blend_window: int = 3          # number of previous frames to blend
    # Optical-flow-guided warping (requires opencv-contrib)
    use_flow_warp: bool = False


# ---------------------------------------------------------------------------
# RIFE interpolation config
# ---------------------------------------------------------------------------

@dataclass
class InterpolationConfig:
    """Controls RIFE frame interpolation for smooth slow-motion output.

    RIFE (Real-Time Intermediate Flow Estimation) doubles or quadruples the
    frame rate by synthesising intermediate frames from optical flow.
    """
    enabled: bool = True
    multiplier: int = 2                   # 2× or 4× frame rate
    # RIFE model variant
    model_version: str = "4.6"            # "4.6" | "4.14" | "4.18"
    # Output FPS after interpolation
    output_fps: float = 50.0              # 25 × 2 = 50 fps
    # Fallback: simple linear blend if RIFE not available
    fallback_blend: bool = True


# ---------------------------------------------------------------------------
# Video export config
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Controls the final MP4 encoding."""
    fps: float = 25.0
    codec: str = "libx264"
    crf: int = 18                         # quality (0=lossless, 23=default, 51=worst)
    pixel_format: str = "yuv420p"
    # Export generated avatar PNG frames next to the final MP4.
    export_frames: bool = True
    frames_path_suffix: str = "_frames"
    # Side-by-side comparison output (pose | avatar)
    export_comparison: bool = True
    comparison_path_suffix: str = "_comparison"


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class AvatarConfig:
    """Master configuration for the photorealistic avatar pipeline.

    Usage:
        cfg = AvatarConfig()                          # defaults
        cfg = AvatarConfig.from_yaml("configs/dgx.yaml")  # from file
        cfg = AvatarConfig(diffusion=DiffusionConfig(resolution=768))
    """
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Paths
    output_dir: str = "outputs/avatar_photorealistic"
    reference_frames_dir: str = "outputs/avatar_photorealistic/reference_frames"
    cache_dir: str = "outputs/avatar_photorealistic/.cache"

    # Device
    device: str = "cuda"                  # "cuda" | "cpu"

    # Logging
    verbose: bool = True

    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AvatarConfig":
        """Load config from a YAML file, merging with defaults."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        if "identity" in raw:
            for k, v in raw["identity"].items():
                setattr(cfg.identity, k, v)
        if "diffusion" in raw:
            for k, v in raw["diffusion"].items():
                setattr(cfg.diffusion, k, v)
        if "temporal" in raw:
            for k, v in raw["temporal"].items():
                setattr(cfg.temporal, k, v)
        if "interpolation" in raw:
            for k, v in raw["interpolation"].items():
                setattr(cfg.interpolation, k, v)
        if "export" in raw:
            for k, v in raw["export"].items():
                setattr(cfg.export, k, v)
        for k in ("output_dir", "reference_frames_dir", "cache_dir", "device", "verbose"):
            if k in raw:
                setattr(cfg, k, raw[k])
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)

    def to_dict(self) -> dict:
        return asdict(self)
