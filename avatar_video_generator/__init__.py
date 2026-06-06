"""avatar_video_generator — photorealistic avatar rendering for MoSL.

Converts skeleton/OpenPose motion sequences produced by the SignLLM pipeline
into photorealistic human avatar videos while preserving the identity of the
original signer from the MoSL dataset.

Architecture overview
---------------------
  SignLLM / OpenPose outputs  (unchanged)
        │
        ▼
  pose_extractor              — loads pose PNGs or skeleton MP4, normalises
        │
        ▼
  identity_encoder            — extracts face embedding from reference frames
        │  (InsightFace ArcFace or IP-Adapter FaceID)
        ▼
  diffusion_renderer          — ControlNet-OpenPose + AnimateDiff + SDXL
        │  conditioned on pose sequence + identity embedding
        ▼
  temporal_smoother           — latent-space Gaussian + pixel-space blend
        │
        ▼
  rife_interpolator           — 2× / 4× frame interpolation (optional)
        │
        ▼
  video_exporter              — H.264 MP4 via imageio-ffmpeg

Public API
----------
    from avatar_video_generator import AvatarPipeline, AvatarConfig

    cfg = AvatarConfig.from_yaml("avatar_video_generator/configs/default.yaml")
    pipeline = AvatarPipeline(cfg)
    pipeline.run(
        pose_source="outputs/pose_control/أَنْتِ_keypoints",
        reference_video=".devcontainer/Dataset/mosl_videos_dataset_Pronouns/أَنْتِ.mp4",
        output_path="outputs/avatar/أَنْتِ_photorealistic.mp4",
    )
"""
__all__ = ["AvatarPipeline", "AvatarConfig"]
__version__ = "1.0.0"


def __getattr__(name: str):
    """Load heavy pipeline modules only when the public API is requested."""
    if name == "AvatarConfig":
        from avatar_video_generator.configs.config import AvatarConfig

        return AvatarConfig
    if name == "AvatarPipeline":
        from avatar_video_generator.pipelines.avatar_pipeline import AvatarPipeline

        return AvatarPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
