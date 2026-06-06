__all__ = ["PoseExtractor", "AvatarPipeline"]


def __getattr__(name: str):
    if name == "PoseExtractor":
        from avatar_video_generator.pipelines.pose_extractor import PoseExtractor

        return PoseExtractor
    if name == "AvatarPipeline":
        from avatar_video_generator.pipelines.avatar_pipeline import AvatarPipeline

        return AvatarPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
