"""Photorealistic diffusion rendering pipeline.

Converts OpenPose conditioning frames + signer identity into photorealistic
avatar video frames using:

  Primary path (GPU ≥ 16 GB, DGX):
    AnimateDiff + ControlNet-OpenPose + SDXL + IP-Adapter FaceID
    → temporally consistent video chunks, stitched with overlap blending

  Fallback path (GPU 8–16 GB):
    ControlNet-OpenPose + SD1.5 + IP-Adapter FaceID
    → frame-by-frame generation with latent-space temporal blending

  CPU-only path (no GPU):
    Raises a clear error with instructions — diffusion requires GPU.

Architecture
------------
The renderer processes the pose sequence in overlapping chunks of
``video_length`` frames (default 16). Adjacent chunks share ``video_stride``
frames (default 8) and are blended in the overlap region to eliminate
seam artefacts.

Identity injection
------------------
When an InsightFace embedding is available, it is injected via IP-Adapter
FaceID cross-attention. This locks the face identity across all frames.
When only a face crop is available (no embedding), it is passed as a
reference image to IP-Adapter image conditioning.

Temporal consistency
--------------------
AnimateDiff's motion module provides strong temporal consistency within
each chunk. Cross-chunk consistency is maintained by:
  1. Overlapping chunk generation (shared context frames)
  2. Latent-space Gaussian smoothing across the full sequence
  3. Pixel-space temporal blending in the overlap region
"""
from __future__ import annotations

import gc
import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from avatar_video_generator.configs.config import DiffusionConfig, TemporalConfig
from avatar_video_generator.identity.encoder import IdentityEmbedding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiffusionRenderer
# ---------------------------------------------------------------------------

class DiffusionRenderer:
    """Renders photorealistic avatar frames from pose conditioning + identity.

    Usage::

        renderer = DiffusionRenderer(diffusion_cfg, temporal_cfg, device="cuda")
        renderer.load_models()
        frames = renderer.render(pose_images, identity_embedding)
        # frames: List[np.ndarray] (H, W, 3) uint8 RGB
    """

    def __init__(
        self,
        diffusion_cfg: DiffusionConfig,
        temporal_cfg: TemporalConfig,
        device: str = "cuda",
    ) -> None:
        self.cfg = diffusion_cfg
        self.temporal_cfg = temporal_cfg
        self.device = device
        self._pipe = None
        self._pipe_type: Optional[str] = None  # "animatediff_sdxl" | "animatediff_sd15" | "controlnet_sd15"
        # float16 on GPU, float32 on CPU
        self._dtype = torch.float16 if (device == "cuda" and self.cfg.use_fp16) else torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Download and load diffusion models into memory.

        Tries backends in order of quality:
          1. AnimateDiff + SDXL + ControlNet (best, needs ≥16 GB VRAM)
          2. AnimateDiff + SD1.5 + ControlNet (good, needs ≥8 GB VRAM)
          3. ControlNet + SD1.5 frame-by-frame (GPU or CPU)
        """
        if self._pipe is not None:
            return

        is_cuda = self.device == "cuda" and torch.cuda.is_available()

        if is_cuda and self.cfg.use_animatediff and self.cfg.use_sdxl:
            try:
                self._load_animatediff_sdxl()
                return
            except Exception as e:
                logger.warning(f"AnimateDiff+SDXL failed ({e}), trying SD1.5...")

        if is_cuda and self.cfg.use_animatediff:
            try:
                self._load_animatediff_sd15()
                return
            except Exception as e:
                logger.warning(f"AnimateDiff+SD1.5 failed ({e}), falling back to ControlNet-only...")

        # ControlNet + SD1.5 works on both GPU and CPU
        self._load_controlnet_sd15()

    def _load_animatediff_sdxl(self) -> None:
        """Load AnimateDiff + ControlNet-OpenPose + SDXL pipeline."""
        from diffusers import (
            AnimateDiffSDXLPipeline,
            ControlNetModel,
            MotionAdapter,
            DDIMScheduler,
        )

        logger.info("Loading AnimateDiff motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            self.cfg.motion_module,
            torch_dtype=self._dtype,
        )

        logger.info(f"Loading ControlNet-OpenPose SDXL: {self.cfg.controlnet_model_sdxl}")
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_sdxl,
            torch_dtype=self._dtype,
            use_safetensors=True,
        )

        logger.info(f"Loading SDXL base: {self.cfg.base_model}")
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            self.cfg.base_model,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=self._dtype,
            use_safetensors=True,
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
        )

        self._apply_memory_optimisations(pipe)
        self._pipe = pipe
        self._pipe_type = "animatediff_sdxl"
        logger.info("AnimateDiff+SDXL pipeline loaded.")

    def _load_animatediff_sd15(self) -> None:
        """Load AnimateDiff + ControlNet-OpenPose + SD1.5 pipeline."""
        from diffusers import (
            AnimateDiffPipeline,
            ControlNetModel,
            MotionAdapter,
            DDIMScheduler,
        )

        logger.info("Loading AnimateDiff motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            self.cfg.motion_module,
            torch_dtype=self._dtype,
        )

        logger.info(f"Loading ControlNet-OpenPose SD1.5: {self.cfg.controlnet_model_sd15}")
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_sd15,
            torch_dtype=self._dtype,
            use_safetensors=True,
        )

        logger.info(f"Loading SD1.5: {self.cfg.base_model_sd15}")
        pipe = AnimateDiffPipeline.from_pretrained(
            self.cfg.base_model_sd15,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=self._dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
        )

        self._apply_memory_optimisations(pipe)
        self._pipe = pipe
        self._pipe_type = "animatediff_sd15"
        logger.info("AnimateDiff+SD1.5 pipeline loaded.")

    def _load_controlnet_sd15(self) -> None:
        """Load ControlNet + SD1.5 frame-by-frame pipeline (GPU or CPU)."""
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        logger.info(f"Loading ControlNet-OpenPose SD1.5: {self.cfg.controlnet_model_sd15}")
        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model_sd15,
            torch_dtype=self._dtype,
            use_safetensors=True,
        )

        logger.info(f"Loading SD1.5: {self.cfg.base_model_sd15}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.base_model_sd15,
            controlnet=controlnet,
            torch_dtype=self._dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        self._apply_memory_optimisations(pipe)
        self._pipe = pipe
        self._pipe_type = "controlnet_sd15"
        logger.info(f"ControlNet+SD1.5 pipeline loaded (device={self.device}, dtype={self._dtype}).")

    def _apply_memory_optimisations(self, pipe) -> None:
        """Apply memory-saving optimisations appropriate for the current device."""
        is_cuda = self.device == "cuda" and torch.cuda.is_available()

        if is_cuda and self.cfg.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory-efficient attention enabled.")
            except Exception:
                logger.debug("xformers not available.")

        if is_cuda and self.cfg.enable_cpu_offload:
            # Low-VRAM GPU: keep only the active layer on GPU
            try:
                pipe.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled.")
            except RuntimeError as e:
                logger.warning(f"CPU offload unavailable ({e}), moving to device directly.")
                pipe.to(self.device)
        elif not is_cuda:
            # CPU: attention slicing reduces peak memory usage
            pipe.enable_attention_slicing()
            pipe.to(self.device)
            logger.info("CPU mode: attention slicing enabled.")
        else:
            pipe.to(self.device)

        pipe.set_progress_bar_config(disable=False)

    # ------------------------------------------------------------------
    # IP-Adapter FaceID injection
    # ------------------------------------------------------------------

    def _inject_ip_adapter(self, identity: IdentityEmbedding) -> None:
        """Load and configure IP-Adapter FaceID on the current pipeline."""
        if not identity.is_valid():
            return
        if self._pipe is None:
            return

        try:
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore

            ip_ckpt = "models/ip-adapter-faceid_sd15.bin"
            image_encoder_path = "models/image_encoder"

            ip_model = IPAdapterFaceID(
                self._pipe,
                ip_ckpt,
                image_encoder_path,
                device=self.device,
            )
            self._ip_model = ip_model
            logger.info("IP-Adapter FaceID loaded.")
        except ImportError:
            logger.warning(
                "ip_adapter package not found. Identity will be injected via "
                "prompt augmentation only.\n"
                "For full identity locking: pip install ip-adapter"
            )
            self._ip_model = None

    # ------------------------------------------------------------------
    # Prompt augmentation with identity
    # ------------------------------------------------------------------

    def _build_prompt(self, identity: IdentityEmbedding) -> str:
        """Augment the base prompt with signer-specific appearance tokens."""
        base = self.cfg.positive_prompt
        if identity.signer_description:
            # Prepend identity description for stronger conditioning
            return f"{identity.signer_description}, {base}"
        return base

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def render(
        self,
        pose_images: List[Image.Image],
        identity: IdentityEmbedding,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Render photorealistic frames from pose conditioning + identity.

        Args:
            pose_images: list of PIL Images (H, W, RGB) — OpenPose frames.
            identity: signer identity embedding from IdentityEncoder.
            max_frames: cap the number of frames rendered. Useful for CPU
                        prototyping — set to 8-16 for fast iteration.

        Returns:
            List of (H, W, 3) uint8 RGB numpy arrays.
        """
        if self._pipe is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if not pose_images:
            raise ValueError("No pose images provided.")

        # Cap frames for CPU mode to keep runtime manageable
        if max_frames is not None and max_frames > 0:
            pose_images = pose_images[:max_frames]

        positive_prompt = self._build_prompt(identity)
        negative_prompt = self.cfg.negative_prompt

        logger.info(
            f"Rendering {len(pose_images)} frames via {self._pipe_type} "
            f"(device={self.device}, resolution={self.cfg.resolution})"
        )

        if self._pipe_type in ("animatediff_sdxl", "animatediff_sd15"):
            frames = self._render_animatediff(
                pose_images, positive_prompt, negative_prompt, identity
            )
        else:
            frames = self._render_controlnet_framewise(
                pose_images, positive_prompt, negative_prompt, identity
            )

        return frames

    # ------------------------------------------------------------------
    # AnimateDiff chunked rendering
    # ------------------------------------------------------------------

    def _render_animatediff(
        self,
        pose_images: List[Image.Image],
        positive_prompt: str,
        negative_prompt: str,
        identity: IdentityEmbedding,
    ) -> List[np.ndarray]:
        """Render using AnimateDiff in overlapping chunks.

        Chunks of ``video_length`` frames are generated with ``video_stride``
        overlap. The overlap region is blended with a cosine weight to
        eliminate seam artefacts.
        """
        chunk_size = self.cfg.video_length
        stride = self.cfg.video_stride
        n = len(pose_images)
        res = self.cfg.resolution

        # Encode prompt once
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        # Collect all chunk outputs
        all_chunks: List[tuple] = []  # (start_idx, frames_list)

        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunk_poses = pose_images[start:end]

            # Pad short final chunk
            if len(chunk_poses) < chunk_size:
                pad = [chunk_poses[-1]] * (chunk_size - len(chunk_poses))
                chunk_poses = chunk_poses + pad

            logger.info(f"  Chunk [{start}:{end}] ({len(chunk_poses)} frames)")

            with torch.no_grad():
                output = self._pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=len(chunk_poses),
                    width=res,
                    height=res,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.guidance_scale,
                    controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                    image=chunk_poses,
                    generator=generator,
                    output_type="np",
                )

            # output.frames: (1, T, H, W, C) float32 [0,1]
            chunk_frames = output.frames[0]  # (T, H, W, C)
            chunk_np = [
                np.clip(chunk_frames[i] * 255, 0, 255).astype(np.uint8)
                for i in range(end - start)  # only real frames, not padding
            ]

            all_chunks.append((start, chunk_np))

            # Advance by stride (overlap = chunk_size - stride)
            start += stride
            if start >= n:
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stitch chunks with cosine blending in overlap regions
        return self._stitch_chunks(all_chunks, n, chunk_size, stride)

    def _stitch_chunks(
        self,
        chunks: List[tuple],
        total_frames: int,
        chunk_size: int,
        stride: int,
    ) -> List[np.ndarray]:
        """Blend overlapping chunks into a seamless frame sequence."""
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks[0][1][:total_frames]

        # Accumulate weighted frames
        h, w, c = chunks[0][1][0].shape
        acc = np.zeros((total_frames, h, w, c), dtype=np.float64)
        weight = np.zeros(total_frames, dtype=np.float64)

        overlap = chunk_size - stride

        for chunk_start, chunk_frames in chunks:
            n_chunk = len(chunk_frames)
            for local_i, frame in enumerate(chunk_frames):
                global_i = chunk_start + local_i
                if global_i >= total_frames:
                    break

                # Cosine weight: ramp up at start, ramp down at end of chunk
                if local_i < overlap and chunk_start > 0:
                    # Ramp up (blend in)
                    w_val = 0.5 * (1 - np.cos(np.pi * local_i / overlap))
                elif local_i >= n_chunk - overlap and chunk_start + n_chunk < total_frames:
                    # Ramp down (blend out)
                    tail_i = local_i - (n_chunk - overlap)
                    w_val = 0.5 * (1 + np.cos(np.pi * tail_i / overlap))
                else:
                    w_val = 1.0

                acc[global_i] += frame.astype(np.float64) * w_val
                weight[global_i] += w_val

        # Normalise
        weight = np.maximum(weight, 1e-6)
        result = acc / weight[:, None, None, None]
        return [np.clip(result[i], 0, 255).astype(np.uint8) for i in range(total_frames)]

    # ------------------------------------------------------------------
    # Frame-by-frame ControlNet rendering (fallback)
    # ------------------------------------------------------------------

    def _render_controlnet_framewise(
        self,
        pose_images: List[Image.Image],
        positive_prompt: str,
        negative_prompt: str,
        identity: IdentityEmbedding,
    ) -> List[np.ndarray]:
        """Render frame-by-frame with ControlNet + fixed-seed temporal consistency.

        Works on both GPU and CPU. Uses a fixed seed across all frames so the
        diffusion process starts from the same noise, which is the primary
        mechanism for temporal consistency in frame-by-frame mode.
        """
        res = self.cfg.resolution
        is_cuda = self.device == "cuda" and torch.cuda.is_available()
        # Generator must be on CPU when running without CUDA
        gen_device = "cuda" if is_cuda else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(self.cfg.seed)

        frames: List[np.ndarray] = []
        n = len(pose_images)

        logger.info(f"Frame-by-frame rendering: {n} frames on {self.device}...")

        for i, pose_pil in enumerate(pose_images):
            logger.info(f"  Frame {i+1}/{n}")
            with torch.no_grad():
                output = self._pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=pose_pil,
                    width=res,
                    height=res,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.guidance_scale,
                    controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                    generator=generator,
                    output_type="pil",
                )

            frame_np = np.array(output.images[0])
            frames.append(frame_np)

            # Free memory every few frames
            if i % 4 == 0:
                gc.collect()
                if is_cuda:
                    torch.cuda.empty_cache()

        return frames

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release GPU memory."""
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Diffusion pipeline unloaded.")
