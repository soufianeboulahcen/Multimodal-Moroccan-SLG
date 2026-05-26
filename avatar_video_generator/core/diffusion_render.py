"""Diffusion rendering: pose conditioning -> photorealistic avatar frames.

Backend selection (tried in order, first that loads wins):
  1. AnimateDiff + ControlNet-OpenPose + SD1.5   (GPU >=8 GB, best for video)
  2. ControlNet-OpenPose + SD1.5 frame-by-frame  (GPU >=6 GB, fallback)
  3. CPU stub — raises clear error with instructions

Identity is injected via:
  - IP-Adapter FaceID (if insightface + ip_adapter installed)
  - Prompt augmentation only (fallback — always works)
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Default prompts tuned for MoSL signer ────────────────────────────────────
POSITIVE_PROMPT = (
    "a Moroccan sign language interpreter, upper body portrait, "
    "natural medium skin tone, dark hair, professional appearance, "
    "studio lighting, sharp focus, photorealistic, 8k DSLR, "
    "cinematic quality, realistic skin texture, "
    "clear expressive hands with detailed fingers, "
    "natural eye contact, neutral dark background"
)
NEGATIVE_PROMPT = (
    "skeleton, stick figure, cartoon, anime, illustration, drawing, "
    "blurry, low quality, deformed hands, extra fingers, missing fingers, "
    "watermark, text, logo, nsfw, ugly, distorted face, "
    "ghosting, flickering, artifacts, CGI, 3D render, plastic skin, "
    "overexposed, underexposed, noise"
)

SD15_MODEL   = "runwayml/stable-diffusion-v1-5"
CN_SD15_MODEL = "lllyasviel/control_v11p_sd15_openpose"
ANIMATEDIFF_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"


class DiffusionRenderer:
    """Renders photorealistic frames from OpenPose conditioning + identity.

    Usage::
        renderer = DiffusionRenderer(device="cuda")
        renderer.load()
        frames = renderer.render(pose_images, face_image=face_crop)
    """

    def __init__(
        self,
        device: str = "cuda",
        resolution: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        seed: int = 42,
        use_animatediff: bool = True,
        use_fp16: bool = True,
        cpu_offload: bool = False,
        positive_prompt: str = POSITIVE_PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
    ) -> None:
        self.device = device
        self.resolution = resolution
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale
        self.seed = seed
        self.use_animatediff = use_animatediff
        self.use_fp16 = use_fp16
        self.cpu_offload = cpu_offload
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt

        self._pipe = None
        self._backend: Optional[str] = None
        self._ip_model = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load diffusion models. Call once before render()."""
        if self._pipe is not None:
            return

        if not torch.cuda.is_available() and self.device == "cuda":
            logger.warning("CUDA not available — switching to CPU (very slow)")
            self.device = "cpu"
            self.use_fp16 = False

        dtype = torch.float16 if (self.use_fp16 and self.device != "cpu") else torch.float32

        if self.use_animatediff:
            try:
                self._load_animatediff(dtype)
                return
            except Exception as e:
                logger.warning(f"AnimateDiff load failed ({e}), trying ControlNet-only")

        self._load_controlnet_sd15(dtype)

    def _load_animatediff(self, dtype: torch.dtype) -> None:
        from diffusers import AnimateDiffPipeline, ControlNetModel, MotionAdapter, DDIMScheduler

        logger.info("Loading AnimateDiff motion adapter...")
        adapter = MotionAdapter.from_pretrained(ANIMATEDIFF_ADAPTER, torch_dtype=dtype)

        logger.info(f"Loading ControlNet OpenPose SD1.5...")
        controlnet = ControlNetModel.from_pretrained(
            CN_SD15_MODEL, torch_dtype=dtype, use_safetensors=True)

        logger.info(f"Loading SD1.5 base...")
        pipe = AnimateDiffPipeline.from_pretrained(
            SD15_MODEL,
            motion_adapter=adapter,
            controlnet=controlnet,
            torch_dtype=dtype,
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
        self._apply_opts(pipe)
        self._pipe = pipe
        self._backend = "animatediff_sd15"
        logger.info("AnimateDiff+SD1.5 loaded.")

    def _load_controlnet_sd15(self, dtype: torch.dtype) -> None:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        logger.info("Loading ControlNet OpenPose SD1.5 (frame-by-frame)...")
        controlnet = ControlNetModel.from_pretrained(
            CN_SD15_MODEL, torch_dtype=dtype, use_safetensors=True)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD15_MODEL,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self._apply_opts(pipe)
        self._pipe = pipe
        self._backend = "controlnet_sd15"
        logger.info("ControlNet+SD1.5 loaded.")

    def _apply_opts(self, pipe) -> None:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled.")
        except Exception:
            pass
        if self.cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=False)

    # ── IP-Adapter FaceID injection ───────────────────────────────────────────

    def load_ip_adapter(self, face_image: Optional[np.ndarray] = None) -> None:
        """Optionally load IP-Adapter FaceID for identity locking."""
        if face_image is None or self._pipe is None:
            return
        try:
            # Try diffusers native IP-Adapter support first
            self._pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter-plus-face_sd15.bin",
            )
            self._pipe.set_ip_adapter_scale(0.7)
            self._ip_face = Image.fromarray(face_image) if isinstance(face_image, np.ndarray) else face_image
            logger.info("IP-Adapter FaceID loaded via diffusers native.")
        except Exception as e:
            logger.warning(f"IP-Adapter not loaded ({e}). Identity via prompt only.")
            self._ip_face = None

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        pose_images: List[Image.Image],
        face_image: Optional[np.ndarray] = None,
        extra_prompt: str = "",
    ) -> List[np.ndarray]:
        """Render photorealistic frames from pose conditioning.

        Args:
            pose_images: OpenPose conditioning PIL Images (512x512 RGB)
            face_image:  (H,W,3) uint8 RGB face crop for identity locking
            extra_prompt: appended to positive prompt (e.g. appearance description)

        Returns:
            List of (H,W,3) uint8 RGB numpy arrays
        """
        if self._pipe is None:
            raise RuntimeError("Call load() first.")

        pos = self.positive_prompt
        if extra_prompt:
            pos = f"{extra_prompt}, {pos}"

        if self._backend == "animatediff_sd15":
            return self._render_animatediff(pose_images, pos)
        else:
            return self._render_framewise(pose_images, pos)

    def _render_animatediff(
        self, pose_images: List[Image.Image], positive_prompt: str
    ) -> List[np.ndarray]:
        """AnimateDiff chunked rendering with cosine overlap blending."""
        chunk = 16
        stride = 8
        n = len(pose_images)
        res = self.resolution
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        chunks_out: List[tuple] = []  # (start_idx, frames)
        start = 0
        while start < n:
            end = min(start + chunk, n)
            batch = pose_images[start:end]
            # Pad to chunk size
            if len(batch) < chunk:
                batch = batch + [batch[-1]] * (chunk - len(batch))

            logger.info(f"  AnimateDiff chunk [{start}:{end}]")
            with torch.no_grad():
                out = self._pipe(
                    prompt=positive_prompt,
                    negative_prompt=self.negative_prompt,
                    num_frames=len(batch),
                    width=res, height=res,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    controlnet_conditioning_scale=self.controlnet_scale,
                    image=batch,
                    generator=gen,
                    output_type="np",
                )
            # out.frames shape: (1, T, H, W, C) float32 [0,1]
            chunk_frames = out.frames[0]
            real_n = end - start
            chunk_np = [
                np.clip(chunk_frames[i] * 255, 0, 255).astype(np.uint8)
                for i in range(real_n)
            ]
            chunks_out.append((start, chunk_np))

            start += stride
            if start >= n:
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return _stitch_chunks(chunks_out, n, chunk, stride)

    def _render_framewise(
        self, pose_images: List[Image.Image], positive_prompt: str
    ) -> List[np.ndarray]:
        """Frame-by-frame ControlNet rendering with fixed seed."""
        res = self.resolution
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        # Encode prompt once
        with torch.no_grad():
            pos_emb, neg_emb = self._pipe.encode_prompt(
                prompt=positive_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=self.negative_prompt,
            )

        frames = []
        ip_kwargs = {}
        if getattr(self, "_ip_face", None) is not None:
            ip_kwargs["ip_adapter_image"] = self._ip_face

        logger.info(f"  Frame-by-frame: {len(pose_images)} frames")
        for i, pose_pil in enumerate(pose_images):
            with torch.no_grad():
                out = self._pipe(
                    prompt_embeds=pos_emb,
                    negative_prompt_embeds=neg_emb,
                    image=pose_pil,
                    width=res, height=res,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                    controlnet_conditioning_scale=self.controlnet_scale,
                    generator=gen,
                    output_type="pil",
                    **ip_kwargs,
                )
            frames.append(np.array(out.images[0]))
            if i % 8 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return frames

    def unload(self) -> None:
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Chunk stitching ───────────────────────────────────────────────────────────

def _stitch_chunks(
    chunks: List[tuple], total: int, chunk_size: int, stride: int
) -> List[np.ndarray]:
    """Blend overlapping AnimateDiff chunks with cosine weights."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks[0][1][:total]

    h, w, c = chunks[0][1][0].shape
    acc = np.zeros((total, h, w, c), dtype=np.float64)
    wgt = np.zeros(total, dtype=np.float64)
    overlap = chunk_size - stride

    for start, chunk_frames in chunks:
        n_c = len(chunk_frames)
        for li, frame in enumerate(chunk_frames):
            gi = start + li
            if gi >= total:
                break
            # Cosine ramp at boundaries
            if li < overlap and start > 0:
                w_val = 0.5 * (1 - np.cos(np.pi * li / max(overlap, 1)))
            elif li >= n_c - overlap and start + n_c < total:
                tail = li - (n_c - overlap)
                w_val = 0.5 * (1 + np.cos(np.pi * tail / max(overlap, 1)))
            else:
                w_val = 1.0
            acc[gi] += frame.astype(np.float64) * w_val
            wgt[gi] += w_val

    wgt = np.maximum(wgt, 1e-8)
    result = acc / wgt[:, None, None, None]
    return [np.clip(result[i], 0, 255).astype(np.uint8) for i in range(total)]
