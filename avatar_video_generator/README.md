# avatar_video_generator

Photorealistic avatar video generation for the Multimodal Moroccan Sign Language (MoSL) system.

Converts skeleton/OpenPose motion sequences produced by the SignLLM pipeline into photorealistic human avatar videos, preserving the visual identity of the real signer from the MoSL dataset.

**The SignLLM pipeline is never modified.** This subsystem only reads its outputs.

---

## Architecture

```
SignLLM / OpenPose outputs  (read-only)
        │
        ▼
  PoseExtractor              loads pose_*.png frames or skeleton MP4
        │
        ▼
  IdentityEncoder            extracts ArcFace embedding from reference video
        │  (InsightFace buffalo_l)
        ▼
  DiffusionRenderer          ControlNet-OpenPose + AnimateDiff + SDXL
        │  conditioned on pose sequence + identity embedding
        ▼
  TemporalSmoother           Gaussian pixel smoothing + optional flow warp
        │
        ▼
  RIFEInterpolator           2× / 4× frame interpolation
        │
        ▼
  VideoExporter              H.264 MP4 + side-by-side comparison
```

### Identity preservation

The signer's face identity is locked using **InsightFace ArcFace** (512-d embedding extracted from the MoSL dataset reference video). The embedding is injected into the diffusion pipeline via **IP-Adapter FaceID** cross-attention, ensuring the generated avatar matches the real person's:

- Face shape and structure
- Skin tone
- Hairstyle
- Eye structure
- Body proportions

### Temporal consistency

Sign language motion requires stable identity across 50–150 frames. Consistency is maintained by:

1. **AnimateDiff motion module** — temporal attention within 16-frame chunks
2. **Overlapping chunk generation** — adjacent chunks share 8 frames, blended with cosine weights
3. **Gaussian temporal smoothing** — 1D Gaussian filter along the time axis (σ=0.8)
4. **RIFE interpolation** — 2× frame synthesis from optical flow

---

## Quick start

```bash
# Install dependencies
pip install -r requirements_avatar.txt

# Generate avatar for أَنْتِ (auto-discovers pose source and reference video)
bash run_avatar_photorealistic.sh --sign أَنْتِ

# Batch: all signs with existing pose frames
bash run_avatar_photorealistic.sh --batch

# DGX high-quality (768px, SDXL, 4× RIFE)
bash run_avatar_photorealistic.sh --sign أَنْتِ --dgx
```

Output: `outputs/avatar_photorealistic/أَنْتِ_photorealistic.mp4`

---

## Python API

```python
from avatar_video_generator import AvatarPipeline, AvatarConfig

cfg = AvatarConfig.from_yaml("avatar_video_generator/configs/default.yaml")
pipeline = AvatarPipeline(cfg)
pipeline.load_models()

result = pipeline.run(
    pose_source="outputs/pose_control/أَنْتِ_keypoints",
    reference_video=".devcontainer/Dataset/mosl_videos_dataset_Pronouns/أَنْتِ.mp4",
    output_path="outputs/avatar_photorealistic/أَنْتِ_photorealistic.mp4",
)
print(result)
```

Or use the high-level convenience method:

```python
result = pipeline.run_sign("أَنْتِ")
```

---

## CLI reference

```
python scripts/generate_photorealistic_avatar.py --help

  --sign ARABIC_SIGN          Arabic sign name (auto-discovers sources)
  --pose-dir DIR              Explicit pose PNG directory
  --skeleton-video MP4        Explicit skeleton MP4
  --batch                     Process all signs in outputs/pose_control/
  --reference-video MP4       Explicit reference video for identity
  --output MP4                Output path (single-sign mode)
  --output-dir DIR            Output directory (batch mode)
  --config YAML               Config file path
  --resolution {512,768}      Output resolution
  --steps N                   Denoising steps (20–50)
  --no-sdxl                   Use SD1.5 instead of SDXL
  --no-animatediff            Frame-by-frame mode (lower VRAM)
  --no-rife                   Skip frame interpolation
  --cpu-offload               Enable sequential CPU offload
  --identity-backend          insightface | ip_adapter | none
  --device {cuda,cpu}
```

---

## Configuration

Three pre-built configs are provided:

| Config | VRAM | Resolution | Quality |
|--------|------|-----------|---------|
| `default.yaml` | ≥8 GB | 512px | Good |
| `dgx.yaml` | ≥16 GB | 768px | Best |
| `cpu_fallback.yaml` | CPU | 512px | Testing only |

Override any field via CLI flags or by editing the YAML.

---

## Directory layout

```
avatar_video_generator/
├── __init__.py                 public API: AvatarPipeline, AvatarConfig
├── configs/
│   ├── config.py               dataclass definitions
│   ├── default.yaml            8 GB GPU config
│   ├── dgx.yaml                DGX / 16+ GB config
│   └── cpu_fallback.yaml       CPU testing config
├── pipelines/
│   ├── avatar_pipeline.py      master orchestration
│   └── pose_extractor.py       OpenPose frame loading
├── identity/
│   └── encoder.py              InsightFace / IP-Adapter identity extraction
├── rendering/
│   ├── diffusion_renderer.py   ControlNet + AnimateDiff + SDXL
│   └── temporal_smoother.py    Gaussian + optical-flow smoothing
├── interpolation/
│   └── rife_interpolator.py    RIFE 2×/4× frame interpolation
└── utils/
    ├── video_io.py             MP4 read/write, comparison video
    └── image_utils.py          frame conversion utilities

scripts/
└── generate_photorealistic_avatar.py   main inference script

run_avatar_photorealistic.sh            shell runner
requirements_avatar.txt                 dependencies
```

---

## Model downloads

Models are downloaded automatically from HuggingFace on first run:

| Model | Size | Purpose |
|-------|------|---------|
| `runwayml/stable-diffusion-v1-5` | ~4 GB | Base diffusion (SD1.5 path) |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | Base diffusion (SDXL path) |
| `lllyasviel/control_v11p_sd15_openpose` | ~1.5 GB | ControlNet SD1.5 |
| `thibaud/controlnet-openpose-sdxl-1.0` | ~2.5 GB | ControlNet SDXL |
| `guoyww/animatediff-motion-adapter-v1-5-2` | ~0.5 GB | AnimateDiff motion |
| InsightFace `buffalo_l` | ~0.3 GB | ArcFace identity |

Total first-run download: ~8–16 GB depending on config.

Set `HF_HOME` to control the cache location:
```bash
export HF_HOME=/data/hf_cache
```

---

## DGX deployment

```bash
# On DGX node
export HF_HOME=/data/hf_cache
export CUDA_VISIBLE_DEVICES=0

bash run_avatar_photorealistic.sh --dgx --batch
```

The pipeline is compatible with multi-GPU setups via `CUDA_VISIBLE_DEVICES`.
For distributed batch processing across multiple GPUs, split the sign list
and run one process per GPU.

---

## Constraints

- Does **not** modify `.skels` files, keypoint JSON, or any SignLLM artifact
- Does **not** retrain any model
- Requires the SignLLM pipeline to have already generated pose frames
- Identity locking requires the MoSL dataset reference videos
- Diffusion rendering requires a CUDA-capable GPU (≥8 GB VRAM)
