# What Prompt2Sign actually provides (and doesn't)

Findings from reading <https://github.com/SignLLM/Prompt2Sign> on 2026-05-08.

## Top-line

The repo is **data-preprocessing tooling, not training code.** No model definition,
no dataloaders, no training loop, no checkpoints.  This confirms the SignLLM
project page: *"This project is being developed in collaboration with commercial
companies and will not be open-sourced."* Only the data side is public.

## Two pipelines — old (OpenPose) vs new (DWPose)

The repo contains two distinct preprocessing pipelines:

### `tools/2D_to_3D/` (2024, OpenPose) — matches the SignLLM paper

- Expected input: directory of CMU OpenPose per-frame JSON files (`*_keypoints.json`)
  with `pose_keypoints_2d`, `hand_left_keypoints_2d`, `hand_right_keypoints_2d`
  flat arrays of `[x, y, confidence, ...]`.  `x==0, y==0` = missing.
- Pipeline: JSON → H5 (`pipeline_demo_01_json2h5.py`) → text
  (`pipeline_demo_02_h5totxt.py`) → skeletal sequences (`pipeline_demo_03_txt2skels.py`).
- Core math: `pose2D.py`, `pose2Dto3D.py`, `pose3D.py`, `skeletalModel.py` —
  pure numpy + math.  **Portable.**  This is the published Step I/II/III implementation.
- Caveat: pipeline orchestration scripts (`pipeline_demo_*.py`) import TensorFlow 1.15
  (`tensorflow.distribute.experimental`).  TF only seems to be referenced incidentally;
  the geometry is in the pure-numpy modules.

### `tools-new-2025/main/` (2025, DWPose) — the authors' current recommendation

- `pipeline00_extract_split_video_to_image.py` — video → frames
- `pipeline01_extract_dwpose_from_video.py` — DWPose keypoint extraction
- `pipeline02_obtain_trg384_data.py` — to "trg384" compressed format

The README is very clear that this is now their official pipeline:

> *"This will be our new processing standard. The previous dataset page has been deprecated."* (2025-07-30)
>
> *"I noticed that DWpose might be a better training method, so unreleased data
> will not be maintained because our time should spent on better data formats."* (2025-03-31)

The HuggingFace datasets they recommend
([`How2Sign-dwpose`](https://huggingface.co/datasets/FangSen9000/How2Sign-dwpose))
are the **DWPose-format compressed data**, which is what the (closed) SignLLM
training code presumably ingests now.

## Implications for our PFE

| Concern | Status |
|---|---|
| Pose extraction | ✓ Done with pytorch-openpose, 33% through, matches the *paper-as-published* 2024 pipeline. |
| Format conversion | NPZ → per-frame JSON exporter needed (~30 lines, trivial). |
| 2D → 3D conversion | Use `tools/2D_to_3D/` numpy modules directly; bypass the TF orchestration scripts. |
| Compressed pose format | Use the H5/text/skels stages of the 2024 pipeline. |
| Model architecture | **Must implement from the paper.** No reference code. |
| Training scripts | **Must write from scratch.** No reference code. |
| Dataloaders | **Must write from scratch.** No reference code. |

## Honest scope check

When we framed the project as *"reproduce SignLLM on MoSL"*, the implicit assumption
was that there was an open SignLLM implementation we'd fine-tune.  There isn't.
What we actually have is:

1. The published architecture description (paper).
2. Public data-prep tooling (this repo).
3. No model weights, no model code, no training scripts.

So a PFE phrased as *"reproduce SignLLM on MoSL"* is really *"reimplement a
SignLLM-inspired text→pose model from the paper, using Prompt2Sign-compatible data
preprocessing, and train it on MoSL."*  That's still a defensible PFE — just bigger
than the original framing implied.

## Three options for next steps

### A. Stay paper-faithful (OpenPose path)
- Continue current OpenPose extraction (already 33% done — sunk cost).
- Add NPZ → OpenPose-JSON exporter.
- Use 2024 `tools/2D_to_3D/` for 2D→3D conversion (numpy-only modules).
- Write model + training from scratch based on the paper.

**Pros:** matches published methodology; extraction effort already invested.
**Cons:** authors have moved on; we're targeting a deprecated format.

### B. Switch to DWPose (authors' current recommendation)
- Stop the OpenPose extraction (lose ~9h compute).
- Install DWPose in the container, re-extract.
- Use 2025 `tools-new-2025/` for trg384 compressed format.
- Write model + training from scratch based on the paper.

**Pros:** aligns with authors' current best practice; format is what closed-source
SignLLM presumably trains on.
**Cons:** wastes 9h compute; user previously instructed *"use OpenPose because it's
in the paper"* — this would reverse that.

### C. Pragmatic alternative — Progressive Transformers
- Pivot to Saunders et al. *Progressive Transformers for End-to-End Sign Language
  Production* (BMVC 2020) — fully open code at
  <https://github.com/BenSaunders27/ProgressiveTransformersSLP>.
- Train on MoSL with the same OpenPose features we're already extracting.
- Compare against our SignLLM-inspired implementation as a baseline.

**Pros:** complete open-source reference; PFE delivers something working faster;
gives a baseline to compare against.
**Cons:** different paper than the one user picked.

## My recommendation

**A**, because:
- User has been firm about paper fidelity to SignLLM as published.
- 33% extraction sunk cost is not enormous but not zero.
- The 2024 `tools/2D_to_3D/` directly accepts our format with minimal glue.
- Writing the model from the paper is a defensible PFE deliverable.

But **C is a real option** worth flagging — if the PFE timeline is tight, having
a fully-open baseline (Progressive Transformers) plus our partial SignLLM
reimplementation is a stronger story than only partial SignLLM.
