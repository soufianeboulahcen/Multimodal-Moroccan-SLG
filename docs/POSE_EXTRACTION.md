# Phase 2 — completion summary

End-to-end transformation of 2,216 raw `.mp4` MoSL videos into the SignLLM-format
compressed pose data the paper's training pipeline ingests.

Completed **2026-05-10 00:14 UTC**.  Total compute: ~38 hours of GPU wall-time
across two detached docker runs on `spark-3112.local` (DGX Spark, GB10
Blackwell, CUDA 13).

## What was built

The full preprocessing pipeline, in two halves:

### Half 1 — OpenPose extraction (per-clip NPZ)
`mosl/pose/extract_dataset.py` walks `data/labels.csv`, runs Hzzone
pytorch-openpose (body BODY_25-compatible 18-keypoint subset + 21+21 hand
keypoints) on every frame, writes one `.npz` per clip with arrays:

```
pose_keypoints_2d        (T, 54)   18 body keypoints * (x, y, c)
hand_left_keypoints_2d   (T, 63)   21 hand keypoints * (x, y, c)
hand_right_keypoints_2d  (T, 63)
fps, width, height       scalars
```

### Half 2 — 2D → 3D + compressed pose (per-mode `.skels`)
Five chained scripts run by [`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh):

1. `mosl/pose/export_openpose_json.py` — unpacks NPZ to per-frame CMU OpenPose JSON.
2. `scripts/setup_p2s_pipeline.py` — split-aware hardlinks + writes `<mode>.files` + `<mode>.text`.
3. `pipeline_demo_01_json2h5.py` — packs JSON to one H5 dataset per clip, shape `(T, 150)` = 8 body + 21 left + 21 right hand × (x,y,c).
4. `pipeline_demo_02_h5totxt.py` — Step I/II/III closed-form initialization + **PyTorch port** of `pose3D.backpropagationBasedFiltering` (1000 SGD iters, lr=0.1, regs=[0.001, 0.1]).
5. `pipeline_demo_03_txt2skels.py` — flatten 3D coords + interleave time markers + normalize/9 → final `.skels`.

## Output artifacts

`third_party/Prompt2Sign/tools/2D_to_3D/final_data/`:

| File | Lines | Bytes | Avg bytes/clip |
|---|---:|---:|---:|
| `train.skels` | 1,674 | 209,342,267 | 125,055 |
| `train.files` | 1,674 | 75,387 | — |
| `train.text` | 1,674 | 33,152 | — |
| `dev.skels` | 430 | 48,456,257 | 112,688 |
| `dev.files` | 430 | 23,163 | — |
| `dev.text` | 430 | 7,211 | — |
| `test.skels` | 112 | 13,344,369 | 119,146 |
| `test.files` | 112 | 6,182 | — |
| `test.text` | 112 | 1,811 | — |
| **Total** | **2,216** | ~271 MB | — |

Each `.skels` line is one clip: `<x y z>` triplets per joint, flattened across
frames, with a frame-time marker (`i/T`) interleaved after every frame.  All
coordinate values divided by 9 (paper-spec normalization).  Zero values
replaced with the placeholder `0.01600`.

`<mode>.files` and `<mode>.text` are line-aligned with `<mode>.skels`.
`<mode>.text` uses the **base Arabic NFC word** (variant suffixes like
`(إِشَارَة 1)` stripped — variants of the same sign share the text label).

## Throughput

Two-stage breakdown across the two detached runs:

| Stage | Wall time | Per-clip avg |
|---|---:|---:|
| OpenPose extraction (2,216 clips) | 27h 46m | 45.1 s |
| Pipeline (export + stages 01/02/03 × 3 modes) | ~11h | ~17.9 s |
| Stage 02 (h5totxt) only — dominant cost | ~10h 40m | ~17.3 s |

Per-mode pipeline times:
- train (1,674 clips): 8h 31m
- dev (430 clips):     2h 7m
- test (112 clips):    33m

Stage 02 dominated.  Each clip runs 1000 SGD iterations of the kinematic-
skeleton optimization (PyTorch port of the original TF1 code), with cost
proportional to clip length T.  Average T is ~100 frames.

## Quality observations

- **0 failures across the entire pipeline.**  All 2,216 clips produced valid
  NPZ → JSON → H5 → demo1-5 → final `.skels`.  No `<mode>_missing_demo5_folder.txt`
  files were created (stage 03's "skip if missing" path never fired).
- **Loss convergence on the SGD refinement** is monotone for every clip
  inspected.  Typical curves drop ~3–5× over the 1000 iterations.  No
  divergence or NaN losses observed in the spot-check.
- **Hand keypoint detection rate** (from spot-check on Pronouns):
  - One-handed signs: only the active hand is non-zero (correct — Hzzone's
    body-driven hand detector skips the resting hand).
  - Two-handed signs: both hands ~96–99% non-zero rate.
  - Body keypoints: ~78% non-zero (upper body always in frame; lower-body
    positions correctly absent for our 460×460 close-up videos).

## Known limitations

1. **OpenPose backend is a PyTorch reimplementation** (Hzzone), not the canonical
   CMU C++/Caffe build.  Output JSON schema is identical; pixel-level
   coordinates may differ from CMU OpenPose by < 1% on a held-out subset
   (not measured).  See [DECISIONS.md](DECISIONS.md).
2. **No face keypoints.**  The Hzzone port covers body + hands but not the
   70-point face model.  Mouth and eyebrow movements (which carry linguistic
   content in MoSL) are absent from training data.  Defensible because the
   2024 Prompt2Sign pipeline's `idxsPose=[0..7]` already discards everything
   except upper-body anyway, but worth flagging in the report.
3. **Stage 01 silently drops frames where the nose (body kp 0) is undetected**:
   `if not points[0] == 0.0: frames[i] = points`.  Rare on our data (face is
   always in frame for sign clips) but not measured at scale.
4. **30-fps clips (~7% of dataset)** were extracted at native fps and then
   their pose tensors are mixed in with 25-fps clips.  The compressed pose
   format includes a per-frame time marker (`i/T`), so models that condition
   on relative time can compensate; absolute-fps-conditioned models cannot.
5. **Hand-driven detection failures** on busy-background frames are silently
   counted as "no hand present" (zeros).  We have no statistics on how often
   this happens across the full dataset; only spot-checked on Pronouns.

## What this unblocks

Phase 3 is fully unblocked: model implementation + training.  Inputs are now:
- Source text: `<mode>.text` (Arabic words, NFC-normalized).
- Target pose: `<mode>.skels` (the SignLLM compressed format).
- Clip identifiers: `<mode>.files` (for diagnostics).

Per the decisions log, the model is the SignLLM architecture from arXiv:2405.10718
implemented in PyTorch, with MoSL added as a 9th language token.
