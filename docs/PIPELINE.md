# Phase 2 — Data preprocessing pipeline

End-to-end: raw `.mp4` videos → SignLLM-format compressed pose sequences (`<mode>.skels`).
Reproduces the published 2024 SignLLM data-prep pipeline on MoSL, with two
surgical adaptations needed to run on our stack (DGX Spark / Python 3.12 / aarch64).

Verified end-to-end on the **dev split of Pronouns (4 clips)** on 2026-05-08.
Run on the **full dataset (2,216 clips, all three splits)** on 2026-05-09/10
in ~11 hours wall-time, 0 failures.  See [POSE_EXTRACTION.md](POSE_EXTRACTION.md)
for the full Phase 2 completion summary with throughput, output sizes, quality
observations, and known limitations.

## Stage diagram

```
data/raw/vedios-dataset/<category>/<clip>.mp4              ┐
                                                            │  Phase 1 (already done)
                                                            ▼
data/labels.csv  +  data/splits.csv  +  data/video_meta.csv
                                                            │
                                                            │  mosl/pose/extract_dataset.py
                                                            │  pytorch-openpose body+hand on GPU, save NPZ per clip
                                                            ▼
data/processed/keypoints_2d/<category>/<clip>.npz
   pose_keypoints_2d        (T, 54)   COCO 18-point body
   hand_left_keypoints_2d   (T, 63)   21 hand keypoints
   hand_right_keypoints_2d  (T, 63)
                                                            │
                                                            │  mosl/pose/export_openpose_json.py
                                                            │  unpack to per-frame OpenPose-style JSON
                                                            ▼
data/processed/openpose_json/<category>/<clip>/
   <clip>_<frame>_keypoints.json    (CMU OpenPose v1.3 schema)
                                                            │
                                                            │  scripts/setup_p2s_pipeline.py --mode {train|dev|test}
                                                            │  hardlinks JSONs into Prompt2Sign expected layout +
                                                            │  generates <mode>.files and <mode>.text from splits.csv
                                                            ▼
third_party/Prompt2Sign/tools/2D_to_3D/output_of_openpose/<mode>/json/<mode>__<clip>/...
                                              + input_data/<mode>.files
                                              + input_data/<mode>.text
                                                            │
                                                            │  pipeline_demo_01_json2h5.py --data_subset <mode>
                                                            │  pack to H5 keeping body[0..7] + 21 left + 21 right hands
                                                            ▼
input_data/<mode>.h5    one dataset per clip, shape (T, 150)
                                                            │
                                                            │  pipeline_demo_02_h5totxt.py --data_subset <mode>
                                                            │  Step I/II/III initialization (pose2Dto3D.initialization)
                                                            │  then PyTorch port of backpropagationBasedFiltering
                                                            │  (1000-iter SGD, lr=0.1, regs=[0.001, 0.1])
                                                            ▼
out_data/<mode>/<clip>/
   demo1.txt   normalized 2D
   demo2.txt   pruned 2D
   demo3.txt   interpolated 2D
   demo4.txt   initial 3D
   demo5.txt   refined 3D (final pose)
                                                            │
                                                            │  pipeline_demo_03_txt2skels.py --data_subset <mode>
                                                            │  flatten + normalize / 9 + interleave time markers
                                                            ▼
final_data/<mode>.skels    one line per clip, the SignLLM-format compressed pose sequence
final_data/<mode>.files    matching clip identifiers
final_data/<mode>.text     matching Arabic word annotations
```

## Adaptations required for our stack

Two surgical changes against the upstream code; both documented in
[DECISIONS.md](DECISIONS.md):

1. **TF stub in `pipeline_demo_02_h5totxt.py`** (~10 lines).  Upstream imports
   TensorFlow only to call `tf.keras.backend.clear_session()` at the end of each
   clip.  Replaced the TF imports with a no-op class.  No algorithmic effect.
2. **PyTorch port of `pose3D.backpropagationBasedFiltering`** (~120 lines of
   meaningful TF1 code).  Upstream uses `tf.placeholder` + `tf.Variable` +
   `tf.train.GradientDescentOptimizer` + `sess.run` — pure TF1 graph mode.
   Reimplemented in PyTorch eager mode keeping the same variables, forward
   pass, loss (weighted reprojection MSE + L1 bone-length reg + 3D temporal
   smoothness reg), hyperparameters, and numerical-stability epsilon.  Same
   public function signature so the upstream pipeline scripts call it
   unchanged.  See `third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py`.

No model code is touched.  No paper-text-described behaviour deviates.

## Throughput (measured on the full-dataset run)

| Stage | Wall time | Per-clip avg |
|-------|----------:|-------------:|
| OpenPose extraction (NPZ) — 2,216 clips | 27 h 46 min | 45.1 s |
| Stages 01 + 02 + 03 over all 3 modes | ~11 h | ~17.9 s |
| ↳ Stage 02 only (dominant) | ~10 h 40 min | ~17.3 s |

Stage 02 dominates re-running cost for any future hyperparameter change to the
3D refinement.  Output shapes are compact (`<mode>.skels` is ~119 KB per clip
on average).

## Per-mode invariants (verified on dev/Pronouns)

- `<mode>.files` and `<mode>.text` are aligned line-for-line.
- Every line in `<mode>.skels` corresponds to exactly one entry in `.files`.
- `<mode>.text` uses the **base Arabic word** (variant suffixes like
  `(إِشَارَة 1/2/3)` are stripped) — variants of the same sign share the same
  text annotation.
- Mode mapping: our split label `val` → upstream pipeline mode `dev`.

## Open follow-ups

- ~~Run the full pipeline for all three modes (train / dev / test) once Phase 2
  pose extraction completes.~~ **Done 2026-05-10.**
- Stage 01 silently drops frames where the nose (body kp 0) is undetected in
  raw OpenPose output (`if not points[0] == 0.0:`).  Worth quantifying on the
  full dataset; rare on the Pronouns sample (no clip dropped any frames).
- Stage 02 silently produces no `demo5.txt` if the optimization fails for any
  reason.  On the full run we got **0 missing demo5.txt** across all 2,216
  clips, so the fail-then-skip path never fired — but worth keeping the
  bookkeeping for future re-runs with different hyperparameters.
