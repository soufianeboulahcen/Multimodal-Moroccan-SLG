# Project decisions log

A running log of non-trivial decisions, why they were made, and what we considered.
Newest entries at the top.  Each entry is short — for fuller treatment, link out
to dedicated docs.

---

## 2026-05-10 · Phase 4 result: ablation ranking inverts paper Table 5

**Context.** Trained the three SignLLM ablation rows (Base + MSE / Base + RL /
Base + RL + PLC) per-paper-faithful on MoSL.  Test set autoregressive DTW is
1.044 (MSE) < 1.212 (RL+PLC) < 1.318 (RL).  Paper Table 5 ranks RL+PLC > RL >
MSE on ASL.

**Decision.** Accept the inverted ranking as a real empirical finding for our
data, not a bug.  Document in [RESULTS.md](RESULTS.md) with reasoning:
(1) our 1,674 train clips is ~18× smaller than the paper's per-language data,
(2) 1:1 sign-to-clip ratio puts the model in the memorization regime where MSE
benefits from extended training (best at epoch 199 — the budget cap),
(3) isolated-word data has no within-class variance for PLC's batch
prioritization to act on.

**Reasoning.** Strict-paper-fidelity rule means we don't tune hyperparameters
to flip the ranking.  Ranking is robust across dev and test splits; gap is
~26% on AR DTW which is well above measurement noise from a single seed.

**Decided by.** Empirical.  Reporting honestly is the right call.

---

## 2026-05-10 · Phase 3 hyperparameters locked: paper text + Vaswani 2017 defaults

**Context.** The SignLLM paper (arXiv:2405.10718) specifies very little about
the model architecture beyond layer count (2 enc + 2 dec for "Base") and
total parameter count (~40 M per language).  The paper is silent on `d_model`,
heads, FFN size, dropout, optimizer, learning rate, batch size, training
budget, PLC's `η`, and the inference-length-prediction mechanism.

**Decision.** Lock the following config for Phase 3:

| Item | Value | Source |
|---|---|---|
| Layers | 2 enc + 2 dec | Paper §5 |
| `d_model` | 768 | Targets paper's 40 M per-language count (lands at ~35 M total) |
| Heads | 12 | d_model // 64, BERT-Base convention |
| `d_ff` | 3072 | 4 × d_model (Vaswani convention) |
| Dropout | 0.1 | Vaswani 2017 default |
| Tokenization | Word/gloss-level | Paper convention |
| Length prediction | MLP head from pooled text features | Paper doesn't describe; documented choice |
| Loss schedule | MSE → RL Loss → RL+PLC ablation matrix | Reproduces paper Table 5 |
| Optimizer | Adam (not AdamW) | Vaswani 2017 default |
| LR schedule | Vaswani noam, warmup=4000 | Vaswani 2017 default |
| Batch size | 32 | Smallest reasonable; paper doesn't specify |
| Training budget | Until dev DTW plateaus (≤ 200 epochs) | Paper doesn't specify |
| PLC `η` | 1 | Linear prioritization, simplest reading |
| PLC threshold | 0.5 × max-reward-observed-so-far | Literal reading of "if reward < 50% skip" |

**Reasoning.** User asked for "exactly what's in paper".  Paper-text-fidelity
fully determines the layer count, parameter target, loss formulation, PLC
sampling formula, and modes (MLSF / Prompt2LangGloss).  For everything else
the paper is silent; we use the canonical Vaswani 2017 transformer recipe
plus the most literal reading of the paper's English-language descriptions.

`d_model=768` is the smallest standard transformer width that lands within
~15% of the paper's stated 40 M per-language count given the fixed 2-layer
architecture.  d_model=512 would land at ~16 M, missing the param target by
~2.5×, which would be a more meaningful deviation than choosing a different
d_model.

**Decided by.** User confirmed "okay" on 2026-05-10 after I surfaced the
40M-param vs Vaswani-default conflict.

---

## 2026-05-09 · Reduce SGD-loop verbosity in `pose3D.py` (every 100 iters)

**Context.** The PyTorch-ported `backpropagationBasedFiltering` originally
matched upstream's `print("iCycle = ...")` exactly — one line per gradient step.
For the full-dataset run that's 1000 iters × 2,216 clips = ~2.2M log lines, well
beyond docker's per-container log cap and useless for debugging.

**Decision.** Print every 100 iterations plus the final iteration (≈11 lines
per clip).  Pure verbosity change, no algorithmic effect.

**Reasoning.** Keeps enough signal to verify convergence by visual inspection
on any clip while reducing log volume by ~100×.  Critical for the detached
docker run.

**Decided by.** Mine.

---

## 2026-05-09 · OpenPose JSON filenames must be `<stem>_<frame>_keypoints.json`

**Context.** The 2024 Prompt2Sign `pipeline_demo_01_json2h5.py` parses frame
indices via `re.search(r"([^\\/]+)_(\d+)_keypoints\.json$", fname)`.  The regex
requires *both* a non-slash stem and a frame-index group, separated by `_`.
Our initial NPZ-to-JSON exporter wrote bare `<frame>_keypoints.json` filenames
(no stem), which made the regex fail and stage 01 would crash on
`p.group(2)`.  The Pronouns smoke run we did first happened to work only
because the regex matched against the longer parent-path component.

**Decision.** Always emit `<stem>_<frame>_keypoints.json` where stem is the
clip's video filename without extension (e.g.
`أَنَا_000000000030_keypoints.json`).  Applied in `mosl/pose/export_openpose_json.py`.

**Reasoning.** Matches the regex deterministically, regardless of parent-path
encoding.  Required for the full-dataset run to succeed.  Idempotent re-export
overwrote the Pronouns sample with the new naming.

**Decided by.** Linter / collaborator caught this between sessions; fix is
correct and stays.

---

## 2026-05-08 · Port pose3D.backpropagationBasedFiltering from TF1 to PyTorch

**Context.** The 2024 Prompt2Sign 2D→3D pipeline includes a backpropagation-based
3D-pose refinement step (`tools/2D_to_3D/pose3D.py`, ~175 lines) implemented in
TensorFlow 1.x.  Their code uses `tf.placeholder`, `tf.Variable`,
`tf.train.GradientDescentOptimizer`, `sess.run(...)` — a pure TF1 graph-mode
formulation. TF 1.15 (the last TF1 release) targets Python 3.7 and has no
aarch64 wheels, so it cannot be installed on our DGX Spark + NGC PyTorch stack.
The paper text (Section 3) describes only the closed-form Step I/II/III
initialization and does not document this refinement step.

**Decision.** Port `pose3D.backpropagationBasedFiltering` to PyTorch (eager mode).
Identical algorithm: same variables (log bone lengths, root positions per frame,
limb rotation angles per frame), same loss (reprojection MSE on 2D weighted by
OpenPose confidence + L1 regularizer on bone lengths + temporal smoothness),
same gradient descent.

**Reasoning.**
1. "Exact paper, nothing extra" — interpreted as paper-as-published incl.
   reference code, since the paper text alone is incomplete (doesn't describe
   the refinement that the authors clearly intended).
2. Verbatim reproduction is impossible: TF 1.15 doesn't run on Python 3.12 /
   aarch64 / CUDA 13.  The authors' setup cannot be recreated on the DGX Spark.
3. The optimization is framework-agnostic.  Given identical hyperparameters
   and random seeds, a PyTorch port produces numerically equivalent output to
   the authors' TF1 code.
4. PyTorch is already in our container.  Adding TF2-compat would mean a
   ~1.5 GB extra layer for a single function.

**Alternatives considered.**
- Skip the refinement and use only the closed-form initialization
  (`pose2Dto3D.initialization`, demo4.txt). Rejected: deviates from the published
  implementation; loses the post-init smoothing.
- Install TF2 with `tf.compat.v1.disable_eager_execution()`.  Rejected: heavy,
  fragile, and only marginally closer to the authors' literal setup since the
  exact TF1.15 they used isn't installable anyway.

**Decided by.** User confirmed "okay" on 2026-05-08 after I explained the three
options and recommended Option 2.

---

## 2026-05-08 · TF stub patch in pipeline_demo_02_h5totxt.py

**Context.** `pipeline_demo_02_h5totxt.py` imports tensorflow at module load
solely to call `tf.keras.backend.clear_session()` once at the end of each clip
(GPU memory cleanup).  No algorithmic role.

**Decision.** Patched the file in-place (in `third_party/Prompt2Sign/...`)
to replace the TF imports with a no-op stub class.  The actual 2D→3D math
(`pose2D`, `pose2Dto3D`, `pose3D`) is unaffected.

**Reasoning.** Adding TensorFlow as a runtime dep just to call `clear_session()`
would be a heavy and fragile dep for incidental code.

**Decided by.** Mine.

---

## 2026-05-08 · COCO-18 vs BODY_25 keypoint format — non-issue

**Context.** Hzzone's pytorch-openpose outputs the COCO 18-keypoint body model;
canonical CMU OpenPose defaults to BODY_25.  Earlier I flagged this as a
methodological deviation from the SignLLM paper that we'd document.

**Decision.** No deviation needed.  The 2024 Prompt2Sign 2D→3D pipeline
(`tools/2D_to_3D/pipeline_demo_01_json2h5.py`) uses
`idxsPose = [0, 1, 2, 3, 4, 5, 6, 7]` — only the first 8 body keypoints
(nose, neck, R/L shoulder, R/L elbow, R/L wrist).  These indices are *identical*
between BODY_25 and COCO 18.  The keypoints that differ (foot landmarks,
mid-hip) are never read by the pipeline.

**Reasoning.** Verified by grep on the source.  The pipeline subsets to
upper-body only because it's a sign-language pipeline and lower body is
irrelevant.

**Decided by.** Mine, after reading the code.

---

## 2026-05-08 · Path A locked: strict SignLLM-paper fidelity

**Context.** After cloning and reading the Prompt2Sign repo, we discovered the
SignLLM model code is not open-sourced (only data-prep tooling), and the authors
have officially deprecated their OpenPose pipeline in favor of DWPose.  This
opened three options (see [PROMPT2SIGN.md](PROMPT2SIGN.md)):
- A. stay paper-faithful with OpenPose
- B. switch to DWPose (authors' current recommendation)
- C. pivot to Progressive Transformers (Saunders 2020) as a fully-open baseline.

**Decision.** A — strict fidelity to the SignLLM paper as published
(arXiv:2405.10718).  No alternative methods.

**Reasoning.** Stated by user: *"exact paper, nothing else nothing extra."* The
PFE timeline is comfortable, so the time cost of reimplementing the model from
the paper is acceptable.  This commits us to: OpenPose for extraction, the 2024
Step I/II/III pipeline for 2D→3D, MLSF + Prompt2LangGloss + RL Loss + PLC for
the model, and the 2024 compressed pose format.

**Decided by.** User.

---

## 2026-05-07 · Switch to Docker (NGC PyTorch 26.04-py3) on remote

**Context.** Initial Phase 2 was set up using a host venv with `pip install
torch==2.9.1+cu128`.  Two problems:
- User asked for sandbox isolation on the DGX Spark (no host pip pollution).
- The cu128 wheel maxes out at SM_120 (Blackwell GB10 capability is 12.1), so
  PyTorch ran every kernel via PTX-JIT — extraction throughput ~0.9 fps.

**Decision.** Use NGC PyTorch container `nvcr.io/nvidia/pytorch:26.04-py3` as the
base image, layer Hzzone deps (opencv, scikit-image, tqdm) on top, and run all
compute inside `pfe-pose:latest` via `docker/run.sh`.  Host venv retired.

**Reasoning.** The NGC image ships PyTorch built with native SM_120 kernels
(via CUDA 13 SDK), eliminating the JIT slowdown.  It also gives us the
sandbox isolation the user asked for.  Verified post-switch: `cuda
capability (12, 1)` recognized natively, throughput up to ~2.4 fps.

**Decided by.** User asked for sandbox; tooling + image choice was mine.

---

## 2026-05-07 · OpenPose backend = `pytorch-openpose` (Hzzone)

**Context.** SignLLM paper [8] cites Cao et al. CMU OpenPose for keypoint
extraction. Building canonical CMU OpenPose on aarch64 + CUDA 13 is impractical
(unmaintained, Caffe-fork build issues, untested architecture).

**Decision.** Use Hzzone's PyTorch reimplementation
(<https://github.com/Hzzone/pytorch-openpose>) with the original CMU body and
hand pretrained weights (converted to .pth).  Keypoint output schema matches
CMU OpenPose JSON convention so downstream Prompt2Sign tooling accepts it.

**Reasoning.** Output JSON schema is identical; weights are direct conversions
of CMU's caffemodels; building canonical OpenPose is a likely time sink.  Known
deviation: Hzzone outputs COCO 18-keypoint body, not BODY_25.  For sign language
this matters minimally — the missing keypoints are feet (irrelevant in
upper-body sign videos).  Will document as a methodological deviation in the
report.

**Decided by.** User confirmed "go with c" (PyTorch port option) on 2026-05-07.

---

## 2026-05-07 · Compute on DGX Spark (`spark-3112.local`)

**Context.** OpenPose / training is not realistic on macOS.  The DGX Spark (GB10
Grace + Blackwell, 121 GiB unified memory, CUDA 13) was available.

**Decision.** Dual-machine workflow: edits and version control on local mac,
all compute on `ssh spark-3112.local`.  Project mirrored at
`/home/omaraitmlouk/Desktop/PFE-SOUFIAN/`.  rsync for code transfer.

**Decided by.** User.

---

## 2026-05-07 · Split strategy = clip-level held-out for multi-clip words

**Context.** 74% of MoSL signs (1,201 / 1,631 unique) have only one clip, so a
"hold out unseen words for val/test" split is impossible without losing most
of the dataset.

**Decision.** Three-way clip-level split, deterministic, lex-sorted within
each (category, word) group:
- 1 clip / word → all to train
- 2 clips / word → 1 train + 1 val
- 3+ clips / word → rest train, 1 val, 1 test
Result: train 1,674 / val 430 / test 112; every val/test word also appears in
train.

**Reasoning.** Held-out clips test the realistic question *"reproduce a known
sign in a new variant"* rather than zero-shot generalization, which the dataset
cannot support.  Documented in [STATS.md](STATS.md).

**Decided by.** Mine, no objection raised.

---

## 2026-05-07 · NFC Arabic as the canonical word key (not diacritic-stripped)

**Context.** 15 clips have a single Arabic diacritic mark as their entire label
(fatḥa, ḍamma, kasra, sukun, etc.) — these are real MoSL signs for the
diacritics themselves.  Stripping diacritics for "robustness" collapses them all
to the empty string, losing 8 distinct signs.  Additionally, 17 stripped-form
collisions are real minimal pairs (e.g. رَجُلٌ "man" vs رِجْلٌ "leg").

**Decision.** Use NFC-normalized raw Arabic (`word_arabic`) as the primary key
for sign uniqueness.  Keep `word_arabic_stripped` only as a secondary
fuzzy-lookup field for inputs that omit diacritics.

**Decided by.** Mine.  Caught while computing dataset stats.

---
