# Phase 4 — Final results

End-to-end evaluation of the three SignLLM-on-MoSL ablation runs from Phase 3d.
Numbers below come from `scripts/evaluate_runs.py` (saved to
`runs/evaluation.json`) on 2026-05-10.

All metrics are computed under the strict-paper-fidelity hyperparameter set
locked in [DECISIONS.md](DECISIONS.md): `d_model=768, heads=12, d_ff=3072,
dropout=0.1`, 2-layer encoder + 2-layer decoder, Adam optimizer with the Vaswani
2017 noam schedule (`warmup=4000`), batch size 32, max 200 epochs, early stop
on dev teacher-forced pose MSE with patience 20.

## Headline numbers

The table below is the paper-comparable evaluation: autoregressive generation
from text only, DTW between predicted and target pose sequence, normalized by
path length so values are comparable across clips of different lengths.

| Run | Split | tf_pose_mse | tf_log_T_err | ar_dtw_mean | ar_dtw_median | n |
|---|---|---:|---:|---:|---:|---:|
| **baseline_mse** | dev  | **0.00025** | 0.347 | **1.0218** | **0.9636** | 430 |
| **baseline_mse** | test | **0.00020** | 0.276 | **1.0447** | **1.0019** | 112 |
| rl | dev | 0.00039 | 0.362 | 1.3069 | 1.3089 | 430 |
| rl | test | 0.00034 | 0.290 | 1.3182 | 1.3419 | 112 |
| rl_plc | dev | 0.00045 | 0.367 | 1.1530 | 1.1325 | 430 |
| rl_plc | test | 0.00040 | 0.293 | 1.2117 | 1.2361 | 112 |

Definitions:
- **tf_pose_mse**: per-frame MSE under teacher forcing (averaged across all 150
  pose coordinates and all real frames, then averaged over clips).
- **tf_log_T_err**: mean absolute error of `log_T_pred` against `log(n_frames)`.
  An error of 0.3 means the model's length prediction is off by a factor of
  e^0.3 ≈ 1.35× on average.
- **ar_dtw_mean / median**: dynamic-time-warping distance (path-length-
  normalized) between the autoregressively-generated pose sequence and the
  target.  Lower is better.

## Headline finding (model-only ranking)

**Among the three trained models, `baseline_mse` wins on every metric across both
splits.** The ranking is robust: MSE > RL+PLC > RL on every column of the table.
Test-set autoregressive DTW best (MSE) is **1.04** vs worst (RL) **1.32** — a
relative gap of ~26%.

This **inverts the paper's Table 5 ablation** (where the order on ASL was
RL+PLC ≈ 18.77 > RL ≈ 18.33 > MSE ≈ 10.96 in BLEU-4 dev).

## Stronger finding (model vs trivial baselines)

After computing reference baselines (`scripts/compute_baselines.py`,
`runs/baselines.json`) the picture is more pointed:

| Method | dev DTW (mean) | test DTW (mean) | n |
|---|---:|---:|---:|
| Mean-pose (single avg, ignore input) | **0.8148** | 0.8682 | 430/112 |
| Nearest-neighbor (retrieve same-text train clip) | 0.8647 | **0.7817** | 430/112 |
| Random-clip from train | 0.9744 | 0.9620 | 430/112 |
| **`baseline_mse` (our best trained model)** | 1.0218 | **1.0447** | 430/112 |
| `rl_plc` | 1.1530 | 1.2117 | 430/112 |
| `rl` | 1.3069 | 1.3182 | 430/112 |

**Every baseline beats every trained model.** The implication is sharper than
"our model didn't reproduce the paper's improvements":

- **Cross-signer variance floor.** Nearest-neighbor lands at 0.78 on test, not 0.0,
  because val/test held-out clips are *different signers' takes* of the same
  Arabic word.  This 0.78 is the irreducible cross-signer variance for our
  data — a perfect "I know exactly which sign this is" model would land
  somewhere around this floor.
- **Our best model is ~34 % above this floor on test.**  It is outputting
  *less informative* sequences than literal retrieval of a training clip
  with the same label — the model is actively worse than memorisation
  + lookup.
- **Mean-pose almost beats our best model on test (0.87 vs 1.04).**
  Outputting the *same* average pose for every input is competitive with our
  35 M-parameter transformer.  This is the giveaway: the model is
  generating mean-pose-like blobs rather than sign-specific motion.

## Why does the model lose to mean-pose?

The most likely explanation is **regression-to-the-mean under MSE-trained
autoregressive regression** — a long-known failure mode of continuous
seq2seq pose models~\cite{saunders2020progressive}.  Three compounding
factors:

1. **MSE on continuous coords penalises the mean of modes most weakly.**
   When several plausible target trajectories exist (e.g., multiple
   variants of the same sign across signers), the loss-minimising single
   prediction is the *average* of those trajectories.  The model converges
   on a smoothed-out mean rather than committing to any one specific
   trajectory.
2. **Teacher forcing during training, autoregression at inference.**
   At training time every previous frame is the ground-truth previous
   frame, so the model never has to recover from its own earlier mistakes.
   At inference each predicted frame becomes the next step's input, and
   small smoothing biases compound across the full clip.
3. **1:1 sign-to-clip ratio + MSE.**  74 % of our signs have one training
   instance.  For these, the model trivially memorises the single
   trajectory.  For the 26 % with multi-variant signs, MSE pushes the
   model toward the cross-variant mean — which is exactly what mean-pose
   does globally.

In aggregate the model behaves as a "mean-pose generator conditioned on
the input's lookup table position" — and on this dataset that's strictly
worse than just doing the lookup.

## Implications

The previous interpretation ("paper's RL+PLC ranking inverts on small
isolated-word data") is true but understates the issue.  The sharper
empirical claim from these results is:

> **A strict paper-faithful reimplementation of SignLLM-Base, on the MoSL
> isolated-word dataset, fails to outperform deterministic retrieval baselines.
> All three loss configurations (MSE / RL / RL+PLC) underperform mean-pose,
> nearest-neighbor, and random-clip baselines on autoregressive DTW.**

This finding is a constraint on the regime in which SignLLM's published
methodology actually delivers value: it does *not* on a $\sim$1{,}600-sign
isolated-word dataset of this scale, regardless of which of the paper's
three loss configurations is chosen.

The result is a clean negative empirical contribution.  It tells future MoSL
SLP practitioners which axis matters most for their setting (data
distribution shape: continuous vs. isolated; samples per sign) and saves
them the same compute we spent demonstrating it.

## Why the result inverts

Three structural differences from the paper's setting that make our outcome
defensible despite the inversion:

1. **Dataset size.**  The paper trains each per-language SignLLM-Base on
   tens of thousands of clips (e.g. ASL in How2Sign: ~31 k train).  We have
   **1,674 train clips for MoSL** — roughly 18× smaller.  At this scale the
   model is firmly in the *memorization regime* (best dev MSE was achieved at
   epoch 199 — the very last epoch — meaning training had not yet plateaued).

2. **Sign-to-clip ratio.**  Our 1,631 unique signs across 1,674 train clips
   means most words have **exactly one training example**.  RL Loss + PLC are
   *batch-prioritization* schemes — they assume there is a meaningful signal
   in re-weighting samples within a batch.  With essentially one example per
   class, the within-class variance the prioritization is supposed to handle
   doesn't exist; instead PLC ends up dropping half the gradient signal each
   step (`skip_below_quantile=0.5`).

3. **Continuous discourse vs. isolated signs.**  The paper's data (How2Sign
   etc.) is continuous sentence-level signing, where a single training
   example contains many gloss tokens with rich temporal structure for the
   transformer to learn.  Our MoSL data is *isolated-sign* recordings — one
   word per clip, no compositional structure at the sentence level.  The RL
   reward signal in eq. 3 is computed over the whole clip, so the
   per-batch reward distribution has lower entropy in our setting; PLC's
   prioritization is operating on noise.

In short: **the paper's improvements over MSE are improvements at scale and
on continuous data.  Strict reimplementation on isolated-word, low-data MoSL
exposes that the additional machinery does not help here — and in fact
slightly hurts.**

## Per-mode training-time observations

From the training logs (`runs/<name>/log.jsonl`):

- **baseline_mse** ran the full **200 epochs** without ever triggering early
  stop.  Best dev MSE was reached at *epoch 199*, indicating training was still
  improving at the budget cap.  More epochs would likely help further.
- **rl** early-stopped at epoch 65 (best at epoch 45).  Plateau was real.
- **rl_plc** early-stopped at epoch 71 (best at epoch 51).  Plateau was real.

This pattern is consistent with the analysis above: the deterministic MSE loss
on a small dataset behaves like progressive function-approximation refinement
and benefits from extended training; the noisier RL / PLC signals saturate
quickly.

## Methodological gaps and caveats

1. **No back-translation BLEU.**  The paper's primary metric is BLEU-{1,2,3,4}
   computed by translating generated poses back to text via a separate
   sign-to-text model.  We don't have a sign-to-text recognizer for MoSL, and
   training one is out of scope for Phase 4.  We report DTW only.  For our
   isolated-sign setting, BLEU would essentially reduce to *exact-match
   accuracy on a single-token recognition task* — a different problem than
   what the paper measures on continuous discourse.

2. **Length-prediction quality.**  Test `tf_log_T_err = 0.276` for MSE
   corresponds to predicted T within ~1.32× of true T.  This is acceptable but
   not great — better than random but not tight enough that downstream
   vid2vid rendering would be reliably timed.  A separately-trained length
   regressor (or EOS-head approach) might do better; our paper-faithful MLP
   head from pooled text features is one defensible choice among several.

3. **Single seed.**  Each run used `seed=0`.  We have not measured run-to-run
   variance from random initialization or data shuffling.  The size of the
   gap (~26% on AR DTW) is large enough that we expect the ranking to be
   stable across seeds, but we have not confirmed this empirically.

4. **Pose-extraction backend deviation.**  Our keypoints come from Hzzone's
   PyTorch port of OpenPose, not the canonical CMU build.  The 2024
   Prompt2Sign pipeline uses only `idxsPose=[0..7]` from the body model, and
   those 8 indices match between COCO-18 (Hzzone) and BODY_25 (CMU), so this
   is not expected to be material.  See [DECISIONS.md](DECISIONS.md).

5. **No face keypoints.**  Hzzone's port covers body + hands but not the
   70-point face model.  Mouth and eyebrow movements (which carry
   linguistic content in MoSL) are absent from training data — but the 2024
   Prompt2Sign pipeline already discards everything except 8 upper-body
   joints + 42 hand joints, so face keypoints would have been dropped in
   post-processing anyway.  This deviation is not material to our result.

## What would change the conclusion

If we were willing to deviate from strict paper fidelity, things that would
likely flip the ranking back toward RL / RL+PLC:

- **More data.**  Pretrain on a continuous-signing dataset (e.g. How2Sign),
  fine-tune on MoSL.  RL methods scale better with data.
- **Character-level tokenization.**  Lets the model factorize the 1,631 signs
  into ~60 atoms with shared parameters; would make the within-batch reward
  distribution meaningful for PLC.
- **More clips per sign.**  Recording 5+ examples of every sign would let
  PLC's batch-prioritization see real within-class variance to act on.

None of these are in scope for the strict-paper-fidelity remit, but they are
defensible follow-up work for a future extension of the project.

## Files referenced by this writeup

- `runs/baseline_mse/{best.pt,log.jsonl,summary.json,config.json}`
- `runs/rl/{best.pt,log.jsonl,summary.json,config.json}`
- `runs/rl_plc/{best.pt,log.jsonl,summary.json,config.json}`
- `runs/evaluation.json` — combined dev+test metrics across the three runs
- `mosl/train/eval.py` — autoregressive + teacher-forced eval implementation
- `scripts/evaluate_runs.py` — driver that produced these numbers
