# Code walkthrough — defense reference

A focused guide to the five most important source files in this project for
a supervisor walkthrough or thesis defense.  Read this before the meeting,
then walk the supervisor through the files in the order below.

The ranking is by **what the supervisor is most likely to probe**, in turn
driven by where the report's claims live.  Files outside the top five matter
too, but they implement standard plumbing — you can describe them in one
sentence each if asked, and only dig in if the supervisor explicitly opens
them.

---

## Tier 1 — must defend line-by-line

### 1. [`mosl/model/signllm.py`](../mosl/model/signllm.py) — the SignLLM architecture

**Why it's important.**  The whole thesis is "did we faithfully reimplement
SignLLM?"  Every line here is where that claim is either true or false.

**What to walk through.**

- **`SignLLMConfig` dataclass** — every hyperparameter in one place;
  cross-reference Table 4.1 in the report, which separates paper-specified
  from Vaswani-default values.
- **`encode_text()`** — implements paper eq. (1): token embedding
  $\times \sqrt{d_{\text{model}}}$ (Vaswani §3.4), sinusoidal positional
  encoding, 2-layer encoder.
- **`predict_length()`** — the MLP length head.  Mean-pool over real text
  tokens, then MLP to predict $\log T$.
- **`forward()`** — teacher-forced training.  Important detail: we shift the
  target right by one and replace position 0 with a learnable
  `start_frame` embedding so the decoder has a non-trivial signal at $t=0$.
- **`generate()`** — autoregressive inference.  $T$ is predicted by the
  length head, then the decoder unrolls for exactly that many steps.
- **`decode()`** — uses a *bool* causal mask (matches the padding-mask
  dtype to avoid torch's deprecation warning about mismatched mask types).

**Questions to expect.**

- *"Why $d_{\text{model}} = 768$?"* — The paper's stated target is ~40 M
  parameters per language for the Base configuration.  With 2 layers
  (paper-fixed), $d_{\text{model}} = 768$ is the smallest standard
  transformer width that lands within ~13 % of that count.  See the
  2026-05-10 entry in [`DECISIONS.md`](DECISIONS.md).
- *"Why 12 heads, dropout = 0.1?"* — Vaswani 2017 defaults for items the
  SignLLM paper does not specify.  Same source for $d_{\text{ff}} = 3072$,
  ReLU activation, post-norm, and sinusoidal positional encoding.
- *"Where is cross-attention?"* — `nn.TransformerDecoderLayer` includes it
  by construction; `decode()` passes `memory=text_features` and the
  decoder's cross-attention attends over those features at every layer.

---

### 2. [`mosl/train/losses.py`](../mosl/train/losses.py) — MSE / RL / RL+PLC

**Why it's important.**  This is where the paper's §4.4 is implemented.  The
PLC interpretation is documented here — the supervisor *will* ask about it.

**What to walk through.**

- **`masked_per_sample_mse()`** — per-frame MSE averaged over the 150 pose
  coordinates and over real (non-padded) frames only.
- **`rl_per_sample_reward()`** — $r = -\text{MSE}$ per the paper's eq.
  Always $\le 0$.
- **`plc_weights()`** — this is the most-likely-probed function in the
  whole project.  Key talking point: the paper says
  $P(i) \propto r(i)^\eta$, but for negative $r$ and non-integer $\eta$
  that's mathematically ill-defined.  We use
  $P(i) \propto \exp(\eta \cdot r)$ — the softmax-style equivalent that is
  well-defined for all $r \in \mathbb{R}$ and matches the *intent* of
  the paper.
- **The 50 % threshold.**  The paper says "skip if reward < 50%" without
  defining the 50 % reference value.  We interpret as "skip samples below
  the within-batch median quantile" — i.e., discard the worse half of each
  batch.  Documented in [`DECISIONS.md`](DECISIONS.md).
- **`signllm_step_loss()`** — top-level dispatcher; selects MSE / RL /
  RL+PLC by config.

**Questions to expect.**

- *"Why $\exp(\eta \cdot r)$ and not $r^\eta$?"* — Negative-reward
  formulation requires it; documented choice.
- *"What does '50%' refer to?"* — Median quantile within each batch;
  documented choice.
- *"Is RL Loss actually different from MSE here?"* — Per the paper's eq.
  (3), at the per-step level RL Loss is numerically identical to MSE.  The
  re-labelling matters only when wrapped in PLC's sample re-weighting; that
  is what `rl_plc` mode does.

---

### 3. [`third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py`](../third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py) — the PyTorch port

**Why it's important.**  This is the most concrete *engineering*
contribution of the project.  The TF1 original could not run on the
DGX Spark's aarch64 + CUDA 13 stack (TF 1.15 targets Python 3.7 only,
no aarch64 wheels), so we re-implemented it in PyTorch eager mode.

**What to walk through.**

- The module docstring explains *why* the port exists and what was
  preserved exactly.
- **Variables** map 1:1 to the TF1 original: `lines` (log bone lengths),
  `rootsx/y/z` (root joint positions per frame), `anglesx/y/z` (limb
  rotation angles per frame).
- **Forward pass.**  Walks the skeletal hierarchy in tree order,
  propagating positions as
  $\text{pos}_{\text{child}} = \text{pos}_{\text{parent}} + L \cdot \frac{(A_x, A_y, A_z)}{|A|}$.
  This is *exactly* the paper's Step III formula.
- **Loss.**  Weighted reprojection MSE on 2D only ($z$ is unobserved) +
  L1 regularisation on $\exp(\text{lines})$ + temporal smoothness in 3D.
- **Hyperparameters.**  `learningRate=0.1`, `nCycles=1000`,
  `regulatorRates=[0.001, 0.1]`, $\varepsilon = 10^{-10}$ — identical to
  the upstream TF1 reference.

**Questions to expect.**

- *"Is the output numerically equivalent to TF1?"* — Same math, same
  hyperparameters, same seed-handling, same numerical-stability
  $\varepsilon$.  Modulo floating-point op-ordering differences, yes.
  We cannot empirically verify against a TF1 reference run on this
  hardware because TF 1.15 doesn't install on it — which is the same
  constraint that motivated the port in the first place.
- *"Why is it called 'backpropagation-based' if there is no neural
  network?"* — The upstream name is misleading.  It is gradient descent
  on the kinematic skeleton's variables (bone lengths, joint angles, root
  positions), implemented as automatic-differentiation through the
  forward pass.  No neural network is involved.

---

### 4. [`scripts/compute_baselines.py`](../scripts/compute_baselines.py) — the headline finding

**Why it's important.**  This is where the report's headline empirical
claim is produced — "every deterministic baseline beats every trained
model on every split".

**What to walk through.**

- **`load_split()`** — parses `<mode>.skels` back to `(T, 150)` arrays.
- **Nearest-Neighbor lookup.**  `nn_lookup` is a `dict[text → first
  matching train pose]`.  For each test text, return that pose
  verbatim.  Deterministic; matches our split's lex-sort ordering.
- **Mean-Pose.**  Global per-coordinate mean across *all training-clip
  frames*, replicated to the median training-clip length.  Same prediction
  for every input.  This is the "ignore the text entirely" floor.
- **Random-Clip.**  Seeded RNG, pick a random training clip per test
  sample.  Reproducible.
- **DTW** is delegated to [`mosl/train/eval.py:dtw_distance`](../mosl/train/eval.py) — the
  same metric the trained-model evaluation uses, so the comparison is
  apples-to-apples.

**Questions to expect.**

- *"Why does the trained model lose to Mean-Pose?"* — Regression-to-the-
  mean under MSE-trained autoregressive regression on a 1:1
  sign-to-clip dataset.  See report §6.3 and [`RESULTS.md`](RESULTS.md).
- *"Is the cross-signer floor really 0.78?"* — That's the
  Nearest-Neighbor mean DTW on test.  It's the lowest DTW achievable by
  any retrieval method on our data; it reflects irreducible variance
  across signers' takes of the same word.

---

### 5. [`mosl/train/train.py`](../mosl/train/train.py) — training orchestration

**Why it's important.**  This is where the paper's training recipe lives.
The supervisor will want to confirm Adam + Noam schedule + early stopping
are correctly wired.

**What to walk through.**

- **`Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9)`** — Vaswani 2017 exact.
- **`NoamLR(d_model=d, warmup_steps=4000)`** — Vaswani 2017 exact
  (implementation in [`mosl/train/scheduler.py`](../mosl/train/scheduler.py)).  Note that
  the base LR in the optimiser is 1.0 — Noam multiplies it by
  $d_{\text{model}}^{-0.5} \cdot \min(s^{-0.5}, s \cdot \text{warmup}^{-1.5})$.
- **Per-batch loop.**  Forward → `signllm_step_loss` → `loss.backward()`
  → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()`.  Standard.
- **Per-epoch eval.**  Teacher-forced dev MSE (fast); the slower
  autoregressive DTW evaluation is deferred to
  [`scripts/evaluate_runs.py`](../scripts/evaluate_runs.py) and only runs at the end of training.
- **Checkpointing.**  `best.pt` on dev-MSE improvement, `last.pt` every
  epoch.  Both saved with the full config dict (so we can reload from any
  checkpoint deterministically).
- **Early stopping.**  20-epoch patience on the dev pose MSE.

**Questions to expect.**

- *"Why Adam, not AdamW?"* — Vaswani 2017 used Adam.  The SignLLM paper is
  silent on the choice, so we defaulted to the canonical recipe.
- *"Why warmup=4000 when total training is ~10 k steps?"* — Literal
  Vaswani default for items the SignLLM paper does not specify.  Could
  have been smaller for our budget; documented as a paper-faithful choice
  rather than a tuned one.

---

## Tier 2 — should know cold, but not walk through line-by-line

If the supervisor opens any of these, be ready to explain in 2–3 sentences
each.

- [`mosl/data/dataset.py`](../mosl/data/dataset.py) — PyTorch `Dataset` reading the
  `.skels` files; `mosl_collate` pads variable-length sequences and emits
  attention masks.
- [`mosl/text/tokenizer.py`](../mosl/text/tokenizer.py) — word-level Arabic tokenizer.
  Multi-word labels (377 of them) are kept as atomic tokens.  NFC is the
  primary key — diacritic stripping would collapse minimal pairs.
- [`mosl/train/eval.py`](../mosl/train/eval.py) — DTW computation (hand-rolled DP using
  `scipy.spatial.distance.cdist` for the pairwise frame matrix) + teacher-
  forced and autoregressive eval drivers.
- [`scripts/evaluate_runs.py`](../scripts/evaluate_runs.py) — driver that loads each
  checkpoint and runs both eval modes; produces `runs/evaluation.json`.

---

## Tier 3 — describe in one sentence each if asked

- [`mosl/pose/extract_dataset.py`](../mosl/pose/extract_dataset.py) — OpenPose
  body+hand keypoint extraction over all 2,216 clips; output is NPZ per
  clip.
- [`mosl/pose/export_openpose_json.py`](../mosl/pose/export_openpose_json.py) — NPZ →
  per-frame OpenPose JSON; required file-naming
  (`<stem>_<frame>_keypoints.json`) matches Prompt2Sign's regex.
- [`scripts/setup_p2s_pipeline.py`](../scripts/setup_p2s_pipeline.py) — split-aware
  hardlinking + writes `<mode>.files` and `<mode>.text` from
  `data/splits.csv`.
- [`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh) — orchestrates the
  entire preprocessing pipeline over all three splits inside a single
  docker container.
- [`mosl/data/build_labels.py`](../mosl/data/build_labels.py),
  [`split.py`](../mosl/data/split.py),
  [`stats.py`](../mosl/data/stats.py) — dataset metadata building.

---

## Quick reference: "show me X"

| If asked to show... | Open... |
|---|---|
| The model architecture | [`mosl/model/signllm.py`](../mosl/model/signllm.py) + report Fig. 4.1 |
| How RL Loss differs from MSE | [`mosl/train/losses.py`](../mosl/train/losses.py) — they're identical at per-step level |
| Where PLC's sample skipping happens | [`mosl/train/losses.py:plc_weights()`](../mosl/train/losses.py) |
| The TF→PyTorch port | [`third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py`](../third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py) |
| How the headline finding was computed | [`scripts/compute_baselines.py`](../scripts/compute_baselines.py) + `runs/baselines.json` |
| The full preprocessing pipeline | [`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh) + report Fig. 3.2 |
| Hyperparameter rationale | [`docs/DECISIONS.md`](DECISIONS.md) — every choice with date + reasoning + alternatives |
| The dataset's long-tail finding | [`docs/STATS.md`](STATS.md) + report Fig. 3.1 |
| Final results and analysis | [`docs/RESULTS.md`](RESULTS.md) + report Ch. 6 |

---

## The single hardest question — and how to answer

**"How do you know your reimplementation is faithful to the paper?"**

The honest answer (which is the right one to give, not a hedge):

> We cannot be perfectly faithful, because the paper underspecifies most
> architectural and training hyperparameters: hidden dimension, attention
> heads, feed-forward dimension, dropout, optimiser, learning rate, batch
> size, training budget, and PLC's $\eta$ are all silent.
>
> What we *can* commit to is:
>
> 1. **Every paper-specified item is implemented exactly.**  2-layer
>    encoder + 2-layer decoder, ~40 M parameter target, MSE / RL = $-$MSE /
>    PLC sampling formula, MLSF and Prompt2LangGloss modes (we use MLSF
>    only since we operate in a single-language setting).
> 2. **Every paper-silent item is filled in using the canonical Vaswani 2017
>    default** — the most defensible inheritance for any unspecified
>    transformer hyperparameter, since SignLLM is itself a transformer
>    architecture.
> 3. **Every decision is logged in `docs/DECISIONS.md`** with date,
>    reasoning, and alternatives considered.
>
> This is what "strict paper fidelity" means in practice.  The negative
> empirical finding — that the trained models lose to deterministic
> retrieval baselines — is therefore a property of the paper-as-published
> *on this data*, not an artefact of choices we made to fill in the gaps.

If the supervisor pushes harder ("how do I know the gap is the paper's fault
and not the Vaswani defaults?"), the honest follow-up is that we did not
re-tune any of the Vaswani defaults — so we cannot rule out that a
carefully-tuned variant would beat the baselines on MoSL.  That is
explicit future work in §7.4 of the report.

---

## Files to have open before the meeting

Have these files already open in tabs so you can switch fast:

1. [`mosl/model/signllm.py`](../mosl/model/signllm.py)
2. [`mosl/train/losses.py`](../mosl/train/losses.py)
3. [`third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py`](../third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py)
4. [`scripts/compute_baselines.py`](../scripts/compute_baselines.py)
5. [`mosl/train/train.py`](../mosl/train/train.py)
6. [`docs/DECISIONS.md`](DECISIONS.md)
7. [`docs/RESULTS.md`](RESULTS.md)
8. The compiled `report/main.pdf` (kept locally; not on GitHub)

And know where to find these if asked:

- `runs/baselines.json` and `runs/evaluation.json` — raw numbers behind the
  headline finding.
- `data/labels.csv`, `data/splits.csv` — the dataset metadata.
