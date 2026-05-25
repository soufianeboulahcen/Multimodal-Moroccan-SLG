# SignLLM model — implementation spec

A faithful translation of arXiv:2405.10718 Sections 4.1–4.4 into a concrete
PyTorch implementation plan, plus the gaps the paper leaves open that we have
to fill ourselves.

This document is the source of truth for Phase 3 architecture and training
decisions.  When the paper conflicts with itself or a published reference
implementation, we follow what we agreed in [DECISIONS.md](DECISIONS.md).

---

## 1. What we're implementing

**Task.** Text-to-Pose (T2P) sign-language production.  Given an input text
sequence (a single Arabic word for our isolated-MoSL data, but the
formulation is general), generate the corresponding 3D skeleton sequence in
the SignLLM-format compressed pose representation our Phase 2 pipeline
produces.

**Inputs:**
- `<mode>.text` — one text string per clip (Arabic word for MoSL).
- `<mode>.files` — line-aligned clip identifiers (for diagnostics only; not
  used by the model).

**Targets:**
- `<mode>.skels` — one space-separated sequence per clip, structured as
  `(T frames) × (150 coords + 1 time marker)` flattened.
  - 150 coords = 50 joints × 3 (x, y, z), where the 50 joints are
    `idxsPose=[0..7]` (8 upper-body) + 21 left hand + 21 right hand.
  - Time marker per frame = `i / T`, ∈ (0, 1].

**Output (at inference):** same shape as targets — a generated pose sequence
of arbitrary length T, decoded autoregressively.

---

## 2. Architecture

### 2.1 The base seq2seq (paper §4.1)

The paper distills sign-language production to a standard seq2seq problem:

```
f_u    = Enc_input2output(x_u  | x_{1:U})       # eq (1)
p_{w+1} = Dec_input2output(p_w | p_{1:w-1}, f_{1:U})  # eq (2)
```

A vanilla encoder-decoder transformer.  `x_{1:U}` is the input text token
sequence; `p_{1:W}` is the output pose-frame sequence.  Decoder is
autoregressive in the frame dimension.

**Module list (Base size, "SignLLM-1x40M-Base"):**

| Module | Layers | Notes |
|---|---:|---|
| Text embedding | — | Token embedding + positional encoding over input vocab |
| Text encoder | 2 | Standard transformer encoder (paper §5: "encoder and decoder of our model versions [...] both have two layers") |
| Pose-frame encoder | — | Linear projection of one frame's 151 coords into model dim, plus learned positional encoding over frame index |
| Pose decoder | 2 | Standard transformer decoder.  Cross-attention over text encoder outputs |
| Pose-frame head | — | Linear projection back from model dim to 151 floats per frame |
| EOS head | — | Scalar predicting end-of-sequence per decoded frame |

**Variable to fix (paper underspecifies — see §6 Open questions):**
- Model dim `d_model`.
- Number of attention heads `h`.
- Feed-forward dim `d_ff`.
- Dropout rate.

The paper says "40M parameters per language".  For a 2-layer enc + 2-layer dec
transformer to land near 40M:
- `d_model=512, d_ff=2048, h=8` ⇒ ≈ 25 M params for the transformer alone, plus
  embeddings — comes in around 30–40 M depending on vocab size.  This is the
  natural fit and matches the convention of comparable T2P papers.

### 2.2 Multilingual modes (paper §4.3)

The paper defines two ways to extend a single-language model to many languages.

#### MLSF — Multi-Language Switching Framework
Per-language **separate parameter sets** for the encoder-decoder pair.  At
runtime, an "Enc_L / Dec_L" pair is selected by the user-specified language L.
For our project this collapses to:

> A single encoder-decoder for MoSL.  No code-paths for other languages.

If we ever extend to other sign languages, we add a new `(Enc, Dec)` group;
we don't modify the existing one.  Parameter count ≈ N_languages × 40 M.

#### Prompt2LangGloss
A **single shared parameter set** but a richer input encoding.  Each gloss
token gets a language-attribute prefix (e.g. `<ASL_xxx>`).  The translation is

```
lg_l = Enc_t2lg(x_u | x_{1:U})  →  Dec_lg2p(lg_w | lg_{1:w-1}, f_{1:U})
```

For our project:
> Vocabulary entries get a `<MoSL_xxx>` prefix at training and inference time.

These two modes are complementary; the paper trains both.  For MoSL alone, MLSF
and Prompt2LangGloss are equivalent in expressive power (single language ⇒
single param set ⇒ single pseudo-language token).  We pick **MLSF** for
simplicity since the prefixing buys us nothing in a single-language setting,
**unless** the user chooses to anticipate future multilingual extension —
see §6.

### 2.3 Tokenization

#### Text side
Source vocabulary built from `data/labels.csv`'s `word_arabic` column
(NFC-normalized Arabic).  1,631 unique signs across the dataset.  Two design
options for tokenization:

- **(a) Word-level** — one token per sign label.  Vocab = 1,631 + 4 specials
  (`<pad>, <bos>, <eos>, <unk>`).  Trivial, exact-match at inference, useless
  for unseen words.
- **(b) Character-level** — one token per Arabic letter + diacritic.  Vocab
  ≈ 60.  Generalizes to unseen words by composing characters.

Since 74% of signs occur in only one clip, (a) collapses to a 1-NN lookup —
the model just learns a vocab-indexed table of pose sequences.  (b) lets the
model decompose words into their constituent letters and learn shared
representations.  Neither is "better" — depends on what we want to test.

**Recommendation:** Start with **(a) word-level**, since the deliverable is
isolated-sign production and the paper itself uses word/gloss-level vocab.
Document this as a known limitation against future generalization.

#### Pose side
Pose frames are continuous-valued — there is no tokenization.  Frames are fed
through a linear projection to `d_model`-dim and decoded back via a linear
head.

---

## 3. Training objective

### 3.1 Base loss (paper implicit)

Per-frame MSE between predicted and target pose vectors:

```
L_MSE = (1/(T·D)) Σ_t Σ_d (y_{t,d} - ŷ_{t,d})²
```

with D = 151 (150 coords + 1 time marker).  Optionally we mask the time-marker
dimension since it's a derivable function of frame position.

### 3.2 RL Loss (paper §4.4)

The paper reformulates supervised learning as cumulative-reward maximization.
Per-step reward:

```
r_t = −(1/N) Σ_i (y_{t,i} - ŷ_{t,i})²       # eq from §4.4 ¶3
```

(negative MSE).  Then

```
θ* = argmax_θ E_θ[Σ_t r_t] = argmin_θ E_θ[Σ_t L(y_t, M(x_t))]   # eq (3)
```

In effect, **at the per-step level, RL Loss is identical to MSE** — the
relabeling matters only at the **batch-prioritization** level, where we feed
it into the Priority Learning Channel.

### 3.3 Priority Learning Channel (paper §4.4)

For each (batch, sample) pair `(j, i)` compute reward `r(i)`.  Convert to
per-sample sampling probabilities

```
P(i) = r(i)^η / Σ_{j∈S} r(j)^η
```

with η a tunable prioritization-intensity hyperparameter and S the dataset.
Then resample the batch from `P(i)`.  "If the reward is less than 50%, skip
the batch" (paper §4.4) — interpret as: if `r(i)` is below a threshold
(literal interpretation: 0.5 × max_reward, since rewards are negative this
needs care; charitable interpretation: a fixed threshold τ, configurable),
skip optimizing on that sample.

The paper's math is **loose here**.  `r(i)^η` is undefined for negative
rewards if η is non-integer.  In practice the published behaviour is most
plausibly:

```
score(i) = max(−r(i), ε)   # positive, smaller is better
P(i)    ∝ score(i)^(−η)   # higher prob for lower-error (= higher-reward) samples
```

We document this explicitly when implementing — see [DECISIONS.md](DECISIONS.md)
when the time comes.

### 3.4 Loss combination

Per the ablation in paper Table 5, results improve cumulatively:

```
Base + Normal MSE Loss          (worst)
Base + RL Loss
Base + RL Loss + PLC            (best on test)
Base + MSE + Prompt2LangGloss
Base + MSE + MLSF               (best on dev)
```

For us with a single language, MLSF and Prompt2LangGloss collapse, so we
implement and compare:

1. Base + MSE (vanilla baseline)
2. Base + RL Loss (= MSE relabel + PLC framework hook, but no PLC sampling)
3. Base + RL Loss + PLC

---

## 4. Inference

Autoregressive decoding:

```
for t in range(T_max):
    pred_t, eos_t = decoder(text_features, prev_frames=p_{1:t-1})
    p_t = pred_t
    if eos_t > threshold: break
```

The paper does not specify how T (target sequence length) is chosen at
inference.  Three workable approaches:

1. **Predict T from the input text** with a small MLP head; train it on the
   ground-truth T values.  Then unroll the decoder exactly T times.
2. **EOS prediction head** at each frame.  Stop when the head fires.
3. **Fixed maximum + truncation** to first low-motion run (rejected — too
   hacky for a published method).

The paper hints at (1) since it predicts a `pose video` from text; the
implementation choice is ours and we should pick one explicitly.

---

## 5. Evaluation

The paper reports BLEU-{1,2,3,4}, ROUGE, and DTW.

### Metrics that work directly for MoSL

- **DTW on the predicted vs. target pose sequence.**  Direct measure of pose
  reconstruction quality.  Works on any sequence pair; no extra model needed.
- **Token / exact-match accuracy** on the target text after back-translation.
  Since our targets are isolated single Arabic words, "BLEU-1" reduces to
  exact-match accuracy in our case.

### Metrics that need an extra model

- **BLEU-n via back-translation** requires a sign→text translator for Arabic
  MoSL.  We don't have one.  Two options:
  - (a) Use exact-match accuracy as a substitute (since signs are isolated,
    the back-translation problem reduces to recognition).  We'd need an
    MoSL recognizer model — out of scope for Phase 3.
  - (b) Skip BLEU and report DTW + exact-match-on-text.

**Recommended for our PFE:**  DTW + exact-match (after a separately-trained
recognizer if time permits).  Document the deviation from the paper's
back-translation BLEU as a known evaluation gap.

---

## 6. Open questions for the user

These genuinely matter for results and the paper underspecifies them.  None
should be set silently by the implementer.

| # | Question | Default I'd suggest | Why it matters |
|---|---|---|---|
| 1 | `d_model` (model hidden dim) | 256 or 384 | Paper says "40 M per language" but doesn't pin dims.  Smaller dim → more regularization, better fit for our 1,674-clip train set.  256 is conservative for low-data; 384 matches typical literature. |
| 2 | Number of attention heads | 8 | Common default; divisible into both 256 and 384. |
| 3 | Feed-forward dim `d_ff` | `4 × d_model` | Standard transformer scaling. |
| 4 | Dropout | 0.1 | Standard.  Could go higher (0.3) given our small dataset. |
| 5 | Tokenization (text side) | (a) word-level | See §2.3.  Word-level is the paper's setting.  Char-level would be a "better than paper" deviation — explicitly excluded by the strict-fidelity rule. |
| 6 | Sequence length prediction | (1) MLP head from text | See §4.  EOS-head is also defensible. |
| 7 | Loss combination to evaluate | All three (MSE, RL, RL+PLC) for ablation | Reproduces the paper's ablation Table 5 in MoSL form.  Costs ~3× training time. |
| 8 | Optimizer + LR | AdamW, lr=1e-4, warmup 4k steps | Paper doesn't specify.  Standard transformer recipe. |
| 9 | Batch size | 32 sequences | Paper doesn't specify.  Memory-bound; safe value for our GB10 with up to ~250-frame clips. |
| 10 | Training budget | 200 epochs (~10 k steps with 1,674 train clips and bs 32) | Paper doesn't specify.  Open to monitoring DTW on dev and stopping when no improvement. |
| 11 | "Reward < 50%" PLC threshold interpretation | Literal: τ = 0.5 × p99(r) on a warmup minibatch | Paper text is ambiguous; we pick the most defensible reading and document it. |

These are the items that need ~5 minutes of conversation tomorrow before any
model code gets written.

---

## 7. Implementation order (Phase 3 sub-tasks)

1. **Tokenizer** (Phase 3c) — word-level vocab from `labels.csv`, `<pad> <bos>
   <eos> <unk>` specials.  Output: `mosl/text/tokenizer.py` + a saved
   `data/processed/vocab.json`.
2. **Dataloader** — reads `<mode>.{skels,text}`, returns
   `(text_token_ids, pose_frames, frame_times)` triplets with proper padding.
3. **Model** (Phase 3a) — MLSF transformer per §2.  Single language for now,
   but with a `language_id` arg so multi-language is a pure scale-up later.
4. **Loss/training** (Phase 3b) — MSE first (validate end-to-end), then add
   RL Loss + PLC.
5. **Training script** (Phase 3d) — config, checkpointing, dev-set DTW
   tracking.

We should resolve §6 questions before starting (3) and beyond.  (1) and (2)
are deterministic and don't need decisions.
