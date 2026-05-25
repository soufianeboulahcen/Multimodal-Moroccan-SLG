# Patches to upstream third-party code

This project depends on two upstream repositories that are **not redistributed**
in this repo (license + size reasons).  Reproducers fetch them via
`scripts/setup_third_party.sh`, which then copies the two files in this
directory over the upstream defaults.

| File | Replaces | Why |
|---|---|---|
| `pose3D.py` | `third_party/Prompt2Sign/tools/2D_to_3D/pose3D.py` | Full PyTorch port of the upstream TensorFlow 1.x `backpropagationBasedFiltering` function.  TF 1.15 cannot run on Python 3.12 / aarch64 / CUDA 13; the port is algorithmically identical (same variables, loss, hyperparameters, $\varepsilon$).  See [`docs/DECISIONS.md`](../docs/DECISIONS.md) entry dated 2026-05-08 for full rationale. |
| `pipeline_demo_02_h5totxt.py` | `third_party/Prompt2Sign/tools/2D_to_3D/pipeline_demo_02_h5totxt.py` | Replaces the file-level `import tensorflow as tf` (used only for the one-call `tf.keras.backend.clear_session()` cleanup) with a no-op stub class.  No algorithmic effect — keeps the container TensorFlow-free.  See [`docs/DECISIONS.md`](../docs/DECISIONS.md) entry dated 2026-05-09. |

Neither patch modifies upstream algorithmic behaviour; both preserve the
public function signatures so the rest of the Prompt2Sign pipeline calls
them unchanged.
