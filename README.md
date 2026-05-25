# Multimodal Moroccan Sign Language Generation

Strict paper-faithful PyTorch reimplementation of **SignLLM**
([Fang et al., 2024](https://arxiv.org/abs/2405.10718)) trained on the
**Moroccan Sign Language (MoSL) video dataset**
([Ben Zaid et al., 2026](https://data.mendeley.com/datasets/23phgyt3mt/1)).

> **Headline finding.**  On the MoSL isolated-word dataset, our strict
> paper-faithful SignLLM-Base reimplementation fails to outperform
> deterministic retrieval baselines.  All three loss configurations from the
> paper's Table 5 ablation (MSE / RL / RL+PLC) underperform Nearest-Neighbor
> (test DTW 0.78), Mean-Pose (0.87), and Random-Clip (0.96), with our best
> trained model at 1.04.  See [`docs/RESULTS.md`](docs/RESULTS.md) for the
> full result table, ablation analysis, and the regression-to-the-mean
> diagnostic that explains the gap.

---

## Repository layout

```
.
├── mosl/                       Python package (model, data, text, pose, train)
├── scripts/                    CLI entry points (training, evaluation, figures)
├── patches/                    Our patches to upstream third-party code
├── data/                       Dataset metadata (CSVs) and tokenizer vocab
├── docker/                     Container build (NGC PyTorch 26.04 base)
├── docs/                       Methodology + decisions + results + walkthroughs
├── predictions/                Sample model outputs (NPZ + animated GIF)
└── runs/                       Tracked: evaluation.json + baselines.json
                                (training logs and checkpoints not redistributed)
```

Each `docs/*.md` file is the source of truth for one aspect of the project:

| File | Contents |
|---|---|
| [`docs/STATS.md`](docs/STATS.md) | Dataset statistics: counts, long-tail, FPS, splits |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | End-to-end preprocessing pipeline |
| [`docs/POSE_EXTRACTION.md`](docs/POSE_EXTRACTION.md) | Phase 2 completion summary |
| [`docs/PROMPT2SIGN.md`](docs/PROMPT2SIGN.md) | What we adapted from upstream |
| [`docs/MODEL.md`](docs/MODEL.md) | Architecture spec + open questions |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Every non-trivial decision, dated |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Final results, baseline comparison, analysis |
| [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) | Guide to defending the code |

---

## Reproducing the work

### 1. System prerequisites

* **Docker** (with NVIDIA Container Toolkit if you want GPU training).  This
  project's compute path is exclusively Dockerised; the container image is
  built from NGC PyTorch 26.04.
* **Git, curl, unzip** (for fetching third-party code + weights).
* **A GPU**.  We used an NVIDIA GB10 (DGX Spark, aarch64, CUDA 13).  Any
  modern CUDA GPU will work; on Blackwell-class hardware you need the
  NGC container image specifically (native SM_120 kernels).

### 2. Clone + fetch third-party code

```bash
git clone https://github.com/omarait-mlouk/multimodal-moroccan-sign-language-generation.git
cd multimodal-moroccan-sign-language-generation
./scripts/setup_third_party.sh
```

The setup script clones `Hzzone/pytorch-openpose` and `SignLLM/Prompt2Sign`,
downloads OpenPose body+hand model weights (~350 MB), and applies our two
small patches (see [`patches/README.md`](patches/README.md)).

### 3. Get the MoSL dataset

Download `vedios-dataset.zip` from
[Mendeley 10.17632/23phgyt3mt.1](https://data.mendeley.com/datasets/23phgyt3mt/1)
and place it at the repo root, then extract:

```bash
python3 scripts/extract_dataset.py        # cp437→UTF-8 Arabic filename fix
```

### 4. Build the container

```bash
./docker/run.sh python --version          # triggers first-time build
```

### 5. Run the full preprocessing pipeline

```bash
docker/run.sh python -m mosl.data.build_labels
docker/run.sh python -m mosl.data.split
docker/run.sh bash scripts/run_full_pipeline.sh
```

This produces `third_party/Prompt2Sign/tools/2D_to_3D/final_data/<mode>.{skels,files,text}`
for `mode ∈ {train, dev, test}`.  Total time on a GB10: roughly 38 hours
across pose extraction and 2D→3D refinement.

### 6. Train the three ablation runs

```bash
docker/run.sh bash scripts/run_ablation.sh
```

Three sequential 200-epoch runs (max), early-stopped on dev pose MSE.  Total
~30 minutes on a GB10.

### 7. Evaluate and produce figures

```bash
docker/run.sh python scripts/evaluate_runs.py      # writes runs/evaluation.json
docker/run.sh python scripts/compute_baselines.py  # writes runs/baselines.json
docker/run.sh python scripts/make_figures.py       # writes report/figures/*.png locally
```

### 8. Build the report (local-only)

The LaTeX thesis report is not redistributed via this repository.  If you
have the local `report/` tree:

```bash
cd report
xelatex -shell-escape main && bibtex main && xelatex main && xelatex main
```

XeLaTeX is required for Arabic typesetting via polyglossia.  Amiri is the
Arabic font; it ships with TeX Live (`texlive-fonts-extra` on Debian/Ubuntu,
`tlmgr install amiri` on macOS).

---

## How to test the model interactively

After training, generate a pose sequence from any Arabic word:

```bash
docker/run.sh python scripts/predict.py --run baseline_mse \
    --clip-stem أَنَا --category Pronouns
docker/run.sh python scripts/visualize_pose.py predictions/baseline_mse_أَنَا.npz \
    --side-by-side --fps 12
```

The second command renders predicted-vs-target as an animated stick-figure
GIF.

---

## License

MIT — see [`LICENSE`](LICENSE).  The upstream third-party projects fetched
by `scripts/setup_third_party.sh` retain their own licenses.
