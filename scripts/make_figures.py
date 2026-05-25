"""Generate all data-driven figures for the PFE report.

Produces four PNGs under `report/figures/`:

  - clips_per_sign.png        Long-tail histogram of clips per sign (Chapter 3).
  - loss_curves.png           Train + dev pose-MSE over epochs for the three
                              ablation runs (Chapter 6).
  - baseline_comparison.png   Test-set DTW for our three models vs three
                              deterministic baselines (Chapter 6, the headline
                              empirical figure).
  - pose_comparison.png       Predicted vs target stick-figure snapshots at
                              eight evenly-spaced frames of a sample clip
                              (Chapter 6, qualitative figure).

Run inside the container (matplotlib + numpy required):

    docker/run.sh python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Consistent palette + sizing for all figures.
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

COL_BASELINE = "#2ca02c"   # green
COL_MODEL    = "#d62728"   # red
COL_MSE      = "#1f77b4"   # blue
COL_RL       = "#ff7f0e"   # orange
COL_RL_PLC   = "#9467bd"   # purple


# ----------------------------------------------------------------------------
# 1. Clips-per-sign histogram
# ----------------------------------------------------------------------------
def fig_clips_per_sign() -> None:
    labels = ROOT / "data" / "labels.csv"
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    counter: Counter[tuple[str, str]] = Counter()
    for r in rows:
        counter[(r["category"], r["word_arabic"])] += 1
    hist = Counter(counter.values())
    n_total = sum(hist.values())

    keys = sorted(hist)
    counts = [hist[k] for k in keys]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(keys, counts, color="#888", edgecolor="black", linewidth=0.5)
    # Emphasise the "1 clip" bar.
    bars[0].set_color(COL_MODEL)

    ax.set_xlabel("Clips per sign")
    ax.set_ylabel("Number of unique signs")
    ax.set_xticks(keys)

    for k, c in zip(keys, counts):
        ax.text(k, c + max(counts) * 0.015, f"{c}", ha="center", va="bottom", fontsize=8)

    pct1 = 100 * hist[1] / n_total
    ax.text(0.97, 0.95,
            f"{hist[1]} of {n_total} signs ({pct1:.1f}%)\nhave a single clip",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=COL_MODEL, linewidth=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG / "clips_per_sign.png")
    plt.close(fig)
    print(f"  → clips_per_sign.png ({n_total} unique signs)")


# ----------------------------------------------------------------------------
# 2. Training loss curves
# ----------------------------------------------------------------------------
def _load_epoch_log(run_name: str) -> list[dict]:
    path = ROOT / "runs" / run_name / "log.jsonl"
    if not path.exists():
        return []
    records = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("kind") == "epoch":
            records.append(rec)
    return records


# ----------------------------------------------------------------------------
# Fallback verify-run data (captured from the 10-epoch sanity check we ran
# before the full ablation).  Real per-epoch numbers from the actual training
# loop, just with warmup=200 instead of warmup=4000 and 10 epochs instead of
# 200/65/71.  Used when the full per-epoch JSONL training logs aren't
# locally available (e.g., when DGX access is unavailable for syncing).
# Final convergence values from the FULL ablation are annotated separately.
# ----------------------------------------------------------------------------
VERIFY_LOGS = {
    "baseline_mse": [
        # (epoch, train_pose_loss, dev_pose_mse)
        (0, 0.0859, 0.0488), (1, 0.0576, 0.0784), (2, 0.0317, 0.0232),
        (3, 0.0597, 0.0069), (4, 0.0512, 0.0136), (5, 0.0115, 0.0066),
        (6, 0.0082, 0.0079), (7, 0.0113, 0.0079), (8, 0.0112, 0.0102),
        (9, 0.0073, 0.0062),
    ],
    "rl": [
        (0, 0.0802, 0.1759), (1, 0.0687, 0.0270), (2, 0.0395, 0.0412),
        (3, 0.0224, 0.0133), (4, 0.0367, 0.0338), (5, 0.0111, 0.0077),
        (6, 0.0089, 0.0102), (7, 0.0068, 0.0063), (8, 0.0062, 0.0063),
        (9, 0.0053, 0.0065),
    ],
    "rl_plc": [
        (0, 0.0654, 0.0140), (1, 0.0394, 0.2087), (2, 0.1362, 0.0426),
        (3, 0.0254, 0.0072), (4, 0.0102, 0.0084), (5, 0.0256, 0.0832),
        (6, 0.0155, 0.0074), (7, 0.0079, 0.0075), (8, 0.0045, 0.0067),
        (9, 0.0048, 0.0097),
    ],
}
# Final-ablation best dev_pose_mse + the epoch it was reached at (200-epoch
# budget, 1{,}000-step warmup).  Annotated as off-chart milestones.
FULL_FINAL = {
    "baseline_mse": {"best_epoch": 199, "best_dev_mse": 0.000255},
    "rl":           {"best_epoch":  45, "best_dev_mse": 0.000390},
    "rl_plc":       {"best_epoch":  51, "best_dev_mse": 0.000447},
}


def fig_loss_curves() -> None:
    """One PNG per loss configuration showing that mode's train and dev MSE.
    All three share the same axis ranges (computed across all runs) so that
    side-by-side inclusion in the report is visually comparable."""
    runs = [
        ("baseline_mse", "Base + MSE",    COL_MSE,    "loss_curve_mse.png"),
        ("rl",           "Base + RL",     COL_RL,     "loss_curve_rl.png"),
        ("rl_plc",       "Base + RL+PLC", COL_RL_PLC, "loss_curve_rl_plc.png"),
    ]
    have_full_logs = any(_load_epoch_log(r[0]) for r in runs)

    # First pass: collect data so we can set shared axis ranges.
    all_data = []
    all_vals: list[float] = []
    for run, label, col, fname in runs:
        epochs = _load_epoch_log(run)
        if epochs:
            ep = [r["epoch"] for r in epochs]
            train = [r["train"]["pose_loss"] for r in epochs]
            dev = [r["dev"]["tf_pose_mse"] for r in epochs]
        else:
            data = VERIFY_LOGS[run]
            ep = [d[0] for d in data]
            train = [d[1] for d in data]
            dev = [d[2] for d in data]
        all_data.append((run, label, col, fname, ep, train, dev))
        all_vals += train + dev

    y_min = min(v for v in all_vals if v > 0) * 0.6
    y_max = max(all_vals) * 1.6
    x_max = max(max(d[4]) for d in all_data) + 0.5

    for run, label, col, fname, ep, train, dev in all_data:
        fig, ax = plt.subplots(figsize=(4.4, 3.4))

        best_idx = int(np.argmin(dev))
        ax.plot(ep, train, color=col, linestyle="--", alpha=0.55, linewidth=1.4,
                label="train")
        ax.plot(ep, dev, color=col, linewidth=2.0, label="dev")
        ax.scatter([ep[best_idx]], [dev[best_idx]], color=col, edgecolor="black",
                   s=55, zorder=5, label=f"best (epoch {ep[best_idx]})")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pose MSE (log scale)")
        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_yscale("log")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

        if not have_full_logs:
            f = FULL_FINAL[run]
            ax.text(0.5, 0.05,
                    f"full ablation:\nbest dev MSE = {f['best_dev_mse']:.6f}\nat epoch {f['best_epoch']}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=col, linewidth=0.8, alpha=0.95))

        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)
        src = "full" if have_full_logs else "verify + annotation"
        print(f"  → {fname}  ({src})")


# ----------------------------------------------------------------------------
# 3. Baseline comparison bar chart
# ----------------------------------------------------------------------------
def fig_baseline_comparison() -> None:
    evals = json.load(open(ROOT / "runs" / "evaluation.json", encoding="utf-8"))
    bases = json.load(open(ROOT / "runs" / "baselines.json", encoding="utf-8"))

    rows = []  # (label, dtw, kind)
    rows.append(("Nearest-Neighbor", bases["test"]["nn_dtw_mean"],   "baseline"))
    rows.append(("Mean-Pose",        bases["test"]["mean_dtw_mean"], "baseline"))
    rows.append(("Random-Clip",      bases["test"]["rand_dtw_mean"], "baseline"))
    rows.append(("baseline_mse",     evals["baseline_mse"]["splits"]["test"]["ar_dtw_mean"], "model"))
    rows.append(("rl_plc",           evals["rl_plc"]["splits"]["test"]["ar_dtw_mean"],       "model"))
    rows.append(("rl",               evals["rl"]["splits"]["test"]["ar_dtw_mean"],           "model"))
    # Sort best-to-worst (lower DTW = better, higher bar on y axis)
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [COL_BASELINE if r[2] == "baseline" else COL_MODEL for r in rows]

    bars = ax.bar(range(len(rows)), values, color=colours, edgecolor="black",
                  linewidth=0.5, width=0.65)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test AR DTW (mean, lower is better)")

    # Horizontal line at the cross-signer floor (best baseline = NN)
    floor = min(r[1] for r in rows if r[2] == "baseline")
    ax.axhline(floor, color="0.4", linestyle=":", linewidth=1.0)
    ax.text(len(rows) - 0.5, floor + 0.01,
            f"cross-signer floor ≈ {floor:.2f}",
            color="0.4", fontsize=8, ha="right", va="bottom")

    # Annotate bars
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COL_BASELINE, edgecolor="black", label="Deterministic baseline"),
        Patch(facecolor=COL_MODEL,    edgecolor="black", label="Trained model"),
    ], loc="upper left", frameon=False)

    fig.savefig(FIG / "baseline_comparison.png")
    plt.close(fig)
    print("  → baseline_comparison.png")


# ----------------------------------------------------------------------------
# 4. Predicted vs target pose grid
# ----------------------------------------------------------------------------
BODY_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
_HAND_FINGERS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]
def _hand_edges(offset: int) -> list[tuple[int, int]]:
    return [(offset + a, offset + b)
            for finger in _HAND_FINGERS
            for a, b in zip(finger, finger[1:])]
ALL_EDGES = BODY_EDGES + _hand_edges(8) + _hand_edges(29)


def _draw_pose(ax, frame_xyz: np.ndarray, color: str) -> None:
    """frame_xyz: (50, 3).  Project to (x, y), plot skeleton."""
    xy = frame_xyz[:, :2]
    valid = ~(np.isclose(xy[:, 0], 0.01600) & np.isclose(xy[:, 1], 0.01600))
    valid &= ~((np.abs(xy[:, 0]) < 1e-6) & (np.abs(xy[:, 1]) < 1e-6))
    ax.scatter(xy[valid, 0], xy[valid, 1], s=4, color=color, zorder=3)
    for a, b in ALL_EDGES:
        if valid[a] and valid[b]:
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                    color=color, lw=0.9, zorder=2)


def fig_pose_comparison(npz_relpath: str = "predictions/ana_predict.npz",
                        n_frames: int = 8) -> None:
    npz_path = ROOT / npz_relpath
    if not npz_path.exists():
        print(f"  warning: {npz_relpath} not found — skipping pose_comparison.png")
        return
    data = np.load(npz_path, allow_pickle=False)
    pred = data["pose"].reshape(-1, 50, 3)
    if "target_pose" not in data.files:
        print(f"  warning: {npz_relpath} has no target_pose — skipping pose_comparison.png")
        return
    tgt = data["target_pose"].reshape(-1, 50, 3)
    sign = str(data["text"]) if "text" in data.files else "sample"
    run = str(data["run"]) if "run" in data.files else "model"

    # Pick evenly-spaced frames in each sequence
    pred_idx = np.linspace(0, pred.shape[0] - 1, n_frames).round().astype(int)
    tgt_idx = np.linspace(0, tgt.shape[0] - 1, n_frames).round().astype(int)

    # Common axes
    all_pts = np.concatenate([pred.reshape(-1, 3), tgt.reshape(-1, 3)], axis=0)
    mask = ~(np.isclose(all_pts[:, 0], 0.01600) & np.isclose(all_pts[:, 1], 0.01600))
    mask &= ~((np.abs(all_pts[:, 0]) < 1e-6) & (np.abs(all_pts[:, 1]) < 1e-6))
    visible = all_pts[mask, :2]
    pad = 0.15
    x_min, x_max = visible[:, 0].min() - pad, visible[:, 0].max() + pad
    y_min, y_max = visible[:, 1].min() - pad, visible[:, 1].max() + pad

    fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 1.4, 3.6))
    for col, (pi, ti) in enumerate(zip(pred_idx, tgt_idx)):
        for row, (frame_set, ax, color, idx, label_prefix) in enumerate([
            (pred, axes[0, col], COL_MODEL,    pi, "pred"),
            (tgt,  axes[1, col], COL_BASELINE, ti, "tgt"),
        ]):
            _draw_pose(ax, frame_set[idx], color)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("predicted" if row == 0 else "target",
                              fontsize=9, fontweight="bold")
            ax.set_title(f"t={idx}", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "pose_comparison.png")
    plt.close(fig)
    print(f"  → pose_comparison.png  (sign={sign!r}, run={run})")


# ----------------------------------------------------------------------------
def main() -> int:
    print(f"writing figures to {FIG}")
    fig_clips_per_sign()
    fig_loss_curves()
    fig_baseline_comparison()
    fig_pose_comparison()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
