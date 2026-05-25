"""Three reference baselines to put our model's DTW numbers in context.

   1. **Nearest-Neighbor (same-text)** — for each test/dev text, retrieve the
      first training clip with the same Arabic word and use *that clip's full
      pose sequence* as the prediction.  This is the "deterministic dictionary
      lookup" baseline: if our model can't beat this, it has learned nothing
      beyond memorisation.

   2. **Mean-Pose** — output the global per-coordinate mean across all
      training-clip frames, replicated to the median training-clip length.
      Same prediction for every input.  This is the "ignore the text" floor.

   3. **Random-Clip** — for each test/dev clip, pick a random training clip
      (with seed=0).  Sanity-check baseline.

For each baseline + split (dev, test) we report mean and median path-length-
normalised DTW, exactly the same metric the model is evaluated on
(`mosl/train/eval.py:dtw_distance`).  Output goes to
`runs/baselines.json` and a comparison table is printed to stdout.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosl.train.eval import dtw_distance


def load_split(mode: str) -> list[tuple[str, np.ndarray]]:
    """Return [(text, pose_array)] for the given split."""
    base = ROOT / "third_party" / "Prompt2Sign" / "tools" / "2D_to_3D" / "final_data"
    text_path = base / f"{mode}.text"
    skels_path = base / f"{mode}.skels"
    with open(text_path, encoding="utf-8") as f:
        texts = [ln.rstrip("\n") for ln in f if ln.strip()]
    poses: list[np.ndarray] = []
    with open(skels_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            floats = np.fromstring(line, sep=" ", dtype=np.float32)
            frames = floats.reshape(-1, 151)
            poses.append(frames[:, :150])
    if len(texts) != len(poses):
        raise ValueError(
            f"length mismatch in {mode}: text={len(texts)} skels={len(poses)}"
        )
    return list(zip(texts, poses))


def main() -> int:
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    print(f"[baselines] train={len(train)}  dev={len(dev)}  test={len(test)}")

    # Nearest-neighbor lookup: text -> first matching train pose (lex order
    # of the training data, which is the same order our split.py imposes).
    nn_lookup: dict[str, np.ndarray] = {}
    for text, pose in train:
        nn_lookup.setdefault(text, pose)
    print(f"[baselines] NN lookup covers {len(nn_lookup)} unique texts "
          f"(of {len({t for t, _ in train})} total in train)")

    # Mean-pose: global mean coord vector across all training-clip *frames*,
    # weighted equally per frame (so longer clips contribute more frames —
    # this is what a model with no knowledge would default to).
    all_train_frames = np.concatenate([p for _, p in train], axis=0)  # (Σ T_i, 150)
    overall_mean = all_train_frames.mean(axis=0)                       # (150,)
    median_T = int(np.median([p.shape[0] for _, p in train]))
    mean_pose = np.tile(overall_mean[None, :], (median_T, 1))          # (median_T, 150)
    print(f"[baselines] mean-pose: {mean_pose.shape}, replicated to median T={median_T}")

    # Random-clip: deterministic seeded RNG so the comparison is reproducible.
    rng = np.random.default_rng(0)

    results: dict[str, dict] = {}
    for split_name, split in [("dev", dev), ("test", test)]:
        nn_d, mean_d, rand_d = [], [], []
        oov_count = 0
        for text, target in split:
            if text in nn_lookup:
                nn_pose = nn_lookup[text]
            else:
                # Should never trigger given our deterministic split design,
                # but defensive: fall back to a random training clip.
                oov_count += 1
                nn_pose = train[rng.integers(0, len(train))][1]
            nn_d.append(dtw_distance(nn_pose, target))
            mean_d.append(dtw_distance(mean_pose, target))
            rand_clip = train[rng.integers(0, len(train))][1]
            rand_d.append(dtw_distance(rand_clip, target))

        results[split_name] = {
            "n_clips": len(split),
            "n_oov_in_nn_lookup": oov_count,
            "nn_dtw_mean":   float(np.mean(nn_d)),
            "nn_dtw_median": float(np.median(nn_d)),
            "mean_dtw_mean":   float(np.mean(mean_d)),
            "mean_dtw_median": float(np.median(mean_d)),
            "rand_dtw_mean":   float(np.mean(rand_d)),
            "rand_dtw_median": float(np.median(rand_d)),
        }
        print(f"[baselines] {split_name}: oov={oov_count}/{len(split)} done")

    # Persist
    out_path = ROOT / "runs" / "baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[baselines] wrote {out_path}")

    # Side-by-side table including the model's numbers from runs/evaluation.json
    eval_path = ROOT / "runs" / "evaluation.json"
    model_results: dict = {}
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            model_results = json.load(f)

    print("\n" + "=" * 86)
    print("Comparison: model vs baselines (path-length-normalised DTW, lower is better)")
    print("=" * 86)
    print(f"{'Method':<22}  {'split':<5}  {'DTW mean':>10}  {'DTW median':>11}  {'n':>5}")
    print("-" * 86)
    for split in ("dev", "test"):
        for run_name in ("baseline_mse", "rl", "rl_plc"):
            if run_name in model_results:
                r = model_results[run_name]["splits"].get(split)
                if r:
                    print(f"  model: {run_name:<14}  {split:<5}  "
                          f"{r['ar_dtw_mean']:>10.4f}  {r['ar_dtw_median']:>11.4f}  "
                          f"{r['n_clips']:>5}")
        b = results[split]
        print(f"  baseline: nearest-nbor  {split:<5}  "
              f"{b['nn_dtw_mean']:>10.4f}  {b['nn_dtw_median']:>11.4f}  "
              f"{b['n_clips']:>5}")
        print(f"  baseline: mean-pose      {split:<5}  "
              f"{b['mean_dtw_mean']:>10.4f}  {b['mean_dtw_median']:>11.4f}  "
              f"{b['n_clips']:>5}")
        print(f"  baseline: random-clip    {split:<5}  "
              f"{b['rand_dtw_mean']:>10.4f}  {b['rand_dtw_median']:>11.4f}  "
              f"{b['n_clips']:>5}")
        print("-" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
