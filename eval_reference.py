"""
eval_reference.py — re-score the reference run under the matrix protocol.

results/ga_selfplay/ is the run trained on the real slimevolleygym environment.
Its historical curve (results/ga_selfplay/eval_curve.jsonl) used 40 episodes at
seed 721, which is not the protocol the matrix runs use. To place the reference
run on the same axes as the matrix, every one of its champion checkpoints is
re-scored here with the validated compiled evaluator at the matrix's protocol
(200 episodes, seed 20260901), plus a 1000-episode held-out re-score of the
final and peak champions at seed 20260902.

The compiled evaluator is bit-identical to the reference one (see
validate_fastvolley.py), so this is a change of protocol, not of measurement.

    python eval_reference.py
"""

import argparse
import glob
import json
import os

import numpy as np

import fastvolley as fv
from run_experiments import SELECT_EPISODES, SELECT_SEED, SWEEP_EPISODES, SWEEP_SEED


def load_champion(path):
    with open(path) as f:
        params, streak = json.load(f)
    return np.array(params, dtype=np.float64), int(streak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ga_selfplay")
    ap.add_argument("--out", default="results/analysis/reference_curve.json")
    ap.add_argument("--episodes", type=int, default=SWEEP_EPISODES)
    ap.add_argument("--seed", type=int, default=SWEEP_SEED)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "ga_*.json")))
    print(f"{len(files)} checkpoints, {args.episodes} episodes each", flush=True)
    w, b = fv.baseline_arrays()

    ts, mean, sd, win, tie, length, streaks = [], [], [], [], [], [], []
    champs = []
    for i, path in enumerate(files):
        params, streak = load_champion(path)
        sc, ln = fv.eval_vs_baseline(params, args.episodes, args.seed, w, b, False)
        ts.append(int(os.path.basename(path)[3:-5]))
        mean.append(float(sc.mean()))
        sd.append(float(sc.std()))
        win.append(float((sc > 0).mean()))
        tie.append(float((sc == 0).mean()))
        length.append(float(ln.mean()))
        streaks.append(streak)
        champs.append(params)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}  t={ts[-1]}  {mean[-1]:+.2f}", flush=True)

    mean = np.array(mean)
    idx_final = len(mean) - 1
    idx_peak = int(np.argmax(mean))
    hold = {}
    for tag, idx in (("final", idx_final), ("peak", idx_peak)):
        sc, ln = fv.eval_vs_baseline(champs[idx], SELECT_EPISODES, SELECT_SEED,
                                     w, b, False)
        hold[tag] = {"tournament": ts[idx], "mean": float(sc.mean()),
                     "sd": float(sc.std()),
                     "sem": float(sc.std() / np.sqrt(len(sc))),
                     "win": float((sc > 0).mean()), "tie": float((sc == 0).mean()),
                     "loss": float((sc < 0).mean()),
                     "meanlen": float(ln.mean()), "episodes": SELECT_EPISODES,
                     "seed": SELECT_SEED}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"tournament": ts, "mean_score": mean.tolist(), "std_score": sd,
               "win_rate": win, "tie_rate": tie, "meanlen": length,
               "streak": streaks, "episodes": args.episodes, "seed": args.seed,
               "holdout": hold}, open(args.out, "w"))

    print(f"\nfinal  t={hold['final']['tournament']}: "
          f"{hold['final']['mean']:+.3f} +/- {hold['final']['sd']:.3f} "
          f"(SEM {hold['final']['sem']:.3f})")
    print(f"peak   t={hold['peak']['tournament']}: "
          f"{hold['peak']['mean']:+.3f} +/- {hold['peak']['sd']:.3f} "
          f"(SEM {hold['peak']['sem']:.3f})")
    print(f"above parity: {int((mean > 0).sum())}/{len(mean)} checkpoints")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
