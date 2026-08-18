"""
resume_fast.py — continue the reference run's snapshot in the compiled
implementation, as a trajectory-level check on the port.

validate_fastvolley.py shows the two implementations produce identical games.
This shows they produce the same *training dynamics*: both continuations start
from the identical committed population (results/ga_selfplay/snapshot.npz) and
are compared over the games that follow.

    python resume_fast.py --tournaments 213000 --seeds 3
"""

import argparse
import json
import os

import numpy as np

import fastvolley as fv
import fastvolley_kernels as fk
from run_experiments import SWEEP_EPISODES, SWEEP_SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="results/ga_selfplay/snapshot.npz")
    ap.add_argument("--tournaments", type=int, default=None,
                    help="default: however many are left to reach 500,000")
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent continuations of the same population")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", default="results/analysis/resume_fast.json")
    args = ap.parse_args()

    snap = np.load(args.snapshot)
    start = int(snap["tournament"])
    n = args.tournaments if args.tournaments else max(0, 500_000 - start)
    print(f"snapshot at tournament {start:,}; continuing {n:,} games "
          f"x {args.seeds} seeds", flush=True)

    w, b = fv.baseline_arrays()
    out = {"snapshot_tournament": start, "tournaments": n,
           "episodes": SWEEP_EPISODES, "seed": SWEEP_SEED, "runs": {}}
    for s in range(args.seeds):
        pop = np.ascontiguousarray(snap["population"].astype(np.float64))
        streak = np.ascontiguousarray(snap["winning_streak"].astype(np.int64))
        champs, streaks, meanlen = fk.run_ga_resume(
            pop, streak, 900 + s, n, args.sigma, args.save_every, w, b)
        scores = []
        for i in range(len(champs)):
            sc, _ = fv.eval_vs_baseline(np.ascontiguousarray(champs[i]),
                                        SWEEP_EPISODES, SWEEP_SEED, w, b, False)
            scores.append(float(sc.mean()))
        t = [start + (i + 1) * args.save_every for i in range(len(champs))]
        out["runs"][f"resume_s{900+s}"] = {
            "tournament": t, "mean_score": scores,
            "streaks": streaks.tolist(), "train_meanlen": meanlen.tolist()}
        print(f"  seed {900+s}: final {scores[-1]:+.2f}, "
              f"above parity {sum(x > 0 for x in scores)}/{len(scores)}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
