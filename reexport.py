"""
reexport.py — can the champion-export rule simply be replaced?

The analysis shows Ha's winning-streak proxy exports an individual near the
median of its own population, and that the gap to the best member is about a
point per episode. That is a diagnosis. This script tests a fix.

For every population snapshot of every control run, three individuals are
identified and scored against the 2015 baseline:

  streak    the individual Ha's rule exports (longest winning lineage).
  internal  the best individual under a short round robin *inside the
            population* — a few hundred games, using no information the
            algorithm does not already have. This is a deployable fix.
  external  the best individual by score against the 2015 baseline. Not
            deployable — it selects on the yardstick — and included only as an
            upper bound on what any re-export rule could recover.

The question is how much of the streak-to-external gap the internal ranking
recovers, and whether the resulting series is also *steadier* — which is the
causal test of the claim that the export rule, not the population, is the main
source of volatility in a champion curve.

    python reexport.py --opponents 8 --episodes 100
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

import fastvolley as fv
import fastvolley_kernels as fk

RANK_SEED = 5150          # disjoint from training, evaluation and tournament seeds


def _job(args):
    path, n_opponents, episodes = args
    z = np.load(path)
    name = os.path.basename(path)[:-4]
    if z["pops"].size == 0:
        return name, None
    pops, streaks = z["pops"], z["pop_streaks"]
    pop_every = int(z["pop_every"][0])
    w, b = fv.baseline_arrays()

    rows = []
    for k in range(len(pops)):
        pop = np.ascontiguousarray(pops[k].astype(np.float64))
        # every member's true score, for the upper bound and the gaps
        ext, _ = fk.eval_population(pop, episodes, RANK_SEED, w, b)
        # the internal ranking: the pool plays itself
        internal = fk.internal_rank(pop, n_opponents, RANK_SEED, w, b)

        i_streak = int(np.argmax(streaks[k]))
        i_internal = int(np.argmax(internal))
        i_external = int(np.argmax(ext))
        order = np.argsort(-ext)
        rank_of = lambda i: int(np.where(order == i)[0][0])

        rows.append({
            "tournament": (k + 1) * pop_every,
            "pop_size": int(len(ext)),
            "games_spent_ranking": int((len(ext) * n_opponents) // 2),
            "streak_score": float(ext[i_streak]),
            "internal_score": float(ext[i_internal]),
            "external_score": float(ext[i_external]),
            "streak_rank": rank_of(i_streak),
            "internal_rank": rank_of(i_internal),
            "recovered": (float((ext[i_internal] - ext[i_streak]) /
                                (ext[i_external] - ext[i_streak]))
                          if ext[i_external] > ext[i_streak] else None),
        })
    return name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/matrix")
    ap.add_argument("--out", default="results/analysis/reexport.json")
    ap.add_argument("--opponents", type=int, default=8,
                    help="peers each individual meets in the internal round robin")
    ap.add_argument("--episodes", type=int, default=100,
                    help="episodes per individual against the 2015 baseline")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.matrix, "control_s*.npz")))
    if not paths:
        print("no control runs with population snapshots yet")
        return
    jobs = [(p, args.opponents, args.episodes) for p in paths]
    print(f"{len(jobs)} runs; internal ranking costs "
          f"{(128 * args.opponents) // 2} games per snapshot", flush=True)

    out = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, rows in pool.imap_unordered(_job, jobs):
            if rows is None:
                continue
            out[name] = rows
            rec = [r["recovered"] for r in rows if r["recovered"] is not None]
            print(f"  {name}: streak {np.mean([r['streak_score'] for r in rows]):+.2f}  "
                  f"internal {np.mean([r['internal_score'] for r in rows]):+.2f}  "
                  f"external {np.mean([r['external_score'] for r in rows]):+.2f}  "
                  f"gap recovered {100*np.mean(rec):.0f}%", flush=True)

    # volatility of each series, per run: the causal test
    summary = {}
    for key in ("streak_score", "internal_score", "external_score"):
        vols, means = [], []
        for rows in out.values():
            v = np.array([r[key] for r in rows])
            vols.append(float(np.abs(np.diff(v)).mean()))
            means.append(float(v.mean()))
        summary[key] = {"volatility_mean": float(np.mean(vols)),
                        "volatility_per_run": vols,
                        "level_mean": float(np.mean(means)),
                        "level_per_run": means}
    allrows = [r for rows in out.values() for r in rows]
    summary["gap_recovered_mean"] = float(np.mean(
        [r["recovered"] for r in allrows if r["recovered"] is not None]))
    summary["ranking_games_per_snapshot"] = allrows[0]["games_spent_ranking"]
    summary["episodes_per_individual"] = args.episodes
    summary["n_runs"] = len(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"per_run": out, "summary": summary}, open(args.out, "w"), indent=1)

    print("\nseries volatility (mean |Δ| between consecutive snapshots)")
    for key in ("streak_score", "internal_score", "external_score"):
        print(f"  {key:<16} level {summary[key]['level_mean']:+.2f}  "
              f"volatility {summary[key]['volatility_mean']:.2f}")
    print(f"\ninternal ranking recovers "
          f"{100*summary['gap_recovered_mean']:.0f}% of the streak-to-best gap "
          f"for {summary['ranking_games_per_snapshot']} games")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
