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
import stats_utils as su

RANK_SEED = 5150          # disjoint from training, evaluation and tournament seeds


def _job(args):
    path, opponent_counts, episodes = args
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

        i_streak = int(np.argmax(streaks[k]))
        i_external = int(np.argmax(ext))
        order = np.argsort(-ext)
        rank_of = lambda i: int(np.where(order == i)[0][0])

        row = {
            "tournament": (k + 1) * pop_every,
            "pop_size": int(len(ext)),
            "streak_score": float(ext[i_streak]),
            "external_score": float(ext[i_external]),
            "median_score": float(np.median(ext)),
            "streak_rank": rank_of(i_streak),
            "rho_streak_external": su.spearman(streaks[k], ext),
        }
        # sweep the internal round-robin budget: does more internal information
        # close more of the gap, or is internal fitness itself the limit?
        for n_opp in opponent_counts:
            internal = fk.internal_rank(pop, n_opp, RANK_SEED + n_opp, w, b)
            i_int = int(np.argmax(internal))
            row[f"internal_score_{n_opp}"] = float(ext[i_int])
            row[f"internal_rank_{n_opp}"] = rank_of(i_int)
            row[f"rho_internal_external_{n_opp}"] = su.spearman(internal, ext)
            row[f"games_{n_opp}"] = int((len(ext) * n_opp) // 2)
        rows.append(row)
    return name, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/matrix")
    ap.add_argument("--out", default="results/analysis/reexport.json")
    ap.add_argument("--opponents", default="4,8,16,32,64",
                    help="comma-separated round-robin sizes to sweep")
    ap.add_argument("--episodes", type=int, default=100,
                    help="episodes per individual against the 2015 baseline")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.matrix, "control_s*.npz")))
    if not paths:
        print("no control runs with population snapshots yet")
        return
    opps = [int(o) for o in str(args.opponents).split(",")]
    jobs = [(p, opps, args.episodes) for p in paths]
    print(f"{len(jobs)} runs; internal ranking sweep over {opps} peers "
          f"({[(128*o)//2 for o in opps]} games per snapshot)", flush=True)

    out = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, rows in pool.imap_unordered(_job, jobs):
            if rows is None:
                continue
            out[name] = rows
            print(f"  {name}: streak {np.mean([r['streak_score'] for r in rows]):+.2f}"
                  f"  external {np.mean([r['external_score'] for r in rows]):+.2f}",
                  flush=True)

    # volatility of each series, per run: the causal test
    allrows = [r for rows in out.values() for r in rows]
    opps = [int(o) for o in str(args.opponents).split(",")]
    series = (["streak_score"] + [f"internal_score_{o}" for o in opps]
              + ["external_score", "median_score"])
    summary = {}
    for key in series:
        vols, means = [], []
        for rows in out.values():
            v = np.array([r[key] for r in rows])
            vols.append(float(np.abs(np.diff(v)).mean()))
            means.append(float(v.mean()))
        summary[key] = {"volatility_mean": float(np.mean(vols)),
                        "volatility_per_run": vols,
                        "level_mean": float(np.mean(means)),
                        "level_per_run": means}
    # The right aggregation is a ratio of means, not a mean of ratios: the
    # per-snapshot denominator (best minus streak) is sometimes near zero, and
    # averaging those ratios is dominated by the small-denominator cases.
    base = summary["streak_score"]["level_mean"]
    ceiling = summary["external_score"]["level_mean"]
    summary["recovered_fraction"] = {
        f"internal_{o}": (float((summary[f"internal_score_{o}"]["level_mean"] - base)
                                / (ceiling - base)) if ceiling > base else None)
        for o in opps}
    summary["rho_streak_external"] = float(np.mean(
        [r["rho_streak_external"] for r in allrows]))
    summary["rho_internal_external"] = {
        f"internal_{o}": float(np.mean(
            [r[f"rho_internal_external_{o}"] for r in allrows])) for o in opps}
    summary["ranking_games_per_snapshot"] = {
        f"internal_{o}": allrows[0][f"games_{o}"] for o in opps}
    summary["episodes_per_individual"] = args.episodes
    summary["n_runs"] = len(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"per_run": out, "summary": summary}, open(args.out, "w"), indent=1)

    print("\nwhich individual you export, and what it costs")
    print(f"  {'rule':<22}{'games':>7}{'level':>8}{'volatility':>12}"
          f"{'gap closed':>12}{'rho(rule,true)':>16}")
    print(f"  {'streak (Ha)':<22}{0:>7}{summary['streak_score']['level_mean']:>+8.2f}"
          f"{summary['streak_score']['volatility_mean']:>12.2f}{'—':>12}"
          f"{summary['rho_streak_external']:>+16.2f}")
    for o in opps:
        k = f"internal_score_{o}"
        rec = summary["recovered_fraction"][f"internal_{o}"]
        print(f"  {'internal round robin':<22}{allrows[0][f'games_{o}']:>7}"
              f"{summary[k]['level_mean']:>+8.2f}"
              f"{summary[k]['volatility_mean']:>12.2f}"
              f"{(f'{100*rec:.0f}%' if rec is not None else '—'):>12}"
              f"{summary['rho_internal_external'][f'internal_{o}']:>+16.2f}")
    print(f"  {'best in pool (oracle)':<22}{'—':>7}"
          f"{summary['external_score']['level_mean']:>+8.2f}"
          f"{summary['external_score']['volatility_mean']:>12.2f}{'100%':>12}"
          f"{1.0:>+16.2f}")
    print(f"  {'population median':<22}{'—':>7}"
          f"{summary['median_score']['level_mean']:>+8.2f}"
          f"{summary['median_score']['volatility_mean']:>12.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
