"""
counter_analysis.py — what each shadow counter would have exported (WP13).

Reads the runs written by run_counter.py. At every population snapshot (10
per run) every member is scored against the 2015 baseline on 60 episodes on
a seed of its own (the measurement the export experiment used for its
oracle), and each rule's pick is re-scored on the held-out seed (1,000
episodes), as every champion in the paper is. The rules:

  the four counters of shadow.py   inherited (Ha's rule), inherited-reset,
                                   own, current: each exports the first
                                   member with its maximum count
  tournament-16                    the preregistered export rule of WP10
                                   (fastvolley_kernels.internal_rank, 16
                                   peers, its seeds), for comparison
  median, best                     the member with the median / the best
                                   60-episode score (best: an oracle, not
                                   deployable)

Per run, for each rule: level (mean held-out score of its picks over the
snapshots) and declines (summed falls between consecutive snapshots), as in
WP10; the rank of its pick in the population; Spearman rho between each
counter and the 60-episode scores, per snapshot. From the learning-curve
sweep stored by run_counter.py, each counter's reported curve at the
5,000-game resolution: mean absolute change between checkpoints and summed
declines. The 2 x 2 of counters splits each difference into the effect of
inheritance at birth and the effect of restarting the count when a tie
mutates the genotype.

    python counter_analysis.py --explore      # results/counter/explore
    python counter_analysis.py                # the preregistered runs
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

import export_analysis as xa
import fastvolley as fv
import fastvolley_kernels as fk
import provenance as pv
import shadow as sh
import stats_utils as su
from run_counter import EXPLORE, OUT as RUNS
from run_experiments import SELECT_EPISODES, SELECT_SEED

PRIMARY = xa.PRIMARY                     # tournament-16, WP10's rule
COUNTERS = sh.RULES
RULES = COUNTERS + (f"tournament-{PRIMARY}", "median", "best")


def _heldout(genome, w, b):
    sc, _ = fv.eval_vs_baseline(np.ascontiguousarray(genome), SELECT_EPISODES,
                                SELECT_SEED, w, b, False)
    return float(sc.mean())


def one_run(path):
    z = np.load(path)
    seed = int(z["seed"][0])
    w, b = fv.baseline_arrays()
    pops, counts = z["pops"], z["pop_counts"]
    snaps = []
    for t in range(len(pops)):
        pop = np.ascontiguousarray(pops[t].astype(np.float64))
        ext, _ = fk.eval_population(pop, xa.RANK_EPISODES, xa.RANK_SEED, w, b)
        pick = {c: int(np.argmax(counts[t, q])) for q, c in enumerate(COUNTERS)}
        internal = fk.internal_rank(pop, PRIMARY, xa.TOURNEY_SEED + 100 * t + PRIMARY, w, b)
        pick[f"tournament-{PRIMARY}"] = int(np.argmax(internal))
        order = np.argsort(-ext)
        pick["median"] = int(order[len(order) // 2])
        pick["best"] = int(order[0])
        held = {i: _heldout(pop[i], w, b) for i in set(pick.values())}
        c = counts[t]
        snaps.append({
            "score": {r: held[i] for r, i in pick.items()},
            "rank": {r: int(np.where(order == i)[0][0]) + 1 for r, i in pick.items()},
            "index": pick,
            "rho": {cn: su.spearman(c[q], ext) for q, cn in enumerate(COUNTERS)},
            "at_max": {cn: int((c[q] == c[q].max()).sum()) for q, cn in enumerate(COUNTERS)},
            "max_count": {cn: int(c[q].max()) for q, cn in enumerate(COUNTERS)},
            "inherited_spread": float((c[sh.INH].max() - np.median(c[sh.INH]))
                                      / max(1, c[sh.INH].max())),
            "median_games_since_birth": float(np.median(z["pop_games"][t])),
            "ext_sd": float(ext.std())})
    out = {"seed": seed, "snapshots": snaps}
    for r in RULES:
        v = np.array([s["score"][r] for s in snaps])
        d = np.diff(v)
        out[f"level_{r}"] = float(v.mean())
        out[f"declines_{r}"] = float(-d[d < 0].sum()) + 0.0
        out[f"rank_{r}"] = float(np.mean([s["rank"][r] for s in snaps]))
    for cn in COUNTERS:
        out[f"rho_{cn}"] = float(np.mean([s["rho"][cn] for s in snaps
                                          if s["rho"][cn] is not None]))
    # each counter's reported curve, 5,000-game resolution
    curve = z["curve"]
    for q, cn in enumerate(COUNTERS):
        d = np.diff(curve[:, q])
        out[f"curve_volatility_{cn}"] = float(np.abs(d).mean())
        out[f"curve_declines_{cn}"] = float(-d[d < 0].sum()) + 0.0
        out[f"curve_mean_{cn}"] = float(curve[:, q].mean())
    tie = z["tie_frac"]
    out["tie_frac_late"] = float(tie[-20:].mean())
    out["matches_stored"] = int(z["matches_stored"][0])
    return f"control_s{seed}", out


def summarise(runs):
    rs = list(runs.values())
    res = {"n_runs": len(rs), "matches_stored": [r["matches_stored"] for r in rs]}
    res["rules"] = {r: {k: float(np.mean([x[f"{k}_{r}"] for x in rs]))
                        for k in ("level", "declines", "rank")} for r in RULES}
    res["rho"] = {cn: float(np.mean([x[f"rho_{cn}"] for x in rs])) for cn in COUNTERS}
    res["curve"] = {cn: {k: float(np.mean([x[f"curve_{k}_{cn}"] for x in rs]))
                         for k in ("volatility", "declines", "mean")} for cn in COUNTERS}

    def paired(a, b, key="level"):
        d = [x[f"{key}_{a}"] - x[f"{key}_{b}"] for x in rs]
        return {"mean": float(np.mean(d)), "runs_higher": int(sum(v > 0 for v in d)),
                "n": len(d), "p_one_sided": su.signflip_greater(d)[1]}
    T = f"tournament-{PRIMARY}"
    res["vs_inherited"] = {c: paired(c, "inherited") for c in RULES if c != "inherited"}
    res["vs_tournament"] = {c: paired(c, T) for c in COUNTERS}
    # the 2 x 2: effect of not inheriting at birth, of restarting at a tie
    lv = {c: np.array([x[f"level_{c}"] for x in rs]) for c in COUNTERS}
    res["factorial_level"] = {
        "no_inheritance": float(np.mean((lv["own"] - lv["inherited"]
                                         + lv["current"] - lv["inherited-reset"]) / 2)),
        "tie_restart": float(np.mean((lv["inherited-reset"] - lv["inherited"]
                                      + lv["current"] - lv["own"]) / 2)),
        "interaction": float(np.mean(lv["current"] - lv["own"]
                                     - lv["inherited-reset"] + lv["inherited"]))}
    snaps = [s for x in rs for s in x["snapshots"]]
    res["diagnostics"] = {
        "inherited_spread_median": float(np.median([s["inherited_spread"] for s in snaps])),
        "at_max_mean": {cn: float(np.mean([s["at_max"][cn] for s in snaps])) for cn in COUNTERS},
        "max_count_median": {cn: float(np.median([s["max_count"][cn] for s in snaps]))
                             for cn in COUNTERS},
        "median_games_since_birth": float(np.median([s["median_games_since_birth"]
                                                     for s in snaps])),
        "tie_frac_late_mean": float(np.mean([x["tie_frac_late"] for x in rs]))}
    return res


def tests(runs, alpha=0.05):
    """The preregistered tests (results/counter/PREREGISTRATION.md): four
    one-sided exact paired sign-flip tests over the runs, Holm at family-wise
    alpha. Positive differences favour the hypothesis."""
    rs = list(runs.values())

    def lv(c):
        return np.array([x[f"level_{c}"] for x in rs])

    def vol(c):
        return np.array([x[f"curve_volatility_{c}"] for x in rs])
    diffs = {
        # H13a: the current genotype's wins export better members than Ha's counter
        "H13a": lv("current") - lv("inherited"),
        # H13b: and report a curve that swings less between checkpoints
        "H13b": vol("inherited") - vol("current"),
        # H13c: not inheriting the count at birth helps (main effect, 2 x 2)
        "H13c": (lv("own") - lv("inherited") + lv("current") - lv("inherited-reset")) / 2,
        # H13d: restarting the count when a tie mutates the genotype helps
        "H13d": (lv("inherited-reset") - lv("inherited") + lv("current") - lv("own")) / 2}
    res = {}
    for h, d in diffs.items():
        mean, p = su.signflip_greater(d)
        res[h] = {"mean": float(mean), "runs_positive": int((d > 0).sum()),
                  "n": len(d), "p_one_sided": p}
    rej = su.holm([res[h]["p_one_sided"] for h in diffs], alpha)
    for h, r in zip(diffs, rej):
        res[h]["rejected"] = bool(r)
    # described, no decision: the free counter against WP10's tournament
    d = lv("current") - lv(f"tournament-{PRIMARY}")
    two = min(1.0, 2 * min(su.signflip_greater(d)[1], su.signflip_greater(-d)[1]))
    res["current_vs_tournament"] = {"mean": float(d.mean()), "runs_higher": int((d > 0).sum()),
                                    "n": len(d), "p_two_sided": two}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    d = EXPLORE if args.explore else RUNS
    paths = sorted(glob.glob(os.path.join(d, "control_s*.npz")))
    with mp.get_context("spawn").Pool(min(args.workers, len(paths))) as pool:
        runs = dict(pool.imap_unordered(one_run, paths))
    runs = dict(sorted(runs.items()))
    res = {"summary": summarise(runs), "tests": tests(runs), "per_run": runs,
           "exploratory": bool(args.explore)}
    out = os.path.join(d, "analysis.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    pv.record(out, "counter_explore" if args.explore else "counter")
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["tests"], indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
