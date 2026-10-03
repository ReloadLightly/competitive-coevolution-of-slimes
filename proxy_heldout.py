"""
proxy_heldout.py — the champion-proxy comparison, re-scored on held-out episodes.

`coevolution_analysis.py --proxy` scores every member of a population snapshot
on 30 episodes (RR_SEED) and compares the exported champion with the member
that scores best *on those same episodes*. The maximum of 128 noisy scores is
biased upward (a winner's curse), and one member's 30-episode score is noisier
than a maximum over 128, so the gap to the best member and the comparison of
their declines both carry measurement noise in the direction of the claim.
Found in a review on 2026-10-03 (results/matrix/decisions.md).

This script keeps every choice exactly as before (exported: the largest streak;
best and median: by the 30-episode scores on RR_SEED) and re-scores each chosen
member on the held-out seed (SELECT_EPISODES on SELECT_SEED), as every champion
in the paper is re-scored. A second, independent 30-episode measurement
(RETEST_SEED) gives the test-retest reliability of the per-member scores on
which the exported member's rank and rho(streak, skill) rest, and the
exported member's rank under that independent measurement.

It covers every control run with population snapshots: the original six
(results/matrix), the replication's twelve and the export experiment's twelve.
A best member chosen on 30 noisy episodes need not be the truly best one, so
its held-out score is, in expectation, a lower bound on the best member's;
the gap reported here is therefore conservative where the old one was
inflated.

    python proxy_heldout.py
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

import coevolution_analysis as ca
import fastvolley as fv
import fastvolley_kernels as fk
import provenance as pv
import stats_utils as su
from run_experiments import SELECT_EPISODES, SELECT_SEED

OUT = "results/analysis/proxy_heldout.json"
RETEST_SEED = 20261015            # disjoint from every training and evaluation seed
GROUPS = {"original": "results/matrix", "replication": "results/replication",
          "export": "results/export"}


def runs():
    out = []
    for group, d in GROUPS.items():
        paths = (pv.matrix_runs(d, "control_s*.npz") if group == "original"
                 else sorted(glob.glob(os.path.join(d, "control_s*.npz"))))
        out += [(group, p) for p in paths]
    return out


def _heldout(genome, w, b):
    sc, _ = fv.eval_vs_baseline(np.ascontiguousarray(genome), SELECT_EPISODES,
                                SELECT_SEED, w, b, False)
    return float(sc.mean())


def _job(job):
    group, path = job
    z = np.load(path)
    w, b = fv.baseline_arrays()
    snaps = []
    for k in range(len(z["pops"])):
        pop = np.ascontiguousarray(z["pops"][k].astype(np.float64))
        a, _ = fk.eval_population(pop, ca.PROXY_EPISODES, ca.RR_SEED, w, b)
        r, _ = fk.eval_population(pop, ca.PROXY_EPISODES, RETEST_SEED, w, b)
        streaks = z["pop_streaks"][k]
        pick = {"exported": int(np.argmax(streaks)), "best": int(np.argmax(a)),
                "median": int(np.argsort(a)[len(a) // 2])}
        held = {m: _heldout(pop[i], w, b) for m, i in pick.items()}
        e = pick["exported"]
        snaps.append({
            "selecting": {m: float(a[i]) for m, i in pick.items()},
            "heldout": held,
            "rank_selecting": int((a > a[e]).sum()) + 1,
            "rank_retest": int((r > r[e]).sum()) + 1,
            "retest_r": float(np.corrcoef(a, r)[0, 1]) if a.std() > 0 and r.std() > 0 else None,
            "rho_streak_selecting": su.spearman(streaks, a),
            "rho_streak_retest": su.spearman(streaks, r),
            "spread": float(a.std())})

    def declines(kind, m):
        d = np.diff([s[kind][m] for s in snaps])
        return float(-d[d < 0].sum())
    out = {"group": group, "snapshots": snaps}
    for kind in ("selecting", "heldout"):
        for m in ("exported", "best", "median"):
            out[f"declines_{kind}_{m}"] = declines(kind, m)
            out[f"level_{kind}_{m}"] = float(np.mean([s[kind][m] for s in snaps]))
    return os.path.basename(path)[:-4], out


def summarise(per_run):
    res = {}
    for group in GROUPS:
        rs = [r for r in per_run.values() if r["group"] == group]
        if not rs:
            continue
        snaps = [s for r in rs for s in r["snapshots"]]
        rel = [s["retest_r"] for s in snaps if s["retest_r"] is not None]
        g = {"runs": len(rs), "snapshots": len(snaps),
             "gap_selecting": float(np.mean([s["selecting"]["best"] - s["selecting"]["exported"]
                                             for s in snaps])),
             "gap_heldout": float(np.mean([s["heldout"]["best"] - s["heldout"]["exported"]
                                           for s in snaps])),
             "best_inflation": float(np.mean([s["selecting"]["best"] - s["heldout"]["best"]
                                              for s in snaps])),
             "retest_r_median": float(np.median(rel)),
             "retest_r_learned_median": float(np.median(
                 [s["retest_r"] for s in snaps if s["retest_r"] is not None and s["spread"] > 0.2])),
             "rank_selecting": float(np.mean([s["rank_selecting"] for s in snaps])),
             "rank_retest": float(np.mean([s["rank_retest"] for s in snaps])),
             "rho_streak_selecting": float(np.mean([s["rho_streak_selecting"] for s in snaps])),
             "rho_streak_retest": float(np.mean([s["rho_streak_retest"] for s in snaps]))}
        for kind in ("selecting", "heldout"):
            for m in ("exported", "best", "median"):
                g[f"declines_{kind}_{m}"] = float(sum(r[f"declines_{kind}_{m}"] for r in rs))
                g[f"level_{kind}_{m}"] = float(np.mean([r[f"level_{kind}_{m}"] for r in rs]))
            ex, be = g[f"declines_{kind}_exported"], g[f"declines_{kind}_best"]
            g[f"decline_ratio_{kind}"] = ex / be if be > 0 else None
            g[f"runs_exported_declines_more_{kind}"] = int(sum(
                r[f"declines_{kind}_exported"] > r[f"declines_{kind}_best"] for r in rs))
        res[group] = g
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    jobs = runs()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        per_run = dict(pool.imap_unordered(_job, jobs))
    res = {"summary": summarise(per_run), "per_run": dict(sorted(per_run.items())),
           "proxy_episodes": ca.PROXY_EPISODES, "heldout_episodes": SELECT_EPISODES}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "proxy_heldout")
    print(json.dumps(res["summary"], indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
