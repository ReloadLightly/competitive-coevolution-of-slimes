"""
run_asymmetric.py — the unequal-power condition.

Two populations play only each other, with the strong side given roughly twice
the policy capacity of the weak side: 12-16-16-3 with 531 parameters against the
study's standard 12-10-10-3 with 273, a ratio of 1.95 : 1. Keeping the weak side
at exactly the standard architecture means its results stay comparable with every
other run in the study, and the variable-capacity forward pass is bit-identical
to the fixed one at the standard size.

Three conditions, because two confounds have to be controlled before an
asymmetry result means anything:

    asym1x        both sides 273 parameters. Whatever happens here is what
                  two-population coevolution does with no asymmetry at all, and
                  is the null distribution the other two are judged against.
    asym2x        531 v 273 at a common per-parameter sigma. The naive reading:
                  a bigger network, all hyperparameters unchanged.
    asym2x-norm   531 v 273 with the larger side's sigma scaled by
                  sqrt(d_small/d_large), so both sides take mutation steps of
                  equal L2 norm. Isolates capacity from search granularity.

Three things are measured rather than the exported champion alone, because
section 2 of the analysis showed the exported champion is a poor estimate of what
a population contains:

    * the cross-population win rate, a property of the populations that needs no
      champion at all;
    * every member of both populations scored against the 2015 baseline at ten
      points in the run, so "did this side learn" is answered by the pool;
    * three promotion rules side by side -- streak, internal round robin and
      best-in-pool -- so a single champion series is never the only evidence.

Each run writes two files in the matrix directory, one per side, in the same
schema as every other run.

    python run_asymmetric.py --workers 3
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import asymmetric as az
import fastvolley as fv
from run_experiments import INIT_SCALE, SAVE_EVERY, SWEEP_EPISODES, SWEEP_SEED

POP = 128           # both sides, so population size is not a second asymmetry
SIGMA = 0.10
CROSS = 0.25        # a quarter of each population's games meet the other side
GAMES = 500_000
POP_EVERY = 50_000  # full-population snapshots, both sides
POP_EPISODES = 40   # episodes per member when scoring a whole population
RANK_OPPONENTS = 8  # peers per member in the internal ranking
RANK_SEED = 5150

CONDITIONS = {
    # name:          (hidden A, hidden B, label A, label B, equalise step norm)
    "asym1x":        (10, 10, "a", "b", False),
    "asym2x":        (16, 10, "strong", "weak", False),
    "asym2x-norm":   (16, 10, "strong", "weak", True),
}
SEEDS = [101, 102, 103, 104, 105, 106]


def one_run(job):
    cond, seed, outdir, n_games = job
    h_a, h_b, lab_a, lab_b, norm = CONDITIONS[cond]
    paths = {lab_a: os.path.join(outdir, f"{cond}-{lab_a}_s{seed}.npz"),
             lab_b: os.path.join(outdir, f"{cond}-{lab_b}_s{seed}.npz")}
    if all(os.path.exists(p) for p in paths.values()):
        return f"{cond}_s{seed}: already done"

    d_a, d_b = az.param_count(h_a), az.param_count(h_b)
    sigma_a = SIGMA * np.sqrt(d_b / d_a) if norm else SIGMA
    sigma_b = SIGMA
    t0 = time.time()
    (champs_a, champs_b, meanlen, a_winrate, a_margin,
     pops_a, pops_b) = az.run_coevo_asym(
        seed, n_games, POP, POP, h_a, h_b, d_a, d_b, sigma_a, sigma_b,
        CROSS, SAVE_EVERY, INIT_SCALE, POP_EVERY)
    train_sec = time.time() - t0

    w, b = fv.baseline_arrays()
    t1 = time.time()
    out = {}
    for lab, champs, pops, h in ((lab_a, champs_a, pops_a, h_a),
                                 (lab_b, champs_b, pops_b, h_b)):
        n = len(champs)
        ms = np.zeros(n); ss = np.zeros(n); win = np.zeros(n)
        tie = np.zeros(n); loss = np.zeros(n); elen = np.zeros(n)
        for i in range(n):
            sc, ln = az.eval_var_vs_baseline(np.ascontiguousarray(champs[i]), h,
                                             SWEEP_EPISODES, SWEEP_SEED, w, b)
            ms[i] = sc.mean(); ss[i] = sc.std()
            win[i] = (sc > 0).mean(); tie[i] = (sc == 0).mean()
            loss[i] = (sc < 0).mean(); elen[i] = ln.mean()

        # the population, not the proxy: every member scored, plus the two
        # alternative promotion rules from the analysis
        n_snap = len(pops)
        pop_best = np.zeros(n_snap); pop_median = np.zeros(n_snap)
        pop_above = np.zeros(n_snap); pop_mean = np.zeros(n_snap)
        pop_internal = np.zeros(n_snap)
        for k in range(n_snap):
            pool = np.ascontiguousarray(pops[k].astype(np.float64))
            ext = az.eval_population_var(pool, h, POP_EPISODES, RANK_SEED, w, b)
            internal = az.internal_rank_var(pool, h, RANK_OPPONENTS,
                                            RANK_SEED + k)
            pop_best[k] = ext.max()
            pop_median[k] = np.median(ext)
            pop_above[k] = (ext > 0).mean()
            pop_internal[k] = ext[int(np.argmax(internal))]
            pop_mean[k] = ext.mean()
        out[lab] = dict(champs=champs, ms=ms, ss=ss, win=win, tie=tie,
                        loss=loss, elen=elen, h=h, pops=pops,
                        pop_best=pop_best, pop_median=pop_median,
                        pop_above=pop_above, pop_internal=pop_internal,
                        pop_mean=pop_mean)
    eval_sec = time.time() - t1

    for lab, o in out.items():
        np.savez_compressed(
            paths[lab],
            champs=o["champs"],
            streaks=np.zeros(len(o["champs"]), dtype=np.int64),
            train_meanlen=meanlen, tie_rate=np.zeros(len(o["champs"])),
            hof_winrate=np.zeros(len(o["champs"])),
            mean_score=o["ms"], std_score=o["ss"], win_rate=o["win"],
            tie_rate_eval=o["tie"], loss_rate=o["loss"], eval_meanlen=o["elen"],
            tournaments=np.array([n_games]), save_every=np.array([SAVE_EVERY]),
            sigma=np.array([sigma_a if lab == lab_a else sigma_b]),
            pop=np.array([POP]), hof_prob=np.array([0.0]),
            seed=np.array([seed]), hof_capacity=np.array([0]),
            hof_every=np.array([0]),
            sweep_episodes=np.array([SWEEP_EPISODES]),
            sweep_seed=np.array([SWEEP_SEED]),
            train_sec=np.array([train_sec]), eval_sec=np.array([eval_sec]),
            pops=o["pops"], pop_streaks=np.zeros((0, 0), dtype=np.int64),
            pop_every=np.array([POP_EVERY]),
            hidden=np.array([o["h"]]),
            param_count=np.array([o["champs"].shape[1]]),
            a_winrate=a_winrate, a_margin=a_margin,
            pop_best=o["pop_best"], pop_median=o["pop_median"],
            pop_above=o["pop_above"], pop_internal=o["pop_internal"],
            pop_mean=o["pop_mean"], cross_prob=np.array([CROSS]))

    wr = a_winrate[-10:].mean()
    return (f"{cond}_s{seed}: {train_sec/60:.0f}+{eval_sec/60:.0f} min, "
            f"larger-side win rate {wr:.3f} | "
            f"{lab_a} best-in-pool {out[lab_a]['pop_best'][-1]:+.2f} "
            f"({100*out[lab_a]['pop_above'][-1]:.0f}% above parity) | "
            f"{lab_b} best-in-pool {out[lab_b]['pop_best'][-1]:+.2f} "
            f"({100*out[lab_b]['pop_above'][-1]:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results/matrix")
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    conds = ([c.strip() for c in args.only.split(",")] if args.only
             else list(CONDITIONS))
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS)
    # interleave by seed so every condition gets its early seeds first: if the
    # session is cut short the design degrades to fewer seeds per condition,
    # rather than to some conditions never having run at all
    jobs = [(c, s, args.outdir, args.games) for s in seeds for c in conds]
    print(f"{len(jobs)} asymmetric runs x {args.games:,} games, "
          f"{len(seeds)} seeds x {len(conds)} conditions "
          f"(capacity ratio {az.param_count(16)}/{az.param_count(10)} = "
          f"{az.param_count(16)/az.param_count(10):.2f}:1)", flush=True)

    az.run_coevo_asym(0, 200, 8, 8, 16, 10, az.param_count(16),
                      az.param_count(10), 0.1, 0.1, 0.25, 100, 0.5, 100)
    w, b = fv.baseline_arrays()
    az.eval_var_vs_baseline(np.zeros(az.param_count(10)), 10, 1, 1, w, b)
    az.eval_population_var(np.zeros((2, az.param_count(10))), 10, 1, 1, w, b)
    az.internal_rank_var(np.zeros((2, az.param_count(10))), 10, 2, 1)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time()-t0)/60:6.1f} min] {msg}", flush=True)
    print(f"asymmetric conditions complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
