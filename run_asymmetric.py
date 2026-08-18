"""
run_asymmetric.py — the unequal-power condition.

Two populations play only each other, with the strong side given roughly twice
the policy capacity of the weak side (12-16-16-3, 531 parameters, against the
study's standard 12-10-10-3, 273 — a ratio of 1.95:1). A symmetric control with
both sides at 273 parameters isolates the effect of the asymmetry itself from
the effect of two-population coevolution.

Each run writes two files in the matrix directory, one per side, in the same
schema as every other run, so the existing analysis picks them up unchanged:

    asym2x-strong_s101.npz   asym2x-weak_s101.npz
    asym1x-a_s101.npz        asym1x-b_s101.npz

    python run_asymmetric.py --workers 3
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import asymmetric as az
import fastvolley as fv
from run_experiments import (INIT_SCALE, SAVE_EVERY, SWEEP_EPISODES, SWEEP_SEED,
                             TOURNAMENTS)

POP = 128           # both sides, so population size is not a second asymmetry
SIGMA = 0.10

CONDITIONS = {
    # name:        (hidden A, hidden B, label A, label B)
    "asym2x": (16, 10, "strong", "weak"),
    "asym1x": (10, 10, "a", "b"),
}
SEEDS = [101, 102, 103]


def one_run(job):
    cond, seed, outdir, n_games = job
    h_a, h_b, lab_a, lab_b = CONDITIONS[cond]
    paths = {lab_a: os.path.join(outdir, f"{cond}-{lab_a}_s{seed}.npz"),
             lab_b: os.path.join(outdir, f"{cond}-{lab_b}_s{seed}.npz")}
    if all(os.path.exists(p) for p in paths.values()):
        return f"{cond}_s{seed}: already done"

    d_a, d_b = az.param_count(h_a), az.param_count(h_b)
    t0 = time.time()
    champs_a, champs_b, meanlen, a_winrate, a_margin = az.run_coevo_asym(
        seed, n_games, POP, POP, h_a, h_b, d_a, d_b, SIGMA, SAVE_EVERY,
        INIT_SCALE)
    train_sec = time.time() - t0

    w, b = fv.baseline_arrays()
    t1 = time.time()
    out = {}
    for lab, champs, h in ((lab_a, champs_a, h_a), (lab_b, champs_b, h_b)):
        n = len(champs)
        mean_score = np.zeros(n)
        std_score = np.zeros(n)
        win = np.zeros(n)
        tie = np.zeros(n)
        loss = np.zeros(n)
        elen = np.zeros(n)
        for i in range(n):
            sc, ln = az.eval_var_vs_baseline(np.ascontiguousarray(champs[i]), h,
                                             SWEEP_EPISODES, SWEEP_SEED, w, b)
            mean_score[i] = sc.mean()
            std_score[i] = sc.std()
            win[i] = (sc > 0).mean()
            tie[i] = (sc == 0).mean()
            loss[i] = (sc < 0).mean()
            elen[i] = ln.mean()
        out[lab] = (champs, mean_score, std_score, win, tie, loss, elen, h)
    eval_sec = time.time() - t1

    for lab, (champs, ms, ss, win, tie, loss, elen, h) in out.items():
        np.savez_compressed(
            paths[lab],
            champs=champs, streaks=np.zeros(len(champs), dtype=np.int64),
            train_meanlen=meanlen, tie_rate=np.zeros(len(champs)),
            hof_winrate=np.zeros(len(champs)),
            mean_score=ms, std_score=ss, win_rate=win, tie_rate_eval=tie,
            loss_rate=loss, eval_meanlen=elen,
            tournaments=np.array([n_games]), save_every=np.array([SAVE_EVERY]),
            sigma=np.array([SIGMA]), pop=np.array([POP]),
            hof_prob=np.array([0.0]), seed=np.array([seed]),
            hof_capacity=np.array([0]), hof_every=np.array([0]),
            sweep_episodes=np.array([SWEEP_EPISODES]),
            sweep_seed=np.array([SWEEP_SEED]),
            train_sec=np.array([train_sec]), eval_sec=np.array([eval_sec]),
            pops=np.zeros((0, 0, 0), dtype=np.float32),
            pop_streaks=np.zeros((0, 0), dtype=np.int64),
            pop_every=np.array([0]),
            hidden=np.array([h]), param_count=np.array([champs.shape[1]]),
            a_winrate=a_winrate, a_margin=a_margin)

    return (f"{cond}_s{seed}: train {train_sec/60:.1f} min, "
            f"eval {eval_sec/60:.1f} min, "
            f"strong-side win rate over the last 100k games "
            f"{a_winrate[-20:].mean():.3f}, "
            f"{lab_a} final {out[lab_a][1][-1]:+.2f}, "
            f"{lab_b} final {out[lab_b][1][-1]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results/matrix")
    ap.add_argument("--games", type=int, default=TOURNAMENTS)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    conds = ([c.strip() for c in args.only.split(",")] if args.only
             else list(CONDITIONS))
    jobs = [(c, s, args.outdir, args.games) for c in conds for s in SEEDS]
    print(f"{len(jobs)} asymmetric runs x {args.games:,} games "
          f"(capacity ratio {az.param_count(16)}/{az.param_count(10)} = "
          f"{az.param_count(16)/az.param_count(10):.2f}:1)", flush=True)

    # compile once in the parent so workers load from the numba cache
    az.run_coevo_asym(0, 200, 8, 8, 16, 10, az.param_count(16),
                      az.param_count(10), 0.1, 100, 0.5)
    w, b = fv.baseline_arrays()
    az.eval_var_vs_baseline(np.zeros(az.param_count(10)), 10, 1, 1, w, b)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time()-t0)/60:6.1f} min] {msg}", flush=True)
    print(f"asymmetric conditions complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
