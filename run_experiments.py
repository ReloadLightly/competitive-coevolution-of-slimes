"""
run_experiments.py — the experiment matrix.

The first version of this repository had one run. One run cannot separate a
property of the algorithm from a property of a seed, which is why it could
report a trajectory but not a comparison. This script runs the design:

  control     Ha's tournament-selection GA, unmodified                (12 seeds)
  hof-0.25    same, but 25% of games are played against an archived
              champion instead of a current population member        (12 seeds)
  hof-0.50    the same intervention at twice the dose                 (3 seeds)
  hof-full    the same dose as hof-0.25 but with an archive spanning
              the whole run instead of the last 64,000 games          (6 seeds)
  sigma-0.05  mutation scale halved                                   (3 seeds)
  sigma-0.20  mutation scale doubled                                  (3 seeds)
  pop-32      population shrunk 4x                                    (3 seeds)
  pop-512     population grown 4x                                     (3 seeds)

Every run is 500,000 tournament games — Ha's budget, and the budget of the
reference run in results/ga_selfplay. Seeds are shared across conditions, so
each condition sees the same set of seed labels (the RNG streams differ by
condition, so this is a labelling convenience, not paired sampling).

Each worker trains one run and immediately evaluates all 100 of its champion
checkpoints against the 2015 baseline policy, so evaluation pipelines behind
training instead of waiting for the whole matrix.

    python run_experiments.py --workers 3
    python run_experiments.py --workers 3 --only control,hof-0.25
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fastvolley as fv

# --------------------------------------------------------------------------
# Pre-registered protocol constants. Fixed before any run was launched.
# --------------------------------------------------------------------------
TOURNAMENTS = 500_000
SAVE_EVERY = 5_000          # -> 100 champion checkpoints per run
POP_EVERY = 50_000          # full-population snapshots (control runs only)
SWEEP_EPISODES = 200        # episodes per checkpoint in the learning-curve sweep
SWEEP_SEED = 20260901       # evaluation seed for the sweep
SELECT_SEED = 20260902      # disjoint seed for re-evaluating selected champions
SELECT_EPISODES = 1_000

SEEDS_MAIN = [101, 102, 103, 104, 105, 106]
SEEDS_SIDE = [101, 102, 103]

# `cap` is the archive size. At one champion archived every HOF_EVERY = 1,000
# games, cap=64 spans only the most recent 64,000 games -- a recency buffer --
# while cap=512 spans the entire 500,000-game run, which is a hall of fame in
# the usual sense. Both are run, because "play recent past selves" and "play
# every past self" are different interventions and it is worth knowing which
# one, if either, does the work.
CONDITIONS = {
    "control":    dict(sigma=0.10, pop=128, hof_prob=0.00, cap=64,  seeds=SEEDS_MAIN, pops=True),
    "hof-0.25":   dict(sigma=0.10, pop=128, hof_prob=0.25, cap=64,  seeds=SEEDS_MAIN, pops=False),
    "hof-0.50":   dict(sigma=0.10, pop=128, hof_prob=0.50, cap=64,  seeds=SEEDS_SIDE, pops=False),
    "hof-full":   dict(sigma=0.10, pop=128, hof_prob=0.25, cap=512, seeds=SEEDS_MAIN, pops=False),
    "sigma-0.05": dict(sigma=0.05, pop=128, hof_prob=0.00, cap=64,  seeds=SEEDS_SIDE, pops=False),
    "sigma-0.20": dict(sigma=0.20, pop=128, hof_prob=0.00, cap=64,  seeds=SEEDS_SIDE, pops=False),
    "pop-32":     dict(sigma=0.10, pop=32,  hof_prob=0.00, cap=64,  seeds=SEEDS_SIDE, pops=False),
    "pop-512":    dict(sigma=0.10, pop=512, hof_prob=0.00, cap=64,  seeds=SEEDS_SIDE, pops=False),
}

HOF_EVERY = 1_000           # a champion is archived this often
HOF_CAPACITY = 64           # default archive size (per-condition `cap` wins)
INIT_SCALE = 0.5            # Ha's population initialisation


def run_id(cond, seed):
    return f"{cond}_s{seed}"


def one_run(job):
    cond, seed, outdir, tournaments = job
    cfg = CONDITIONS[cond]
    path = os.path.join(outdir, run_id(cond, seed) + ".npz")
    if os.path.exists(path):
        return f"{run_id(cond, seed)}: already done"

    w, b = fv.baseline_arrays()
    t0 = time.time()
    if cfg["pops"]:
        champs, streaks, meanlen, pops, pop_streaks = fv.run_ga_with_pops(
            seed, tournaments, cfg["pop"], cfg["sigma"], SAVE_EVERY,
            cfg["hof_prob"], HOF_EVERY, cfg.get("cap", HOF_CAPACITY), w, b,
            INIT_SCALE, POP_EVERY)
        ties = np.zeros(len(champs))
        hofwins = np.zeros(len(champs))
    else:
        champs, streaks, meanlen, ties, hofwins = fv.run_ga(
            seed, tournaments, cfg["pop"], cfg["sigma"], SAVE_EVERY,
            cfg["hof_prob"], HOF_EVERY, cfg.get("cap", HOF_CAPACITY), w, b,
            INIT_SCALE)
        pops = np.zeros((0, 0, 0), dtype=np.float32)
        pop_streaks = np.zeros((0, 0), dtype=np.int64)
    train_sec = time.time() - t0

    # ---- learning-curve sweep: every checkpoint, held-out evaluation seed --
    n = len(champs)
    mean_score = np.zeros(n)
    std_score = np.zeros(n)
    win = np.zeros(n)
    tie = np.zeros(n)
    loss = np.zeros(n)
    mean_len = np.zeros(n)
    t1 = time.time()
    for i in range(n):
        sc, ln = fv.eval_vs_baseline(np.ascontiguousarray(champs[i]),
                                     SWEEP_EPISODES, SWEEP_SEED, w, b, False)
        mean_score[i] = sc.mean()
        std_score[i] = sc.std()
        win[i] = (sc > 0).mean()
        tie[i] = (sc == 0).mean()
        loss[i] = (sc < 0).mean()
        mean_len[i] = ln.mean()
    eval_sec = time.time() - t1

    np.savez_compressed(
        path,
        champs=champs.astype(np.float64), streaks=streaks, train_meanlen=meanlen,
        tie_rate=ties, hof_winrate=hofwins,
        mean_score=mean_score, std_score=std_score,
        win_rate=win, tie_rate_eval=tie, loss_rate=loss, eval_meanlen=mean_len,
        tournaments=np.array([tournaments]), save_every=np.array([SAVE_EVERY]),
        sigma=np.array([cfg["sigma"]]), pop=np.array([cfg["pop"]]),
        hof_prob=np.array([cfg["hof_prob"]]), seed=np.array([seed]),
        hof_capacity=np.array([cfg.get("cap", HOF_CAPACITY)]),
        hof_every=np.array([HOF_EVERY]),
        sweep_episodes=np.array([SWEEP_EPISODES]),
        sweep_seed=np.array([SWEEP_SEED]),
        train_sec=np.array([train_sec]), eval_sec=np.array([eval_sec]),
        pops=pops, pop_streaks=pop_streaks,
        pop_every=np.array([POP_EVERY if cfg["pops"] else 0]))

    return (f"{run_id(cond, seed)}: train {train_sec/60:.1f} min, "
            f"eval {eval_sec/60:.1f} min, final {mean_score[-1]:+.2f}, "
            f"best {mean_score.max():+.2f}, "
            f"above parity {int((mean_score > 0).sum())}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results/matrix")
    ap.add_argument("--only", default=None,
                    help="comma-separated condition names")
    ap.add_argument("--tournaments", type=int, default=TOURNAMENTS)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds, overriding the condition's "
                         "own list (used to add a second wave of seeds; runs "
                         "whose .npz already exists are skipped)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    conds = list(CONDITIONS)
    if args.only:
        conds = [c.strip() for c in args.only.split(",")]
    override = ([int(s) for s in args.seeds.split(",")] if args.seeds else None)

    jobs = []
    for cond in conds:
        for seed in (override or CONDITIONS[cond]["seeds"]):
            jobs.append((cond, seed, args.outdir, args.tournaments))

    with open(os.path.join(args.outdir, "protocol.json"), "w") as f:
        json.dump({"tournaments": args.tournaments, "save_every": SAVE_EVERY,
                   "pop_every": POP_EVERY, "sweep_episodes": SWEEP_EPISODES,
                   "sweep_seed": SWEEP_SEED, "select_seed": SELECT_SEED,
                   "select_episodes": SELECT_EPISODES,
                   "hof_every": HOF_EVERY, "hof_capacity": HOF_CAPACITY,
                   "init_scale": INIT_SCALE,
                   "conditions": {k: {kk: vv for kk, vv in v.items()}
                                  for k, v in CONDITIONS.items()}}, f, indent=1)

    print(f"{len(jobs)} runs x {args.tournaments:,} tournaments "
          f"on {args.workers} workers", flush=True)

    # compile once in the parent so workers load from the numba cache
    w, b = fv.baseline_arrays()
    fv.run_ga(0, 200, 8, 0.1, 100, 0.0, 100, 4, w, b, 0.5)
    fv.run_ga_with_pops(0, 200, 8, 0.1, 100, 0.0, 100, 4, w, b, 0.5, 100)
    fv.eval_vs_baseline(np.zeros(fv.PARAM_COUNT), 1, 1, w, b, False)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time()-t0)/60:6.1f} min] {msg}", flush=True)
    print(f"matrix complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
