"""
run_counter.py — control runs with shadow counters (WP13).

Each run is the paper's control (Ha's GA, population 128, sigma 0.10,
500,000 games, a champion every 5,000 games, the whole population every
50,000), replayed by shadow.run_ga_shadow, which is bit-identical to the
frozen kernel and also keeps four counters (shadow.RULES). Each counter's
pick at every checkpoint is scored on the learning-curve sweep (200 episodes
on the sweep seed, as every champion curve in the paper), so each counter's
reported curve can be compared at the 5,000-game resolution.

  --explore   replay the six original control runs (seeds 101-106) into
              results/counter/explore/ and check each against its stored
              file (results/matrix/control_s<seed>.npz): champions,
              populations and counters must be identical. Exploratory.
  (default)   the preregistered fresh runs (results/counter/PREREGISTRATION.md),
              written into results/counter/.

Runs whose file exists are skipped, so an interrupted session can be
restarted.

    python run_counter.py --explore
    python run_counter.py --seeds 601,602
    python run_counter.py --protocol     # the protocol record only
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fastvolley as fv
import run_experiments as RE
import shadow as sh

OUT = "results/counter"
EXPLORE = os.path.join(OUT, "explore")
EXPLORE_SEEDS = [101, 102, 103, 104, 105, 106]
FRESH_SEEDS = list(range(601, 613))
CFG = RE.CONDITIONS["control"]


def _sweep(genome, w, b):
    sc, ln = fv.eval_vs_baseline(np.ascontiguousarray(genome, dtype=np.float64),
                                 RE.SWEEP_EPISODES, RE.SWEEP_SEED, w, b, False)
    return float(sc.mean()), float(ln.mean())


def one_run(job):
    seed, outdir, tournaments = job
    path = os.path.join(outdir, f"control_s{seed}.npz")
    if os.path.exists(path):
        return f"control_s{seed}: already done"
    w, b = fv.baseline_arrays()
    t0 = time.time()
    r = sh.replay(seed, tournaments, CFG["pop"], CFG["sigma"], RE.SAVE_EVERY,
                  RE.INIT_SCALE, RE.POP_EVERY)
    train_sec = time.time() - t0

    # each counter's reported curve: its pick at every checkpoint, swept as
    # the paper sweeps every champion (identical picks are scored once)
    t1 = time.time()
    n_ckpt = len(r["champs"])
    curve = np.zeros((n_ckpt, sh.N_RULES))
    curve_len = np.zeros((n_ckpt, sh.N_RULES))
    for k in range(n_ckpt):
        done = {}
        for q in range(sh.N_RULES):
            i = int(r["pick_idx"][k, q])
            if i not in done:
                done[i] = _sweep(r["picks"][k, q], w, b)
            curve[k, q], curve_len[k, q] = done[i]
    eval_sec = time.time() - t1

    matches = np.array([-1])
    stored = os.path.join("results/matrix", f"control_s{seed}.npz")
    if outdir == EXPLORE and os.path.exists(stored):
        z = np.load(stored)
        same = (np.array_equal(z["champs"], r["champs"])
                and np.array_equal(z["pops"], r["pops"])
                and np.array_equal(z["pop_streaks"], r["pop_streaks"])
                and np.array_equal(z["mean_score"], curve[:, sh.INH]))
        matches = np.array([int(same)])

    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, **r, curve=curve, curve_len=curve_len, rules=np.array(sh.RULES),
            seed=np.array([seed]), tournaments=np.array([tournaments]),
            save_every=np.array([RE.SAVE_EVERY]), pop_every=np.array([RE.POP_EVERY]),
            sigma=np.array([CFG["sigma"]]), pop=np.array([CFG["pop"]]),
            init_scale=np.array([RE.INIT_SCALE]),
            sweep_episodes=np.array([RE.SWEEP_EPISODES]),
            sweep_seed=np.array([RE.SWEEP_SEED]),
            matches_stored=matches, train_sec=np.array([train_sec]),
            eval_sec=np.array([eval_sec]))
    os.replace(tmp, path)
    note = {1: ", identical to the stored run", 0: ", DIFFERS from the stored run",
            -1: ""}[int(matches[0])]
    return (f"control_s{seed}: replay {train_sec / 60:.1f} min, sweep "
            f"{eval_sec / 60:.1f} min, final INH {curve[-1, sh.INH]:+.2f} "
            f"CUR {curve[-1, sh.CUR]:+.2f}{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--seeds", default=None, help="comma-separated subset")
    ap.add_argument("--tournaments", type=int, default=RE.TOURNAMENTS)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--protocol", action="store_true",
                    help="write the protocol record and exit (no run)")
    args = ap.parse_args()
    outdir = EXPLORE if args.explore else OUT
    seeds = EXPLORE_SEEDS if args.explore else FRESH_SEEDS
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(outdir, exist_ok=True)
    proto = os.path.join(outdir, "protocol.json")
    if not os.path.exists(proto):
        json.dump({"tournaments": args.tournaments, "save_every": RE.SAVE_EVERY,
                   "pop_every": RE.POP_EVERY, "sigma": CFG["sigma"], "pop": CFG["pop"],
                   "init_scale": RE.INIT_SCALE, "sweep_episodes": RE.SWEEP_EPISODES,
                   "sweep_seed": RE.SWEEP_SEED, "rules": list(sh.RULES),
                   "seeds": EXPLORE_SEEDS if args.explore else FRESH_SEEDS,
                   "kernel": "shadow.run_ga_shadow"},
                  open(proto, "w"), indent=1)
    if args.protocol:
        print(f"-> {proto}")
        return
    jobs = [(s, outdir, args.tournaments) for s in seeds]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
