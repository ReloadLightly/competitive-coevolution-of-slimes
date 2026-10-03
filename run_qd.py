"""
run_qd.py — the niche-archive experiment (WP9): does an archive organised by
behaviour, instead of by time, do what the time-ordered one did not?

The archive is used as a test, as in the paper's `hof-eval-v2`, but it is a
MAP-Elites-style grid (lab.kernels.HOF_NICHE): 8 x 8 cells over a two-number
behaviour descriptor, one champion per cell, the newest to land there.
Design, hypotheses and analysis are fixed in results/qd/PREREGISTRATION.md
before any run exists.

  slime-niche          Slime Volleyball, seeds 201-212: the seeds of the
                       replication, whose control and archive-as-test runs
                       are the comparison. Evaluated exactly as
                       run_experiments.py evaluates every matrix run.
  discmix-<lam>-niche  the lab's discmix game at the four lambdas of WP8,
                       seeds 301-312; WP8's control and archive-as-test runs
                       are the comparison.

The runner is restartable: a run whose file exists is skipped.

    python run_qd.py --only slime-niche --seeds 201
    python run_qd.py --only discmix-0.75-niche
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fastvolley as fv
import run_experiments as RE
import run_lab as RL
from lab import games as G
from lab import kernels as K

OUT = "results/qd"
HOF_PROB, HOF_EVERY = 0.25, 1_000             # as hof-eval-v2 and WP8's test
CAPACITY = K.NICHE_GRID * K.NICHE_GRID        # one slot per cell
# Grid bounds, fixed before any niche run: the 1st to 99th percentile of each
# descriptor over all exported champions of runs that already existed
# (slime: the replication's control and archive-as-test runs; discmix: all
# WP8 runs), rounded outward to 0.05.
BOUNDS = {"slime": (-0.35, 0.95, -0.95, 0.35),
          "discmix": (-0.80, 0.80, -0.75, 0.80)}
SLIME_SEEDS = list(range(201, 213))
LAB_SEEDS = RL.SEEDS
CONDITIONS = {"slime-niche": dict(game="slime", seeds=SLIME_SEEDS)}
CONDITIONS.update({f"discmix-{lam:.2f}-niche": dict(game="discmix", lam=lam,
                                                    seeds=LAB_SEEDS)
                   for lam in RL.LAMBDAS})


def _gp(game, lam=0.0):
    g, gp, X1, X2, X3 = G.make(game, lam)
    return g, np.concatenate([gp, np.array(BOUNDS[game], dtype=np.float64)]), X1, X2, X3


def one_run(job):
    cond, seed, outdir = job
    path = os.path.join(outdir, f"{cond}_s{seed}.npz")
    if os.path.exists(path):
        return f"{cond}_s{seed}: already done"
    cfg = CONDITIONS[cond]
    w, b = fv.baseline_arrays()
    t0 = time.time()
    if cfg["game"] == "slime":
        g, gp, X1, X2, X3 = _gp("slime")
        champs, streaks, meanlen, ties, hofwins, _, _, arch = K.run(
            g, gp, X1, X2, X3, seed, RE.TOURNAMENTS, 128, 0.10, RE.SAVE_EVERY,
            K.HOF_NICHE, HOF_PROB, HOF_EVERY, CAPACITY, w, b, RE.INIT_SCALE, 0)
        train_sec = time.time() - t0
        # the learning-curve sweep of run_experiments.one_run, unchanged
        n = len(champs)
        mean_score, std_score = np.zeros(n), np.zeros(n)
        win, tie, loss, mean_len = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        t1 = time.time()
        for i in range(n):
            sc, ln = fv.eval_vs_baseline(np.ascontiguousarray(champs[i]),
                                         RE.SWEEP_EPISODES, RE.SWEEP_SEED, w, b, False)
            mean_score[i], std_score[i] = sc.mean(), sc.std()
            win[i], tie[i], loss[i] = (sc > 0).mean(), (sc == 0).mean(), (sc < 0).mean()
            mean_len[i] = ln.mean()
        data = dict(
            champs=champs.astype(np.float64), streaks=streaks, train_meanlen=meanlen,
            tie_rate=ties, hof_winrate=hofwins, mean_score=mean_score,
            std_score=std_score, win_rate=win, tie_rate_eval=tie, loss_rate=loss,
            eval_meanlen=mean_len, tournaments=np.array([RE.TOURNAMENTS]),
            save_every=np.array([RE.SAVE_EVERY]), sigma=np.array([0.10]),
            pop=np.array([128]), hof_prob=np.array([HOF_PROB]), seed=np.array([seed]),
            hof_capacity=np.array([CAPACITY]), hof_every=np.array([HOF_EVERY]),
            algo=np.array(["lab_niche"]), sweep_episodes=np.array([RE.SWEEP_EPISODES]),
            sweep_seed=np.array([RE.SWEEP_SEED]), train_sec=np.array([train_sec]),
            eval_sec=np.array([time.time() - t1]),
            pops=np.zeros((0, 0, 0), dtype=np.float32),
            pop_streaks=np.zeros((0, 0), dtype=np.int64), pop_every=np.array([0]),
            arch_size=arch, niche_bounds=np.array(BOUNDS["slime"]))
        note = f"final {mean_score[-1]:+.2f}, archive {arch[-1]} cells"
    else:
        g, gp, X1, X2, X3 = _gp("discmix", cfg["lam"])
        champs, streaks, meanlen, ties, hofwins, pops, pop_streaks, arch = K.run(
            g, gp, X1, X2, X3, seed, RL.TOURNAMENTS, RL.POP, RL.SIGMA, RL.SAVE_EVERY,
            K.HOF_NICHE, HOF_PROB, HOF_EVERY, CAPACITY, w, b, RL.INIT_SCALE,
            RL.POP_EVERY)
        pop_feats = np.array([G.features(p.astype(np.float64)) for p in pops])
        data = dict(
            champs=champs, champ_feats=G.features(champs), streaks=streaks, ties=ties,
            hof_winrate=hofwins, pop_feats=pop_feats, pop_streaks=pop_streaks,
            lam=np.array([cfg["lam"]]), mode=np.array(["niche"]), seed=np.array([seed]),
            gp=gp, tournaments=np.array([RL.TOURNAMENTS]),
            save_every=np.array([RL.SAVE_EVERY]), pop_every=np.array([RL.POP_EVERY]),
            arch_size=arch, train_sec=np.array([time.time() - t0]))
        note = f"final skill {data['champ_feats'][-1, 0]:+.2f}, archive {arch[-1]} cells"
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, path)
    return f"{cond}_s{seed}: {(time.time() - t0) / 60:.1f} min, {note}"


def write_protocol(path):
    """Record the design, additively; refuse to change a recorded entry."""
    new = json.loads(json.dumps({
        "hof_prob": HOF_PROB, "hof_every": HOF_EVERY, "capacity": CAPACITY,
        "grid": K.NICHE_GRID, "bounds": BOUNDS, "conditions": CONDITIONS,
        "slime_probe_seed": G.SLIME_PROBE_SEED, "slime_probes": G.SLIME_PROBES,
        "slime_probe_every": G.SLIME_PROBE_EVERY}))
    if os.path.exists(path) and json.load(open(path)) != new:
        raise SystemExit(f"{path} differs from this design; refusing to overwrite it")
    with open(path, "w") as f:
        json.dump(new, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--only", default=None, help="comma-separated conditions")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_protocol(os.path.join(args.outdir, "protocol.json"))
    conds = args.only.split(",") if args.only else list(CONDITIONS)
    jobs = [(c, s, args.outdir) for c in conds
            for s in ([int(x) for x in args.seeds.split(",")] if args.seeds
                      else CONDITIONS[c]["seeds"])]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
