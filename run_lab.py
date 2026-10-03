"""
run_lab.py — the discmix experiment (WP8): does skill structure change C3, C4
and C5b?

Ha's GA (control) and the archive used as a test, exactly as in the paper's
`control` and `hof-eval-v2` conditions, on the discmix game of lab/games.py at
four mixtures of transitive and cyclic skill (lambda). Design, hypotheses and
analysis are fixed in results/lab/PREREGISTRATION.md before any run exists.

A run file holds the 100 exported champions (one per 5,000 games) and, for
every population snapshot (one per 50,000 games), the features (skill, u0,
u1) and streak counter of every member. The game is defined by those
features alone, so every analysis is exact and needs no genomes beyond the
champions.

The runner is restartable: a run whose file exists is skipped.

    python run_lab.py                          # everything, one worker per core
    python run_lab.py --only discmix-0.50-test --seeds 301,302
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fastvolley as fv
from lab import games as G
from lab import kernels as K

OUT = "results/lab"
TOURNAMENTS = 500_000
SAVE_EVERY = 5_000
POP_EVERY = 50_000
POP = 128
SIGMA = 0.10
INIT_SCALE = 0.5
HOF_PROB, HOF_EVERY, HOF_CAP = 0.25, 1_000, 512      # as hof-eval-v2
SEEDS = list(range(301, 313))
LAMBDAS = (0.0, 0.25, 0.5, 0.75)
MODES = {"control": (K.HOF_NONE, 0.0), "test": (K.HOF_TEST, HOF_PROB)}

CONDITIONS = {f"discmix-{lam:.2f}-{mode}": dict(lam=lam, mode=mode)
              for lam in LAMBDAS for mode in MODES}


def one_run(job):
    cond, seed, outdir = job
    path = os.path.join(outdir, f"{cond}_s{seed}.npz")
    if os.path.exists(path):
        return f"{cond}_s{seed}: already done"
    cfg = CONDITIONS[cond]
    hof_mode, hof_prob = MODES[cfg["mode"]]
    game, gp, X1, X2, X3 = G.make("discmix", cfg["lam"])
    w, b = fv.baseline_arrays()
    t0 = time.time()
    champs, streaks, meanlen, ties, hofwins, pops, pop_streaks = K.run(
        game, gp, X1, X2, X3, seed, TOURNAMENTS, POP, SIGMA, SAVE_EVERY,
        hof_mode, hof_prob, HOF_EVERY, HOF_CAP, w, b, INIT_SCALE, POP_EVERY)
    pop_feats = np.array([G.features(p.astype(np.float64)) for p in pops])
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, champs=champs, champ_feats=G.features(champs), streaks=streaks,
            ties=ties, hof_winrate=hofwins, pop_feats=pop_feats,
            pop_streaks=pop_streaks, lam=np.array([cfg["lam"]]),
            mode=np.array([cfg["mode"]]), seed=np.array([seed]),
            gp=gp, tournaments=np.array([TOURNAMENTS]),
            save_every=np.array([SAVE_EVERY]), pop_every=np.array([POP_EVERY]),
            train_sec=np.array([time.time() - t0]))
    os.replace(tmp, path)
    return (f"{cond}_s{seed}: {(time.time() - t0) / 60:.1f} min, final skill "
            f"{G.features(champs[-1:])[0, 0]:+.2f}")


def write_protocol(path):
    """Record the design, additively; refuse to change a recorded entry."""
    new = json.loads(json.dumps({
        "tournaments": TOURNAMENTS, "save_every": SAVE_EVERY,
        "pop_every": POP_EVERY, "pop": POP, "sigma": SIGMA,
        "init_scale": INIT_SCALE, "hof_prob": HOF_PROB, "hof_every": HOF_EVERY,
        "hof_capacity": HOF_CAP, "seeds": SEEDS, "conditions": CONDITIONS,
        "game": {"probe_seed": G.PROBE_SEED, "n_skill": G.N_SKILL,
                 "n_style": G.N_STYLE, "teacher_scale": G.TEACHER_SCALE,
                 "alpha": G.ALPHA, "beta": G.BETA, "noise": G.NOISE,
                 "tie": G.TIE}}))
    if os.path.exists(path):
        old = json.load(open(path))
        if old != new:
            raise SystemExit(f"{path} differs from this design; refusing to "
                             f"overwrite it")
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
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = [(c, s, args.outdir) for c in conds for s in seeds]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
