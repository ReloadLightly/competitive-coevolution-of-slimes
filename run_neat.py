"""
run_neat.py — NEAT as a fifth family (WP9, part 2).

NEAT (lab/neat.py) under the generational GA's self-play evaluation, 500,000
games per run, evaluated exactly as every matrix run: each of the 100
exported champions plays 200 episodes against the 2015 baseline on the
sweep seed. Design and analysis are fixed in results/neat/PREREGISTRATION.md
before any run exists.

A run file has the keys analyze_matrix.metrics reads, plus the champions'
genomes (variable-size, stored flat with offsets) and the structure of the
search: species, hidden nodes and enabled connections per checkpoint.

    python run_neat.py                  # all seeds, one worker per core
    python run_neat.py --seeds 101
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fastvolley as fv
import run_experiments as RE
from lab import neat as N

OUT = "results/neat"
SEEDS = list(range(101, 113))


def one_run(job):
    seed, outdir = job
    path = os.path.join(outdir, f"neat_s{seed}.npz")
    if os.path.exists(path):
        return f"neat_s{seed}: already done"
    w, b = fv.baseline_arrays()
    t0 = time.time()
    out = N.run(seed, RE.TOURNAMENTS, RE.SAVE_EVERY, w, b)
    train_sec = time.time() - t0
    champs = out["champs"]
    n = len(champs)
    P = N.pack([N.compile_genome(g) for g in champs])
    mean_score, std_score = np.zeros(n), np.zeros(n)
    win, tie, loss, mean_len = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    t1 = time.time()
    for i in range(n):
        sc, ln = N.eval_vs_baseline(P, i, RE.SWEEP_EPISODES, RE.SWEEP_SEED, w, b)
        mean_score[i], std_score[i] = sc.mean(), sc.std()
        win[i], tie[i], loss[i] = (sc > 0).mean(), (sc == 0).mean(), (sc < 0).mean()
        mean_len[i] = ln.mean()
    arrays = [N.genome_arrays(g) for g in champs]
    node_off = np.cumsum([0] + [len(a[0]) for a in arrays])
    conn_off = np.cumsum([0] + [len(a[1]) for a in arrays])
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            champ_nodes=np.concatenate([a[0] for a in arrays]), champ_node_off=node_off,
            champ_conns=np.concatenate([a[1] for a in arrays]), champ_conn_off=conn_off,
            champ_fit=np.array(out["champ_fit"]), n_species=np.array(out["n_species"]),
            hidden=np.array(out["hidden"]), connections=np.array(out["connections"]),
            threshold=np.array(out["threshold"]),
            streaks=np.zeros(n, dtype=np.int64), train_meanlen=np.array(out["meanlen"]),
            tie_rate=np.zeros(n), hof_winrate=np.zeros(n),
            mean_score=mean_score, std_score=std_score, win_rate=win,
            tie_rate_eval=tie, loss_rate=loss, eval_meanlen=mean_len,
            tournaments=np.array([RE.TOURNAMENTS]), save_every=np.array([RE.SAVE_EVERY]),
            sigma=np.array([N.PARAMS["sigma"]]), pop=np.array([N.PARAMS["pop"]]),
            hof_prob=np.array([0.0]), seed=np.array([seed]), algo=np.array(["neat"]),
            sweep_episodes=np.array([RE.SWEEP_EPISODES]),
            sweep_seed=np.array([RE.SWEEP_SEED]), train_sec=np.array([train_sec]),
            eval_sec=np.array([time.time() - t1]),
            pops=np.zeros((0, 0, 0), dtype=np.float32),
            pop_streaks=np.zeros((0, 0), dtype=np.int64), pop_every=np.array([0]))
    os.replace(tmp, path)
    return (f"neat_s{seed}: train {train_sec / 60:.1f} min, final {mean_score[-1]:+.2f}, "
            f"best {mean_score.max():+.2f}, {out['hidden'][-1]} hidden, "
            f"{out['connections'][-1]} connections")


def champions(path):
    """The exported champions of a run file, as genomes."""
    z = np.load(path)
    no, co = z["champ_node_off"], z["champ_conn_off"]
    return [N.genome_from_arrays(z["champ_nodes"][no[i]:no[i + 1]],
                                 z["champ_conns"][co[i]:co[i + 1]])
            for i in range(len(no) - 1)]


def write_protocol(path):
    new = json.loads(json.dumps({
        "tournaments": RE.TOURNAMENTS, "save_every": RE.SAVE_EVERY,
        "sweep_episodes": RE.SWEEP_EPISODES, "sweep_seed": RE.SWEEP_SEED,
        "seeds": SEEDS, "params": N.PARAMS}))
    if os.path.exists(path) and json.load(open(path)) != new:
        raise SystemExit(f"{path} differs from this design; refusing to overwrite it")
    with open(path, "w") as f:
        json.dump(new, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_protocol(os.path.join(args.outdir, "protocol.json"))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, [(s, args.outdir) for s in seeds]):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
