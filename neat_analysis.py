"""
neat_analysis.py — the preregistered analysis of NEAT as a fifth family (WP9).

Fixed in results/neat/PREREGISTRATION.md before any NEAT run existed.

  * per run: analyze_matrix.metrics, and the final and best champions
    re-scored on the held-out seed (1,000 episodes), as every matrix run;
  * the family table: NEAT next to the four families of the paper (C6);
  * H9d: NEAT against the generational GA, which shares its selection scheme
    and differs only in evolving a fixed topology: end-of-run champion (held
    out), two-sided exact Mann-Whitney;
  * a cross-run tournament: the final champion of every NEAT run and of every
    single-population matrix run, all played through the NEAT path (a matrix
    champion encoded as a NEAT genome plays exactly as itself), 25 games per
    pair and side, Bradley-Terry ratings on the Elo scale.

    python neat_analysis.py
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from numba import njit

import analyze_matrix as am
import coevolution_analysis as ca
import fastvolley as fv
import provenance as pv
import stats_utils as su
from lab import neat as N
from run_experiments import SELECT_EPISODES, SELECT_SEED
from run_neat import OUT as RUNS, SEEDS, champions

OUT = "results/neat/analysis.json"
FAMILIES = ("control", "ga2015", "es", "hof-eval-v2")
TOURNEY_GAMES = 25          # per pair and side, as coevolution_analysis --across
ALPHA = 0.05


def _neat_run(path):
    run = am.load_run(path)
    m = am.metrics(run)
    ch = champions(path)
    P = N.pack([N.compile_genome(g) for g in ch])
    w, b = fv.baseline_arrays()
    s = run["mean_score"]
    for tag, i in (("final", len(ch) - 1), ("peak", int(np.argmax(s)))):
        sc, _ = N.eval_vs_baseline(P, i, SELECT_EPISODES, SELECT_SEED, w, b)
        m[f"{tag}_holdout"] = float(sc.mean())
    z = np.load(path)
    m.update(final_hidden=int(z["hidden"][-1]), final_connections=int(z["connections"][-1]),
             species_mean=float(np.mean(z["n_species"])))
    return os.path.basename(path)[:-4], m


@njit(cache=True)
def round_robin(P, k, games, seed, w, b):
    """All-play-all among networks 0..k-1 of pack P, both sides, as
    fastvolley_kernels.round_robin: (wins, draws, losses) of i against j."""
    np.random.seed(seed)
    wins = np.zeros((k, k))
    draws = np.zeros((k, k))
    losses = np.zeros((k, k))
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    tr = np.zeros((1, 7))
    for i in range(k):
        for j in range(i + 1, k):
            for _ in range(games):
                for right, left, sign in ((i, j, 1), (j, i, -1)):
                    sc, _ = N.play(P, right, N.KIND_NEAT, left, N.KIND_NEAT, w, b,
                                   rnn_a, rnn_b, empty, empty, False, tr)
                    sc = sc * sign
                    if sc > 0:
                        wins[i, j] += 1
                        losses[j, i] += 1
                    elif sc < 0:
                        losses[i, j] += 1
                        wins[j, i] += 1
                    else:
                        draws[i, j] += 1
                        draws[j, i] += 1
    return wins, draws, losses


def tournament(neat_paths):
    inno = N.Innovations()
    names, genomes = [], []
    for p in pv.SCOPES["single"]():
        names.append(os.path.basename(p)[:-4])
        genomes.append(N.from_mlp(np.load(p)["champs"][-1].astype(np.float64), inno))
    for p in neat_paths:
        names.append(os.path.basename(p)[:-4])
        genomes.append(champions(p)[-1])
    P = N.pack([N.compile_genome(g) for g in genomes])
    w, b = fv.baseline_arrays()
    wins, draws, losses = round_robin(P, len(genomes), TOURNEY_GAMES, ca.RR_SEED, w, b)
    elo = ca.bradley_terry_elo(wins, draws, losses)
    return dict(zip(names, map(float, elo)))


def analyse(neat, per_run, elo):
    def cond_of(n):
        return n.rsplit("_s", 1)[0]
    fam = {f: [r for n, r in per_run.items() if cond_of(n) == f] for f in FAMILIES}
    fam["neat"] = list(neat.values())
    table = {}
    for f, rows in fam.items():
        fh = [r["final_holdout"] for r in rows]
        table[f] = {"runs": len(rows), "learned": sum(r["reached"] for r in rows),
                    "reached_parity": sum(r["t_parity"] is not None for r in rows),
                    "final_mean": float(np.mean(fh)), "final_sd": float(np.std(fh, ddof=1)),
                    "peak_mean": float(np.mean([r["peak_holdout"] for r in rows])),
                    "best_final": float(max(fh)),
                    "above_parity": float(np.mean([r["above_parity"] for r in rows])),
                    "elo_median": float(np.median([elo[n] for n in elo
                                                   if cond_of(n) == f]))}
    for f in table:
        table[f]["sd_ratio_vs_control"] = table[f]["final_sd"] / table["control"]["final_sd"]
    a = [r["final_holdout"] for r in fam["neat"]]
    g = [r["final_holdout"] for r in fam["ga2015"]]
    h9d = {"neat_mean": float(np.mean(a)), "ga2015_mean": float(np.mean(g)),
           "cliffs_delta": float(su.cliffs_delta(a, g)),
           "p_two_sided": su.mannwhitney_dp(a, g)[1]}
    h9d["detectable_difference"] = h9d["p_two_sided"] < ALPHA
    structure = {k: [r[k] for r in fam["neat"]]
                 for k in ("final_hidden", "final_connections", "species_mean")}
    return {"families": table, "H9d": h9d, "structure": structure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    paths = [os.path.join(RUNS, f"neat_s{s}.npz") for s in SEEDS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} NEAT runs missing")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        neat = dict(pool.imap_unordered(_neat_run, paths))
    per_run = json.load(open("results/analysis/per_run.json"))
    elo = tournament(paths)
    res = analyse(neat, per_run, elo)
    res["per_run"] = neat
    res["elo"] = elo
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "neat")
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_run", "elo")}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
