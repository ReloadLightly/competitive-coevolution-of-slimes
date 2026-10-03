"""
transitivity_fine.py — the within-run tournament at 5,000-game spacing (WP7).

`coevolution_analysis.py --within` plays checkpoints 50,000 games apart, so a
cycle that opens and closes within 50,000 games is invisible to it, and "the
population is not cycling" (C3) holds only at that resolution. Every run
stores all 100 of its champions, one per 5,000 games, so the same round robin
can be played on all of them. This script does that for the control runs, the
condition C3 is stated for, with the same tournament code, seed and deadband;
only the games per pair are fewer, because there are 4,950 pairs per run
instead of 45.

Besides the paper's two statistics (rho(Elo, time) and the cyclic share of
decided triads) it reports the two the coarse tournament could not:

  short triads   the cyclic share among decided triads spanning at most
                 50,000 games, i.e. the cycles the coarse tournament missed;
  next beats     how often a champion beats the one exported 5,000 games
                 earlier, among decided adjacent pairs.

"Decided" is computed two ways. The paper's rule (mean margin outside
+/-0.25) is kept for comparability, but at 20 games per pair the margin
between two near-equal policies has a standard error of the order of the
deadband, so sampling noise decides pairs and noise makes cycles. The
second rule decides a pair only if an exact sign test on its wins against
its losses gives p < 0.05.

    python transitivity_fine.py
"""

import argparse
import json
import math
import multiprocessing as mp
import os

import numpy as np

import coevolution_analysis as ca
import fastvolley as fv
import fastvolley_kernels as fk
import provenance as pv
import stats_utils as su

OUT = "results/analysis/within_fine.json"
EVERY = 5_000
GAMES = 10              # per side, so 20 per pair
SHORT = 50_000          # the coarse tournament's spacing
DEADBAND = 0.25         # as in coevolution_analysis.triad_stats
ALPHA = 0.05            # sign-test rule: a pair is decided if p < ALPHA


def sign_p(w, l):
    """Exact two-sided binomial p for w wins against l losses (draws dropped)."""
    n = int(w + l)
    if n == 0:
        return 1.0
    k = int(min(w, l))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def triads(beats, decided, ts, span=None):
    """(decided triads, cyclic triads), optionally only those within `span`."""
    k = beats.shape[0]
    total = cyclic = 0
    for i in range(k):
        for j in range(i + 1, k):
            if span is not None and ts[j] - ts[i] > span:
                break
            if not decided[i, j]:
                continue
            for l in range(j + 1, k):
                if span is not None and ts[l] - ts[i] > span:
                    break
                if not (decided[i, l] and decided[j, l]):
                    continue
                total += 1
                w = [int(beats[i, j]) + int(beats[i, l]),
                     int(beats[j, i]) + int(beats[j, l]),
                     int(beats[l, i]) + int(beats[l, j])]
                cyclic += sorted(w) == [1, 1, 1]
    return total, cyclic


def _job(path):
    """The round robin of coevolution_analysis._within_job at 5,000-game
    spacing (same kernel, seed and checkpoint selection), scored two ways."""
    z = np.load(path)
    name = os.path.basename(path)[:-4]
    save_every = int(z["save_every"][0])
    step = max(1, EVERY // save_every)
    idx = list(range(step - 1, len(z["champs"]), step))
    ts = [(i + 1) * save_every for i in idx]
    w, b = fv.baseline_arrays()
    margin, wins, draws, losses = fk.round_robin(
        np.ascontiguousarray(z["champs"][idx]), GAMES, ca.RR_SEED, w, b)
    elo = ca.bradley_terry_elo(wins, draws, losses)
    k = len(idx)
    p = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            p[i, j] = p[j, i] = sign_p(wins[i, j], losses[i, j])
    rules = {
        # the paper's rule: decided if the mean margin leaves +/- DEADBAND
        "deadband": (margin > DEADBAND, np.abs(margin) > DEADBAND),
        # noise-aware: decided if the exact sign test on wins vs losses says so
        "sign_test": (wins > losses, p < ALPHA),
    }
    out = {"checkpoints": ts, "games_per_pair": 2 * GAMES,
           "spearman_elo_vs_time": su.spearman(ts, elo)}
    for rule, (beats, decided) in rules.items():
        np.fill_diagonal(decided, False)
        tot, cyc = triads(beats, decided, ts)
        stot, scyc = triads(beats, decided, ts, SHORT)
        adj = [(i + 1, i) for i in range(k - 1) if decided[i + 1, i]]
        out[rule] = {"triads_decided": tot, "cyclic": cyc,
                     "short_triads_decided": stot, "short_cyclic": scyc,
                     "pairs_decided": int(decided[np.triu_indices(k, 1)].sum()),
                     "adjacent_decided": len(adj),
                     "adjacent_later_wins": int(sum(beats[a, b_] for a, b_ in adj))}
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    paths = pv.SCOPES["control"]()
    out = {"every": EVERY, "games_per_pair": 2 * GAMES, "deadband": DEADBAND,
           "alpha": ALPHA, "short_span": SHORT, "runs": {}}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, res in pool.imap_unordered(_job, paths):
            out["runs"][name] = res
            d, t = res["deadband"], res["sign_test"]
            print(f"  {name}: rho {res['spearman_elo_vs_time']:+.2f} | deadband "
                  f"{d['cyclic']}/{d['triads_decided']} short {d['short_cyclic']}/"
                  f"{d['short_triads_decided']} | sign test {t['cyclic']}/"
                  f"{t['triads_decided']} short {t['short_cyclic']}/"
                  f"{t['short_triads_decided']} next>prev {t['adjacent_later_wins']}/"
                  f"{t['adjacent_decided']}", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    pv.record(OUT, "control", params={"every": EVERY, "games": GAMES,
                                      "alpha": ALPHA})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
