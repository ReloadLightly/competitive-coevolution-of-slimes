"""
coevolution_analysis.py — the three measurements the external yardstick
cannot make.

The score-versus-2015-baseline curve says how a champion fares against one
frozen opponent. Three questions about coevolution are invisible to it:

1. INTRANSITIVITY (`--within`). Do later champions actually beat earlier ones?
   Every checkpoint of a run plays every other checkpoint, both court sides.
   A transitive tournament with a rising Elo means the swings in the external
   curve are noise in the champion-selection proxy. Cyclic triads (A beats B
   beats C beats A) mean genuine coevolutionary cycling, which is the
   empirical argument for a hall of fame.

2. WHOSE CHAMPIONS ARE BETTER (`--across`). Every run's final champion plays
   every other run's final champion. This ranks the conditions with no frozen
   opponent involved at all, so it cannot be gamed by a policy that happens to
   specialise against the 2015 baseline.

3. IS THE EXPORTED CHAMPION THE BEST ONE? (`--proxy`). Ha exports the
   individual with the longest winning lineage, "without actually computing
   who is best to save time". The control runs keep full population snapshots,
   so every member can be scored against the baseline and compared with the
   one the proxy picked.

    python coevolution_analysis.py --within --across --proxy
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

import fastvolley as fv
import fastvolley_kernels as fk
import stats_utils as su

RR_SEED = 4242          # tournament seed, disjoint from training and evaluation
PROXY_EPISODES = 40


# --------------------------------------------------------------------------
def bradley_terry_elo(wins, draws, losses, iters=500, prior=1.0):
    """Elo ratings from a win/draw/loss matrix by MM (Zermelo) iteration.

    A draw counts as half a win to each side. `prior` adds that many virtual
    drawn games between every pair, which keeps a player who never won (or
    never lost) from running off to +/- infinity.
    """
    k = wins.shape[0]
    w = wins + 0.5 * draws + 0.5 * prior
    n = wins + draws + losses + prior
    np.fill_diagonal(w, 0.0)
    np.fill_diagonal(n, 0.0)
    wi = w.sum(axis=1)
    p = np.ones(k)
    for _ in range(iters):
        denom = np.zeros(k)
        for i in range(k):
            for j in range(k):
                if i != j and n[i, j] > 0:
                    denom[i] += n[i, j] / (p[i] + p[j])
        newp = np.where(denom > 0, wi / np.maximum(denom, 1e-12), p)
        newp = np.maximum(newp, 1e-12)
        newp /= np.exp(np.log(newp).mean())
        if np.max(np.abs(np.log(newp) - np.log(p))) < 1e-12:
            p = newp
            break
        p = newp
    return 400.0 * np.log10(p)


def triad_stats(margin, deadband=0.25):
    """Fraction of decided triads that are cyclic.

    A pair whose mean margin is inside +/-deadband is treated as undecided and
    every triad containing it is skipped, so the number reports cycles among
    relationships the data actually resolves.
    """
    k = margin.shape[0]
    beats = margin > deadband
    undecided = np.abs(margin) <= deadband
    total = 0
    cyclic = 0
    for i in range(k):
        for j in range(i + 1, k):
            if undecided[i, j]:
                continue
            for l in range(j + 1, k):
                if undecided[i, l] or undecided[j, l]:
                    continue
                total += 1
                # cyclic iff every player in the triad has exactly one win
                d = [int(beats[i, j]) + int(beats[i, l]),
                     int(beats[j, i]) + int(beats[j, l]),
                     int(beats[l, i]) + int(beats[l, j])]
                if sorted(d) == [1, 1, 1]:
                    cyclic += 1
    return {"triads_decided": total, "cyclic": cyclic,
            "cyclic_frac": (cyclic / total) if total else None,
            "pairs_undecided": int(undecided.sum() // 2)}


# --------------------------------------------------------------------------
def _within_job(job):
    path, every, games = job
    z = np.load(path)
    name = os.path.basename(path)[:-4]
    champs = z["champs"]
    if champs.shape[1] != fv.PARAM_COUNT:
        return name, None          # variable-capacity genomes: see run_asymmetric.py
    save_every = int(z["save_every"][0])
    step = max(1, every // save_every)
    idx = list(range(step - 1, len(champs), step))
    pop = np.ascontiguousarray(champs[idx])
    ts = [(i + 1) * save_every for i in idx]

    w, b = fv.baseline_arrays()
    margin, wins, draws, losses = fk.round_robin(pop, games, RR_SEED, w, b)
    elo = bradley_terry_elo(wins, draws, losses)
    tri = triad_stats(margin)
    return name, {
        "checkpoints": ts,
        "games_per_pair": 2 * games,
        "elo": elo.tolist(),
        "spearman_elo_vs_time": su.spearman(ts, elo),
        "spearman_elo_vs_external": su.spearman(
            elo, [z["mean_score"][i] for i in idx]),
        "external_score": [float(z["mean_score"][i]) for i in idx],
        **tri,
        "margin": margin.tolist(),
    }


def _proxy_job(job):
    path, episodes = job
    z = np.load(path)
    name = os.path.basename(path)[:-4]
    if z["pops"].size == 0:
        return name, None
    pops = z["pops"]
    pop_streaks = z["pop_streaks"]
    pop_every = int(z["pop_every"][0])
    w, b = fv.baseline_arrays()
    out = []
    for k in range(len(pops)):
        pop = np.ascontiguousarray(pops[k].astype(np.float64))
        means, wins = fk.eval_population(pop, episodes, RR_SEED, w, b)
        streaks = pop_streaks[k]
        exported = int(np.argmax(streaks))
        order = np.argsort(-means)
        rank = int(np.where(order == exported)[0][0])

        # How much of a collective is left? The update rule overwrites the
        # loser with a mutant of the winner, so the pool collapses towards one
        # lineage; these two numbers say how far that has gone.
        centroid = pop.mean(axis=0)
        spread = float(np.sqrt(((pop - centroid) ** 2).sum(axis=1).mean()))
        sq = (pop * pop).sum(axis=1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * pop @ pop.T, 0.0))
        iu = np.triu_indices(len(pop), 1)
        out.append({
            "genotype_spread": spread,
            "mean_pairwise_distance": float(d[iu].mean()),
            "min_pairwise_distance": float(d[iu].min()),
            "score_spread": float(means.std()),
            "tournament": (k + 1) * pop_every,
            "exported_idx": exported,
            "exported_score": float(means[exported]),
            "best_score": float(means.max()),
            "median_score": float(np.median(means)),
            "gap": float(means.max() - means[exported]),
            "exported_rank": rank,
            "pop_size": int(len(means)),
            "spearman_streak_vs_score": su.spearman(streaks, means),
            "n_above_parity": int((means > 0).sum()),
        })
    return name, out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/matrix")
    ap.add_argument("--outdir", default="results/analysis")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--within", action="store_true")
    ap.add_argument("--across", action="store_true")
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--every", type=int, default=50_000,
                    help="checkpoint spacing for the within-run round robin")
    ap.add_argument("--games", type=int, default=25,
                    help="games per ordered pair")
    ap.add_argument("--proxy-episodes", type=int, default=PROXY_EPISODES)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.matrix, "*_s*.npz")))
    if not paths:
        print("no runs yet")
        return

    if args.within:
        jobs = [(p, args.every, args.games) for p in paths]
        out = {}
        print(f"within-run round robins: {len(jobs)} runs", flush=True)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for name, res in pool.imap_unordered(_within_job, jobs):
                if res is None:
                    continue
                out[name] = res
                print(f"  {name}: rho(elo,time)={res['spearman_elo_vs_time']:+.2f} "
                      f"cyclic {res['cyclic']}/{res['triads_decided']}", flush=True)
        json.dump(out, open(os.path.join(args.outdir, "within_run.json"), "w"))

    if args.across:
        names, finals = [], []
        for p in paths:
            z = np.load(p)
            if z["champs"].shape[1] != fv.PARAM_COUNT:
                continue           # variable-capacity genomes cannot enter this pool
            names.append(os.path.basename(p)[:-4])
            finals.append(z["champs"][-1])
        pop = np.ascontiguousarray(np.array(finals))
        w, b = fv.baseline_arrays()
        print(f"cross-run tournament: {len(names)} final champions, "
              f"{args.games * 2} games per pair", flush=True)
        margin, wins, draws, losses = fk.round_robin(pop, args.games, RR_SEED, w, b)
        elo = bradley_terry_elo(wins, draws, losses)
        res = {"runs": names, "elo": elo.tolist(),
               "games_per_pair": 2 * args.games,
               "margin": margin.tolist(), **triad_stats(margin)}
        json.dump(res, open(os.path.join(args.outdir, "across_runs.json"), "w"))
        order = np.argsort(-elo)
        print("  Elo (best first):")
        for i in order:
            print(f"    {names[i]:<18} {elo[i]:+8.1f}")

    if args.proxy:
        jobs = [(p, args.proxy_episodes) for p in paths]
        out = {}
        print("champion-selection proxy check", flush=True)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for name, res in pool.imap_unordered(_proxy_job, jobs):
                if res is None:
                    continue
                out[name] = res
                gaps = [r["gap"] for r in res]
                print(f"  {name}: mean gap {np.mean(gaps):.2f} points "
                      f"(exported vs best in pool)", flush=True)
        json.dump(out, open(os.path.join(args.outdir, "champion_proxy.json"), "w"),
                  indent=1)

    print(f"-> {args.outdir}")


if __name__ == "__main__":
    main()
