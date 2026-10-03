"""
export_analysis.py — the preregistered test of alternative export rules (WP10).

Fixed in results/export/PREREGISTRATION.md before any run existed. In the
control GA the exported individual never feeds back into training, so every
rule is applied to the same stored population snapshots and each run is its
own paired comparison.

Rules, at every population snapshot (10 per run, one per 50,000 games):

  streak        argmax of the winning-streak counter (Ha's rule)
  tournament-k  argmax of the mean margin in an internal round robin with k
                random peers per member, (128 k) / 2 games: in Slime
                Volleyball fastvolley_kernels.internal_rank, the rule of the
                paper's post hoc re-export analysis; in discmix the same
                scheme with each game's outcome drawn from the game's own model
                (lab.kernels.discmix_play: the sign of the margin plus noise).
                k = 16 is the preregistered rule; k = 4 and 64 describe the
                budget
  random        one member drawn at random (a null)
  best          the member that scores best on the outcome itself (an oracle:
                not deployable, an upper bound only)

Outcomes:

  Slime Volleyball  the exported member's score against the 2015 baseline on
                    the held-out seed (SELECT_EPISODES on SELECT_SEED), as
                    every champion in the paper is re-scored; `best` is chosen
                    on RANK_EPISODES on a different seed and then re-scored on
                    the held-out seed like every other rule
  discmix           outsider strength: the exported member's exact mean
                    expected score against every member of the other runs'
                    pools at the same lambda and snapshot

Per run: level (mean over the 10 snapshots) and declines (summed falls
between consecutive snapshots) of each rule's series.

  H10a  Slime: tournament-16 level > streak level
  H10b  Slime: tournament-16 declines < streak declines
        (both one-sided paired sign-flip tests over the 12 runs; Holm)
  H10c  discmix: tournament-16 outsider level > streak, pooled over lambda
        (one-sided paired sign-flip test over the 48 runs)
  H10d  discmix: the advantage (tournament-16 minus streak, outsider level)
        falls as lambda rises: Spearman rho(lambda, advantage) < 0, one-sided
        permutation test (Holm with H10c)

    python export_analysis.py
"""

import argparse
import json
import math
import multiprocessing as mp
import os

import numpy as np
from numba import njit

import fastvolley as fv
import fastvolley_kernels as fk
import provenance as pv
import stats_utils as su
import yardsticks as ys
from lab import kernels as K
from run_experiments import SELECT_EPISODES, SELECT_SEED
from run_export import LAB_SEEDS, OUT as RUNS, SLIME_SEEDS
from run_lab import LAMBDAS

OUT = "results/export/analysis.json"
BUDGETS = (4, 16, 64)
PRIMARY = 16
RANK_EPISODES, RANK_SEED = 60, 20261011     # per member, for the oracle only
TOURNEY_SEED = 20261012                     # + 100 * snapshot + k
RANDOM_SEED = 20261013                      # + run seed
PERMUTATIONS, PERM_SEED = 20_000, 20261014
ALPHA = 0.05
RULES = ("streak",) + tuple(f"tournament-{k}" for k in BUDGETS) + ("random", "best")


# --------------------------------------------------------------------------
# Slime Volleyball
# --------------------------------------------------------------------------
def _heldout(genome, w, b):
    sc, _ = fv.eval_vs_baseline(np.ascontiguousarray(genome), SELECT_EPISODES,
                                SELECT_SEED, w, b, False)
    return float(sc.mean())


def slime_run(path):
    z = np.load(path)
    seed = int(z["seed"][0])
    pops, streaks = z["pops"], z["pop_streaks"]
    w, b = fv.baseline_arrays()
    rng = np.random.default_rng(RANDOM_SEED + seed)
    snaps = []
    for t in range(len(pops)):
        pop = np.ascontiguousarray(pops[t].astype(np.float64))
        ext, _ = fk.eval_population(pop, RANK_EPISODES, RANK_SEED, w, b)
        pick = {"streak": int(np.argmax(streaks[t]))}
        for k in BUDGETS:
            internal = fk.internal_rank(pop, k, TOURNEY_SEED + 100 * t + k, w, b)
            pick[f"tournament-{k}"] = int(np.argmax(internal))
        pick["random"] = int(rng.integers(len(pop)))
        pick["best"] = int(np.argmax(ext))
        held = {i: _heldout(pop[i], w, b) for i in set(pick.values())}
        order = np.argsort(-ext)
        snaps.append({"score": {r: held[i] for r, i in pick.items()},
                      "rank": {r: int(np.where(order == i)[0][0]) + 1
                               for r, i in pick.items()},
                      "index": pick})
    # a stronger yardstick at the last snapshot (descriptive): the zoo GA
    p, h = ys.load_zoo("zoo-ga")
    last = np.ascontiguousarray(pops[-1].astype(np.float64))
    zoo = {r: float(ys.match(np.ascontiguousarray(last[snaps[-1]["index"][r]]), 10,
                             p, h, ys.GAMES, ys.SEED).mean())
           for r in ("streak", f"tournament-{PRIMARY}", "best")}
    return os.path.basename(path)[:-4], {"seed": seed, "snapshots": snaps,
                                         "zoo_final": zoo, **_series(snaps)}


def _series(snaps):
    out = {}
    for r in RULES:
        v = np.array([s["score"][r] for s in snaps])
        d = np.diff(v)
        out[f"level_{r}"] = float(v.mean())
        out[f"declines_{r}"] = float(-d[d < 0].sum())
    return out


# --------------------------------------------------------------------------
# discmix
# --------------------------------------------------------------------------
@njit(cache=True)
def discmix_tournament(feats, n_opponents, gp, seed):
    """internal_rank's scheme, each game drawn as lab.kernels.discmix_play
    draws it: +1 / -1 / 0 by the sign of margin + noise, a tie in +/- tie."""
    np.random.seed(seed)
    k = feats.shape[0]
    fitness = np.zeros(k)
    played = np.zeros(k)
    for _ in range((k * n_opponents) // 2):
        i = np.random.randint(0, k)
        j = np.random.randint(0, k)
        while j == i:
            j = np.random.randint(0, k)
        x = K.discmix_margin(gp, feats[i], feats[j]) + np.random.normal(0.0, 1.0) * gp[3]
        sc = 1.0 if x > gp[4] else (-1.0 if x < -gp[4] else 0.0)
        fitness[i] += sc
        fitness[j] -= sc
        played[i] += 1.0
        played[j] += 1.0
    for i in range(k):
        if played[i] > 0:
            fitness[i] /= played[i]
    return fitness


@njit(cache=True)
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def outsider_strength(feats, others, gp):
    """Mean exact expected score of every row of `feats` against `others`
    (as lab.games.expected_score)."""
    out = np.zeros(feats.shape[0])
    for i in range(feats.shape[0]):
        s = 0.0
        for j in range(others.shape[0]):
            m = K.discmix_margin(gp, feats[i], others[j])
            s += (1.0 - _phi((gp[4] - m) / gp[3])) - _phi((-gp[4] - m) / gp[3])
        out[i] = s / others.shape[0]
    return out


def discmix_lambda(lam):
    zs = {s: np.load(os.path.join(RUNS, f"discmix-{lam:.2f}-control_s{s}.npz"))
          for s in LAB_SEEDS}
    gp = zs[LAB_SEEDS[0]]["gp"].astype(np.float64)
    runs = {}
    for s, z in zs.items():
        rng = np.random.default_rng(RANDOM_SEED + s)
        snaps = []
        for t in range(len(z["pop_feats"])):
            pf = np.ascontiguousarray(z["pop_feats"][t].astype(np.float64))
            others = np.ascontiguousarray(np.concatenate(
                [zs[o]["pop_feats"][t] for o in LAB_SEEDS if o != s]).astype(np.float64))
            strength = outsider_strength(pf, others, gp)
            pick = {"streak": int(np.argmax(z["pop_streaks"][t]))}
            for k in BUDGETS:
                internal = discmix_tournament(pf, k, gp, TOURNEY_SEED + 100 * t + k)
                pick[f"tournament-{k}"] = int(np.argmax(internal))
            pick["random"] = int(rng.integers(len(pf)))
            pick["best"] = int(np.argmax(strength))
            order = np.argsort(-strength)
            snaps.append({"score": {r: float(strength[i]) for r, i in pick.items()},
                          "rank": {r: int(np.where(order == i)[0][0]) + 1
                                   for r, i in pick.items()}})
        runs[f"discmix-{lam:.2f}-control_s{s}"] = {"seed": s, "lam": lam,
                                                    "snapshots": snaps, **_series(snaps)}
    return runs


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------
def _rule_table(runs):
    return {r: {"level": float(np.mean([x[f"level_{r}"] for x in runs])),
                "declines": float(np.mean([x[f"declines_{r}"] for x in runs])),
                "mean_rank": float(np.mean([s["rank"][r] for x in runs
                                            for s in x["snapshots"]]))}
            for r in RULES}


def analyse(slime, lab):
    T = f"tournament-{PRIMARY}"
    res = {}
    s = [slime[f"control_s{x}"] for x in SLIME_SEEDS]
    d_level = [x[f"level_{T}"] - x["level_streak"] for x in s]
    d_decl = [x["declines_streak"] - x[f"declines_{T}"] for x in s]
    a_mean, a_p = su.signflip_greater(d_level)
    b_mean, b_p = su.signflip_greater(d_decl)
    rej = su.holm([a_p, b_p], ALPHA)
    rt = _rule_table(s)
    gap = rt["best"]["level"] - rt["streak"]["level"]
    res["slime"] = {
        "rules": rt,
        "H10a": {"mean_gain": a_mean, "runs_improved": int(sum(x > 0 for x in d_level)),
                 "n": len(d_level), "p_one_sided": a_p, "rejected": rej[0]},
        "H10b": {"mean_fewer_declines": b_mean,
                 "runs_improved": int(sum(x > 0 for x in d_decl)),
                 "n": len(d_decl), "p_one_sided": b_p, "rejected": rej[1]},
        "recovered_fraction": {f"tournament-{k}": ((rt[f"tournament-{k}"]["level"]
                                                   - rt["streak"]["level"]) / gap
                                                  if gap > 0 else None) for k in BUDGETS},
        "zoo_final": {r: float(np.mean([x["zoo_final"][r] for x in s]))
                      for r in s[0]["zoo_final"]}}

    L = [lab[f"discmix-{lam:.2f}-control_s{x}"] for lam in LAMBDAS for x in LAB_SEEDS]
    adv = [x[f"level_{T}"] - x["level_streak"] for x in L]
    lams = [x["lam"] for x in L]
    c_mean, c_p = su.signflip_greater(adv, seed=PERM_SEED)
    rho = su.spearman(lams, adv)
    rng = np.random.default_rng(PERM_SEED)
    hits = sum(su.spearman(lams, rng.permutation(adv)) <= rho + 1e-12
               for _ in range(PERMUTATIONS))
    d_p = (hits + 1) / (PERMUTATIONS + 1)
    rej = su.holm([c_p, d_p], ALPHA)
    res["discmix"] = {
        "by_lambda": {f"{lam:.2f}": {
            "rules": _rule_table([x for x in L if x["lam"] == lam]),
            "advantage": float(np.mean([a for a, l in zip(adv, lams) if l == lam])),
            "runs_improved": int(sum(a > 0 for a, l in zip(adv, lams) if l == lam))}
            for lam in LAMBDAS},
        "H10c": {"mean_advantage": c_mean, "runs_improved": int(sum(a > 0 for a in adv)),
                 "n": len(adv), "p_one_sided": c_p, "rejected": rej[0]},
        "H10d": {"rho": rho, "p_one_sided": d_p, "rejected": rej[1]}}
    res["verdicts"] = {"H10a": res["slime"]["H10a"]["rejected"],
                       "H10b": res["slime"]["H10b"]["rejected"],
                       "H10c": res["discmix"]["H10c"]["rejected"],
                       "H10d": res["discmix"]["H10d"]["rejected"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    sp = [os.path.join(RUNS, f"control_s{s}.npz") for s in SLIME_SEEDS]
    lp = [os.path.join(RUNS, f"discmix-{lam:.2f}-control_s{s}.npz")
          for lam in LAMBDAS for s in LAB_SEEDS]
    missing = [p for p in sp + lp if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} run files missing, e.g. {missing[0]}")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        slime = dict(pool.imap_unordered(slime_run, sp))
        lab = {}
        for runs in pool.imap_unordered(discmix_lambda, LAMBDAS):
            lab.update(runs)
    res = analyse(slime, lab)
    res["per_run"] = {"slime": dict(sorted(slime.items())), "discmix": dict(sorted(lab.items()))}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "export")
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
