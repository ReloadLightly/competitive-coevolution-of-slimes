"""
lab_analysis.py — the preregistered analysis of the discmix experiment (WP8).

Fixed in results/lab/PREREGISTRATION.md before any run existed. Every
quantity is computed from exact expected scores (lab.games.expected_score),
so nothing here is sampled except the permutation nulls, whose seeds are
fixed.

Per run:
  C3  cyclic share of triads among the 100 exported champions (exact payoff
      matrix; a pair whose expected score is exactly 0 is undecided), and
      rho(row strength, time);
  C4  at every population snapshot: the exported individual's rank (1 =
      best) by true strength in its pool (mean expected score against the
      rest of the pool), rho(streak, strength), and the summed declines of the
      exported individual and of the pool's best member against a fixed
      external panel (64 genomes from the initial distribution, never seen in
      training: the analogue of the 2015 baseline);
  C5b the final champion's cross-run strength: its mean expected score
      against the final champions of every other run at the same lambda.

    python lab_analysis.py
"""

import argparse
import json
import os

import numpy as np

import coevolution_analysis as ca
import provenance as pv
import stats_utils as su
from lab import games as G
from run_lab import CONDITIONS, LAMBDAS, OUT as RUNS, SEEDS

OUT = "results/lab/analysis.json"
PANEL_SEED, PANEL_N = 20261005, 64
PERMUTATIONS, PERM_SEED = 20_000, 20261006
ALPHA = 0.05
TOP_QUARTER = 32
EPS = 1e-12            # an expected score this close to 0 is a draw


def panel_features():
    rng = np.random.default_rng(PANEL_SEED)
    return G.features(rng.normal(size=(PANEL_N, G.K.PARAM_COUNT)) * 0.5)


def strength_vs(feats, others, lam):
    """Mean expected score of each row of `feats` against `others`."""
    return np.array([np.mean([G.expected_score(f, o, lam) for o in others])
                     for f in feats])


def per_run(path, panel):
    z = np.load(path)
    lam = float(z["lam"][0])
    cf = z["champ_feats"]
    M = G.payoff_matrix(cf, lam)
    tri = ca.triad_stats(M, deadband=EPS)
    ts = (np.arange(len(cf)) + 1) * int(z["save_every"][0])
    out = {"lam": lam, "mode": str(z["mode"][0]),
           "cyclic": tri["cyclic"], "triads_decided": tri["triads_decided"],
           "rho_strength_time": su.spearman(ts, M.mean(axis=1)),
           "final_skill": float(cf[-1, 0])}
    snaps = []
    for pf, st in zip(z["pop_feats"], z["pop_streaks"]):
        P = G.payoff_matrix(pf, lam)
        strength = P.sum(axis=1) / (len(pf) - 1)
        ext = strength_vs(pf, panel, lam)
        e = int(np.argmax(st))
        order = np.argsort(-strength)
        snaps.append({"exported_rank": int(np.where(order == e)[0][0]) + 1,
                      "rho_streak_strength": su.spearman(st, strength),
                      "exported_ext": float(ext[e]),
                      "best_ext": float(ext.max()),
                      "median_ext": float(np.median(ext))})
    out["snapshots"] = snaps

    def decline(key):
        d = np.diff([s[key] for s in snaps])
        return float(-d[d < 0].sum())
    out["mean_rank"] = float(np.mean([s["exported_rank"] for s in snaps]))
    out["mean_rho_streak"] = float(np.mean([s["rho_streak_strength"] for s in snaps]))
    out["decline_exported"] = decline("exported_ext")
    out["decline_best"] = decline("best_ext")
    out["final_feats"] = cf[-1].tolist()
    return out


def perm_spearman(x, y, rng, n):
    """One-sided permutation p for Spearman rho(x, y) > 0."""
    obs = su.spearman(x, y)
    y = np.asarray(y)
    hits = sum(su.spearman(x, rng.permutation(y)) >= obs - 1e-12 for _ in range(n))
    return obs, (hits + 1) / (n + 1)


def analyse(runs):
    rng = np.random.default_rng(PERM_SEED)
    res = {}

    # cross-run strength of every final champion, within each lambda
    for lam in LAMBDAS:
        names = sorted(n for n, r in runs.items() if r["lam"] == lam)
        F = np.array([runs[n]["final_feats"] for n in names])
        P = G.payoff_matrix(F, lam)
        for i, n in enumerate(names):
            runs[n]["cross_strength"] = float(P[i].sum() / (len(names) - 1))

    # H8a: within-run cyclic share rises with lambda (per mode)
    res["H8a"] = {}
    for mode in ("control", "test"):
        rs = [r for r in runs.values() if r["mode"] == mode]
        x = [r["lam"] for r in rs]
        y = [r["cyclic"] / r["triads_decided"] if r["triads_decided"] else 0.0
             for r in rs]
        rho, p = perm_spearman(x, y, rng, PERMUTATIONS)
        res["H8a"][mode] = {"rho": rho, "p_one_sided": p,
                            "share_by_lambda": {f"{l:.2f}": float(np.mean(
                                [yy for xx, yy in zip(x, y) if xx == l]))
                                for l in LAMBDAS}}

    # H8b: the exported individual sits outside its pool's top quarter, per
    # lambda (control runs), Holm over the four lambdas
    ps, rows = [], {}
    for lam in LAMBDAS:
        rs = [r for r in runs.values() if r["mode"] == "control" and r["lam"] == lam]
        k, n, p = su.sign_test_greater([r["mean_rank"] - TOP_QUARTER for r in rs])
        rows[f"{lam:.2f}"] = {"above_top_quarter": k, "informative": n, "p": p,
                              "mean_rank": float(np.mean([r["mean_rank"] for r in rs])),
                              "mean_rho_streak": float(np.mean([r["mean_rho_streak"] for r in rs])),
                              "decline_exported": float(sum(r["decline_exported"] for r in rs)),
                              "decline_best": float(sum(r["decline_best"] for r in rs))}
        ps.append(p)
    for (lam, row), rej in zip(rows.items(), su.holm(ps, ALPHA)):
        row["rejected"] = rej
    res["H8b"] = rows

    # H8c: the archive-as-test effect on cross-run strength grows with lambda
    deltas, by_lam = {}, {}
    for lam in LAMBDAS:
        a = [r["cross_strength"] for r in runs.values() if r["lam"] == lam and r["mode"] == "test"]
        b = [r["cross_strength"] for r in runs.values() if r["lam"] == lam and r["mode"] == "control"]
        u, p = su.mannwhitney_dp(a, b)
        deltas[lam] = su.cliffs_delta(a, b)
        by_lam[f"{lam:.2f}"] = {"cliffs_delta": float(deltas[lam]), "p_two_sided": p,
                                "test_mean": float(np.mean(a)), "control_mean": float(np.mean(b))}
    lc = np.array(LAMBDAS) - np.mean(LAMBDAS)
    obs = float(sum(c * deltas[l] for c, l in zip(lc, LAMBDAS)))
    hits = 0
    for _ in range(PERMUTATIONS):
        t = 0.0
        for c, lam in zip(lc, LAMBDAS):
            vals = [r["cross_strength"] for r in runs.values() if r["lam"] == lam]
            lab = rng.permutation([r["mode"] for r in runs.values() if r["lam"] == lam])
            a = [v for v, m in zip(vals, lab) if m == "test"]
            b = [v for v, m in zip(vals, lab) if m == "control"]
            t += c * su.cliffs_delta(a, b)
        hits += t >= obs - 1e-12
    res["H8c"] = {"by_lambda": by_lam, "trend": obs,
                  "p_one_sided": (hits + 1) / (PERMUTATIONS + 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    paths = [os.path.join(args.runs, f"{c}_s{s}.npz") for c in CONDITIONS for s in SEEDS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} runs missing")
    panel = panel_features()
    runs = {os.path.basename(p)[:-4]: per_run(p, panel) for p in paths}
    res = analyse(runs)
    res["per_run"] = runs
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "lab")
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
