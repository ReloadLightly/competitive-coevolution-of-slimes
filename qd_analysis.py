"""
qd_analysis.py — the preregistered analysis of the niche-archive experiment
(WP9). Fixed in results/qd/PREREGISTRATION.md before any niche run existed.

Everything is recomputed from raw run files, the comparison runs included:

  Slime Volleyball   results/qd/slime-niche_s201-212 against the
                     replication's control_s201-212 and hof-eval-v2_s201-212,
                     with analyze_matrix.metrics and the held-out re-scoring
                     (the paper's own measurements);
  discmix            results/qd/discmix-<lam>-niche_s301-312 against WP8's
                     control and test runs at the same lambda; strength is
                     exact (lab.games.expected_score).

    python qd_analysis.py
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

import analyze_matrix as am
import coevolution_analysis as ca
import provenance as pv
import stats_utils as su
from lab import games as G
from run_lab import LAMBDAS
from run_qd import LAB_SEEDS, SLIME_SEEDS

OUT = "results/qd/analysis.json"
ALPHA = 0.05
OUTCOMES = ("final_holdout", "peak_holdout", "above_parity", "late_mean")
PERMUTATIONS, PERM_SEED = 20_000, 20261008
EPS = 1e-12
TOP_LAMBDA = max(LAMBDAS)


def slime_paths():
    out = {}
    for s in SLIME_SEEDS:
        out[f"niche_s{s}"] = f"results/qd/slime-niche_s{s}.npz"
        out[f"control_s{s}"] = f"results/replication/control_s{s}.npz"
        out[f"fifo_s{s}"] = f"results/replication/hof-eval-v2_s{s}.npz"
    return out


def _slime_job(job):
    name, path = job
    run = am.load_run(path)
    m = am.metrics(run)
    m.update(am._holdout_job((path,))[1])
    if "arch_size" in run:
        m["archive_cells_final"] = int(run["arch_size"][-1])
    return name, m


def lab_paths():
    out = {}
    for lam in LAMBDAS:
        for s in LAB_SEEDS:
            out[f"{lam:.2f}-niche_s{s}"] = f"results/qd/discmix-{lam:.2f}-niche_s{s}.npz"
            out[f"{lam:.2f}-control_s{s}"] = f"results/lab/discmix-{lam:.2f}-control_s{s}.npz"
            out[f"{lam:.2f}-test_s{s}"] = f"results/lab/discmix-{lam:.2f}-test_s{s}.npz"
    return out


def lab_run(path):
    z = np.load(path)
    lam = float(z["lam"][0])
    cf = z["champ_feats"]
    tri = ca.triad_stats(G.payoff_matrix(cf, lam), deadband=EPS)
    out = {"lam": lam, "final_feats": cf[-1].tolist(),
           "cyclic_share": (tri["cyclic"] / tri["triads_decided"]
                            if tri["triads_decided"] else 0.0)}
    if "arch_size" in z.files:
        out["archive_cells_final"] = int(z["arch_size"][-1])
    return out


def analyse_slime(per):
    def arm(k):
        return [per[f"{k}_s{s}"] for s in SLIME_SEEDS]
    niche, ctrl, fifo = arm("niche"), arm("control"), arm("fifo")
    res = {"learned": {k: sum(r["reached"] for r in arm(k))
                       for k in ("niche", "control", "fifo")},
           "n": len(niche)}
    for ref_name, ref in (("vs_control", ctrl), ("vs_fifo", fifo)):
        cmp = {}
        for o in OUTCOMES:
            a, b = [r[o] for r in niche], [r[o] for r in ref]
            cmp[o] = {"niche_mean": float(np.mean(a)), "ref_mean": float(np.mean(b)),
                      "cliffs_delta": float(su.cliffs_delta(a, b)),
                      "p_two_sided": su.mannwhitney_dp(a, b)[1]}
        res[ref_name] = cmp
    res["H9a_no_detectable_effect"] = all(
        c["p_two_sided"] >= ALPHA for c in res["vs_control"].values())
    res["archive_cells_final"] = [r["archive_cells_final"] for r in niche]
    res["archive_late_winrate"] = {
        "niche": [r["hof_winrate_late"] for r in niche],
        "fifo": [r["hof_winrate_late"] for r in fifo]}
    return res


def analyse_lab(runs):
    rng = np.random.default_rng(PERM_SEED)
    # cross-run strength within each lambda: control, test and niche finals
    for lam in LAMBDAS:
        names = sorted(n for n, r in runs.items() if r["lam"] == lam)
        P = G.payoff_matrix(np.array([runs[n]["final_feats"] for n in names]), lam)
        for i, n in enumerate(names):
            runs[n]["cross_strength"] = float(P[i].sum() / (len(names) - 1))

    def vals(lam, mode):
        return [runs[f"{lam:.2f}-{mode}_s{s}"]["cross_strength"] for s in LAB_SEEDS]

    by_lam, deltas = {}, {}
    for lam in LAMBDAS:
        row = {}
        for ref in ("control", "test"):
            a, b = vals(lam, "niche"), vals(lam, ref)
            row[f"vs_{ref}"] = {"cliffs_delta": float(su.cliffs_delta(a, b)),
                                "p_two_sided": su.mannwhitney_dp(a, b)[1]}
        row["archive_cells_final"] = float(np.mean(
            [runs[f"{lam:.2f}-niche_s{s}"]["archive_cells_final"] for s in LAB_SEEDS]))
        row["cyclic_share"] = {m: float(np.mean(
            [runs[f"{lam:.2f}-{m}_s{s}"]["cyclic_share"] for s in LAB_SEEDS]))
            for m in ("control", "test", "niche")}
        deltas[lam] = row["vs_control"]["cliffs_delta"]
        by_lam[f"{lam:.2f}"] = row

    # H9b: the niche effect (vs control) grows with lambda
    lc = np.array(LAMBDAS) - np.mean(LAMBDAS)
    obs = float(sum(c * deltas[l] for c, l in zip(lc, LAMBDAS)))
    hits = 0
    for _ in range(PERMUTATIONS):
        t = 0.0
        for c, lam in zip(lc, LAMBDAS):
            pool = vals(lam, "niche") + vals(lam, "control")
            lab = rng.permutation([1] * len(LAB_SEEDS) + [0] * len(LAB_SEEDS))
            a = [v for v, m in zip(pool, lab) if m == 1]
            b = [v for v, m in zip(pool, lab) if m == 0]
            t += c * su.cliffs_delta(a, b)
        hits += t >= obs - 1e-12
    h9b = (hits + 1) / (PERMUTATIONS + 1)
    # H9c: at the most cyclic lambda, niche beats the time-ordered archive
    h9c = su.mannwhitney_dp(vals(TOP_LAMBDA, "niche"), vals(TOP_LAMBDA, "test"),
                            "greater")[1]
    rej = su.holm([h9b, h9c], ALPHA)
    return {"by_lambda": by_lam, "H9b": {"trend": obs, "p_one_sided": h9b,
                                         "rejected": rej[0]},
            "H9c": {"lambda": TOP_LAMBDA, "p_one_sided": h9c, "rejected": rej[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    sp, lp = slime_paths(), lab_paths()
    missing = [p for p in list(sp.values()) + list(lp.values()) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} run files missing, e.g. {missing[0]}")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        slime = dict(pool.imap_unordered(_slime_job, sp.items()))
    lab = {n: lab_run(p) for n, p in lp.items()}
    res = {"slime": analyse_slime(slime), "discmix": analyse_lab(lab)}
    res["verdicts"] = {"H9a": res["slime"]["H9a_no_detectable_effect"],
                       "H9b": res["discmix"]["H9b"]["rejected"],
                       "H9c": res["discmix"]["H9c"]["rejected"]}
    res["per_run"] = {"slime": slime, "discmix": lab}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "qd")
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
