"""
export_robustness.py — the discmix tests of the export experiment, with the
seed as the unit.

The preregistered H10c and H10d (export_analysis.py) treat the 48 discmix runs
as independent. They are not entirely: the four runs that share a seed (one
per lambda) start from the same initial population, because lab.kernels draws
it from the seed alone, and the runs at one lambda are scored against each
other's populations. Raised in an external review on 2026-10-03
(results/matrix/decisions.md). This script repeats both tests with the seed as
the unit. It describes robustness; it does not replace the preregistered
tests, whose verdicts stand.

  H10c by seed  the mean advantage (tournament-16 minus streak, outsider
                level) over each seed's four lambda runs; one-sided exact
                sign-flip test over the 12 seeds (2^12 patterns).
  H10d by seed  Spearman rho(lambda, advantage) over the 48 runs, with lambda
                permuted only within each seed (20,000 permutations, the
                preregistered seed), so the null keeps a seed's four runs
                together.

The shared outsider panels remain: every run at one lambda is scored against
the populations of the other eleven. No test here removes that dependence.

    python export_robustness.py
"""

import argparse
import json

import numpy as np

import export_analysis as xa
import provenance as pv
import stats_utils as su

OUT = "results/analysis/export_robustness.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    a = json.load(open(xa.OUT))
    lab = a["per_run"]["discmix"]
    T = f"tournament-{xa.PRIMARY}"
    seeds = list(xa.LAB_SEEDS)
    lams = list(xa.LAMBDAS)
    adv = np.array([[lab[f"discmix-{lam:.2f}-control_s{s}"][f"level_{T}"]
                     - lab[f"discmix-{lam:.2f}-control_s{s}"]["level_streak"]
                     for lam in lams] for s in seeds])          # seeds x lambdas

    by_seed = adv.mean(axis=1)
    c_mean, c_p = su.signflip_greater(by_seed)

    flat_lam = np.tile(lams, len(seeds))
    rho = su.spearman(flat_lam, adv.ravel())
    rng = np.random.default_rng(xa.PERM_SEED)
    hits = 0
    for _ in range(xa.PERMUTATIONS):
        perm = np.concatenate([rng.permutation(lams) for _ in seeds])
        hits += su.spearman(perm, adv.ravel()) <= rho + 1e-12
    d_p = (hits + 1) / (xa.PERMUTATIONS + 1)

    res = {
        "preregistered": {"H10c_p": a["discmix"]["H10c"]["p_one_sided"],
                          "H10d_p": a["discmix"]["H10d"]["p_one_sided"],
                          "H10d_rho": a["discmix"]["H10d"]["rho"]},
        "H10c_by_seed": {"n_seeds": len(seeds), "mean_advantage": c_mean,
                         "seeds_improved": int((by_seed > 0).sum()),
                         "p_one_sided": c_p},
        "H10d_within_seed": {"rho": rho, "permutations": xa.PERMUTATIONS,
                             "p_one_sided": d_p,
                             "seeds_with_negative_trend": int(sum(
                                 su.spearman(lams, adv[i]) < 0 for i in range(len(seeds))))},
        "advantage_by_seed_and_lambda": {str(s): {f"{lam:.2f}": float(adv[i, j])
                                                  for j, lam in enumerate(lams)}
                                         for i, s in enumerate(seeds)}}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "export")
    print(json.dumps({k: v for k, v in res.items() if k != "advantage_by_seed_and_lambda"},
                     indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
