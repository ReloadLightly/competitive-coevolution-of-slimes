"""
replication.py — the preregistered confirmatory replication (WP6).

Every test, threshold and decision rule below was fixed in
results/replication/PREREGISTRATION.md and committed before the first
replication run was started. Change nothing here after data exists; a change
would need a dated entry in results/matrix/decisions.md and would turn the
affected hypothesis into an exploratory one.

The measurements are the paper's own, imported rather than re-implemented:
analyze_matrix.metrics and the held-out re-scoring, the within-run round robin
of coevolution_analysis, and the snapshot scoring of reexport.

    python replication.py                     # all 36 runs must exist
    python replication.py --partial           # progress view, decides nothing
    python replication.py --runs results/matrix --seeds 101-106 \\
        --out /tmp/original.json              # the same analysis, original seeds
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

import analyze_matrix as am
import coevolution_analysis as ca
import provenance as pv
import reexport as rx
import stats_utils as su

RUNS = "results/replication"
OUT = "results/replication/analysis.json"
CONTROL, PARENT, TEST = "control", "hof-0.25", "hof-eval-v2"
CONDITIONS = (CONTROL, PARENT, TEST)
SEEDS = list(range(201, 213))
ALPHA = 0.05

# measurement settings, identical to the paper's committed analyses
WITHIN_EVERY, WITHIN_GAMES = 50_000, 25          # within_run.json
REEXPORT_OPPONENTS, REEXPORT_EPISODES = [4, 8, 16, 32, 64], 60   # reexport.json

# decision thresholds (PREREGISTRATION.md, section 3)
MIN_LEARNED = (10, 12)      # H2a: at least 10 of 12 control runs learn to rally
                            # (the same share for other run counts: 5 of 6)
TIMING_RATIO = 3.0          # H2b: latest / earliest internal transition
CYCLIC_MAX = 0.01           # H3b: pooled cyclic share of decided triads
TOP_QUARTER = 32            # H4a: rank 1 = best of 128; above 32 = outside top quarter
RHO_BOUND = 0.2             # H4b: equivalence bound for rho(streak, skill)
C5B_OUTCOMES = ("final_holdout", "peak_holdout", "above_parity", "late_mean")
ARCHIVE_LEARNED_MAX = 0.20  # H6 description: late archive win rate, runs that learned
ARCHIVE_FAILED_MIN = 0.40   # H6 description: late archive win rate, runs that did not


def _seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def _per_run(job):
    """Every per-run measurement for one run file."""
    path, cond = job
    run = am.load_run(path)
    out = am.metrics(run)
    out.update(am._holdout_job((path,))[1])
    if cond == CONTROL:
        out["within"] = ca._within_job((path, WITHIN_EVERY, WITHIN_GAMES))[1]
        out["snapshots"] = rx._job((path, REEXPORT_OPPONENTS,
                                    REEXPORT_EPISODES))[1]
    return run["name"], out


def _decline(series):
    d = np.diff(series)
    return float(-d[d < 0].sum())


def analyse(per_run, seeds):
    def rows(cond):
        return [per_run[f"{cond}_s{s}"] for s in seeds
                if f"{cond}_s{s}" in per_run]

    ctrl, parent, test = rows(CONTROL), rows(PARENT), rows(TEST)
    res = {"n": {c: len(rows(c)) for c in CONDITIONS}}

    # ---- H1 (C1): internal improvement precedes external transfer --------
    lags = [r["lag_internal_to_parity"] for r in ctrl
            if r["lag_internal_to_parity"] is not None]
    k, n, p = su.sign_test_greater(lags)
    res["H1"] = {"lags": lags, "positive": k, "informative": n, "p": p}

    # ---- H2 (C2): the transition is robust, its timing is not ------------
    learned = sum(r["reached"] for r in ctrl)
    t_int = [r["t_internal"] for r in ctrl if r["t_internal"] is not None]
    ratio = (max(t_int) / min(t_int)) if t_int else None
    res["H2"] = {"learned": learned, "of": len(ctrl), "t_internal": t_int,
                 "timing_ratio": ratio,
                 "a_holds": MIN_LEARNED[1] * learned >= MIN_LEARNED[0] * len(ctrl),
                 "b_holds": ratio is not None and ratio >= TIMING_RATIO}

    # ---- H3 (C3): the population is not cycling --------------------------
    within = [r["within"] for r in ctrl if r.get("within")]
    rhos = [w["spearman_elo_vs_time"] for w in within]
    k, n, p = su.sign_test_greater(rhos)
    cyc = sum(w["cyclic"] for w in within)
    dec = sum(w["triads_decided"] for w in within)
    res["H3"] = {"rho_elo_time": rhos, "rho_mean": float(np.mean(rhos)) if rhos else None,
                 "positive": k, "informative": n, "p": p,
                 "cyclic": cyc, "triads_decided": dec,
                 "cyclic_share": (cyc / dec) if dec else None,
                 "b_holds": dec > 0 and cyc / dec < CYCLIC_MAX}

    # ---- H4 (C4): the export rule is the noise source --------------------
    snaps = [r["snapshots"] for r in ctrl if r.get("snapshots")]
    mean_rank = [float(np.mean([s["streak_rank"] + 1 for s in sn])) for sn in snaps]
    mean_rho = [float(np.mean([s["rho_streak_external"] for s in sn])) for sn in snaps]
    dec_exp = [_decline([s["streak_score"] for s in sn]) for sn in snaps]
    dec_best = [_decline([s["external_score"] for s in sn]) for sn in snaps]
    dec_med = [_decline([s["median_score"] for s in sn]) for sn in snaps]
    ka, na, pa = su.sign_test_greater([x - TOP_QUARTER for x in mean_rank])
    lo, hi = (su.bootstrap_ci(mean_rho, alpha=0.10) if len(mean_rho) > 1
              else (float("nan"), float("nan")))
    kc, nc, pc = su.sign_test_greater([e - b for e, b in zip(dec_exp, dec_best)])
    res["H4"] = {
        "a": {"mean_rank": mean_rank, "above_top_quarter": ka,
              "informative": na, "p": pa},
        "b": {"mean_rho": mean_rho,
              "rho_mean": float(np.mean(mean_rho)) if mean_rho else None,
              "ci90": [lo, hi],
              "holds": bool(-RHO_BOUND < lo and hi < RHO_BOUND)},
        "c": {"decline_exported": dec_exp, "decline_best": dec_best,
              "decline_median": dec_med,
              "sum_exported": float(sum(dec_exp)), "sum_best": float(sum(dec_best)),
              "sum_median": float(sum(dec_med)),
              "exported_larger": kc, "informative": nc, "p": pc},
    }

    # ---- H5 (C5a): archive as parent destroys learning -------------------
    kp = sum(r["reached"] for r in parent)
    res["H5"] = {"learned_parent": kp, "of_parent": len(parent),
                 "learned_control": learned, "of_control": len(ctrl),
                 "p": su.fisher_less(kp, len(parent), learned, len(ctrl))}

    # ---- H6 (C5b): archive as test neither harms nor helps detectably ----
    cmp = {}
    for o in C5B_OUTCOMES:
        a = [r[o] for r in test]
        b = [r[o] for r in ctrl]
        u, p = su.mannwhitney_dp(a, b)
        cmp[o] = {"test_mean": float(np.mean(a)), "control_mean": float(np.mean(b)),
                  "cliffs_delta": float(su.cliffs_delta(a, b)), "p_two_sided": p}
    late_l = [r["hof_winrate_late"] for r in test if r["reached"]]
    late_f = [r["hof_winrate_late"] for r in test if not r["reached"]]
    res["H6"] = {
        "comparisons": cmp,
        "learned_test": sum(r["reached"] for r in test), "of_test": len(test),
        "no_detectable_effect": all(c["p_two_sided"] >= ALPHA for c in cmp.values()),
        "archive_late_learned": late_l, "archive_late_failed": late_f,
        "description_holds": (all(x <= ARCHIVE_LEARNED_MAX for x in late_l)
                              and all(x >= ARCHIVE_FAILED_MIN for x in late_f)),
    }

    # ---- the confirmatory family: Holm over the five directional tests ---
    fam = {"H1": res["H1"]["p"], "H3a": res["H3"]["p"], "H4a": pa,
           "H4c": pc, "H5": res["H5"]["p"]}
    rej = su.holm(list(fam.values()), ALPHA)
    res["holm"] = {k: {"p": p, "rejected": r} for (k, p), r in zip(fam.items(), rej)}
    R = {k: v["rejected"] for k, v in res["holm"].items()}

    res["verdict"] = {
        "C1": R["H1"],
        "C2": res["H2"]["a_holds"] and res["H2"]["b_holds"],
        "C3": R["H3a"] and res["H3"]["b_holds"],
        "C4": R["H4a"] and res["H4"]["b"]["holds"] and R["H4c"],
        "C5a": R["H5"],
        "C5b": res["H6"]["no_detectable_effect"],
    }
    return res


def table(res):
    """The verdicts as a markdown table (one row per criterion)."""
    H = res["holm"]
    n = res["n"]

    def yes(b):
        return "yes" if b else "**no**"

    def p(x):
        return f"{x:.2g}" if x >= 0.001 else f"{x:.1e}"

    h2, h3, h4, h5, h6 = (res[k] for k in ("H2", "H3", "H4", "H5", "H6"))
    rows = [
        ("C1", "internal transition before parity (lag > 0)",
         f"{res['H1']['positive']}/{res['H1']['informative']} runs, "
         f"p = {p(H['H1']['p'])}", yes(H["H1"]["rejected"])),
        ("C2", "control runs that learn to rally",
         f"{h2['learned']}/{h2['of']}", yes(h2["a_holds"])),
        ("C2", "latest / earliest internal transition ≥ 3",
         f"{h2['timing_ratio']:.1f}×" if h2["timing_ratio"] else "–",
         yes(h2["b_holds"])),
        ("C3", "ρ(Elo, time) > 0 within runs",
         f"{h3['positive']}/{h3['informative']} runs (mean {h3['rho_mean']:+.2f}), "
         f"p = {p(H['H3a']['p'])}", yes(H["H3a"]["rejected"])),
        ("C3", "cyclic share of decided triads < 1%",
         f"{h3['cyclic']}/{h3['triads_decided']} "
         f"({100 * (h3['cyclic_share'] or 0):.1f}%)", yes(h3["b_holds"])),
        ("C4", "exported individual outside its pool's top quarter",
         f"{h4['a']['above_top_quarter']}/{h4['a']['informative']} runs "
         f"(mean rank {np.mean(h4['a']['mean_rank']):.0f}), "
         f"p = {p(H['H4a']['p'])}", yes(H["H4a"]["rejected"])),
        ("C4", "ρ(streak, skill) inside ±0.2 (90% CI)",
         f"{h4['b']['rho_mean']:+.2f} [{h4['b']['ci90'][0]:+.2f}, "
         f"{h4['b']['ci90'][1]:+.2f}]", yes(h4["b"]["holds"])),
        ("C4", "exported declines more than the best member",
         f"{h4['c']['exported_larger']}/{h4['c']['informative']} runs "
         f"({h4['c']['sum_exported']:.1f} vs {h4['c']['sum_best']:.1f}), "
         f"p = {p(H['H4c']['p'])}", yes(H["H4c"]["rejected"])),
        ("C5a", "archive as parent learns less often than control",
         f"{h5['learned_parent']}/{h5['of_parent']} vs "
         f"{h5['learned_control']}/{h5['of_control']}, p = {p(H['H5']['p'])}",
         yes(H["H5"]["rejected"])),
        ("C5b", "archive as test vs control: all four p ≥ 0.05",
         ", ".join(f"{o.split('_')[0]} δ {c['cliffs_delta']:+.2f} p {c['p_two_sided']:.2f}"
                   for o, c in h6["comparisons"].items()),
         yes(h6["no_detectable_effect"])),
    ]
    out = [f"Runs per condition: control {n['control']}, archive as parent "
           f"{n['hof-0.25']}, archive as test {n['hof-eval-v2']}. Tests in "
           f"the Holm family count as holding when rejected at family-wise "
           f"α = {ALPHA}.", "",
           "| claim | criterion | result | holds |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    out += ["", "Verdicts: " + ", ".join(
        f"{k} {'replicated' if v else '**not replicated**'}"
        for k, v in res["verdict"].items()) + "."]
    out += [f"Archive as test (description, no decision): late archive win "
            f"rate {', '.join(f'{x:.2f}' for x in h6['archive_late_learned'])} "
            f"in runs that learned"
            + (f"; {', '.join(f'{x:.2f}' for x in h6['archive_late_failed'])} "
               f"in runs that did not" if h6["archive_late_failed"] else "")
            + f" ({h6['learned_test']}/{h6['of_test']} learned)."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", metavar="JSON",
                    help="print the verdict table of an existing analysis file "
                         "(decisions recomputed from its per-run values)")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]}")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--partial", action="store_true",
                    help="analyse whatever exists; prints, writes nothing")
    args = ap.parse_args()

    if args.table:
        saved = json.load(open(args.table))
        print(table(analyse(saved["per_run"], saved["seeds"])))
        return

    seeds = _seeds(args.seeds)
    jobs, missing = [], []
    for c in CONDITIONS:
        for s in seeds:
            p = os.path.join(args.runs, f"{c}_s{s}.npz")
            (jobs if os.path.exists(p) else missing).append((p, c))
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} of {len(jobs) + len(missing)} runs "
                         f"missing; the preregistered analysis runs once, on "
                         f"all of them (use --partial for a progress view)")

    per_run = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, out in pool.imap_unordered(_per_run, jobs):
            per_run[name] = out
            print(f"  {name}: reached={out['reached']} "
                  f"final={out['final_holdout']:+.2f}", flush=True)

    res = analyse(per_run, seeds)
    res["runs"] = args.runs
    res["seeds"] = seeds
    res["per_run"] = per_run
    print(table(res))
    if args.partial:
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.out == OUT:
        pv.record(args.out, "replication",
                  params={"seeds": args.seeds, "within_every": WITHIN_EVERY,
                          "within_games": WITHIN_GAMES,
                          "reexport_episodes": REEXPORT_EPISODES,
                          "reexport_opponents": REEXPORT_OPPONENTS})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
