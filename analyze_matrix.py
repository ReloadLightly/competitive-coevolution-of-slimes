"""
analyze_matrix.py — turn the experiment matrix into numbers a paper can cite.

Metric definitions (fixed before the runs were inspected; the window is the
last 100,000 games, i.e. the last 20 of each run's 100 checkpoints):

  final          score of the t=500,000 champion. The primary endpoint: it is
                 the only one with no checkpoint selection in it.
  peak           best checkpoint score under the sweep seed. Selected, and
                 therefore optimistically biased -- which is why both `final`
                 and `peak` champions are re-scored on a disjoint evaluation
                 seed with 5x the episodes (`*_holdout`).
  above_parity   fraction of the 100 checkpoints scoring above 0.
  late_mean      mean checkpoint score over the last 100,000 games. "How good
                 is this population, on average, once it is trained?"
  volatility     mean absolute change between consecutive checkpoints in that
                 window. "How much does it swing?"
  drawdown       mean gap between the best checkpoint so far and the current
                 one, in that window. "How much of its best does it hold?"
  t_parity       first checkpoint above parity, in games. Speed of transfer.

Usage:
  python analyze_matrix.py --holdout      # re-score final/peak, then aggregate
  python analyze_matrix.py                # aggregate whatever exists
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

import fastvolley as fv
import stats_utils as su
from run_experiments import CONDITIONS, SELECT_EPISODES, SELECT_SEED

WINDOW = 20  # checkpoints = 100,000 games


def load_run(path):
    z = np.load(path)
    return {k: z[k] for k in z.files} | {"path": path,
                                         "name": os.path.basename(path)[:-4]}


def metrics(run):
    s = run["mean_score"]
    n = len(s)
    every = int(run["save_every"][0])
    late = s[-WINDOW:]
    runmax = np.maximum.accumulate(s)
    above = s > 0
    t_par = int(np.argmax(above) + 1) * every if above.any() else None
    return {
        "final": float(s[-1]),
        "peak": float(s.max()),
        "peak_t": int((np.argmax(s) + 1) * every),
        "above_parity": float(above.mean()),
        "late_mean": float(late.mean()),
        "late_sd": float(late.std()),
        "volatility": float(np.abs(np.diff(late)).mean()),
        "drawdown": float((runmax[-WINDOW:] - late).mean()),
        "t_parity": t_par,
        "final_win_rate": float(run["win_rate"][-1]),
        "final_meanlen": float(run["eval_meanlen"][-1]),
        "train_meanlen_final": float(run["train_meanlen"][-1]),
        "final_streak": int(run["streaks"][-1]),
    }


def _holdout_job(args):
    path, = args
    run = load_run(path)
    w, b = fv.baseline_arrays()
    s = run["mean_score"]
    out = {}
    for tag, idx in (("final", len(s) - 1), ("peak", int(np.argmax(s)))):
        sc, ln = fv.eval_vs_baseline(np.ascontiguousarray(run["champs"][idx]),
                                     SELECT_EPISODES, SELECT_SEED, w, b, False)
        out[tag + "_holdout"] = float(sc.mean())
        out[tag + "_holdout_sd"] = float(sc.std())
        out[tag + "_holdout_sem"] = float(sc.std() / np.sqrt(len(sc)))
        out[tag + "_holdout_win"] = float((sc > 0).mean())
        out[tag + "_holdout_tie"] = float((sc == 0).mean())
        out[tag + "_holdout_len"] = float(ln.mean())
    return run["name"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/matrix")
    ap.add_argument("--outdir", default="results/analysis")
    ap.add_argument("--holdout", action="store_true",
                    help="re-score final and peak champions on the disjoint seed")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.matrix, "*_s*.npz")))
    if not paths:
        print("no runs yet")
        return

    per_run = {}
    for p in paths:
        run = load_run(p)
        cond, seed = run["name"].rsplit("_s", 1)
        per_run[run["name"]] = {"condition": cond, "seed": int(seed),
                                "sigma": float(run["sigma"][0]),
                                "pop": int(run["pop"][0]),
                                "hof_prob": float(run["hof_prob"][0]),
                                "train_sec": float(run["train_sec"][0]),
                                **metrics(run)}

    hold_path = os.path.join(args.outdir, "holdout.json")
    holdout = json.load(open(hold_path)) if os.path.exists(hold_path) else {}
    if args.holdout:
        todo = [(p,) for p in paths
                if os.path.basename(p)[:-4] not in holdout]
        if todo:
            print(f"re-scoring {len(todo)} runs on the held-out seed "
                  f"({SELECT_EPISODES} episodes)", flush=True)
            with mp.get_context("spawn").Pool(args.workers) as pool:
                for name, out in pool.imap_unordered(_holdout_job, todo):
                    holdout[name] = out
                    print(f"  {name}: final {out['final_holdout']:+.3f} "
                          f"peak {out['peak_holdout']:+.3f}", flush=True)
            json.dump(holdout, open(hold_path, "w"), indent=1)
    for name, out in holdout.items():
        if name in per_run:
            per_run[name].update(out)

    json.dump(per_run, open(os.path.join(args.outdir, "per_run.json"), "w"),
              indent=1)

    # ---- per-condition aggregates ---------------------------------------
    keys = ["final", "peak", "above_parity", "late_mean", "late_sd",
            "volatility", "drawdown", "final_win_rate", "final_meanlen"]
    if holdout:
        keys += ["final_holdout", "peak_holdout"]

    conds = {}
    for cond in CONDITIONS:
        rows = [v for v in per_run.values() if v["condition"] == cond]
        if not rows:
            continue
        agg = {"n_runs": len(rows), "seeds": sorted(r["seed"] for r in rows)}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if vals:
                agg[k] = su.describe(vals)
        tp = [r["t_parity"] for r in rows]
        agg["t_parity"] = {"reached": int(sum(t is not None for t in tp)),
                           "median": (float(np.median([t for t in tp if t is not None]))
                                      if any(t is not None for t in tp) else None)}
        conds[cond] = agg

    # ---- comparisons against the control --------------------------------
    comparisons = {}
    ctrl = [v for v in per_run.values() if v["condition"] == "control"]
    for cond in CONDITIONS:
        if cond == "control":
            continue
        rows = [v for v in per_run.values() if v["condition"] == cond]
        if not rows or not ctrl:
            continue
        comparisons[cond] = {}
        for k in keys:
            a = [r[k] for r in rows if k in r]
            b = [r[k] for r in ctrl if k in r]
            if len(a) > 1 and len(b) > 1:
                comparisons[cond][k] = su.compare(a, b)

    json.dump({"conditions": conds, "vs_control": comparisons},
              open(os.path.join(args.outdir, "conditions.json"), "w"), indent=1)

    # ---- console summary -------------------------------------------------
    hdr = f"{'condition':<12}{'n':>3}  {'final':>16}  {'late_mean':>16}  " \
          f"{'volatility':>12}  {'drawdown':>10}  {'above':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for cond, agg in conds.items():
        f = agg.get("final", {})
        lm = agg.get("late_mean", {})
        v = agg.get("volatility", {})
        d = agg.get("drawdown", {})
        ap_ = agg.get("above_parity", {})
        print(f"{cond:<12}{agg['n_runs']:>3}  "
              f"{f.get('mean', float('nan')):>7.3f} +/- {f.get('sem', float('nan')):<6.3f}  "
              f"{lm.get('mean', float('nan')):>7.3f} +/- {lm.get('sem', float('nan')):<6.3f}  "
              f"{v.get('mean', float('nan')):>12.3f}  {d.get('mean', float('nan')):>10.3f}  "
              f"{ap_.get('mean', float('nan')):>7.2f}")

    if comparisons:
        print("\nvs control (Mann-Whitney, exact, two-sided)")
        for cond, c in comparisons.items():
            for k in ("late_mean", "volatility", "drawdown", "final_holdout"):
                if k in c:
                    r = c[k]
                    print(f"  {cond:<11} {k:<15} diff {r['diff_of_means']:+.3f}  "
                          f"delta {r['cliffs_delta']:+.2f}  p={r['p_two_sided']:.3f}")
    print(f"\n-> {args.outdir}/per_run.json, conditions.json")


if __name__ == "__main__":
    main()
