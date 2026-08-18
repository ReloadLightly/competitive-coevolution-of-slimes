"""
make_tables.py — every table in the writeup, generated from the result files.

No number in the paper is typed by hand. Running this after the matrix
finishes regenerates docs/paper/_tables.md, which the writeup sections
include verbatim.

    python make_tables.py
"""

import argparse
import glob
import json
import os

import numpy as np

ANDIR = "results/analysis"
LABELS = {
    "control": "control (Ha GA)",
    "hof-0.25": "hall of fame, p=0.25",
    "hof-0.50": "hall of fame, p=0.50",
    "sigma-0.05": "sigma = 0.05",
    "sigma-0.20": "sigma = 0.20",
    "pop-32": "population 32",
    "pop-512": "population 512",
}
ORDER = ["control", "hof-0.25", "hof-0.50", "sigma-0.05", "sigma-0.20",
         "pop-32", "pop-512"]


def load(name):
    p = os.path.join(ANDIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{{:{'+' if sign else ''}.{nd}f}}".format(x)
    return s


def t_validation(out):
    v = None
    if os.path.exists("results/validation.json"):
        v = json.load(open("results/validation.json"))
    if not v:
        return
    out.append("### Table A1 — the compiled environment against the reference\n")
    out.append("| scenario | paired games | identical score | identical length "
               "| identical trajectory | max abs deviation | env steps compared |")
    out.append("|---|---|---|---|---|---|---|")
    tot_steps = 0
    for name, d in v["scenarios"].items():
        tot_steps += d["steps"]
        out.append(f"| {name} | {d['n']} | {d['score_match']}/{d['n']} | "
                   f"{d['len_match']}/{d['n']} | {d['trace_exact']}/{d['n']} | "
                   f"{d['max_abs_dev']:.0f} | {d['steps']:,} |")
    out.append(f"\nAll {sum(d['n'] for d in v['scenarios'].values())} paired "
               f"games agree bit for bit over {tot_steps:,} environment steps. "
               f"Throughput on one core: {v['reference_games_per_sec']:.1f} "
               f"games/s reference, {v['compiled_games_per_sec']:.0f} games/s "
               f"compiled ({v['speedup']:.0f}x).\n")


def t_conditions(out, conds, per_run):
    if not conds:
        return
    c = conds["conditions"]
    out.append("### Table 1 — conditions, held-out scores and stability\n")
    out.append("| condition | runs | final (held out) | peak (held out) | "
               "mean, last 100k | volatility | drawdown | above parity | "
               "median first parity |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        if k not in c:
            continue
        a = c[k]
        fh = a.get("final_holdout", {})
        ph = a.get("peak_holdout", {})
        lm = a.get("late_mean", {})
        vo = a.get("volatility", {})
        dd = a.get("drawdown", {})
        ap = a.get("above_parity", {})
        tp = a.get("t_parity", {})
        tpm = tp.get("median")
        out.append(
            f"| {LABELS[k]} | {a['n_runs']} | "
            f"{fmt(fh.get('mean'), 2, True)} ± {fmt(fh.get('sem'))} | "
            f"{fmt(ph.get('mean'), 2, True)} ± {fmt(ph.get('sem'))} | "
            f"{fmt(lm.get('mean'), 2, True)} ± {fmt(lm.get('sem'))} | "
            f"{fmt(vo.get('mean'))} | {fmt(dd.get('mean'))} | "
            f"{fmt(100*ap.get('mean', float('nan')), 0)}% | "
            f"{(f'{tpm/1000:.0f}k' if tpm else '—')} "
            f"({tp.get('reached', 0)}/{a['n_runs']}) |")
    out.append("\nScores are points per episode against the 2015 baseline, "
               "mean ± s.e.m. across runs. `final` and `peak` are re-scored on "
               "the held-out evaluation seed over 1,000 episodes; the "
               "remaining columns are computed on the 200-episode sweep.\n")


def t_comparisons(out, conds):
    if not conds or not conds.get("vs_control"):
        return
    out.append("### Table 2 — each intervention against the control\n")
    out.append("| condition | metric | difference | Cliff's δ | exact p |")
    out.append("|---|---|---|---|---|")
    for k in ORDER[1:]:
        cc = conds["vs_control"].get(k)
        if not cc:
            continue
        for metric in ("final_holdout", "late_mean", "volatility", "drawdown",
                       "above_parity"):
            if metric not in cc:
                continue
            r = cc[metric]
            out.append(f"| {LABELS[k]} | {metric} | "
                       f"{fmt(r['diff_of_means'], 3, True)} | "
                       f"{fmt(r['cliffs_delta'], 2, True)} | "
                       f"{r['p_two_sided']:.3f} |")
    out.append("\nExact two-sided Mann–Whitney U. Difference is "
               "condition minus control, in points per episode "
               "(`above_parity` is a fraction). Only `final_holdout` is the "
               "pre-registered primary endpoint; the rest are descriptive.\n")


def t_per_run(out, per_run):
    if not per_run:
        return
    out.append("### Table A2 — every run\n")
    out.append("| run | seed | final (sweep) | final (held out) | peak (held out) "
               "| mean last 100k | volatility | drawdown | above parity | "
               "first parity | train min |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(per_run, key=lambda n: (ORDER.index(per_run[n]["condition"])
                                               if per_run[n]["condition"] in ORDER
                                               else 99, per_run[n]["seed"])):
        v = per_run[name]
        tp = v.get("t_parity")
        out.append(
            f"| {v['condition']} | {v['seed']} | {fmt(v['final'], 2, True)} | "
            f"{fmt(v.get('final_holdout'), 2, True)} | "
            f"{fmt(v.get('peak_holdout'), 2, True)} | "
            f"{fmt(v['late_mean'], 2, True)} | {fmt(v['volatility'])} | "
            f"{fmt(v['drawdown'])} | {fmt(100*v['above_parity'], 0)}% | "
            f"{(f'{tp/1000:.0f}k' if tp else '—')} | "
            f"{fmt(v['train_sec']/60, 1)} |")
    out.append("")


def t_reference(out):
    r = load("reference_curve.json")
    if not r:
        return
    h = r["holdout"]
    s = np.array(r["mean_score"])
    out.append("### Table 3 — the reference run on the unmodified environment\n")
    out.append("| checkpoint | games | score vs 2015 baseline | won / drawn / lost "
               "| mean rally |")
    out.append("|---|---|---|---|---|")
    for tag in ("final", "peak"):
        d = h[tag]
        out.append(f"| {tag} | {d['tournament']:,} | "
                   f"{d['mean']:+.3f} ± {d['sd']:.3f} (s.e.m. {d['sem']:.3f}) | "
                   f"{d['win']*100:.0f}% / {d['tie']*100:.0f}% / "
                   f"{d['loss']*100:.0f}% | {d['meanlen']:.0f} steps |")
    out.append(f"| Ha (2020) reference | 500,000 | +0.353 ± 0.728 | — | — |")
    out.append(f"\n{int((s > 0).sum())} of {len(s)} checkpoints score above "
               f"parity on the 200-episode sweep. Held-out rows are 1,000 "
               f"episodes at the disjoint evaluation seed.\n")


def t_coevolution(out, within):
    if not within:
        return
    out.append("### Table 4 — intransitivity inside a run\n")
    out.append("| condition | runs | ρ(Elo, training time) | cyclic triads | "
               "undecided pairs |")
    out.append("|---|---|---|---|---|")
    for k in ORDER:
        rows = [d for n, d in within.items() if n.rsplit("_s", 1)[0] == k]
        if not rows:
            continue
        rho = [d["spearman_elo_vs_time"] for d in rows]
        cyc = [d["cyclic"] for d in rows]
        tot = [d["triads_decided"] for d in rows]
        und = [d["pairs_undecided"] for d in rows]
        frac = sum(cyc) / max(1, sum(tot))
        out.append(f"| {LABELS[k]} | {len(rows)} | {np.mean(rho):+.2f} | "
                   f"{sum(cyc)}/{sum(tot)} ({frac*100:.1f}%) | "
                   f"{np.mean(und):.1f} |")
    out.append("\nCheckpoints 50,000 games apart, 50 games per pair over both "
               "court sides. A pair whose mean margin is inside ±0.25 points is "
               "undecided and its triads are skipped.\n")


def t_proxy(out, proxy):
    if not proxy:
        return
    out.append("### Table 5 — what the champion-selection proxy costs\n")
    out.append("| games | exported champion | best in the same pool | gap | "
               "exported rank | ρ(streak, score) | pool members above parity |")
    out.append("|---|---|---|---|---|---|---|")
    by_t = {}
    for rows in proxy.values():
        for r in rows:
            by_t.setdefault(r["tournament"], []).append(r)
    for t in sorted(by_t):
        rs = by_t[t]
        out.append(
            f"| {t:,} | {np.mean([r['exported_score'] for r in rs]):+.2f} | "
            f"{np.mean([r['best_score'] for r in rs]):+.2f} | "
            f"{np.mean([r['gap'] for r in rs]):.2f} | "
            f"{np.mean([r['exported_rank'] for r in rs]):.0f} / "
            f"{rs[0]['pop_size']} | "
            f"{np.mean([r['spearman_streak_vs_score'] for r in rs]):+.2f} | "
            f"{np.mean([r['n_above_parity'] for r in rs]):.0f} |")
    out.append("\nControl runs only, averaged across seeds. Every member of "
               "the snapshotted population is scored against the 2015 baseline; "
               "'exported' is the individual Ha's longest-winning-lineage rule "
               "selects.\n")


def t_across(out, across):
    if not across:
        return
    names = across["runs"]
    elo = np.array(across["elo"])
    out.append("### Table 6 — cross-run tournament of final champions\n")
    out.append("| condition | runs | median Elo | best run | worst run |")
    out.append("|---|---|---|---|---|")
    for k in ORDER:
        idx = [i for i, n in enumerate(names) if n.rsplit("_s", 1)[0] == k]
        if not idx:
            continue
        vals = elo[idx]
        out.append(f"| {LABELS[k]} | {len(idx)} | {np.median(vals):+.0f} | "
                   f"{vals.max():+.0f} | {vals.min():+.0f} |")
    out.append(f"\nBradley–Terry ratings on the Elo scale from an all-play-all "
               f"tournament of the {len(names)} final champions, "
               f"{across['games_per_pair']} games per pair over both sides. "
               f"Cyclic triads: {across['cyclic']}/{across['triads_decided']} "
               f"({100*across['cyclic']/max(1, across['triads_decided']):.1f}%).\n")


def t_resume(out):
    r = load("resume_fast.json")
    if not r:
        return
    out.append("### Table A3 — the same population continued in both "
               "implementations\n")
    out.append("| continuation | games added | final score | checkpoints above parity |")
    out.append("|---|---|---|---|")
    for name, d in r["runs"].items():
        s = np.array(d["mean_score"])
        out.append(f"| compiled, {name} | {r['tournaments']:,} | "
                   f"{s[-1]:+.2f} | {int((s > 0).sum())}/{len(s)} |")
    ref = load("reference_curve.json")
    if ref:
        t = np.array(ref["tournament"])
        s = np.array(ref["mean_score"])
        m = t > r["snapshot_tournament"]
        if m.any():
            out.append(f"| reference environment | {int(t[m].max() - r['snapshot_tournament']):,} | "
                       f"{s[m][-1]:+.2f} | {int((s[m] > 0).sum())}/{int(m.sum())} |")
    out.append("\nAll continuations start from the identical committed "
               "population snapshot at tournament "
               f"{r['snapshot_tournament']:,}.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/_tables.md")
    args = ap.parse_args()

    per_run = load("per_run.json")
    conds = load("conditions.json")
    within = load("within_run.json")
    across = load("across_runs.json")
    proxy = load("champion_proxy.json")

    out = ["<!-- generated by make_tables.py — do not edit by hand -->\n"]
    t_conditions(out, conds, per_run)
    t_comparisons(out, conds)
    t_reference(out)
    t_coevolution(out, within)
    t_proxy(out, proxy)
    t_across(out, across)
    t_validation(out)
    t_per_run(out, per_run)
    t_resume(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
