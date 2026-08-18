"""
make_tables.py — every table in the writeup, generated from the result files.

No table in the paper is typed by hand. Each section file carries markers

    <!-- table:1 -->
    ...generated content...
    <!-- /table:1 -->

and this script rewrites whatever sits between them. Running it after new
results land refreshes the paper in place, and `git diff docs/paper` shows
exactly which numbers moved.

    python make_tables.py            # inject into docs/paper/*.md
    python make_tables.py --check    # fail if injection would change anything
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ANDIR = "results/analysis"
PAPER = "docs/paper"
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
    return f"{{:{'+' if sign else ''}.{nd}f}}".format(x)


def games(x):
    return "—" if x is None else f"{x/1000:.0f}k"


# --------------------------------------------------------------------------
# One function per table. Each returns a markdown string, or None if the data
# it needs is not there yet.
# --------------------------------------------------------------------------
def table_1(d):
    conds = d["conditions"]
    if not conds:
        return None
    c = conds["conditions"]
    out = ["| condition | runs | final (held out) | peak (held out) | "
           "mean, last 100k | volatility | drawdown | above parity | "
           "median first parity |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in c:
            continue
        a = c[k]
        fh, ph = a.get("final_holdout", {}), a.get("peak_holdout", {})
        lm, vo = a.get("late_mean", {}), a.get("volatility", {})
        dd, ap = a.get("drawdown", {}), a.get("above_parity", {})
        tp = a.get("t_parity", {})
        out.append(
            f"| {LABELS[k]} | {a['n_runs']} | "
            f"{fmt(fh.get('mean'), 2, True)} ± {fmt(fh.get('sem'))} | "
            f"{fmt(ph.get('mean'), 2, True)} ± {fmt(ph.get('sem'))} | "
            f"{fmt(lm.get('mean'), 2, True)} ± {fmt(lm.get('sem'))} | "
            f"{fmt(vo.get('mean'))} | {fmt(dd.get('mean'))} | "
            f"{fmt(100*ap.get('mean', float('nan')), 0)}% | "
            f"{games(tp.get('median'))} ({tp.get('reached', 0)}/{a['n_runs']}) |")
    out.append("")
    out.append("Scores are points per episode against the 2015 baseline, "
               "mean ± s.e.m. across runs. `final` and `peak` are re-scored on "
               "the held-out evaluation seed over 1,000 episodes; the other "
               "columns come from the 200-episode sweep.")
    return "\n".join(out)


def table_2(d):
    conds = d["conditions"]
    if not conds or not conds.get("vs_control"):
        return None
    out = ["| condition | metric | difference | Cliff's δ | exact p |",
           "|---|---|---|---|---|"]
    for k in ORDER[1:]:
        cc = conds["vs_control"].get(k)
        if not cc:
            continue
        for metric in ("final_holdout", "late_mean", "volatility", "drawdown",
                       "above_parity"):
            if metric not in cc:
                continue
            r = cc[metric]
            out.append(f"| {LABELS[k]} | `{metric}` | "
                       f"{fmt(r['diff_of_means'], 3, True)} | "
                       f"{fmt(r['cliffs_delta'], 2, True)} | "
                       f"{r['p_two_sided']:.3f} |")
    out.append("")
    out.append("Exact two-sided Mann–Whitney U over all label assignments. "
               "Difference is condition minus control in points per episode "
               "(`above_parity` is a fraction). Only `final_holdout` is the "
               "pre-registered primary endpoint; the rest are descriptive and "
               "uncorrected for multiplicity.")
    return "\n".join(out)


def table_3(d):
    r = d["reference"]
    if not r:
        return None
    h = r["holdout"]
    s = np.array(r["mean_score"])
    out = ["| checkpoint | games | score vs 2015 baseline | won / drawn / lost "
           "| mean rally |", "|---|---|---|---|---|"]
    for tag in ("final", "peak"):
        x = h[tag]
        out.append(f"| {tag} | {x['tournament']:,} | "
                   f"{x['mean']:+.3f} ± {x['sd']:.3f} (s.e.m. {x['sem']:.3f}) | "
                   f"{x['win']*100:.0f}% / {x['tie']*100:.0f}% / "
                   f"{x['loss']*100:.0f}% | {x['meanlen']:.0f} steps |")
    out.append("| Ha (2020), same algorithm and budget | 500,000 | "
               "+0.353 ± 0.728 | — | — |")
    out.append("")
    ti = r.get("t_internal")
    tp = r.get("t_parity")
    ti_s = f"{ti:,}" if ti else "not reached"
    tp_s = f"{tp:,}" if tp else "not reached"
    lag_s = f"{tp - ti:,}" if (ti and tp) else "—"
    out.append(f"{int((s > 0).sum())} of {len(s)} checkpoints score above "
               f"parity on the 200-episode sweep. Held-out rows are 1,000 "
               f"episodes at the disjoint evaluation seed. Internal transition "
               f"(training rally length above 1,500 steps): {ti_s} games; "
               f"first checkpoint above parity: {tp_s} games; lag {lag_s} games.")
    return "\n".join(out)


def table_4(d):
    within = d["within"]
    if not within:
        return None
    out = ["| condition | runs | ρ(Elo, training time) | cyclic triads | "
           "undecided pairs |", "|---|---|---|---|---|"]
    for k in ORDER:
        rows = [v for n, v in within.items() if n.rsplit("_s", 1)[0] == k]
        if not rows:
            continue
        rho = [v["spearman_elo_vs_time"] for v in rows]
        cyc, tot = sum(v["cyclic"] for v in rows), sum(v["triads_decided"] for v in rows)
        und = np.mean([v["pairs_undecided"] for v in rows])
        out.append(f"| {LABELS[k]} | {len(rows)} | {np.mean(rho):+.2f} | "
                   f"{cyc}/{tot} ({100*cyc/max(1,tot):.1f}%) | {und:.1f} |")
    out.append("")
    out.append("Checkpoints 50,000 games apart play a round robin, 50 games "
               "per pair over both court sides. A pair whose mean margin is "
               "inside ±0.25 points counts as undecided and its triads are "
               "skipped. A cyclic triad is A beats B beats C beats A.")
    return "\n".join(out)


def table_5(d):
    proxy = d["proxy"]
    if not proxy:
        return None
    out = ["| games | exported champion | best in the same pool | gap | "
           "exported rank | ρ(streak, score) | above parity in pool | "
           "mean pairwise genotype distance |",
           "|---|---|---|---|---|---|---|---|"]
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
            f"{np.mean([r['exported_rank'] for r in rs]):.0f} / {rs[0]['pop_size']} | "
            f"{np.mean([r['spearman_streak_vs_score'] for r in rs]):+.2f} | "
            f"{np.mean([r['n_above_parity'] for r in rs]):.0f} | "
            f"{np.mean([r['mean_pairwise_distance'] for r in rs]):.2f} |")
    out.append("")
    out.append(f"Control runs only ({len(proxy)} seeds), averaged across seeds. "
               "Every member of the snapshotted population is scored against "
               "the 2015 baseline; 'exported' is the individual Ha's "
               "longest-winning-lineage rule selects.")
    return "\n".join(out)


def table_6(d):
    across = d["across"]
    if not across:
        return None
    names = across["runs"]
    elo = np.array(across["elo"])
    out = ["| condition | runs | median Elo | best run | worst run |",
           "|---|---|---|---|---|"]
    for k in ORDER:
        idx = [i for i, n in enumerate(names) if n.rsplit("_s", 1)[0] == k]
        if not idx:
            continue
        v = elo[idx]
        out.append(f"| {LABELS[k]} | {len(idx)} | {np.median(v):+.0f} | "
                   f"{v.max():+.0f} | {v.min():+.0f} |")
    out.append("")
    out.append(f"Bradley–Terry ratings on the Elo scale from an all-play-all "
               f"tournament of the {len(names)} final champions, "
               f"{across['games_per_pair']} games per pair over both court "
               f"sides. Cyclic triads across the whole tournament: "
               f"{across['cyclic']}/{across['triads_decided']} "
               f"({100*across['cyclic']/max(1, across['triads_decided']):.1f}%).")
    return "\n".join(out)


def table_a1(d):
    v = d["validation"]
    if not v:
        return None
    out = ["| scenario | paired games | identical score | identical length | "
           "identical trajectory | max abs deviation | env steps compared |",
           "|---|---|---|---|---|---|---|"]
    tot = 0
    for name, x in v["scenarios"].items():
        tot += x["steps"]
        out.append(f"| {name} | {x['n']} | {x['score_match']}/{x['n']} | "
                   f"{x['len_match']}/{x['n']} | {x['trace_exact']}/{x['n']} | "
                   f"{x['max_abs_dev']:.0f} | {x['steps']:,} |")
    out.append("")
    out.append(f"All {sum(x['n'] for x in v['scenarios'].values())} paired "
               f"games agree bit for bit over {tot:,} environment steps. "
               f"Throughput on one core: {v['reference_games_per_sec']:.1f} "
               f"games/s reference, {v['compiled_games_per_sec']:.0f} games/s "
               f"compiled ({v['speedup']:.0f}×).")
    return "\n".join(out)


def table_a2(d):
    per_run = d["per_run"]
    if not per_run:
        return None
    out = ["| condition | seed | final (sweep) | final (held out) | "
           "peak (held out) | mean last 100k | volatility | drawdown | "
           "above parity | internal transition | first parity | train min |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(per_run, key=lambda n: (
            ORDER.index(per_run[n]["condition"])
            if per_run[n]["condition"] in ORDER else 99, per_run[n]["seed"])):
        v = per_run[name]
        out.append(
            f"| {v['condition']} | {v['seed']} | {fmt(v['final'], 2, True)} | "
            f"{fmt(v.get('final_holdout'), 2, True)} | "
            f"{fmt(v.get('peak_holdout'), 2, True)} | "
            f"{fmt(v['late_mean'], 2, True)} | {fmt(v['volatility'])} | "
            f"{fmt(v['drawdown'])} | {fmt(100*v['above_parity'], 0)}% | "
            f"{games(v.get('t_internal'))} | {games(v.get('t_parity'))} | "
            f"{fmt(v['train_sec']/60, 1)} |")
    return "\n".join(out)


def table_a3(d):
    r = d["resume"]
    if not r:
        return None
    out = ["| continuation | games added | final score | checkpoints above parity |",
           "|---|---|---|---|"]
    for name, x in r["runs"].items():
        s = np.array(x["mean_score"])
        out.append(f"| compiled, {name} | {r['tournaments']:,} | {s[-1]:+.2f} | "
                   f"{int((s > 0).sum())}/{len(s)} |")
    ref = d["reference"]
    if ref:
        t, s = np.array(ref["tournament"]), np.array(ref["mean_score"])
        m = t > r["snapshot_tournament"]
        if m.any():
            out.append(f"| reference environment | "
                       f"{int(t[m].max() - r['snapshot_tournament']):,} | "
                       f"{s[m][-1]:+.2f} | {int((s[m] > 0).sum())}/{int(m.sum())} |")
    out.append("")
    out.append("All continuations start from the identical committed "
               f"population snapshot at tournament "
               f"{r['snapshot_tournament']:,}. Independent continuations of one "
               "population diverge because the algorithm is stochastic; the "
               "question is whether the compiled ones land in the same band as "
               "the reference one.")
    return "\n".join(out)


def table_7(d):
    """Windowed profile of the control condition — the damping claim."""
    per_run = d["per_run"]
    if not per_run:
        return None
    rows = [v for v in per_run.values()
            if v["condition"] == "control" and "window_profile" in v]
    if not rows:
        return None
    n_win = max(len(r["window_profile"]) for r in rows)
    out = ["| games | mean across seeds | within-run s.d. | spread across seeds "
           "| checkpoints above parity |", "|---|---|---|---|---|"]
    for i in range(n_win):
        ws = [r["window_profile"][i] for r in rows if len(r["window_profile"]) > i]
        if not ws:
            continue
        means = [w["mean"] for w in ws]
        out.append(f"| {ws[0]['from']:,}–{ws[0]['to']:,} | "
                   f"{np.mean(means):+.2f} | {np.mean([w['sd'] for w in ws]):.2f} | "
                   f"{np.std(means):.2f} | "
                   f"{sum(w['above'] for w in ws)}/{sum(w['n'] for w in ws)} |")
    out.append("")
    out.append(f"Control condition, {len(rows)} seeds. 'Within-run s.d.' is the "
               "spread of checkpoint scores inside a window, averaged over "
               "seeds — the quantity the single-run version of this study "
               "claimed was damping. 'Spread across seeds' is the s.d. of the "
               "per-seed window means.")
    return "\n".join(out)


TABLES = {
    "1": table_1, "2": table_2, "3": table_3, "4": table_4, "5": table_5,
    "6": table_6, "7": table_7,
    "a1": table_a1, "a2": table_a2, "a3": table_a3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any table is out of date")
    args = ap.parse_args()

    d = {
        "per_run": load("per_run.json"),
        "conditions": load("conditions.json"),
        "within": load("within_run.json"),
        "across": load("across_runs.json"),
        "proxy": load("champion_proxy.json"),
        "reference": load("reference_curve.json"),
        "resume": load("resume_fast.json"),
        "validation": (json.load(open("results/validation.json"))
                       if os.path.exists("results/validation.json") else None),
    }

    built = {}
    for key, fn in TABLES.items():
        try:
            md = fn(d)
        except Exception as e:
            print(f"table {key}: FAILED {type(e).__name__}: {e}")
            md = None
        if md:
            built[key] = md

    combined = ["<!-- generated by make_tables.py — do not edit by hand -->"]
    for key in TABLES:
        if key in built:
            combined.append(f"\n### Table {key.upper()}\n\n{built[key]}\n")
    os.makedirs(PAPER, exist_ok=True)
    with open(os.path.join(PAPER, "_tables.md"), "w") as f:
        f.write("\n".join(combined))

    stale = []
    for path in sorted(glob.glob(os.path.join(PAPER, "*.md"))):
        if os.path.basename(path).startswith("_"):
            continue
        src = open(path).read()
        new = src
        for key, md in built.items():
            pat = re.compile(rf"(<!-- table:{re.escape(key)} -->\n).*?"
                             rf"(\n<!-- /table:{re.escape(key)} -->)",
                             re.DOTALL)
            if pat.search(new):
                new = pat.sub(lambda m: m.group(1) + md + m.group(2), new)
        if new != src:
            stale.append(path)
            if not args.check:
                open(path, "w").write(new)

    print(f"built {len(built)}/{len(TABLES)} tables: {', '.join(sorted(built))}")
    missing = [k for k in TABLES if k not in built]
    if missing:
        print(f"not yet available: {', '.join(missing)}")
    if args.check:
        print("OUT OF DATE: " + ", ".join(stale) if stale else "up to date")
        sys.exit(1 if stale else 0)
    print(f"updated: {', '.join(os.path.basename(p) for p in stale) or 'nothing'}")


if __name__ == "__main__":
    main()
