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
    python make_tables.py --check    # fail unless every table follows from disk

`--check` fails when

  * any table named by a marker (README.md, docs/paper/*.md) or present in
    docs/paper/_tables.md cannot be built — a table is never silently
    skipped because its data is missing;
  * any analysis file a table is built from does not follow from the raw
    files on disk (provenance.py: an input deleted, added or changed, or the
    analysis file edited by hand);
  * injecting the tables would change any file.

Numbers in running text are handled the same way: prose_numbers.py defines
each one, markdown carries `<!-- n:key -->value<!-- /n -->` markers, and
paper/numbers.tex is written from the same values. An unknown key, a stale
value or a stale numbers.tex fails the check.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import prose_numbers as pn
import provenance as pv
from run_experiments import SELECT_EPISODES

ANDIR = "results/analysis"
PAPER = "docs/paper"
NUMBERS_TEX = "paper/numbers.tex"
NUM = re.compile(r"(<!-- n:([A-Za-z_]+) -->)(.*?)(<!-- /n -->)")
LABELS = {
    "control": "control (Ha 2020 GA)",
    "hof-0.25": "archive as parent, p=0.25",
    "hof-0.50": "archive as parent, p=0.50",
    "hof-full": "archive as parent, full span",
    "hof-eval-v2": "archive as test, full span",
    "ga2015": "generational GA (Ha 2015)",
    "es": "self-play ES",
    "sigma-0.05": "sigma = 0.05",
    "sigma-0.20": "sigma = 0.20",
    "pop-32": "population 32",
    "pop-512": "population 512",
}
ORDER = ["control", "hof-eval-v2", "hof-0.25", "hof-0.50", "hof-full",
         "ga2015", "es", "sigma-0.05", "sigma-0.20", "pop-32", "pop-512"]


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
    proxy, held = d["proxy"], d["proxy_heldout"]
    if not proxy or not held:
        return None
    hruns = {n: r for n, r in held["per_run"].items() if r["group"] == "original"}
    out = ["| games | exported champion | best in the same pool | gap | "
           "exported rank | ρ(streak, score) | above parity in pool | "
           "mean pairwise genotype distance |",
           "|---|---|---|---|---|---|---|---|"]
    by_t = {}
    for name, rows in proxy.items():
        for k, r in enumerate(rows):
            by_t.setdefault(r["tournament"], []).append((r, hruns[name]["snapshots"][k]))
    for t in sorted(by_t):
        rs = [r for r, _ in by_t[t]]
        hs = [h["heldout"] for _, h in by_t[t]]
        out.append(
            f"| {t:,} | {np.mean([h['exported'] for h in hs]):+.2f} | "
            f"{np.mean([h['best'] for h in hs]):+.2f} | "
            f"{np.mean([h['best'] - h['exported'] for h in hs]):.2f} | "
            f"{np.mean([r['exported_rank'] for r in rs]):.0f} / {rs[0]['pop_size']} | "
            f"{np.mean([r['spearman_streak_vs_score'] for r in rs]):+.2f} | "
            f"{np.mean([r['n_above_parity'] for r in rs]):.0f} | "
            f"{np.mean([r['mean_pairwise_distance'] for r in rs]):.2f} |")
    out.append("")
    out.append(f"Control runs only ({len(proxy)} seeds), averaged across seeds. "
               f"Every member of the snapshotted population is scored against "
               f"the 2015 baseline on {held['proxy_episodes']} episodes; 'exported' "
               f"is the individual Ha's longest-winning-lineage rule selects, "
               f"'best' the member with the best of those scores. Both are then "
               f"re-scored on {held['heldout_episodes']:,} held-out episodes, which "
               f"is what the score and gap columns show: on its own selecting "
               f"episodes the best member's score is inflated (the maximum of "
               f"{rs[0]['pop_size']} noisy scores). Rank, ρ and the parity count "
               f"use the {held['proxy_episodes']}-episode scores.")
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


def table_8(d):
    """When the phase change happens, and the internal-to-external lag."""
    per_run = d["per_run"]
    if not per_run:
        return None
    out = ["| condition | runs | reached long rallies | internal transition "
           "(median, range) | first parity (median, range) | lag (median) |",
           "|---|---|---|---|---|---|"]
    for k in ORDER:
        rows = [v for v in per_run.values() if v["condition"] == k]
        if not rows:
            continue
        ti = [r["t_internal"] for r in rows if r.get("t_internal")]
        tp = [r["t_parity"] for r in rows if r.get("t_parity")]
        lg = [r["lag_internal_to_parity"] for r in rows
              if r.get("lag_internal_to_parity")]
        rng = lambda v: (f"{np.median(v)/1000:.0f}k ({min(v)/1000:.0f}k–"
                         f"{max(v)/1000:.0f}k)") if v else "never"
        out.append(f"| {LABELS[k]} | {len(rows)} | {len(ti)}/{len(rows)} | "
                   f"{rng(ti)} | {rng(tp)} ({len(tp)}/{len(rows)}) | "
                   f"{f'{np.median(lg)/1000:.0f}k' if lg else '—'} |")
    ref = d["reference"]
    if ref and ref.get("t_internal"):
        out.append(f"| *reference run (1 run, real environment)* | 1 | 1/1 | "
                   f"*{ref['t_internal']/1000:.0f}k* | "
                   f"*{ref['t_parity']/1000:.0f}k* | "
                   f"*{(ref['t_parity']-ref['t_internal'])/1000:.0f}k* |")
    out.append("")
    out.append("'Internal transition' is the first checkpoint at which the "
               "population's own training games average more than 1,500 steps — "
               "measured with no external opponent involved. 'First parity' is "
               "the first checkpoint scoring above 0 against the 2015 baseline. "
               "The lag between them is how far internal progress runs ahead of "
               "anything an external evaluation can see.")
    return "\n".join(out)


def table_r(d):
    """The compact headline table for the repository README."""
    conds = d["conditions"]
    per_run = d["per_run"]
    if not conds or not per_run:
        return None
    c = conds["conditions"]
    out = ["| condition | runs | learned to rally | best champion (held out) | "
           "end-of-run champion | checkpoints above parity |",
           "|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in c:
            continue
        a = c[k]
        ph, fh = a.get("peak_holdout", {}), a.get("final_holdout", {})
        ap = a.get("above_parity", {})
        out.append(f"| {LABELS[k]} | {a['n_runs']} | "
                   f"{a.get('n_reached', 0)}/{a['n_runs']} | "
                   f"{fmt(ph.get('mean'), 2, True)} ± {fmt(ph.get('sem'))} | "
                   f"{fmt(fh.get('mean'), 2, True)} ± {fmt(fh.get('sem'))} | "
                   f"{fmt(100*ap.get('mean', float('nan')), 0)}% |")
    ref = d["reference"]
    if ref:
        h = ref["holdout"]
        out.append(f"| *reference run, unmodified environment* | 1 | 1/1 | "
                   f"*{h['peak']['mean']:+.2f} ± {h['peak']['sem']:.2f}* | "
                   f"*{h['final']['mean']:+.2f} ± {h['final']['sem']:.2f}* | "
                   f"*{int((np.array(ref['mean_score']) > 0).mean()*100):.0f}%* |")
    out.append("| *Ha (2020), same algorithm and budget* | 1 | — | "
               "*+0.35 ± 0.02* | — | — |")
    out.append("")
    out.append("Points per episode against the 2015 champion policy, which is "
               "never seen during training. Held-out columns are 1,000 episodes "
               "on an evaluation seed disjoint from the one used to pick the "
               "checkpoint. 'Learned to rally' counts runs whose population "
               "ever held 1,500-step rallies against itself.")
    return "\n".join(out)


def table_9(d):
    """Promotion rules: what each one costs and what it recovers."""
    r = d["reexport"]
    if not r:
        return None
    su_ = r["summary"]
    opps = sorted(int(k.split("_")[1]) for k in su_["recovered_fraction"])
    out = ["| promotion rule | games spent ranking | mean score of the exported "
           "individual | volatility of the reported series | gap to best closed "
           "| ρ(rule statistic, true skill) |",
           "|---|---|---|---|---|---|"]
    out.append(f"| winning streak (Ha's rule) | 0 | "
               f"{su_['streak_score']['level_mean']:+.2f} | "
               f"{su_['streak_score']['volatility_mean']:.2f} | — | "
               f"{su_['rho_streak_external']:+.2f} |")
    out.append(f"| *pick the population median* | *0* | "
               f"*{su_['median_score']['level_mean']:+.2f}* | "
               f"*{su_['median_score']['volatility_mean']:.2f}* | *—* | *—* |")
    for o in opps:
        k = f"internal_score_{o}"
        rec = su_["recovered_fraction"][f"internal_{o}"]
        out.append(f"| internal round robin, {o} peers each | "
                   f"{su_['ranking_games_per_snapshot'][f'internal_{o}']:,} | "
                   f"{su_[k]['level_mean']:+.2f} | {su_[k]['volatility_mean']:.2f} | "
                   f"{(f'{100*rec:.0f}%' if rec is not None else '—')} | "
                   f"{su_['rho_internal_external'][f'internal_{o}']:+.2f} |")
    out.append(f"| *best in pool (oracle, not deployable)* | *—* | "
               f"*{su_['external_score']['level_mean']:+.2f}* | "
               f"*{su_['external_score']['volatility_mean']:.2f}* | *100%* | "
               f"*+1.00* |")
    big = max(opps)
    games = su_["ranking_games_per_snapshot"][f"internal_{big}"]
    pop = next(iter(r["per_run"].values()))[0]["pop_size"]
    budget = json.load(open("results/matrix/protocol.json"))["tournaments"]
    out.append("")
    out.append(f"Control runs only ({su_['n_runs']} seeds), across all population "
               f"snapshots. Every population member is scored against the 2015 "
               f"baseline over {su_['episodes_per_individual']} episodes, which "
               f"stand in for true skill; the promotion rules then pick a member "
               f"using only what they are entitled to see, so their scores on "
               f"those episodes are unbiased. The oracle row is not: it is the "
               f"best of {pop} scores on the very episodes that "
               f"chose it, which inflates it (a winner's curse), so the gap to "
               f"best is overstated and the share closed understated. The "
               f"held-out re-scoring of the export check and the preregistered "
               f"export test do not have this bias. "
               f"'Volatility' is the mean absolute change in the exported "
               f"individual's score between consecutive snapshots. For scale, "
               f"{games:,} ranking games is {100 * games / budget:.1f}% of a "
               f"{budget:,}-game run.")
    return "\n".join(out)


def table_10(d):
    """Unequal power: who dominates, and whose population learns."""
    import glob as _glob
    import numpy as _np
    rows = {}
    for f in sorted(_glob.glob("results/matrix/asym*_s*.npz")):
        name = os.path.basename(f)[:-4]
        base, seed = name.rsplit("_s", 1)
        cond, side = base.rsplit("-", 1)
        rows.setdefault(cond, {}).setdefault(seed, {})[side] = _np.load(f)
    if not rows:
        return None
    label = {"asym1x": "symmetric control (273 v 273)",
             "asym2x": "2:1 capacity, common σ",
             "asym2x-norm": "2:1 capacity, matched step norm"}
    out = ["| condition | seeds | larger side wins cross-play | larger side's "
           "pool, best member | smaller side's pool, best member | runs where "
           "only one side's pool learned |",
           "|---|---|---|---|---|---|"]
    for cond in ("asym1x", "asym2x", "asym2x-norm"):
        if cond not in rows:
            continue
        wr, ba, bb, one_sided, dom = [], [], [], 0, 0
        for seed, sides in sorted(rows[cond].items()):
            ka = "strong" if "strong" in sides else "a"
            kb = "weak" if "weak" in sides else "b"
            if "pop_best" not in sides[ka]:
                continue
            w = float(sides[ka]["a_winrate"][-10:].mean())
            a = float(sides[ka]["pop_best"][-1])
            b = float(sides[kb]["pop_best"][-1])
            wr.append(w); ba.append(a); bb.append(b)
            if (a > 0) != (b > 0):
                one_sided += 1
            if w > 0.5:
                dom += 1
        if not wr:
            continue
        out.append(f"| {label[cond]} | {len(wr)} | "
                   f"{_np.median(wr):.2f} (range {min(wr):.2f}–{max(wr):.2f}); "
                   f"larger side ahead in {dom}/{len(wr)} | "
                   f"{_np.median(ba):+.2f} | {_np.median(bb):+.2f} | "
                   f"{one_sided}/{len(wr)} |")
    out.append("")
    out.append("Two populations of 128 playing only each other for 500,000 "
               "games; a quarter of each population's games are crossed with "
               "the other side. Win rate is over cross-population games in the "
               "last 50,000 games — 0.5 means the sides are holding each other. "
               "'Pool, best member' is the best individual the population "
               "contains at the end, scored against the 2015 baseline, not the "
               "exported champion. In the symmetric control both sides have "
               "identical architecture, so any departure from 0.5 there is "
               "spontaneous symmetry breaking and is the null the other two "
               "rows are judged against.")
    return "\n".join(out)


def table_c(d):
    """The design: every condition analysed, what it changes, how many runs.

    Parameters come from protocol.json (single population) and from the run
    files themselves (two populations), so the table describes what ran.
    """
    per_run = d["per_run"]
    if not per_run:
        return None
    proto = json.load(open("results/matrix/protocol.json"))
    pc = proto["conditions"]
    ctrl = pc["control"]
    runs = {}
    for v in per_run.values():
        c = v["condition"]
        base = c.rsplit("-", 1)[0] if c.startswith("asym") else c
        runs.setdefault(base, set()).add(v["seed"])

    def single(c):
        a = pc[c]
        algo = a.get("algo", "ga")
        if algo == "ga2015":
            return (f"generational GA: population {a['pop']}, {a['n_opponents']} "
                    f"games per individual per generation, top {a['n_elite']} "
                    f"kept, uniform crossover, σ = {a['sigma']}")
        if algo == "es":
            return (f"self-play OpenAI-ES: {a['pop']} mirrored perturbations, "
                    f"{a['n_opponents']} games each, learning rate {a['alpha']}, "
                    f"σ = {a['sigma']}; reports the mean")
        if algo == "hof_eval":
            return (f"archive as test: with p = {a['hof_prob']} the opponent is "
                    f"an archived champion (capacity {a['cap']}); genes come "
                    f"only from the living pool")
        if a["hof_prob"] > 0:
            return (f"archive as parent: with p = {a['hof_prob']} the opponent "
                    f"is an archived champion (capacity {a['cap']}) whose "
                    f"mutant replaces a member it beats")
        diff = []
        if a["sigma"] != ctrl["sigma"]:
            diff.append(f"mutation scale σ = {a['sigma']}")
        if a["pop"] != ctrl["pop"]:
            diff.append(f"population {a['pop']}")
        return ", ".join(diff) or (
            f"Ha's 2020 GA: population {a['pop']}, σ = {a['sigma']}, "
            f"champion = longest winning streak")

    out = ["| condition | what it changes relative to the control | runs |",
           "|---|---|---|"]
    for k in ORDER:
        if k in runs and k in pc:
            out.append(f"| `{k}` | {single(k)} | {len(runs[k])} |")
    for base in ("asym1x", "asym2x", "asym2x-norm"):
        if base not in runs:
            continue
        sides = {}
        for f in sorted(glob.glob("results/matrix/asym*_s*.npz")):
            cond, side = os.path.basename(f)[:-4].rsplit("_s", 1)[0].rsplit("-", 1)
            if cond == base and side not in sides:
                z = np.load(f)
                sides[side] = (int(z["param_count"][0]), float(z["sigma"][0]),
                               float(z["cross_prob"][0]))
        if len(sides) != 2:
            continue
        # larger side first; the symmetric control's sides are equal
        (pa, sa, cp), (pb, sb, _) = sorted(sides.values(), reverse=True)
        out.append(f"| `{base}` | two populations of {pc['control']['pop']}, "
                   f"{pa} v {pb} parameters, σ = {sa:.3f} v {sb:.3f}; a share "
                   f"{cp} of games crosses populations | {len(runs[base])} |")
    out.append("")
    out.append("Every run plays 500,000 games. Two-population runs count once "
               "per seed.")
    return "\n".join(out)


def table_rep(d):
    """The preregistered replication's verdicts (WP6)."""
    rep = d["replication"]
    if not rep:
        return None
    import replication
    # decisions recomputed from the stored per-run values, exactly as
    # replication.py --table does; they must equal the stored verdicts
    res = replication.analyse(rep["per_run"], rep["seeds"])
    assert res["verdict"] == rep["verdict"], "replication verdicts do not recompute"
    return replication.table(res)


def _p(x):
    return "< 0.001" if x < 0.001 else f"= {x:.3f}"


def table_lab(d):
    """The discmix experiment (WP8), one row per lambda."""
    a = d["lab"]
    if not a:
        return None
    rows = ["| λ | cyclic triads within runs (control / test) | exported rank in pool of 128 "
            "| ρ(streak, strength) | decline: exported / best | archive as test vs control, δ (p) |",
            "|---|---|---|---|---|---|"]
    for lam in sorted(a["H8b"]):
        b = a["H8b"][lam]
        c = a["H8c"]["by_lambda"][lam]
        cc = a["H8a"]["control"]["share_by_lambda"][lam]
        ct = a["H8a"]["test"]["share_by_lambda"][lam]
        rows.append(f"| {float(lam):.2f} | {100 * cc:.1f}% / {100 * ct:.1f}% "
                    f"| {b['mean_rank']:.0f} | {b['mean_rho_streak']:+.2f} "
                    f"| {b['decline_exported']:.2f} / {b['decline_best']:.2f} "
                    f"| {c['cliffs_delta']:+.2f} ({c['p_two_sided']:.3f}) |")
    h8a, h8c = a["H8a"], a["H8c"]
    rows += ["", "Discmix game, 12 runs per cell, all quantities exact. Exported rank "
             "and ρ: control runs, mean over 10 population snapshots (rank 1 = "
             "strongest). Declines: summed falls between snapshots against a fixed "
             "external panel, control runs. Archive effect: Cliff's δ of the final "
             "champions' cross-run strength, archive as test minus control, with the "
             f"two-sided exact Mann–Whitney p. Trend tests (one-sided permutation): "
             f"cycling vs λ ρ = {h8a['control']['rho']:+.2f} "
             f"(p {_p(h8a['control']['p_one_sided'])}) in control and "
             f"{h8a['test']['rho']:+.2f} (p {_p(h8a['test']['p_one_sided'])}) with the "
             f"archive; archive effect vs λ p {_p(h8c['p_one_sided'])}."]
    return "\n".join(rows)


QD_OUTCOMES = (("final_holdout", "final champion, held out", "{:+.2f}"),
               ("peak_holdout", "best champion, held out", "{:+.2f}"),
               ("above_parity", "checkpoints above parity", "{:.2f}"),
               ("late_mean", "mean score, last 100,000 games", "{:+.2f}"))


def table_qd(d):
    """The niche archive in Slime Volleyball (WP9, H9a)."""
    q = d["qd"]
    if not q:
        return None
    s = q["slime"]
    rows = ["| outcome | niche archive | control | δ (p) | time-ordered archive | δ (p) |",
            "|---|---|---|---|---|---|"]
    for key, label, fmt in QD_OUTCOMES:
        c, f = s["vs_control"][key], s["vs_fifo"][key]
        rows.append(f"| {label} | {fmt.format(c['niche_mean'])} | {fmt.format(c['ref_mean'])} "
                    f"| {c['cliffs_delta']:+.2f} ({c['p_two_sided']:.3f}) "
                    f"| {fmt.format(f['ref_mean'])} "
                    f"| {f['cliffs_delta']:+.2f} ({f['p_two_sided']:.3f}) |")
    L, n = s["learned"], s["n"]
    cells = s["archive_cells_final"]
    wn, wf = s["archive_late_winrate"]["niche"], s["archive_late_winrate"]["fifo"]
    rows.append(f"| learned to rally | {L['niche']}/{n} | {L['control']}/{n} | — "
                f"| {L['fifo']}/{n} | — |")
    rows += ["", f"{n} runs per arm on the same seeds (the replication's control and "
             f"archive-as-test runs). Scores: points per episode against the 2015 "
             f"baseline; δ: Cliff's δ, niche archive minus the comparison, with the "
             f"two-sided exact Mann–Whitney p. H9a (niche vs control, all four "
             f"p ≥ 0.05): **{'holds' if q['verdicts']['H9a'] else 'does not hold'}**. "
             f"Occupied cells at the end: {min(cells)}–{max(cells)} of 64. Late win "
             f"rate against the archive: niche {np.mean(wn):.2f}, time-ordered "
             f"{np.mean(wf):.2f} (means over runs)."]
    return "\n".join(rows)


def table_qdm(d):
    """The niche archive in discmix, one row per lambda (WP9, H9b and H9c)."""
    q = d["qd"]
    if not q:
        return None
    m = q["discmix"]
    rows = ["| λ | niche vs control, δ (p) | niche vs time-ordered archive, δ (p) "
            "| occupied cells (of 64) | cyclic triads: control / time-ordered / niche |",
            "|---|---|---|---|---|"]
    for lam in sorted(m["by_lambda"], key=float):
        r = m["by_lambda"][lam]
        c, t, cy = r["vs_control"], r["vs_test"], r["cyclic_share"]
        rows.append(f"| {float(lam):.2f} | {c['cliffs_delta']:+.2f} ({c['p_two_sided']:.3f}) "
                    f"| {t['cliffs_delta']:+.2f} ({t['p_two_sided']:.3f}) "
                    f"| {r['archive_cells_final']:.0f} "
                    f"| {100 * cy['control']:.1f}% / {100 * cy['test']:.1f}% "
                    f"/ {100 * cy['niche']:.1f}% |")
    b, c = m["H9b"], m["H9c"]
    rows += ["", "Discmix game, 12 runs per cell, all quantities exact. δ: Cliff's δ of "
             "the final champions' cross-run strength (mean expected score against the "
             "final champions of the other 35 runs at the same λ), niche archive minus "
             "the comparison, with the two-sided exact Mann–Whitney p; no decision "
             f"rests on these per-λ values. H9b, the effect vs control grows with λ: "
             f"one-sided permutation p {_p(b['p_one_sided'])}, "
             f"**{'holds' if b['rejected'] else 'does not hold'}**. H9c, at "
             f"λ = {c['lambda']:.2f} the niche archive beats the time-ordered one: "
             f"one-sided exact Mann–Whitney p {_p(c['p_one_sided'])}, "
             f"**{'holds' if c['rejected'] else 'does not hold'}**. Both under Holm."]
    return "\n".join(rows)


NEAT_ORDER = ("control", "ga2015", "es", "hof-eval-v2", "neat")


def table_neat(d):
    """NEAT as a fifth family next to the four of C6 (WP9, H9d)."""
    a = d["neat"]
    if not a:
        return None
    import fastvolley as fv
    fam = a["families"]
    rows = ["| family | runs | learned to rally | reached parity | final (held out) "
            "| spread vs 2020 GA | best final | checkpoints above parity "
            "| median cross-run Elo |", "|---|---|---|---|---|---|---|---|---|"]
    for f in NEAT_ORDER:
        r = fam[f]
        rows.append(f"| {'NEAT' if f == 'neat' else LABELS[f]} | {r['runs']} "
                    f"| {r['learned']}/{r['runs']} | {r['reached_parity']}/{r['runs']} "
                    f"| {r['final_mean']:+.2f} ± {r['final_sd']:.2f} "
                    f"| {r['sd_ratio_vs_control']:.2f}× | {r['best_final']:+.2f} "
                    f"| {r['above_parity']:.2f} | {r['elo_median']:+.0f} |")
    h, st = a["H9d"], a["structure"]
    rows += ["", "Final: end-of-run champion against the 2015 baseline, held-out seed, "
             "mean ± SD over runs; spread: that SD relative to the 2020 GA's. Elo: "
             "Bradley–Terry ratings of every run's final champion in one all-play-all "
             "tournament (NEAT finals and the final champion of every "
             "single-population run of the matrix), median per family. H9d, NEAT vs "
             f"the generational GA on the final champion: Cliff's δ "
             f"{h['cliffs_delta']:+.2f}, two-sided exact Mann–Whitney p "
             f"{_p(h['p_two_sided'])}, "
             f"**{'a detectable difference' if h['detectable_difference'] else 'no detectable difference'}**. "
             f"NEAT's final champions have {min(st['final_hidden'])}–"
             f"{max(st['final_hidden'])} hidden nodes and "
             f"{min(st['final_connections'])}–{max(st['final_connections'])} "
             f"enabled connections (the other families: a fixed 12-10-10-3 "
             f"network, {fv.PARAM_COUNT} weights and biases)."]
    return "\n".join(rows)


EXPORT_LABELS = {"streak": "streak (Ha's rule)", "tournament-4": "tournament, 4 peers",
                 "tournament-16": "**tournament, 16 peers**",
                 "tournament-64": "tournament, 64 peers", "random": "random member",
                 "best": "best member (oracle)"}
EXPORT_GAMES = {"streak": "0", "tournament-4": "256", "tournament-16": "1,024",
                "tournament-64": "4,096", "random": "0", "best": "—"}


def table_x(d):
    """Which member to export: the rules on the same populations (WP10)."""
    a = d["export"]
    if not a:
        return None
    s = a["slime"]
    n_snap = len(next(iter(a["per_run"]["slime"].values()))["snapshots"])
    rows = ["| rule | games per export | level | declines | rank in population "
            "| final vs zoo GA |", "|---|---|---|---|---|---|"]
    for r, lab in EXPORT_LABELS.items():
        x = s["rules"][r]
        zoo = s["zoo_final"].get(r)
        rows.append(f"| {lab} | {EXPORT_GAMES[r]} | {x['level']:+.2f} | {x['declines']:.2f} "
                    f"| {x['mean_rank']:.0f} | {'—' if zoo is None else f'{zoo:+.2f}'} |")
    h, k = s["H10a"], s["H10b"]
    rows += ["", f"Slime Volleyball, {h['n']} fresh control runs, every rule applied to the "
             f"same {n_snap} population snapshots per run. Level: mean held-out score of the "
             "exported member against the 2015 baseline over the snapshots; declines: "
             "summed falls between consecutive snapshots; rank: by score among the "
             "population (1 = best); zoo GA: the final exported member against the "
             "slimevolleygym zoo GA. Preregistered tests, tournament-16 "
             f"against streak: H10a level, {h['runs_improved']}/{h['n']} runs higher, "
             f"one-sided exact sign-flip p {_p(h['p_one_sided'])}, "
             f"**{'holds' if h['rejected'] else 'does not hold'}**; H10b declines, "
             f"{k['runs_improved']}/{k['n']} runs fewer, p {_p(k['p_one_sided'])}, "
             f"**{'holds' if k['rejected'] else 'does not hold'}** (Holm)."]
    return "\n".join(rows)


def table_xm(d):
    """The export rules in discmix, judged by outsiders (WP10)."""
    a = d["export"]
    if not a:
        return None
    m = a["discmix"]
    per = a["per_run"]["discmix"]
    n_lam = len(per) // len(m["by_lambda"])
    n_snap = len(next(iter(per.values()))["snapshots"])
    rows = ["| λ | streak | tournament, 16 peers | best member (oracle) "
            "| tournament higher | rank: streak / tournament |", "|---|---|---|---|---|---|"]
    for lam in sorted(m["by_lambda"], key=float):
        b = m["by_lambda"][lam]
        r = b["rules"]
        rows.append(f"| {float(lam):.2f} | {r['streak']['level']:+.3f} "
                    f"| {r['tournament-16']['level']:+.3f} | {r['best']['level']:+.3f} "
                    f"| {b['runs_improved']}/{n_lam} | {r['streak']['mean_rank']:.0f} / "
                    f"{r['tournament-16']['mean_rank']:.0f} |")
    c, e = m["H10c"], m["H10d"]
    rows += ["", f"Discmix game, {n_lam} fresh control runs per λ. Outsider strength: the "
             "exported member's exact mean expected score against every member of the "
             f"other {n_lam - 1} runs' populations at the same λ and snapshot, averaged over "
             f"the {n_snap} snapshots; rank among its own population by the same measure. "
             "Preregistered "
             f"tests: H10c, tournament-16 above streak over all {c['n']} runs "
             f"({c['runs_improved']} higher), one-sided sign-flip p {_p(c['p_one_sided'])}, "
             f"**{'holds' if c['rejected'] else 'does not hold'}**; H10d, the advantage "
             f"shrinks with λ, ρ = {e['rho']:+.2f}, one-sided p {_p(e['p_one_sided'])}, "
             f"**{'holds' if e['rejected'] else 'does not hold'}** (Holm)."]
    return "\n".join(rows)


def table_t(d):
    """Within-run transitivity at 5,000-game spacing, control runs (WP7)."""
    f = d["within_fine"]
    if not f:
        return None

    def pct(c, n):
        return f"{100 * c / n:.2f}% ({c:,}/{n:,})" if n else "—"
    rows = ["| run | ρ(Elo, time) | cyclic, ±0.25 rule | cyclic, sign test "
            "| within 50k games, sign test | next beats previous |",
            "|---|---|---|---|---|---|"]
    tot = {k: 0 for k in ("dc", "dn", "sc", "sn", "ssc", "ssn", "aw", "an")}
    for name in sorted(f["runs"]):
        r = f["runs"][name]
        db, st = r["deadband"], r["sign_test"]
        rows.append(f"| {name} | {r['spearman_elo_vs_time']:+.2f} "
                    f"| {pct(db['cyclic'], db['triads_decided'])} "
                    f"| {pct(st['cyclic'], st['triads_decided'])} "
                    f"| {pct(st['short_cyclic'], st['short_triads_decided'])} "
                    f"| {st['adjacent_later_wins']}/{st['adjacent_decided']} |")
        for k, v in (("dc", db["cyclic"]), ("dn", db["triads_decided"]),
                     ("sc", st["cyclic"]), ("sn", st["triads_decided"]),
                     ("ssc", st["short_cyclic"]), ("ssn", st["short_triads_decided"]),
                     ("aw", st["adjacent_later_wins"]), ("an", st["adjacent_decided"])):
            tot[k] += v
    rows.append(f"| *all control runs* | — | {pct(tot['dc'], tot['dn'])} "
                f"| {pct(tot['sc'], tot['sn'])} | {pct(tot['ssc'], tot['ssn'])} "
                f"| {tot['aw']}/{tot['an']} |")
    rows += ["", f"Every one of the 100 champions of each control run (one per "
             f"{f['every']:,} games) played every other, {f['games_per_pair']} "
             f"games per pair. A triad counts when all three of its pairs are "
             f"decided: by the paper's rule (mean margin outside "
             f"±{f['deadband']}) or by an exact sign test on wins against "
             f"losses (p < {f['alpha']}). 'Next beats previous': adjacent "
             f"champions whose difference the sign test decides, and how often "
             f"the later one wins."]
    return "\n".join(rows)


def table_z(d):
    """Final champions against the slimevolleygym zoo policies (WP7)."""
    y, per_run = d["yardsticks"], d["per_run"]
    if not y or not per_run:
        return None
    rows = ["| condition | runs | vs 2015 baseline | vs zoo GA | vs zoo CMA-ES "
            "| beat zoo GA | beat zoo CMA-ES |", "|---|---|---|---|---|---|---|"]
    for c in ORDER:
        names = sorted(n for n in y["per_run"] if per_run[n]["condition"] == c)
        if not names:
            continue
        base = [per_run[n]["final_holdout"] for n in names]
        ga = [y["per_run"][n]["zoo-ga"]["mean"] for n in names]
        cma = [y["per_run"][n]["zoo-cma"]["mean"] for n in names]
        rows.append(f"| {LABELS.get(c, c)} | {len(names)} | {np.mean(base):+.2f} "
                    f"| {np.mean(ga):+.2f} | {np.mean(cma):+.2f} "
                    f"| {sum(x > 0 for x in ga)}/{len(names)} "
                    f"| {sum(x > 0 for x in cma)}/{len(names)} |")
    zb, zz = y["zoo_vs_baseline"], y["zoo_ga_vs_zoo_cma"]
    rows += ["", f"Final (t = 500,000) champion of every single-population run, "
             f"mean points per episode. Baseline column: held out, "
             f"{SELECT_EPISODES:,} episodes; "
             f"zoo columns: {2 * y['games_per_side']} games per champion, half "
             f"on each side. 'Beat' counts runs whose champion scores above 0. "
             f"For scale, against the 2015 baseline the zoo GA scores "
             f"{zb['zoo-ga']['mean']:+.2f} and the zoo CMA-ES "
             f"{zb['zoo-cma']['mean']:+.2f}; head to head the zoo GA scores "
             f"{zz['mean']:+.2f} against the zoo CMA-ES."]
    return "\n".join(rows)


# The analysis files each table is built from. Table 10 reads the raw
# two-population runs directly; per_run.json's provenance covers those files.
FILES = {
    "per_run": f"{ANDIR}/per_run.json",
    "conditions": f"{ANDIR}/conditions.json",
    "within": f"{ANDIR}/within_run.json",
    "across": f"{ANDIR}/across_runs.json",
    "proxy": f"{ANDIR}/champion_proxy.json",
    "proxy_heldout": f"{ANDIR}/proxy_heldout.json",
    "reference": f"{ANDIR}/reference_curve.json",
    "reexport": f"{ANDIR}/reexport.json",
    "resume": f"{ANDIR}/resume_fast.json",
    "yardsticks": f"{ANDIR}/yardsticks.json",
    "within_fine": f"{ANDIR}/within_fine.json",
    "replication": "results/replication/analysis.json",
    "lab": "results/lab/analysis.json",
    "qd": "results/qd/analysis.json",
    "neat": "results/neat/analysis.json",
    "neat_explore": "results/neat/explore/summary.json",
    "export": "results/export/analysis.json",
    "validation": "results/validation.json",
}
DEPS = {
    "c": ["per_run"],
    "1": ["conditions"], "2": ["conditions"], "3": ["reference"],
    "4": ["within"], "5": ["proxy", "proxy_heldout"], "6": ["across"], "7": ["per_run"],
    "8": ["per_run", "reference"], "9": ["reexport"], "10": ["per_run"],
    "r": ["conditions", "per_run", "reference"], "a1": ["validation"],
    "a2": ["per_run"], "a3": ["resume", "reference"],
    "z": ["yardsticks", "per_run"], "t": ["within_fine"],
    "rep": ["replication"], "lab": ["lab"], "qd": ["qd"], "qdm": ["qd"],
    "neat": ["neat", "per_run"], "x": ["export"], "xm": ["export"],
}

TABLES = {
    "c": table_c,
    "1": table_1, "2": table_2, "3": table_3, "4": table_4, "5": table_5,
    "6": table_6, "7": table_7, "8": table_8, "9": table_9, "10": table_10,
    "r": table_r, "z": table_z, "t": table_t, "rep": table_rep,
    "lab": table_lab, "qd": table_qd, "qdm": table_qdm,
    "neat": table_neat, "x": table_x, "xm": table_xm,
    "a1": table_a1, "a2": table_a2, "a3": table_a3,
}


# Tables the LaTeX paper includes, written to paper/tables/<key>.tex.
PAPER_TABLES = ["c", "r", "3", "4", "5", "7", "9", "10", "1", "2", "6", "8", "a1", "a3",
                "z", "t", "rep", "x", "xm", "lab", "qd", "qdm", "neat"]
TEX_MAP = [("±", r"$\pm$"), ("—", "---"), ("–", "--"), ("σ", r"$\sigma$"),
           ("δ", r"$\delta$"), ("ρ", r"$\rho$"), ("λ", r"$\lambda$"), ("×", r"$\times$"),
           ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("%", r"\%"), ("&", r"\&"),
           ("#", r"\#")]


def md_cell_to_tex(cell):
    cell = cell.strip()
    code = re.findall(r"`([^`]*)`", cell)
    cell = re.sub(r"`([^`]*)`", "\x00", cell)
    cell = cell.replace("_", r"\_")
    for a, b in TEX_MAP:
        cell = cell.replace(a, b)
    cell = re.sub(r"(?<![\w.$-])-(\d)", r"$-$\1", cell)   # not the 2nd '-' of '--'
    cell = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", cell)
    cell = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", cell)
    for c in code:
        cell = cell.replace("\x00", r"\texttt{" + c.replace("_", r"\_") + "}", 1)
    return cell


# column types where the default (left, then right-aligned) does not fit
_L = r">{\raggedright\arraybackslash}p{%s}"
_R = r">{\raggedleft\arraybackslash}p{%s}"
TEX_SPEC = {
    "c": r"l>{\raggedright\arraybackslash}p{0.66\linewidth}r",
    "r": _L % "0.24\\linewidth" + "r" + "".join(_R % "0.13\\linewidth"
                                              for _ in range(4)),
    "10": _L % "0.17\\linewidth" + "r" + _L % "0.27\\linewidth"
          + "".join(_R % "0.12\\linewidth" for _ in range(3)),
}


def md_table_to_tex(md, spec=None):
    """The table part of a generated markdown table as a booktabs tabular."""
    rows = [l for l in md.split("\n") if l.startswith("|")]
    head = [c for c in rows[0].strip("|").split("|")]
    body = [[c for c in r.strip("|").split("|")] for r in rows[2:]]
    spec = spec or "l" + "r" * (len(head) - 1)
    out = ["% generated by make_tables.py from the markdown table -- do not edit",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
           " & ".join(md_cell_to_tex(c) for c in head) + r" \\", "\\midrule"]
    out += [" & ".join(md_cell_to_tex(c) for c in r) + r" \\" for r in body]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def targets():
    """Every file that carries table markers."""
    out = [p for p in sorted(glob.glob(os.path.join(PAPER, "*.md")))
           if not os.path.basename(p).startswith("_")]
    return out + [p for p in ("README.md",) if os.path.exists(p)]


def required_tables(paths):
    """Tables some document expects: a marker, or a section of _tables.md."""
    keys = set()
    for path in paths:
        keys |= set(re.findall(r"<!-- table:([\w-]+) -->", open(path).read()))
    tm = os.path.join(PAPER, "_tables.md")
    if os.path.exists(tm):
        keys |= {k.lower() for k in
                 re.findall(r"^### Table (\S+)$", open(tm).read(), re.M)}
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless every table follows from disk")
    args = ap.parse_args()

    d = {k: (json.load(open(p)) if os.path.exists(p) else None)
         for k, p in FILES.items()}

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
    combined = "\n".join(combined)

    paths = targets()
    required = required_tables(paths)
    errors = []
    for k in sorted(required - set(TABLES)):
        errors.append(f"table {k}: named in a document but no generator exists")
    for k in sorted((required & set(TABLES)) - set(built)):
        errors.append(f"table {k}: named in a document but cannot be built "
                      f"from the data on disk "
                      f"(needs {', '.join(FILES[f] for f in DEPS[k])})")
    # numbers in running text
    values = pn.compute(d)
    known = set(pn.definitions())
    used = set()
    for path in paths:
        for m in NUM.finditer(open(path).read()):
            used.add(m.group(2))
    for k in sorted(used - known):
        errors.append(f"number {k}: used in a document but not defined in "
                      f"prose_numbers.py")
    for k in sorted((used & known) - set(values)):
        errors.append(f"number {k}: used in a document but cannot be "
                      f"computed from the data on disk")

    deps = {FILES[f] for k in (required | set(built)) & set(DEPS)
            for f in DEPS[k]}
    deps |= {FILES[f] for k in used & set(values) for f in values[k][1]}
    if os.path.isdir(os.path.dirname(NUMBERS_TEX)):    # numbers.tex uses all
        deps |= {FILES[f] for v in values.values() for f in v[1]}
    errors += pv.verify(sorted(deps))

    # README.md carries markers too: the repository is public, so its headline
    # numbers must come from the same generator as the paper's.
    stale = []
    tm = os.path.join(PAPER, "_tables.md")
    if not os.path.exists(tm) or open(tm).read() != combined:
        stale.append(tm)
        if not args.check:
            os.makedirs(PAPER, exist_ok=True)
            open(tm, "w").write(combined)
    if os.path.isdir("paper"):
        os.makedirs("paper/tables", exist_ok=True)
        for key in PAPER_TABLES:
            path = f"paper/tables/{key}.tex"
            if key not in built:
                errors.append(f"table {key}: needed by the paper but cannot be "
                              f"built")
                continue
            tex = md_table_to_tex(built[key], TEX_SPEC.get(key))
            if not os.path.exists(path) or open(path).read() != tex:
                stale.append(path)
                if not args.check:
                    open(path, "w").write(tex)
    if os.path.isdir(os.path.dirname(NUMBERS_TEX)):
        tex = pn.write_tex(values)
        if not os.path.exists(NUMBERS_TEX) or open(NUMBERS_TEX).read() != tex:
            stale.append(NUMBERS_TEX)
            if not args.check:
                open(NUMBERS_TEX, "w").write(tex)
    for path in paths:
        src = open(path).read()
        new = src
        for key, md in built.items():
            # the block may be empty (a marker pair just added), which a
            # pattern requiring a newline on both sides of the content would
            # silently skip -- and --check would then pass on an empty table
            pat = re.compile(rf"(<!-- table:{re.escape(key)} -->\n)(?:.*?\n)?"
                             rf"(<!-- /table:{re.escape(key)} -->)",
                             re.DOTALL)
            if pat.search(new):
                new = pat.sub(lambda m: m.group(1) + md + "\n" + m.group(2), new)
        new = NUM.sub(lambda m: (m.group(1) + values[m.group(2)][0] + m.group(4))
                      if m.group(2) in values else m.group(0), new)
        if new != src:
            stale.append(path)
            if not args.check:
                open(path, "w").write(new)

    print(f"built {len(built)}/{len(TABLES)} tables: {', '.join(sorted(built))}")
    for e in errors:
        print(f"ERROR {e}")
    if args.check:
        if stale:
            print("OUT OF DATE: " + ", ".join(stale))
        ok = not stale and not errors
        print("up to date" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)
    print(f"updated: {', '.join(os.path.basename(p) for p in stale) or 'nothing'}")


if __name__ == "__main__":
    main()
