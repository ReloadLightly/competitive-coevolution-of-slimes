"""
make_figures.py — every figure in the writeup, regenerated from the data.

    python make_figures.py                 # all figures that have data
    python make_figures.py --only fig3

Reads results/matrix/*.npz, results/analysis/*.json and the reference run's
re-scored curve. Writes results/figures/*.png.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = "results/figures"
ANDIR = "results/analysis"
MATRIX = "results/matrix"

COLORS = {
    "reference": "#222222",
    "control": "#3B6EA8",
    "hof-eval": "#1F7A5A",
    "hof-0.25": "#E08B3C",
    "hof-0.50": "#B0413E",
    "hof-full": "#8C5A2B",
    "ga2015": "#7A5EA6",
    "es": "#C1445E",
    "sigma-0.05": "#4C956C",
    "sigma-0.20": "#946A9E",
    "pop-32": "#8A6A55",
    "pop-512": "#4FA3B8",
}
LABELS = {
    "control": "control (Ha 2020 GA)",
    "hof-eval": "archive as test",
    "hof-0.25": "archive as parent, p=0.25",
    "hof-0.50": "archive as parent, p=0.50",
    "hof-full": "archive as parent, full span",
    "ga2015": "generational GA (Ha 2015)",
    "es": "self-play ES",
    "sigma-0.05": r"$\sigma=0.05$",
    "sigma-0.20": r"$\sigma=0.20$",
    "pop-32": "population 32",
    "pop-512": "population 512",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def load_matrix():
    runs = {}
    for p in sorted(glob.glob(os.path.join(MATRIX, "*_s*.npz"))):
        name = os.path.basename(p)[:-4]
        cond, seed = name.rsplit("_s", 1)
        z = np.load(p)
        runs.setdefault(cond, []).append({
            "seed": int(seed), "name": name,
            "t": (np.arange(len(z["mean_score"])) + 1) * int(z["save_every"][0]),
            "score": z["mean_score"], "win": z["win_rate"],
            "trainlen": z["train_meanlen"], "evallen": z["eval_meanlen"],
            "streaks": z["streaks"], "hofwin": z["hof_winrate"],
        })
    for v in runs.values():
        v.sort(key=lambda r: r["seed"])
    return runs


def load_json(name):
    p = os.path.join(ANDIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def band(ax, runs, color, label, lw=1.4, alpha=0.16, stat="median"):
    """Median (or mean) across seeds with an inter-quartile band."""
    t = runs[0]["t"]
    n = min(len(r["score"]) for r in runs)
    m = np.array([r["score"][:n] for r in runs])
    centre = np.median(m, axis=0) if stat == "median" else m.mean(axis=0)
    lo = np.percentile(m, 25, axis=0)
    hi = np.percentile(m, 75, axis=0)
    ax.fill_between(t[:n] / 1000, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(t[:n] / 1000, centre, color=color, lw=lw, label=label)
    return centre


def parity(ax):
    ax.axhline(0, color="#999999", lw=0.8, ls=(0, (4, 3)), zorder=1)


# --------------------------------------------------------------------------
def fig1_reference():
    ref = load_json("reference_curve.json")
    if not ref:
        return "fig1: no reference curve yet"
    t = np.array(ref["tournament"]) / 1000
    s = np.array(ref["mean_score"])
    ln = np.array(ref["meanlen"])

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    ax = axes[0]
    parity(ax)
    ax.plot(t, s, color=COLORS["reference"], lw=0.8, alpha=0.45)
    k = 9
    if len(s) > k:
        sm = np.convolve(s, np.ones(k) / k, mode="valid")
        ax.plot(t[k // 2: k // 2 + len(sm)], sm, color=COLORS["reference"], lw=1.8,
                label=f"{k}-checkpoint moving average")
    above = s > 0
    if above.any():
        first = t[np.argmax(above)]
        ax.axvline(first, color="#C0392B", lw=0.9, ls=":")
        ax.annotate(f"first checkpoint above parity: {first*1000:,.0f} games",
                    xy=(first, -0.15), xytext=(t.max() * 0.02, -1.9), fontsize=7,
                    color="#C0392B", va="center",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.7,
                                    shrinkA=2, shrinkB=2))
    ax.set_ylabel("score vs 2015 baseline\n(points per episode)")
    ax.set_title(f"Reference run on slimevolleygym: {t.max()*1000:,.0f} "
                 f"self-play games", loc="left")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    ax2.plot(t, ln, color="#3B6EA8", lw=1.2)
    ax2.set_ylabel("evaluation rally\nlength (steps)")
    ax2.set_xlabel("self-play games (thousands)")
    ax2.set_ylim(0, 3100)
    fig.savefig(f"{FIGDIR}/fig1_reference_trajectory.png")
    plt.close(fig)
    return "fig1_reference_trajectory.png"


def fig2_control(runs):
    if "control" not in runs:
        return "fig2: no control runs yet"
    ref = load_json("reference_curve.json")
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    parity(ax)
    for r in runs["control"]:
        ax.plot(r["t"] / 1000, r["score"], color=COLORS["control"], lw=0.7,
                alpha=0.45)
    band(ax, runs["control"], COLORS["control"],
         f"control, median of {len(runs['control'])} seeds", lw=2.0)
    if ref:
        # the reference run checkpoints every 1,000 games against the matrix's
        # 5,000, so subsample it onto the same grid before overlaying
        rt = np.array(ref["tournament"])
        rs = np.array(ref["mean_score"])
        keep = (rt % 5000) == 0
        ax.plot(rt[keep] / 1000, rs[keep], color=COLORS["reference"], lw=1.1,
                ls=(0, (5, 2)), alpha=0.85,
                label="reference run (slimevolleygym)")
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel("score vs 2015 baseline")
    n_seeds = len(runs["control"])
    ax.set_title(f"The same algorithm, {n_seeds} seeds: the phase change is "
                 f"real, its timing is not reproducible", loc="left")
    ax.legend(loc="lower right")
    fig.savefig(f"{FIGDIR}/fig2_control_seeds.png")
    plt.close(fig)
    return "fig2_control_seeds.png"


def _strip(ax, per_run, conds, key, ylabel, title):
    for i, cond in enumerate(conds):
        vals = [v[key] for v in per_run.values() if v["condition"] == cond]
        if not vals:
            continue
        x = np.full(len(vals), i, dtype=float)
        x += np.linspace(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=16, color=COLORS[cond], alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.plot([i - 0.26, i + 0.26], [np.median(vals)] * 2,
                color=COLORS[cond], lw=2.0, zorder=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], rotation=18,
                       ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def fig3_hof(runs, per_run):
    conds = [c for c in ("control", "hof-eval", "hof-0.25", "hof-0.50",
                         "hof-full") if c in runs]
    if len(conds) < 2:
        return "fig3: need control and a hall-of-fame condition"
    fig = plt.figure(figsize=(6.8, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    parity(ax)
    for c in conds:
        band(ax, runs[c], COLORS[c], f"{LABELS[c]}  (n={len(runs[c])})", lw=1.8)
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel("score vs 2015 baseline")
    ax.set_title("An archive of past champions: as a test, and as a parent",
                 loc="left")
    ax.legend(loc="lower right")

    if per_run:
        for j, (key, lab, ttl) in enumerate([
                ("late_mean", "points/episode", "mean, last 100k games"),
                ("volatility", "|Δ| between checkpoints", "volatility"),
                ("drawdown", "best-so-far − current", "drawdown")]):
            _strip(fig.add_subplot(gs[1, j]), per_run, conds, key, lab, ttl)
    fig.savefig(f"{FIGDIR}/fig3_hall_of_fame.png")
    plt.close(fig)
    return "fig3_hall_of_fame.png"


def fig4_ablations(runs, per_run):
    groups = [(["sigma-0.05", "control", "sigma-0.20"], "mutation scale"),
              (["pop-32", "control", "pop-512"], "population size")]
    groups = [([c for c in g if c in runs], t) for g, t in groups]
    groups = [(g, t) for g, t in groups if len(g) >= 2]
    if not groups:
        return "fig4: no ablation runs yet"
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 3.1),
                             squeeze=False)
    for ax, (conds, title) in zip(axes[0], groups):
        parity(ax)
        for c in conds:
            lab = "control ($\\sigma$=0.10, pop 128)" if c == "control" else LABELS[c]
            band(ax, runs[c], COLORS[c], f"{lab} (n={len(runs[c])})", lw=1.6)
        ax.set_xlabel("self-play games (thousands)")
        ax.set_ylabel("score vs 2015 baseline")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right")
    fig.savefig(f"{FIGDIR}/fig4_ablations.png")
    plt.close(fig)
    return "fig4_ablations.png"


def fig5_coevolution(within, per_run):
    if not within:
        return "fig5: no within-run tournaments yet"
    fig = plt.figure(figsize=(6.8, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.1, 1, 1])

    ax = fig.add_subplot(gs[0, 0])
    for name, d in within.items():
        cond = name.rsplit("_s", 1)[0]
        if cond != "control":
            continue
        ax.plot(np.array(d["checkpoints"]) / 1000, d["elo"],
                color=COLORS["control"], lw=1.0, alpha=0.7, marker="o", ms=2.2)
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel("Elo within its own run")
    ax.set_title("Later beats earlier?", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    name = next((n for n in within if n.startswith("control")), None)
    m = np.array(within[name]["margin"])
    ts = np.array(within[name]["checkpoints"]) / 1000
    im = ax.imshow(m, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower")
    ax.set_xticks(range(0, len(ts), max(1, len(ts) // 4)))
    ax.set_xticklabels([f"{ts[i]:.0f}" for i in range(0, len(ts), max(1, len(ts) // 4))])
    ax.set_yticks(range(0, len(ts), max(1, len(ts) // 4)))
    ax.set_yticklabels([f"{ts[i]:.0f}" for i in range(0, len(ts), max(1, len(ts) // 4))])
    ax.set_xlabel("opponent (thousands of games)")
    ax.set_ylabel("champion")
    ax.set_title(f"margin matrix, {name}", loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="points")

    ax = fig.add_subplot(gs[0, 2])
    conds, fracs = [], []
    for cond in COLORS:
        vals = [d["cyclic_frac"] for n, d in within.items()
                if n.rsplit("_s", 1)[0] == cond and d["cyclic_frac"] is not None]
        if vals:
            conds.append(cond)
            fracs.append(vals)
    for i, (c, vals) in enumerate(zip(conds, fracs)):
        x = np.full(len(vals), i, dtype=float) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=14, color=COLORS[c], zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=COLORS[c], lw=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS.get(c, c) for c in conds], rotation=25, ha="right")
    ax.set_ylabel("cyclic triads (fraction)")
    ax.set_title("Intransitivity", loc="left")
    fig.savefig(f"{FIGDIR}/fig5_coevolution.png")
    plt.close(fig)
    return "fig5_coevolution.png"


def fig6_proxy(proxy):
    if not proxy:
        return "fig6: no proxy data yet"
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axes[0]
    for name, rows in proxy.items():
        t = [r["tournament"] / 1000 for r in rows]
        ax.plot(t, [r["exported_score"] for r in rows], color="#B0413E", lw=0.9,
                alpha=0.7)
        ax.plot(t, [r["best_score"] for r in rows], color="#3B6EA8", lw=0.9,
                alpha=0.7)
    parity(ax)
    ax.plot([], [], color="#B0413E", lw=1.6, label="exported champion")
    ax.plot([], [], color="#3B6EA8", lw=1.6, label="best in the same pool")
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel("score vs 2015 baseline")
    ax.set_title("What the champion-export rule costs", loc="left")
    ax.legend(loc="lower right", handlelength=1.4)

    ax = axes[1]
    allrows = [r for rows in proxy.values() for r in rows]
    ax.scatter([r["tournament"] / 1000 for r in allrows],
               [r["spearman_streak_vs_score"] for r in allrows],
               s=16, color="#4C956C", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#999999", lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel(r"Spearman $\rho$(streak, score)")
    ax.set_title("Does the streak counter track quality?", loc="left")
    ax.set_ylim(-1, 1)
    fig.savefig(f"{FIGDIR}/fig6_champion_proxy.png")
    plt.close(fig)
    return "fig6_champion_proxy.png"


def fig7_cross_run(across):
    if not across:
        return "fig7: no cross-run tournament yet"
    names = across["runs"]
    elo = np.array(across["elo"])
    conds = sorted({n.rsplit("_s", 1)[0] for n in names},
                   key=lambda c: -np.median([elo[i] for i, n in enumerate(names)
                                             if n.rsplit("_s", 1)[0] == c]))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, c in enumerate(conds):
        vals = [elo[j] for j, n in enumerate(names) if n.rsplit("_s", 1)[0] == c]
        y = np.full(len(vals), i, dtype=float) + np.linspace(-0.14, 0.14, len(vals))
        ax.scatter(vals, y, s=20, color=COLORS.get(c, "#666"), zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.plot([np.median(vals)] * 2, [i - 0.28, i + 0.28],
                color=COLORS.get(c, "#666"), lw=2.2)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABELS.get(c, c) for c in conds])
    ax.set_xlabel("Elo in the all-runs tournament of final champions")
    ax.set_title("Which condition's champions actually beat the others?",
                 loc="left")
    fig.savefig(f"{FIGDIR}/fig7_cross_run_elo.png")
    plt.close(fig)
    return "fig7_cross_run_elo.png"


def fig8_families(runs, per_run):
    conds = [c for c in ("control", "ga2015", "es") if c in runs]
    if len(conds) < 2:
        return "fig8: need at least two algorithm families"
    fig = plt.figure(figsize=(6.8, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.55, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    parity(ax)
    for c in conds:
        for r in runs[c]:
            ax.plot(r["t"] / 1000, r["score"], color=COLORS[c], lw=0.6, alpha=0.3)
        band(ax, runs[c], COLORS[c], f"{LABELS[c]}  (n={len(runs[c])})", lw=1.9)
    ax.set_xlabel("self-play games (thousands)")
    ax.set_ylabel("score vs 2015 baseline")
    ax.set_title("Three ways to turn a population into the next population",
                 loc="left")
    ax.legend(loc="lower right")
    if per_run:
        for j, (key, lab, ttl) in enumerate([
                ("late_mean", "points/episode", "mean, last 100k games"),
                ("volatility", "|Δ| between checkpoints", "volatility"),
                ("above_parity", "fraction of checkpoints", "above parity")]):
            _strip(fig.add_subplot(gs[1, j]), per_run, conds, key, lab, ttl)
    fig.savefig(f"{FIGDIR}/fig8_algorithm_families.png")
    plt.close(fig)
    return "fig8_algorithm_families.png"


def fig9_reexport(reexp):
    """What you get for replacing the champion-export rule."""
    if not reexp:
        return "fig9: no re-export data yet"
    su_ = reexp["summary"]
    opps = sorted(int(k.split("_")[1]) for k in su_["recovered_fraction"])
    games = [su_["ranking_games_per_snapshot"][f"internal_{o}"] for o in opps]
    lvl = [su_[f"internal_score_{o}"]["level_mean"] for o in opps]
    vol = [su_[f"internal_score_{o}"]["volatility_mean"] for o in opps]
    rho = [su_["rho_internal_external"][f"internal_{o}"] for o in opps]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    ax.axhline(su_["streak_score"]["level_mean"], color="#B0413E", lw=1.4,
               ls=(0, (4, 2)), label="streak rule (Ha)")
    ax.axhline(su_["external_score"]["level_mean"], color="#3B6EA8", lw=1.4,
               ls=(0, (1, 2)), label="best in pool (oracle)")
    ax.axhline(su_["median_score"]["level_mean"], color="#999999", lw=1.0,
               ls=(0, (2, 2)), label="population median")
    ax.plot(games, lvl, color="#1F7A5A", lw=1.6, marker="o", ms=3.4,
            label="internal round robin")
    ax.set_xscale("log")
    ax.set_xlabel("ranking games")
    ax.set_ylabel("score vs 2015 baseline")
    ax.set_title("Level of the exported individual", loc="left", fontsize=8.5)
    ax.legend(loc="center right", fontsize=6.2, handlelength=1.5,
              borderpad=0.2, labelspacing=0.3)

    ax = axes[1]
    ax.axhline(su_["streak_score"]["volatility_mean"], color="#B0413E", lw=1.4,
               ls=(0, (4, 2)))
    ax.axhline(su_["external_score"]["volatility_mean"], color="#3B6EA8", lw=1.4,
               ls=(0, (1, 2)))
    ax.plot(games, vol, color="#1F7A5A", lw=1.6, marker="o", ms=3.4)
    ax.set_xscale("log")
    ax.set_xlabel("ranking games")
    ax.set_ylabel(r"mean $|\Delta|$ per snapshot")
    ax.set_title("Volatility of the curve", loc="left", fontsize=8.5)
    ax.set_ylim(0.55, 0.90)

    ax = axes[2]
    ax.axhline(su_["rho_streak_external"], color="#B0413E", lw=1.4, ls=(0, (4, 2)),
               label="streak counter")
    ax.plot(games, rho, color="#1F7A5A", lw=1.6, marker="o", ms=3.4,
            label="internal margin")
    ax.axhline(0, color="#CCCCCC", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("ranking games")
    ax.set_ylabel(r"$\rho$ with true skill")
    ax.set_title("Does the rule know?", loc="left", fontsize=8.5)
    ax.legend(loc="center right", fontsize=6.2, handlelength=1.5,
              borderpad=0.2, labelspacing=0.3)
    fig.savefig(f"{FIGDIR}/fig9_reexport.png")
    plt.close(fig)
    return "fig9_reexport.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    style()
    os.makedirs(FIGDIR, exist_ok=True)

    runs = load_matrix()
    per_run = load_json("per_run.json")
    within = load_json("within_run.json")
    across = load_json("across_runs.json")
    proxy = load_json("champion_proxy.json")

    todo = {
        "fig1": lambda: fig1_reference(),
        "fig2": lambda: fig2_control(runs),
        "fig3": lambda: fig3_hof(runs, per_run),
        "fig4": lambda: fig4_ablations(runs, per_run),
        "fig5": lambda: fig5_coevolution(within, per_run),
        "fig6": lambda: fig6_proxy(proxy),
        "fig7": lambda: fig7_cross_run(across),
        "fig8": lambda: fig8_families(runs, per_run),
        "fig9": lambda: fig9_reexport(load_json("reexport.json")),
    }
    for k, fn in todo.items():
        if args.only and k not in args.only.split(","):
            continue
        try:
            print(f"{k}: {fn()}")
        except Exception as e:  # a missing condition should not kill the batch
            print(f"{k}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
