"""
plot_curve.py — learning curve: champion score vs the 2015 baseline over
self-play training, from eval_curve.jsonl (produced by eval_vs_baseline --all).

Usage:
  python plot_curve.py results/ga_selfplay/eval_curve.jsonl --out results/figures/learning_curve.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette (validated): series blue on light surface, recessive chrome
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

HA_REFERENCE = 0.353  # Ha's GA self-play result after 500k tournaments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curve", help="eval_curve.jsonl")
    ap.add_argument("--out", default="results/figures/learning_curve.png")
    ap.add_argument("--title",
                    default="Score vs the 2015 baseline during self-play evolution")
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.curve)]
    recs.sort(key=lambda r: r["tournament"])
    x = np.array([r["tournament"] for r in recs]) / 1000.0
    y = np.array([r["mean_score"] for r in recs])
    n = np.array([r["episodes"] for r in recs])
    se = np.array([r["std_score"] for r in recs]) / np.sqrt(np.maximum(n, 1))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # recessive grid, baseline-weight axes
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)

    # reference lines: parity and Ha's published result
    ax.axhline(0, color=BASELINE, linewidth=1.2, linestyle=(0, (5, 4)))
    ax.axhline(HA_REFERENCE, color=INK2, linewidth=1.2, linestyle=(0, (2, 3)))
    label_box = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
    ax.text(x.max() * 1.11, 0, "parity", va="center", ha="right",
            color=MUTED, fontsize=8, bbox=label_box)
    ax.text(x.max() * 1.11, HA_REFERENCE, f"Ha 500k ({HA_REFERENCE})",
            va="center", ha="right", color=INK2, fontsize=8, bbox=label_box)

    # the series: 2px line, standard-error band
    ax.fill_between(x, y - se, y + se, color=SERIES, alpha=0.15, linewidth=0)
    ax.plot(x, y, color=SERIES, linewidth=2, solid_capstyle="round")

    ax.set_xlabel("tournament games played (thousands)", color=INK2, fontsize=9)
    ax.set_ylabel("mean score per episode  (−5 … +5)", color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlim(0, x.max() * 1.12)  # headroom for reference labels

    subtitle = args.subtitle or (
        f"mean over {int(n[0])} episodes per checkpoint · "
        "the baseline is never seen during training")
    ax.set_title(args.title, loc="left", color=INK, fontsize=12, pad=18,
                 fontweight="bold")
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {args.out} ({len(recs)} checkpoints, "
          f"latest t={int(x[-1]*1000)}: {y[-1]:+.3f})")


if __name__ == "__main__":
    main()
