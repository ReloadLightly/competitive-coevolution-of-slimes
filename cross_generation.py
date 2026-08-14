"""
cross_generation.py — do later champions actually beat earlier ones?

The learning curve measures every champion against ONE frozen opponent (the
2015 baseline). That cannot distinguish two very different explanations for
the violent swings in that curve:

  (a) the population genuinely forgets skills no current opponent punishes
      (real coevolutionary cycling — the classic argument for a hall of fame);
  (b) the exported "champion" is simply the wrong individual — it is picked by
      longest winning lineage, a cheap proxy Ha adopted "to save time".

This script plays saved champions against EACH OTHER in a round-robin. If
skill is transitive (later beats earlier, consistently), the swings are mostly
proxy noise. If the results are cyclic (A beats B beats C beats A), that is
genuine intransitivity and the archive argument is empirically motivated.

Usage:
  python cross_generation.py --every 10000 --games 20
  python cross_generation.py --checkpoints 25000,100000,175000 --games 40
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from slimevolleygym import SlimeVolleyEnv
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym.mlp import Model, games


def load(path):
    with open(path) as f:
        params, _ = json.load(f)
    m = Model(games["slimevolleylite"])
    m.set_model_params(np.array(params))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ga_selfplay")
    ap.add_argument("--every", type=int, default=10000,
                    help="take one champion every N games")
    ap.add_argument("--checkpoints", default=None,
                    help="explicit comma-separated tournament numbers")
    ap.add_argument("--games", type=int, default=20,
                    help="games per ordered pair (each pair plays both sides)")
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--out", default="results/cross_generation.json")
    args = ap.parse_args()

    if args.checkpoints:
        ts = [int(t) for t in args.checkpoints.split(",")]
    else:
        allf = sorted(glob.glob(os.path.join(args.dir, "ga_*.json")))
        ts = [int(os.path.basename(f)[3:-5]) for f in allf]
        ts = [t for t in ts if t % args.every == 0]

    paths = {t: os.path.join(args.dir, f"ga_{t:08d}.json") for t in ts}
    ts = [t for t in ts if os.path.exists(paths[t])]
    print(f"{len(ts)} champions: {ts}", flush=True)

    models = {t: load(paths[t]) for t in ts}
    env = SlimeVolleyEnv()
    env.seed(args.seed)
    np.random.seed(args.seed)

    # score[a][b] = mean points margin of champion a against champion b,
    # averaged over both court sides (removes any side bias).
    results = {}
    for a, b in itertools.combinations(ts, 2):
        margins = []
        for _ in range(args.games):
            s, _ = rollout(env, models[a], models[b])   # a on the right
            margins.append(s)
            s, _ = rollout(env, models[b], models[a])   # a on the left
            margins.append(-s)
        m = float(np.mean(margins))
        results[f"{a}v{b}"] = {"a": a, "b": b, "margin_a": m,
                               "games": 2 * args.games}
        print(f"  {a:>7d} vs {b:>7d}: {m:+.2f}", flush=True)

    # transitivity check: rank by average margin, then count upsets
    avg = {t: 0.0 for t in ts}
    for r in results.values():
        avg[r["a"]] += r["margin_a"]
        avg[r["b"]] -= r["margin_a"]
    order = sorted(ts, key=lambda t: avg[t], reverse=True)

    upsets = 0
    for r in results.values():
        stronger = r["a"] if r["margin_a"] > 0 else r["b"]
        # an upset = the pairwise winner is ranked below the loser overall
        loser = r["b"] if stronger == r["a"] else r["a"]
        if order.index(stronger) > order.index(loser):
            upsets += 1

    summary = {
        "checkpoints": ts,
        "games_per_pair": 2 * args.games,
        "ranking_best_first": order,
        "avg_margin": {str(t): round(avg[t] / max(1, len(ts) - 1), 3)
                       for t in ts},
        "monotone_in_training_time": order == sorted(ts, reverse=True),
        "pairwise_upsets_vs_ranking": upsets,
        "pairs": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\nranking (best first): {order}")
    print(f"later-is-better throughout: {summary['monotone_in_training_time']}")
    print(f"pairwise upsets vs overall ranking: {upsets} / {len(results)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
