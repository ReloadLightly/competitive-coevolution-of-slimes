"""
eval_vs_baseline.py — the external yardstick.

Plays an evolved champion against the 2015 baseline policy (the tiny
120-parameter recurrent network from https://otoro.net/slimevolley/, shipped
inside slimevolleygym) and reports the mean per-episode score.

Score = points won minus points lost per episode, range -5 .. +5.
The baseline is NEVER used during training — it exists purely to measure.

Reference number to beat (Ha, TRAINING.md): the GA self-play champion after
500k tournaments scored 0.353 +/- 0.728 over 1000 episodes.

Usage:
  # single checkpoint, full evaluation
  python eval_vs_baseline.py results/ga_selfplay/ga_00500000.json --episodes 1000

  # sweep every checkpoint (learning-curve data -> eval_curve.jsonl)
  python eval_vs_baseline.py --all results/ga_selfplay --episodes 100
"""

import argparse
import glob
import json
import os

import numpy as np

from slimevolleygym import SlimeVolleyEnv
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym.mlp import Model, games

try:
    from slimevolleygym import BaselinePolicy
except ImportError:  # depending on package __init__ exports
    from slimevolleygym.slimevolley import BaselinePolicy


def load_champion(path):
    with open(path) as f:
        params, streak = json.load(f)
    m = Model(games["slimevolleylite"])
    m.set_model_params(np.array(params))
    return m, streak


def evaluate(champion_path, episodes, seed):
    # February lesson #1 again: direct env construction, no gym.make.
    env = SlimeVolleyEnv()
    env.seed(seed)
    np.random.seed(seed)
    champ, streak = load_champion(champion_path)
    base = BaselinePolicy()
    scores, lengths = [], []
    for _ in range(episodes):
        score, length = rollout(env, champ, base)  # champion plays right
        scores.append(score)
        lengths.append(length)
    return np.array(scores), np.array(lengths), streak


def main():
    ap = argparse.ArgumentParser(description="Evaluate champion vs 2015 baseline")
    ap.add_argument("target", help="champion .json OR (with --all) a directory")
    ap.add_argument("--all", action="store_true",
                    help="sweep every ga_*.json in the directory")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=721)
    args = ap.parse_args()

    if not args.all:
        scores, lengths, streak = evaluate(args.target, args.episodes, args.seed)
        print(f"{os.path.basename(args.target)}  (winning_streak={streak})")
        print(f"episodes: {args.episodes}")
        print(f"score vs baseline: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"episode won/tied/lost: "
              f"{(scores > 0).sum()}/{(scores == 0).sum()}/{(scores < 0).sum()}")
        print(f"mean episode length: {lengths.mean():.0f} steps")
        print("reference (Ha, 500k GA self-play): 0.353 +/- 0.728")
        return

    files = sorted(glob.glob(os.path.join(args.target, "ga_*.json")))
    out = os.path.join(args.target, "eval_curve.jsonl")

    # Skip checkpoints already evaluated at this episode count, so the curve
    # can be extended incrementally as training continues (and so the curve
    # never silently lags behind the checkpoint directory).
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r.get("episodes") == args.episodes:
                done.add(r["tournament"])
    files = [p for p in files
             if int(os.path.basename(p)[3:-5]) not in done]

    print(f"sweeping {len(files)} new checkpoints "
          f"({len(done)} already done), {args.episodes} episodes each")
    with open(out, "a") as f:
        for path in files:
            scores, lengths, streak = evaluate(path, args.episodes, args.seed)
            tournament = int(os.path.basename(path)[3:-5])
            rec = {"tournament": tournament,
                   "mean_score": float(scores.mean()),
                   "std_score": float(scores.std()),
                   "win_rate": float((scores > 0).mean()),
                   "episodes": args.episodes,
                   "winning_streak": int(streak)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  t={tournament:>8d}  score={rec['mean_score']:+.3f} "
                  f"+/- {rec['std_score']:.3f}  win_rate={rec['win_rate']:.2f}",
                  flush=True)
    print(f"curve data -> {out}")


if __name__ == "__main__":
    main()
