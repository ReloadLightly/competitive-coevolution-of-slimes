"""
train_ga_selfplay.py — exact replication of David Ha's self-play genetic
algorithm for Slime Volleyball, on the real slimevolleygym environment.

Original: https://github.com/hardmaru/slimevolleygym
          training_scripts/train_ga_selfplay.py (David Ha, 2020, Apache-2.0).
Ported with the smallest possible diff; every deviation is listed below.

The algorithm in plain language
-------------------------------
Genotype        : the flat weight vector (273 numbers) of a tiny fixed
                  feed-forward network, "slimevolleylite". Nothing else
                  evolves — no topology change, no crossover.
Population      : 128 random individuals.
One tournament  : two individuals are drawn at random and play ONE full game
                  (up to 3000 steps, 5 lives each side).
Selection       : the winner survives untouched; the loser is overwritten by
                  a copy of the winner plus Gaussian noise (sigma=0.1) — the
                  mutation operator. On a tie, the first-drawn individual is
                  nudged with noise instead.
Fitness         : implicit and purely relative — win games, stay in the pool.
                  The 2015 baseline policy is NEVER seen during training; it
                  serves only as an external yardstick afterwards
                  (see eval_vs_baseline.py).

Deliberate deviations from the original (each one is a February lesson —
see docs/postmortem-february.md):
1. `SlimeVolleyEnv()` is constructed directly — never `gym.make(...)`.
   Under modern gym, `gym.make` wraps the env; the wrapper breaks two-player
   `step()` calls and silently swallows attribute assignments such as
   `env.policy = ...`. That silent swallow is precisely what reduced
   February's "coevolution" to no coevolution at all.
2. Pinned dependencies (requirements.txt) instead of a floating stack.
3. Crash-safe bookkeeping: champion JSON every --save-freq tournaments
   (Ha's exact format), a full-population .npz snapshot every
   --snapshot-freq, and an append-only history.jsonl for learning curves.
"""

import argparse
import json
import os
import time

import numpy as np

from slimevolleygym import SlimeVolleyEnv
from slimevolleygym import multiagent_rollout as rollout
from slimevolleygym.mlp import Model, games


def main():
    ap = argparse.ArgumentParser(
        description="Exact-Ha GA self-play on SlimeVolley")
    ap.add_argument("--seed", type=int, default=612,
                    help="random seed (612 = Ha's)")
    ap.add_argument("--tournaments", type=int, default=500_000,
                    help="total games to play (Ha: 500k)")
    ap.add_argument("--population", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="mutation noise scale")
    ap.add_argument("--outdir", default="results/ga_selfplay")
    ap.add_argument("--save-freq", type=int, default=1000,
                    help="save champion JSON every N tournaments")
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--snapshot-freq", type=int, default=25_000,
                    help="full population snapshot every N tournaments")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <outdir>/snapshot.npz (RNG reseeds)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    policy_left = Model(games["slimevolleylite"])
    policy_right = Model(games["slimevolleylite"])
    param_count = policy_left.param_count
    print(f"param_count={param_count} population={args.population} "
          f"tournaments={args.tournaments} seed={args.seed}", flush=True)

    # February lesson #1: direct construction, no gym.make wrapper.
    env = SlimeVolleyEnv()

    start = 1
    snap_path = os.path.join(args.outdir, "snapshot.npz")
    if args.resume and os.path.exists(snap_path):
        snap = np.load(snap_path)
        population = snap["population"]
        winning_streak = list(snap["winning_streak"])
        start = int(snap["tournament"]) + 1
        # RNG state is not preserved across restarts: reseed deterministically
        # from (seed, resume point). Every restart is visible in history.jsonl
        # (elapsed_sec resets) and gets reported in the writeup.
        env.seed(args.seed + start)
        np.random.seed(args.seed + start)
        print(f"resumed from snapshot at tournament {start - 1}", flush=True)
    else:
        env.seed(args.seed)
        np.random.seed(args.seed)
        population = np.random.normal(
            size=(args.population, param_count)) * 0.5
        winning_streak = [0] * args.population  # proxy for quality (Ha's trick)

    history = []
    t0 = time.time()
    for tournament in range(start, args.tournaments + 1):
        m, n = np.random.choice(args.population, 2, replace=False)

        policy_left.set_model_params(population[m])
        policy_right.set_model_params(population[n])

        # the match between the m-th (left) and n-th (right) individual
        score, length = rollout(env, policy_right, policy_left)
        history.append(length)

        if score == 0:      # tie: nudge the left individual
            population[m] += np.random.normal(size=param_count) * args.sigma
        elif score > 0:     # right (n) won: overwrite loser m with mutant of n
            population[m] = population[n] + \
                np.random.normal(size=param_count) * args.sigma
            winning_streak[m] = winning_streak[n]
            winning_streak[n] += 1
        else:               # left (m) won: overwrite loser n with mutant of m
            population[n] = population[m] + \
                np.random.normal(size=param_count) * args.sigma
            winning_streak[n] = winning_streak[m]
            winning_streak[m] += 1

        if tournament % args.save_freq == 0:
            rh = int(np.argmax(winning_streak))
            fname = os.path.join(args.outdir, f"ga_{tournament:08d}.json")
            with open(fname, "wt") as f:
                json.dump([population[rh].tolist(), winning_streak[rh]], f,
                          sort_keys=True, indent=0, separators=(",", ": "))

        if tournament % args.snapshot_freq == 0:
            np.savez_compressed(
                os.path.join(args.outdir, "snapshot.npz"),
                population=population,
                winning_streak=np.array(winning_streak),
                tournament=tournament)

        if tournament % args.log_freq == 0:
            rh = int(np.argmax(winning_streak))
            rec = {"tournament": tournament,
                   "best_winning_streak": int(winning_streak[rh]),
                   "mean_duration": float(np.mean(history)),
                   "stdev": float(np.std(history)),
                   "elapsed_sec": round(time.time() - t0, 1)}
            with open(os.path.join(args.outdir, "history.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"tournament: {tournament} "
                  f"best_streak: {rec['best_winning_streak']} "
                  f"mean_dur: {rec['mean_duration']:.0f} "
                  f"elapsed: {rec['elapsed_sec']:.0f}s", flush=True)
            history = []

    print(f"done in {(time.time() - t0) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
