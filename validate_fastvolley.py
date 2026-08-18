"""
validate_fastvolley.py — is the compiled environment the same game?

A fast reimplementation of a benchmark is worthless unless it is the benchmark.
This script drives `slimevolleygym` and `fastvolley` from an identical stream
of serve velocities and compares them step by step:

  * per-step ball position/velocity and both agents' positions,
  * the point at which every rally ends,
  * the final score and episode length.

Three populations of policies are checked, because they exercise different
parts of the state space: freshly initialised random genomes (short rallies,
many scoring frames), mid-training champions, and the final champion against
the 2015 baseline RNN (3000-step rallies, thousands of collisions).

Also reports the throughput ratio, which is the reason the port exists.

    python validate_fastvolley.py --games 100
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import fastvolley as fv


class ServeStub:
    """Stands in for gym's np_random inside Game, serving a fixed sequence.

    Game draws exactly two uniforms per serve: vx on [-20, 20], then vy on
    [10, 25]. Feeding both implementations the same sequence isolates the
    comparison from the two different RNG families.
    """

    def __init__(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.i = 0

    def uniform(self, low=0.0, high=1.0):
        j = self.i // 2
        if self.i % 2 == 0:
            v = self.vx[j]
        else:
            v = self.vy[j]
        self.i += 1
        return v


def reference_game(params_r, params_l, left_is_baseline, vx, vy):
    """multiagent_rollout() against the real environment, with a trace."""
    from slimevolleygym import SlimeVolleyEnv
    from slimevolleygym.mlp import Model, games as mlp_games
    try:
        from slimevolleygym import BaselinePolicy
    except ImportError:
        from slimevolleygym.slimevolley import BaselinePolicy

    env = SlimeVolleyEnv()
    env.game.np_random = ServeStub(vx, vy)

    pr = Model(mlp_games["slimevolleylite"])
    pr.set_model_params(np.array(params_r))
    if left_is_baseline:
        pl = BaselinePolicy()
    else:
        pl = Model(mlp_games["slimevolleylite"])
        pl.set_model_params(np.array(params_l))

    obs_r = env.reset()
    obs_l = obs_r
    done = False
    total = 0
    t = 0
    trace = np.zeros((fv.T_LIMIT, 7))
    while not done:
        a_r = pr.predict(obs_r)
        a_l = pl.predict(obs_l)
        obs_r, reward, done, info = env.step(a_r, a_l)
        obs_l = info["otherObs"]
        g = env.game
        trace[t] = (g.ball.x, g.ball.y, g.ball.vx, g.ball.vy,
                    g.agent_left.x, g.agent_left.y, g.agent_right.x)
        total += reward
        t += 1
    return total, t, trace[:t]


def compiled_game(params_r, params_l, left_is_baseline, vx, vy):
    w, b = fv.baseline_arrays()
    rnn_r = np.zeros(7)
    rnn_l = np.zeros(7)
    trace = np.zeros((fv.T_LIMIT, 7))
    tl = fv.POLICY_BASELINE if left_is_baseline else fv.POLICY_MLP
    total, t = fv.play_game_trace(
        np.ascontiguousarray(params_r), fv.POLICY_MLP,
        np.ascontiguousarray(params_l), tl,
        w, b, rnn_r, rnn_l, vx, vy, trace)
    return total, t, trace[:t]


def load_champion(path):
    with open(path) as f:
        params, _ = json.load(f)
    return np.array(params, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60,
                    help="games per scenario")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--ckpt-dir", default="results/ga_selfplay")
    ap.add_argument("--out", default="results/validation.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ga_*.json")))
    early = load_champion(ckpts[len(ckpts) // 8]) if ckpts else None
    late = load_champion(ckpts[-1]) if ckpts else None

    scenarios = []
    # 1. two freshly initialised genomes: short rallies, many serves
    for _ in range(args.games):
        scenarios.append(("random vs random",
                          rng.normal(size=fv.PARAM_COUNT) * 0.5,
                          rng.normal(size=fv.PARAM_COUNT) * 0.5,
                          False))
    # 2. a trained champion against a fresh genome
    if late is not None:
        for _ in range(args.games):
            scenarios.append(("champion vs random", late,
                              rng.normal(size=fv.PARAM_COUNT) * 0.5, False))
    # 3. champion vs earlier champion: long, competitive rallies
    if late is not None and early is not None:
        for _ in range(args.games):
            scenarios.append(("champion vs champion", late, early, False))
    # 4. champion vs the 2015 baseline RNN: the evaluation protocol itself
    if late is not None:
        for _ in range(args.games):
            scenarios.append(("champion vs 2015 baseline", late,
                              np.zeros(fv.PARAM_COUNT), True))

    stats = {}
    print(f"{len(scenarios)} paired games\n")
    for name, pr, pl, is_base in scenarios:
        n_serves = 64
        vx = rng.uniform(-20.0, 20.0, size=n_serves)
        vy = rng.uniform(10.0, 25.0, size=n_serves)

        rs, rt, rtr = reference_game(pr, pl, is_base, vx, vy)
        cs, ct, ctr = compiled_game(pr, pl, is_base, vx, vy)

        d = stats.setdefault(name, {"n": 0, "score_match": 0, "len_match": 0,
                                    "trace_exact": 0, "max_abs_dev": 0.0,
                                    "steps": 0, "first_div_step": []})
        d["n"] += 1
        d["steps"] += rt
        d["score_match"] += int(rs == cs)
        d["len_match"] += int(rt == ct)
        if rt == ct:
            dev = np.abs(rtr - ctr)
            d["max_abs_dev"] = max(d["max_abs_dev"], float(dev.max()))
            exact = bool((dev == 0).all())
            d["trace_exact"] += int(exact)
            if not exact:
                d["first_div_step"].append(int(np.argmax(dev.max(axis=1) > 0)))

    print(f"{'scenario':<26} {'games':>6} {'score':>7} {'length':>7} "
          f"{'trace':>7} {'max|dev|':>10} {'steps':>9}")
    ok = True
    for name, d in stats.items():
        print(f"{name:<26} {d['n']:>6} {d['score_match']:>7} {d['len_match']:>7} "
              f"{d['trace_exact']:>7} {d['max_abs_dev']:>10.3g} {d['steps']:>9}")
        ok = ok and d["score_match"] == d["n"] and d["trace_exact"] == d["n"]

    # ---- throughput ------------------------------------------------------
    print("\nthroughput (champion vs 2015 baseline, self-play-length games)")
    w, b = fv.baseline_arrays()
    probe = late if late is not None else rng.normal(size=fv.PARAM_COUNT) * 0.5
    fv.eval_vs_baseline(probe, 2, 1, w, b, False)      # warm up / compile

    t0 = time.time()
    n_ref = 6
    for i in range(n_ref):
        vxr = rng.uniform(-20.0, 20.0, size=64)
        vyr = rng.uniform(10.0, 25.0, size=64)
        reference_game(probe, np.zeros(fv.PARAM_COUNT), True, vxr, vyr)
    ref_rate = n_ref / (time.time() - t0)

    t0 = time.time()
    n_fast = 400
    fv.eval_vs_baseline(probe, n_fast, 7, w, b, False)
    fast_rate = n_fast / (time.time() - t0)

    print(f"  reference : {ref_rate:8.2f} games/s")
    print(f"  compiled  : {fast_rate:8.2f} games/s   ({fast_rate / ref_rate:.0f}x)")

    summary = {"scenarios": stats,
               "reference_games_per_sec": ref_rate,
               "compiled_games_per_sec": fast_rate,
               "speedup": fast_rate / ref_rate,
               "all_exact": ok}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{'PASS' if ok else 'FAIL'} -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
