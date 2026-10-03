"""
yardsticks.py — every final champion against stronger, independent opponents.

The study's only external yardstick is the 2015 baseline: a 112-parameter
recurrent policy (7 x 15 weights and 7 biases; slimevolleygym's docstring says
120) that never sees its opponent. "Above parity" against it is a
low bar, and a population can specialise against it without being good. This
script adds the two trained feed-forward policies Ha published with
slimevolleygym (WP7):

  zoo-ga    zoo/ga_sp/ga.json, a 12-10-10-3 tanh network (273 parameters, the
            same policy class as every agent in this study), trained by Ha's
            self-play GA (training_scripts/train_ga_selfplay.py).
  zoo-cma   zoo/cmaes/slimevolley.cma.64.96.best.json, a 12-20-20-3 tanh
            network (743 parameters), trained by CMA-ES against the 2015
            baseline (estool).

Both files are copied verbatim into results/zoo/ from slimevolleygym at the
commit requirements.txt pins (8ac22434), Apache-2.0; results/zoo/SOURCE.md
records their hashes.

Games are played on the validated compiled physics through
asymmetric.play_game_asym, which runs 12-h-h-3 networks of any hidden size.
`--validate` first checks that path bit for bit against the reference
slimevolleygym environment for zoo policies of both sizes, the way
validate_fastvolley.py checks the 273-parameter path.

    python yardsticks.py --validate      # bit-level check, then score
    python yardsticks.py                 # score only
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os

import numpy as np
from numba import njit

import asymmetric as az
import fastvolley as fv
import provenance as pv
from fastvolley import (BVX, BVY, BX, BY, LIFE_L, LIFE_R, LX, LY, OBS_SIZE,
                        RX, STATE_SIZE, T_LIMIT, _game_step, _reset_game)

ZOO = "results/zoo"
OUT = "results/analysis/yardsticks.json"
OPPONENTS = {
    # name: (file, hidden size)
    "zoo-ga": ("ga.json", 10),
    "zoo-cma": ("slimevolley.cma.64.96.best.json", 20),
}
GAMES = 100             # per side, so 200 games per champion and opponent
SEED = 20261004         # disjoint from training, sweep, selection and ranking seeds
BASELINE_EPISODES = 1_000


def load_zoo(name):
    fname, h = OPPONENTS[name]
    params, _ = json.load(open(os.path.join(ZOO, fname)))
    p = np.array(params, dtype=np.float64)
    assert len(p) == az.param_count(h), (name, len(p))
    return p, h


@njit(cache=True)
def match(p_a, h_a, p_b, h_b, games, seed):
    """`games` games with a on the right and `games` with a on the left.

    Returns the per-game scores from a's point of view (2 * games values)."""
    np.random.seed(seed)
    out = np.empty(2 * games, dtype=np.int64)
    for g in range(games):
        sc, _ = az.play_game_asym(p_a, h_a, p_b, h_b)
        out[2 * g] = sc
        sc, _ = az.play_game_asym(p_b, h_b, p_a, h_a)
        out[2 * g + 1] = -sc
    return out


@njit(cache=True)
def trace_game(p_r, h_r, p_l, h_l, bvx_buf, bvy_buf, trace):
    """play_game_asym with the serves taken from buffers, recording the state.

    Same loop as asymmetric.play_game_asym; serves come from the buffers as in
    fastvolley.play_game_trace, so the reference environment can be driven
    with the identical sequence."""
    s = np.zeros(STATE_SIZE)
    obs_l = np.zeros(OBS_SIZE)
    obs_r = np.zeros(OBS_SIZE)
    a_l = np.zeros(3)
    a_r = np.zeros(3)
    b1r = np.zeros(az.MAX_H)
    b2r = np.zeros(az.MAX_H)
    b1l = np.zeros(az.MAX_H)
    b2l = np.zeros(az.MAX_H)
    k = 0
    _reset_game(s, obs_l, obs_r, bvx_buf[0], bvy_buf[0])
    k += 1
    obs_l_view = obs_r.copy()
    total = 0
    t = 0
    while True:
        az.mlp_forward_var(p_r, obs_r, a_r, h_r, b1r, b2r)
        az.mlp_forward_var(p_l, obs_l_view, a_l, h_l, b1l, b2l)
        result = _game_step(s, obs_l, obs_r, a_l, a_r, bvx_buf[k], bvy_buf[k])
        if result != 0:
            k += 1
        for i in range(OBS_SIZE):
            obs_l_view[i] = obs_l[i]
        trace[t, 0] = s[BX]
        trace[t, 1] = s[BY]
        trace[t, 2] = s[BVX]
        trace[t, 3] = s[BVY]
        trace[t, 4] = s[LX]
        trace[t, 5] = s[LY]
        trace[t, 6] = s[RX]
        total += result
        t += 1
        if t >= T_LIMIT:
            break
        if s[LIFE_L] <= 0.0 or s[LIFE_R] <= 0.0:
            break
    return total, t


def reference_game(p_r, h_r, p_l, h_l, vx, vy):
    """The same game in the reference environment, with the same serves."""
    from slimevolleygym import SlimeVolleyEnv
    from slimevolleygym.mlp import Model, games as mlp_games
    from validate_fastvolley import ServeStub
    kind = {10: "slimevolleylite", 20: "slimevolley"}
    env = SlimeVolleyEnv()
    env.game.np_random = ServeStub(vx, vy)
    pr = Model(mlp_games[kind[h_r]])
    pr.set_model_params(np.array(p_r))
    pl = Model(mlp_games[kind[h_l]])
    pl.set_model_params(np.array(p_l))
    obs_r = env.reset()
    obs_l = obs_r
    done, total, t = False, 0, 0
    trace = np.zeros((T_LIMIT, 7))
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


def validate(champion, games=10, seed=20261003):
    """Compiled vs reference, game by game, step by step."""
    rng = np.random.default_rng(seed)
    ga, cma = load_zoo("zoo-ga"), load_zoo("zoo-cma")
    pairings = [("zoo-cma vs zoo-ga", cma, ga), ("zoo-ga vs zoo-cma", ga, cma),
                ("champion vs zoo-cma", (champion, 10), cma),
                ("zoo-cma vs champion", cma, (champion, 10))]
    report, ok = [], True
    for name, (pr, hr), (pl, hl) in pairings:
        same, steps = 0, 0
        for _ in range(games):
            vx = rng.uniform(-20.0, 20.0, size=64)
            vy = rng.uniform(10.0, 25.0, size=64)
            rs, rt, rtr = reference_game(pr, hr, pl, hl, vx, vy)
            tr = np.zeros((T_LIMIT, 7))
            cs, ct = trace_game(pr, hr, pl, hl, vx, vy, tr)
            match_ok = rs == cs and rt == ct and bool((rtr == tr[:ct]).all())
            same += match_ok
            steps += rt
        ok &= same == games
        report.append({"pairing": name, "games": games, "identical": same,
                       "steps": steps})
        print(f"  {name}: {same}/{games} games identical, {steps} steps",
              flush=True)
    return ok, report


def _score(job):
    name, champ = job
    out = {}
    for opp in OPPONENTS:
        p, h = load_zoo(opp)
        sc = match(np.ascontiguousarray(champ), 10, p, h, GAMES, SEED)
        out[opp] = {"mean": float(sc.mean()), "sem": float(sc.std() / np.sqrt(len(sc))),
                    "win": float((sc > 0).mean()), "tie": float((sc == 0).mean()),
                    "loss": float((sc < 0).mean())}
    return name, out


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    paths = pv.SCOPES["single"]()
    finals = {}
    for p in paths:
        z = np.load(p)
        finals[os.path.basename(p)[:-4]] = z["champs"][-1].astype(np.float64)

    res = {"games_per_side": GAMES, "seed": SEED,
           "zoo": {n: {"file": f, "hidden": h,
                       "sha256": sha256(os.path.join(ZOO, f))}
                   for n, (f, h) in OPPONENTS.items()}}

    if args.validate:
        print("bit-level check against slimevolleygym:", flush=True)
        ok, report = validate(finals[sorted(finals)[0]])
        res["validation"] = report
        if not ok:
            raise SystemExit("compiled zoo games differ from the reference")

    # scale: each zoo policy against the 2015 baseline and against each other
    w, b = fv.baseline_arrays()
    res["zoo_vs_baseline"] = {}
    for n in OPPONENTS:
        p, h = load_zoo(n)
        sc, _ = az.eval_var_vs_baseline(p, h, BASELINE_EPISODES, SEED, w, b)
        res["zoo_vs_baseline"][n] = {"mean": float(sc.mean()),
                                     "win": float((sc > 0).mean()),
                                     "loss": float((sc < 0).mean())}
    (pg, hg), (pc, hc) = load_zoo("zoo-ga"), load_zoo("zoo-cma")
    sc = match(pg, hg, pc, hc, GAMES, SEED)
    res["zoo_ga_vs_zoo_cma"] = {"mean": float(sc.mean()),
                                "win": float((sc > 0).mean()),
                                "loss": float((sc < 0).mean())}
    print(f"zoo vs baseline: {res['zoo_vs_baseline']}; "
          f"zoo-ga vs zoo-cma {res['zoo_ga_vs_zoo_cma']['mean']:+.2f}", flush=True)

    res["per_run"] = {}
    jobs = sorted(finals.items())
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, out in pool.imap_unordered(_score, jobs):
            res["per_run"][name] = out
            print(f"  {name}: zoo-ga {out['zoo-ga']['mean']:+.2f}  "
                  f"zoo-cma {out['zoo-cma']['mean']:+.2f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    pv.record(OUT, "yardsticks", params={"games_per_side": GAMES, "seed": SEED,
                                         "baseline_episodes": BASELINE_EPISODES})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
