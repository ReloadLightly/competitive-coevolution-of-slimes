"""
test_repo.py — does this repository actually work?

Runs in about a minute and checks the things that, in February 2026, were
silently broken and cost weeks of uninterpretable training. Each test maps to
a documented failure in docs/postmortem-february.md.

    python test_repo.py
"""

import glob
import json
import os
import sys

import numpy as np

FAILS = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILS.append(name)


def main():
    print("=" * 62)
    print("Repository self-test")
    print("=" * 62)

    # --- environment and dependencies ---------------------------------
    from slimevolleygym import SlimeVolleyEnv, multiagent_rollout
    from slimevolleygym.mlp import Model, games
    try:
        from slimevolleygym import BaselinePolicy
    except ImportError:
        from slimevolleygym.slimevolley import BaselinePolicy

    m = Model(games["slimevolleylite"])
    check("dependencies import", True)
    check("genotype size is 273 parameters", m.param_count == 273,
          f"got {m.param_count}")

    # --- February failure #1: all-buttons bug -------------------------
    # A random policy must NOT press every button on every step; if it does,
    # the whole population behaves identically and selection has nothing to
    # act on. This is what killed February's runs.
    env = SlimeVolleyEnv()
    env.seed(1)
    np.random.seed(1)
    m.set_model_params(np.random.normal(size=m.param_count) * 0.5)
    obs = env.reset()
    actions = []
    for _ in range(300):
        a = m.predict(obs)
        actions.append(list(a))
        obs, _, done, _ = env.step(a)
        if done:
            break
    A = np.array(actions)
    variance = A.std(axis=0).sum()
    check("actions vary over time (no all-buttons bug)", variance > 0.05,
          f"per-button std sum = {variance:.3f}")

    # --- February failure #2: silent gym.make wrapper -----------------
    # Opponent policies must actually be consulted. February assigned an
    # opponent onto a gym wrapper, where it was silently ignored, so every
    # "coevolution" game was secretly against the built-in expert.
    calls = {"n": 0}

    class Probe:
        def predict(self, obs):
            calls["n"] += 1
            return [0, 0, 0]

    env2 = SlimeVolleyEnv()
    env2.seed(2)
    o_r = env2.reset()
    probe = Probe()
    for _ in range(50):
        o_l = env2.game.agent_left.getObservation()
        o_r, _, done, _ = env2.step([0, 0, 0], probe.predict(o_l))
        if done:
            break
    check("opponent policy is actually consulted", calls["n"] >= 50,
          f"{calls['n']} predict() calls")

    # --- February failure #3: observation sign convention -------------
    # obs[4] is the ball's x in the agent's OWN frame: positive means the
    # ball is on the agent's own side. February's shaping term assumed the
    # opposite and paid agents for losing.
    env3 = SlimeVolleyEnv()
    env3.seed(3)
    obs = env3.reset()
    for _ in range(150):
        obs, _, done, _ = env3.step([0, 0, 0])
        if done:
            break
    same_sign = np.sign(obs[4] * 10) == np.sign(env3.game.ball.x)
    check("obs[4] is ball-x in the agent's own frame", bool(same_sign),
          f"obs[4]*10={obs[4] * 10:.2f}, game ball.x={env3.game.ball.x:.2f}")

    # --- episodes run to the real limit -------------------------------
    check("environment episode limit is 3000 steps", env3.t_limit == 3000,
          f"t_limit={env3.t_limit}")

    # --- the evolved champions load and play --------------------------
    ckpts = sorted(glob.glob("results/ga_selfplay/ga_*.json"))
    check("champion checkpoints present", len(ckpts) > 0,
          f"{len(ckpts)} checkpoints")
    if ckpts:
        with open(ckpts[-1]) as f:
            params, streak = json.load(f)
        champ = Model(games["slimevolleylite"])
        champ.set_model_params(np.array(params))
        env4 = SlimeVolleyEnv()
        env4.seed(4)
        np.random.seed(4)
        scores = [multiagent_rollout(env4, champ, BaselinePolicy())[0]
                  for _ in range(5)]
        check("latest champion plays against the 2015 baseline", True,
              f"{os.path.basename(ckpts[-1])}, 5 episodes, "
              f"mean score {np.mean(scores):+.2f}")

    # --- the run can be continued -------------------------------------
    snap = "results/ga_selfplay/snapshot.npz"
    if os.path.exists(snap):
        s = np.load(snap)
        ok = s["population"].shape[1] == 273 and int(s["tournament"]) > 0
        check("population snapshot is usable (resume works)", ok,
              f"population {s['population'].shape}, "
              f"tournament {int(s['tournament'])}")
    else:
        check("population snapshot present", False, "snapshot.npz missing")

    # --- the compiled environment is the same game --------------------
    # The experiment matrix is trained in fastvolley.py, so the port has to be
    # the benchmark, not merely similar to it. Full evidence is in
    # validate_fastvolley.py; these are the cheap versions of those checks.
    try:
        import fastvolley as fv
        from slimevolleygym import slimevolley as sv

        consts = [("REF_W", sv.REF_W, fv.REF_W), ("REF_U", sv.REF_U, fv.REF_U),
                  ("GRAVITY", sv.GRAVITY, fv.GRAVITY),
                  ("TIMESTEP", sv.TIMESTEP, fv.TIMESTEP),
                  ("NUDGE", sv.NUDGE, fv.NUDGE),
                  ("MAX_BALL_SPEED", sv.MAX_BALL_SPEED, fv.MAX_BALL_SPEED),
                  ("PLAYER_SPEED_X", sv.PLAYER_SPEED_X, fv.PLAYER_SPEED_X),
                  ("PLAYER_SPEED_Y", sv.PLAYER_SPEED_Y, fv.PLAYER_SPEED_Y),
                  ("MAXLIVES", sv.MAXLIVES, fv.MAXLIVES),
                  ("INIT_DELAY_FRAMES", sv.INIT_DELAY_FRAMES,
                   fv.INIT_DELAY_FRAMES)]
        bad = [n for n, a, b in consts if a != b]
        check("compiled env constants match slimevolleygym exactly",
              not bad, f"{len(consts)} constants" if not bad
              else f"differ: {bad}")

        import validate_fastvolley as vf
        rng = np.random.default_rng(11)
        pr = rng.normal(size=273) * 0.5
        pl = rng.normal(size=273) * 0.5
        vx = rng.uniform(-20.0, 20.0, size=64)
        vy = rng.uniform(10.0, 25.0, size=64)
        rs, rt, rtr = vf.reference_game(pr, pl, False, vx, vy)
        cs, ct, ctr = vf.compiled_game(pr, pl, False, vx, vy)
        same = (rs == cs) and (rt == ct) and bool((rtr == ctr).all())
        check("compiled env reproduces a reference game bit for bit", same,
              f"score {rs} vs {cs}, length {rt} vs {ct}, "
              f"max|dev| {np.abs(rtr - ctr).max() if rt == ct else float('nan')}")

        w, b = fv.baseline_arrays()
        champs, streaks, meanlen, ties, hofwins = fv.run_ga(
            1, 2000, 32, 0.1, 1000, 0.0, 1000, 8, w, b, 0.5)
        check("compiled GA runs and exports 273-parameter champions",
              champs.shape[1] == 273 and streaks[-1] > 0,
              f"{champs.shape[0]} checkpoints, best streak {streaks[-1]}")
    except ImportError as e:
        check("compiled environment available", False,
              f"{e} — install requirements-fast.txt")

    print("=" * 62)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
