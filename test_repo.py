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


def coev_triads(margin):
    """Cyclic triads of an exact payoff matrix (no deadband)."""
    from coevolution_analysis import triad_stats
    return triad_stats(margin, deadband=0.0)


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

        # --- hof-eval: an archive loss must not credit a bystander ------
        # When an archived champion beats population member m, m is replaced
        # by a mutant of a living peer n and inherits n's streak. n did not
        # play, so no other individual's streak may change. The streak is not
        # cosmetic: argmax(streak) decides what enters the archive.
        #
        # The kernel's own source (py_func) is driven with a stubbed game in
        # which the right-hand side always wins. Game 1 is an ordinary game
        # (the archive is empty) and leaves one streak at 1; every later game
        # is an archive loss, which may only copy existing streak values. The
        # best streak in the pool must therefore never exceed 1.
        import algorithms as alg
        real_play = alg.play_game
        try:
            alg.play_game = lambda *args: (1, 100, 0)
            _, st, _, _ = alg.run_ga_hof_eval.py_func(
                7, 200, 4, 0.1, 1, 1.0, 1, 512, w, b, 0.5)
        finally:
            alg.play_game = real_play
        check("hof-eval: an archive loss leaves every other streak unchanged",
              int(st.max()) <= 1,
              f"best streak after 199 archive losses: {int(st.max())}")

        # The compiled kernel cannot be compared with py_func bit for bit:
        # play_game draws its serves from numba's RNG, which py_func does not
        # seed. So the compiled version is only checked to run and to play
        # real archive games.
        ch, st, _, aw = alg.run_ga_hof_eval(3, 2000, 16, 0.1, 500, 0.5, 100,
                                            32, w, b, 0.5)
        check("hof-eval: compiled kernel runs and plays archive games",
              ch.shape == (4, 273) and 0.0 < aw[-1] < 1.0,
              f"{ch.shape[0]} checkpoints, archive win rates "
              f"{np.round(aw, 2).tolist()}")
        # --- the lab wraps the study, it does not rewrite it ---------------
        # lab.kernels.run on Slime Volleyball must reproduce the paper's three
        # GA kernels and the population snapshots bit for bit.
        from lab import kernels as LK, games as LG
        g, gp, X1, X2, X3 = LG.make("slime")
        same = []
        ref = fv.run_ga(7, 2000, 16, 0.1, 500, 0.0, 100, 8, w, b, 0.5)
        lab = LK.run(g, gp, X1, X2, X3, 7, 2000, 16, 0.1, 500, LK.HOF_NONE,
                     0.0, 100, 8, w, b, 0.5, 0)
        same.append(all(np.array_equal(r, l) for r, l in zip(ref, lab[:5])))
        ref = fv.run_ga(7, 2000, 16, 0.1, 500, 0.25, 100, 8, w, b, 0.5)
        lab = LK.run(g, gp, X1, X2, X3, 7, 2000, 16, 0.1, 500, LK.HOF_PARENT,
                     0.25, 100, 8, w, b, 0.5, 0)
        same.append(all(np.array_equal(r, l) for r, l in zip(ref, lab[:5])))
        ref = alg.run_ga_hof_eval(7, 2000, 16, 0.1, 500, 0.25, 100, 8, w, b, 0.5)
        lab = LK.run(g, gp, X1, X2, X3, 7, 2000, 16, 0.1, 500, LK.HOF_TEST,
                     0.25, 100, 8, w, b, 0.5, 0)
        same.append(all(np.array_equal(r, lab[i]) for r, i in zip(ref, (0, 1, 2, 4))))
        ref = fv.run_ga_with_pops(7, 2000, 16, 0.1, 500, 0.0, 100, 8, w, b, 0.5, 1000)
        lab = LK.run(g, gp, X1, X2, X3, 7, 2000, 16, 0.1, 500, LK.HOF_NONE,
                     0.0, 100, 8, w, b, 0.5, 1000)
        same.append(all(np.array_equal(r, lab[i]) for r, i in zip(ref, (0, 1, 2, 5, 6))))
        check("lab GA reproduces the paper's kernels bit for bit", all(same),
              "control, archive as parent, archive as test, snapshots: "
              + ", ".join("same" if s else "DIFFERENT" for s in same))

        # the niche archive (WP9): the compiled descriptor binning agrees
        # with a plain reimplementation, a short run keeps its archive within
        # the grid, and the guards refuse a too-small archive or missing bounds
        rng = np.random.default_rng(5)
        probes = rng.normal(size=(16, 12))
        bounds = (-0.35, 0.95, -0.95, 0.35)
        out = np.empty(3)
        agree = True
        for _ in range(50):
            p = rng.normal(size=273) * 0.5
            d = np.zeros(2)
            for x in probes:
                fv.mlp_forward(p, x, out)
                d += out[:2]
            d /= len(probes)
            c0 = min(max(int(np.floor((d[0] - bounds[0]) / (bounds[1] - bounds[0]) * 8)), 0), 7)
            c1 = min(max(int(np.floor((d[1] - bounds[2]) / (bounds[3] - bounds[2]) * 8)), 0), 7)
            agree &= LK.niche_cell(p, probes, 8, *bounds) == c0 * 8 + c1
        gpn = np.concatenate([gp, np.array(bounds)])
        res = LK.run(g, gpn, X1, X2, X3, 9, 3000, 16, 0.1, 500, LK.HOF_NICHE,
                     0.25, 100, 64, w, b, 0.5, 0)
        arch = res[7]
        refused = 0
        for cap, gpx in ((63, gpn), (64, gp)):
            try:
                LK.run(g, gpx, X1, X2, X3, 9, 1000, 8, 0.1, 500, LK.HOF_NICHE,
                       0.25, 100, cap, w, b, 0.5, 0)
            except ValueError:
                refused += 1
        check("niche archive: descriptor cells, archive size and guards",
              agree and (np.diff(arch) >= 0).all() and 1 <= arch[-1] <= 64
              and refused == 2,
              f"cells agree on 50 genomes: {agree}; archive {arch.tolist()} cells; "
              f"bad settings refused: {refused}/2")

        # discmix: the expected margin is antisymmetric, and with lambda = 0
        # (transitive part only) no triad can be cyclic
        pool = LG.features(np.random.default_rng(3).normal(size=(24, 273)) * 0.5)
        m0, m1 = LG.payoff_matrix(pool, 0.0), LG.payoff_matrix(pool, 1.0)
        t0 = coev_triads(m0)
        t1 = coev_triads(m1)
        check("discmix: antisymmetric, transitive at lambda 0, cyclic at 1",
              np.allclose(m0, -m0.T) and np.allclose(m1, -m1.T)
              and t0["cyclic"] == 0 and t1["cyclic"] > 0,
              f"cyclic triads {t0['cyclic']}/{t0['triads_decided']} at lambda 0, "
              f"{t1['cyclic']}/{t1['triads_decided']} at lambda 1")
    except ImportError as e:
        check("compiled environment available", False,
              f"{e} — install requirements-fast.txt")

    # --- every table follows from the raw files, and notices if one goes -
    # make_tables.py --check refuses tables whose analysis files fail
    # provenance verification. Here: the tree as committed must verify, and
    # hiding any single run file (simulated, nothing on disk is touched) must
    # make verification fail.
    import provenance as pv
    clean = pv.verify()
    check("analysis files follow from the raw files on disk", not clean,
          f"{len(pv._load())} outputs verified" if not clean
          else f"{len(clean)} problems, first: {clean[0]}")
    if not clean:
        import glob as glob_mod
        real_glob, real_exists, real_sha = glob_mod.glob, os.path.exists, pv.sha256
        cache = {}
        pv.sha256 = lambda p: cache.setdefault(p, real_sha(p))
        runs = sorted(real_glob("results/matrix/*_s*.npz"))
        caught = []
        try:
            for hidden in runs:
                glob_mod.glob = lambda pat, **k: [p for p in real_glob(pat, **k)
                                                  if p != hidden]
                os.path.exists = lambda p: p != hidden and real_exists(p)
                caught.append(bool(pv.verify()))
        finally:
            glob_mod.glob, os.path.exists, pv.sha256 = real_glob, real_exists, real_sha
        missed = [r for r, c in zip(runs, caught) if not c]
        check("removing any single run file fails the consistency check",
              runs and not missed,
              f"{len(runs)} files, each caught" if not missed
              else f"not caught: {missed[:3]}")

    print("=" * 62)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
