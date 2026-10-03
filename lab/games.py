"""
lab/games.py — the games of the lab, as data for lab.kernels.run.

slime    Slime Volleyball, the paper's game, through fastvolley unchanged.

discmix  A game whose skill structure is set by one number, lambda.
         Each genome is the same 273-parameter 12-10-10-3 tanh network as in
         Slime Volleyball, mutated the same way. What a network "is" in this
         game is read off two fixed sets of probe inputs:

           skill  how well its third output classifies 256 skill probes by
                  the sign of a fixed random teacher network's (mean of
                  label * output, in [-1, 1]). A property of the genome
                  alone, so it orders any population transitively;
           u      its mean first two outputs on 16 style probes, a point in
                  [-1, 1]^2.

         A game between right (r) and left (l) is won by r when
         M + noise > tie, lost when M + noise < -tie, tied otherwise, with

           M = (1 - lambda) * alpha * (skill_r - skill_l)
             +      lambda  * beta  * (u_r x u_l)          (2-D cross product)

         lambda = 0 is purely transitive (better skill wins more often);
         lambda = 1 is Balduzzi et al.'s disc game, purely cyclic; values in
         between mix the two. Because M is known exactly, so is the expected
         score of any genome against any other: true skill and true
         intransitivity are measured without sampling noise.

The probe inputs, the teacher and the constants are fixed here, before any
discmix experiment exists, and recorded in every run file.
"""

import math

import numpy as np

from lab import kernels as K

PROBE_SEED = 20261003       # skill probes, style probes and the teacher
N_SKILL, N_STYLE = 256, 16
TEACHER_SCALE = 1.0         # a harder rule than a typical initial network

# Constants, fixed on 2026-10-03 before any discmix experiment:
#   ALPHA, BETA  give the two parts unit spread over random pairs of an
#                initial population (calibrate()), so lambda = 0.5 weighs
#                them equally at the start;
#   the task     (256 probes, teacher-sign labels, teacher scale 1.0) and
#   NOISE, TIE   were chosen in pilot runs at lambda = 0 only (seed 11), as
#                the first setting tried in which skill keeps rising over a
#                500,000-game run instead of stalling (regression to a
#                teacher) or saturating within 50,000 games (a linear rule).
#                No pilot looked at cycles, archives or the export rule.
ALPHA = 4.62               # calibrate(): 4.6245
BETA = 3.32                # calibrate(): 3.3207
NOISE = 0.3
TIE = 0.05


def probes():
    """The fixed skill probes, their labels, the style probes, the teacher.

    A label is the sign of a fixed random teacher network's third output, so
    the transitive task is a nonlinear classification a perfect member of the
    network class solves."""
    rng = np.random.default_rng(PROBE_SEED)
    skill_x = rng.normal(size=(N_SKILL, 12))
    style_x = rng.normal(size=(N_STYLE, 12))
    teacher = rng.normal(size=K.PARAM_COUNT) * TEACHER_SCALE
    skill_y = np.zeros((N_SKILL, 3))
    out = np.empty(3)
    for k in range(N_SKILL):
        K.mlp_forward(teacher, skill_x[k], out)
        skill_y[k, 0] = 1.0 if out[2] > 0 else -1.0
    return skill_x, skill_y, style_x, teacher


SLIME_PROBE_SEED, SLIME_PROBES, SLIME_PROBE_EVERY = 20261007, 256, 25


def slime_probes():
    """256 observations of real Slime Volleyball states.

    The 2015 baseline plays itself in the reference environment (seeded);
    the right player's observation is kept every 25 steps. Only the niche
    archive (lab.kernels.HOF_NICHE) uses them, as the inputs on which a
    network's behaviour descriptor is read."""
    from slimevolleygym import SlimeVolleyEnv
    try:
        from slimevolleygym import BaselinePolicy
    except ImportError:
        from slimevolleygym.slimevolley import BaselinePolicy
    env = SlimeVolleyEnv()
    env.seed(SLIME_PROBE_SEED)
    policy = BaselinePolicy()
    obs, t, out = env.reset(), 0, []
    while len(out) < SLIME_PROBES:
        obs, _, done, _ = env.step(policy.predict(obs))
        t += 1
        if t % SLIME_PROBE_EVERY == 0:
            out.append(np.array(obs, dtype=np.float64))
        if done:
            obs = env.reset()
    return np.array(out)


def make(name, lam=0.0):
    """(game id, gp, X1, X2, X3) for lab.kernels.run.

    For slime, X3 holds the niche archive's probe states (unused by the
    game itself, so every other mode is unaffected)."""
    if name == "slime":
        z = np.zeros((1, 1))
        return K.GAME_SLIME, np.zeros(5), z, z, slime_probes()
    if name == "discmix":
        skill_x, skill_y, style_x, _ = probes()
        gp = np.array([lam, ALPHA, BETA, NOISE, TIE], dtype=np.float64)
        return K.GAME_DISCMIX, gp, skill_x, skill_y, style_x
    raise ValueError(name)


def features(genomes):
    """(n, 3) array of (skill, u0, u1) for every genome."""
    skill_x, skill_y, style_x, _ = probes()
    return np.array([K.discmix_features(np.ascontiguousarray(g, dtype=np.float64),
                                        skill_x, skill_y, style_x)
                     for g in genomes])


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_score(fr, fl, lam):
    """E[score of r against l] = P(win) - P(loss), exactly."""
    gp = np.array([lam, ALPHA, BETA, NOISE, TIE])
    m = K.discmix_margin(gp, np.asarray(fr, float), np.asarray(fl, float))
    p_win = 1.0 - _phi((TIE - m) / NOISE)
    p_loss = _phi((-TIE - m) / NOISE)
    return p_win - p_loss


def payoff_matrix(feats, lam):
    """Exact expected-score matrix between all pairs (antisymmetric)."""
    n = len(feats)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = expected_score(feats[i], feats[j], lam)
            out[j, i] = -out[i, j]
    return out


def calibrate(n=128, pairs=20000, seed=1):
    """alpha, beta that give the two parts equal spread, unit s.d. each.

    Computed over random pairs of an initial population (init scale 0.5)."""
    rng = np.random.default_rng(seed)
    pop = rng.normal(size=(n, K.PARAM_COUNT)) * TEACHER_SCALE
    f = features(pop)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    trans = f[i, 0] - f[j, 0]
    cyc = f[i, 1] * f[j, 2] - f[i, 2] * f[j, 1]
    return 1.0 / trans.std(), 1.0 / cyc.std()
