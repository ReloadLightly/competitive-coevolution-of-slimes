"""
fastvolley.py — a compiled, line-for-line port of the slimevolleygym physics,
the 273-parameter MLP policy and the 120-parameter 2015 baseline RNN.

Why this file exists
--------------------
The reference environment (`slimevolleygym`, David Ha 2020) runs at roughly
10 games/second on one core, which puts a single 500,000-game self-play run at
about twelve core-hours. That is affordable once. It is not affordable for a
seeded, multi-condition experiment — which is why the first version of this
repository had one run and no error bars.

This module is the same game compiled with numba. Everything below is a
transcription of `slimevolleygym/slimevolley.py` and `slimevolleygym/mlp.py`
with the control flow preserved statement for statement, including the parts
that look like bugs (they are load-bearing: see NOTE comments). It is
validated against the reference implementation in `validate_fastvolley.py`.

What is deliberately *not* identical
------------------------------------
1. Random number generation. The reference draws the ball's initial velocity
   from gym's PCG64 `np_random`; this module draws from numba's internal
   Mersenne Twister. Both are uniform on the same intervals. Seeds therefore
   label runs, they do not transfer between implementations.
2. Tournament pair sampling. The reference calls
   `np.random.choice(pop, 2, replace=False)`; this module draws two distinct
   uniform integers by rejection. Same distribution, different stream.
3. Floating point of the neural network forward pass. NumPy dispatches
   `matmul` to BLAS and uses a SIMD `tanh`; the compiled version uses a plain
   loop and libm. Results differ in the last one or two units in the last
   place. This cannot change a trajectory unless it flips the *sign* of a
   network output (the game only ever sees `action[i] > 0`), which is why
   bit-identical trajectories are still the expected outcome and are what
   `validate_fastvolley.py` measures.

Everything else — physics, collision resolution, scoring, the delay screen,
the observation vector, episode termination — is bit-exact by construction.
"""

import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# Game constants. Written as the same arithmetic expressions as the reference
# so that the resulting doubles are identical bit patterns.
# --------------------------------------------------------------------------
REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5
REF_WALL_WIDTH = 1.0
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10 * 1.75
PLAYER_SPEED_Y = 10 * 1.35
MAX_BALL_SPEED = 15 * 1.5
TIMESTEP = 1 / 30.
NUDGE = 0.1
FRICTION = 1.0
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8 * 2 * 1.5
MAXLIVES = 5
T_LIMIT = 3000

BALL_R = 0.5
AGENT_R = 1.5
STUB_X = 0.0
STUB_Y = REF_WALL_HEIGHT
STUB_R = REF_WALL_WIDTH / 2

PARAM_COUNT = 273          # 12*10+10 + 10*10+10 + 10*3+3
OBS_SIZE = 12

# Policy type codes.
POLICY_MLP = 0
POLICY_BASELINE = 1

# The 2015 champion, verbatim from slimevolleygym.slimevolley.BaselinePolicy.
BASELINE_WEIGHT = np.array(
    [7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689,
     1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887,
     2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353,
     -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809,
     7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695,
     -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952,
     -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377]
).reshape(7, 15)
BASELINE_BIAS = np.array([2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685])

# --------------------------------------------------------------------------
# Game state layout (one float64 array per game, so helpers stay inlineable)
# --------------------------------------------------------------------------
BX, BY, BPX, BPY, BVX, BVY = 0, 1, 2, 3, 4, 5
LX, LY, LVX, LVY, LDX, LDY = 6, 7, 8, 9, 10, 11
RX, RY, RVX, RVY, RDX, RDY = 12, 13, 14, 15, 16, 17
DELAY, LIFE_L, LIFE_R = 18, 19, 20
STATE_SIZE = 21


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def mlp_forward(p, obs, out):
    """Model.predict() from slimevolleygym/mlp.py: 12-10-10-3, tanh throughout.

    `p` is the flat genotype in estool layout: w1 b1 w2 b2 w3 b3.
    """
    h1 = np.empty(10)
    for j in range(10):
        s = 0.0
        for i in range(12):
            s += obs[i] * p[i * 10 + j]
        h1[j] = np.tanh(s + p[120 + j])
    h2 = np.empty(10)
    for j in range(10):
        s = 0.0
        for i in range(10):
            s += h1[i] * p[130 + i * 10 + j]
        h2[j] = np.tanh(s + p[230 + j])
    for j in range(3):
        s = 0.0
        for i in range(10):
            s += h2[i] * p[240 + i * 3 + j]
        out[j] = np.tanh(s + p[270 + j])


@njit(cache=True, inline="always")
def baseline_forward(w, b, rnn, obs, out):
    """BaselinePolicy.predict(): 8 game inputs + 7 recurrent, tanh, >0.75.

    `rnn` (length 7) is the persistent output state and is updated in place.
    NOTE: the reference never resets this between episodes when a single
    BaselinePolicy instance is reused (as eval_vs_baseline.py does), so it is
    the caller's job to decide whether to carry it over.
    """
    inp = np.empty(15)
    for i in range(8):
        inp[i] = obs[i]
    for i in range(7):
        inp[8 + i] = rnn[i]
    for j in range(7):
        s = 0.0
        for i in range(15):
            s += w[j, i] * inp[i]
        rnn[j] = np.tanh(s + b[j])
    # thresholds at 0.75; the game reads action[i] > 0
    out[0] = 1.0 if rnn[0] > 0.75 else 0.0
    out[1] = 1.0 if rnn[1] > 0.75 else 0.0
    out[2] = 1.0 if rnn[2] > 0.75 else 0.0


@njit(cache=True, inline="always")
def act(ptype, params, w, b, rnn, obs, out):
    if ptype == POLICY_MLP:
        mlp_forward(params, obs, out)
    else:
        baseline_forward(w, b, rnn, obs, out)


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _is_colliding(bx, by, px, py, pr):
    r = BALL_R + pr
    dy = py - by
    dx = px - bx
    return r * r > (dx * dx + dy * dy)


@njit(cache=True, inline="always")
def _bounce(s, px, py, pvx, pvy, pr):
    """Particle.bounce(): elastic bounce off a moving disc, with the
    reference's `while colliding: nudge apart` separation loop."""
    abx = s[BX] - px
    aby = s[BY] - py
    abd = np.sqrt(abx * abx + aby * aby)
    abx /= abd
    aby /= abd
    nx = abx
    ny = aby
    abx *= NUDGE
    aby *= NUDGE
    while _is_colliding(s[BX], s[BY], px, py, pr):
        s[BX] += abx
        s[BY] += aby
    ux = s[BVX] - pvx
    uy = s[BVY] - pvy
    un = ux * nx + uy * ny
    unx = nx * (un * 2.)
    uny = ny * (un * 2.)
    ux -= unx
    uy -= uny
    s[BVX] = ux + pvx
    s[BVY] = uy + pvy


@njit(cache=True, inline="always")
def _check_edges(s):
    """Particle.checkEdges(). Returns -1 / +1 when the ball lands (left /
    right of centre), 0 otherwise.

    NOTE: the ground test returns immediately, so the ceiling and fence tests
    are skipped on a scoring frame. Preserved deliberately.
    """
    if s[BX] <= (BALL_R - REF_W / 2):
        s[BVX] *= -FRICTION
        s[BX] = BALL_R - REF_W / 2 + NUDGE * TIMESTEP
    if s[BX] >= (REF_W / 2 - BALL_R):
        s[BVX] *= -FRICTION
        s[BX] = REF_W / 2 - BALL_R - NUDGE * TIMESTEP
    if s[BY] <= (BALL_R + REF_U):
        s[BVY] *= -FRICTION
        s[BY] = BALL_R + REF_U + NUDGE * TIMESTEP
        if s[BX] <= 0:
            return -1
        else:
            return 1
    if s[BY] >= (REF_H - BALL_R):
        s[BVY] *= -FRICTION
        s[BY] = REF_H - BALL_R - NUDGE * TIMESTEP
    # fence
    if (s[BX] <= (REF_WALL_WIDTH / 2 + BALL_R)) and (s[BPX] > (REF_WALL_WIDTH / 2 + BALL_R)) and (s[BY] <= REF_WALL_HEIGHT):
        s[BVX] *= -FRICTION
        s[BX] = REF_WALL_WIDTH / 2 + BALL_R + NUDGE * TIMESTEP
    if (s[BX] >= (-REF_WALL_WIDTH / 2 - BALL_R)) and (s[BPX] < (-REF_WALL_WIDTH / 2 - BALL_R)) and (s[BY] <= REF_WALL_HEIGHT):
        s[BVX] *= -FRICTION
        s[BX] = -REF_WALL_WIDTH / 2 - BALL_R - NUDGE * TIMESTEP
    return 0


@njit(cache=True, inline="always")
def _agent_update(s, xi, yi, vxi, vyi, dxi, dyi, direction):
    """Agent.update(): gravity, ground clamp, own-half clamp."""
    s[vyi] += GRAVITY * TIMESTEP
    if s[yi] <= REF_U + NUDGE * TIMESTEP:
        s[vyi] = s[dyi]
    s[vxi] = s[dxi] * direction
    s[xi] += s[vxi] * TIMESTEP
    s[yi] += s[vyi] * TIMESTEP
    if s[yi] <= REF_U:
        s[yi] = REF_U
        s[vyi] = 0.0
    if s[xi] * direction <= (REF_WALL_WIDTH / 2 + AGENT_R):
        s[vxi] = 0.0
        s[xi] = direction * (REF_WALL_WIDTH / 2 + AGENT_R)
    if s[xi] * direction >= (REF_W / 2 - AGENT_R):
        s[vxi] = 0.0
        s[xi] = direction * (REF_W / 2 - AGENT_R)


@njit(cache=True, inline="always")
def _set_action(s, dxi, dyi, a, direction):
    """Agent.setAction(): three booleans -> desired velocities."""
    forward = a[0] > 0
    backward = a[1] > 0
    jump = a[2] > 0
    s[dxi] = 0.0
    s[dyi] = 0.0
    if forward and (not backward):
        s[dxi] = -PLAYER_SPEED_X
    if backward and (not forward):
        s[dxi] = PLAYER_SPEED_X
    if jump:
        s[dyi] = PLAYER_SPEED_Y


@njit(cache=True, inline="always")
def _update_obs(s, obs_l, obs_r):
    """Agent.updateState() + RelativeState.getObservation() for both sides."""
    # left agent: dir = -1, opponent is right
    obs_l[0] = -s[LX] / 10.0
    obs_l[1] = s[LY] / 10.0
    obs_l[2] = -s[LVX] / 10.0
    obs_l[3] = s[LVY] / 10.0
    obs_l[4] = -s[BX] / 10.0
    obs_l[5] = s[BY] / 10.0
    obs_l[6] = -s[BVX] / 10.0
    obs_l[7] = s[BVY] / 10.0
    obs_l[8] = s[RX] / 10.0
    obs_l[9] = s[RY] / 10.0
    obs_l[10] = s[RVX] / 10.0
    obs_l[11] = s[RVY] / 10.0
    # right agent: dir = +1, opponent is left
    obs_r[0] = s[RX] / 10.0
    obs_r[1] = s[RY] / 10.0
    obs_r[2] = s[RVX] / 10.0
    obs_r[3] = s[RVY] / 10.0
    obs_r[4] = s[BX] / 10.0
    obs_r[5] = s[BY] / 10.0
    obs_r[6] = s[BVX] / 10.0
    obs_r[7] = s[BVY] / 10.0
    obs_r[8] = -s[LX] / 10.0
    obs_r[9] = s[LY] / 10.0
    obs_r[10] = -s[LVX] / 10.0
    obs_r[11] = s[LVY] / 10.0


@njit(cache=True, inline="always")
def _new_ball(s, bvx, bvy):
    s[BX] = 0.0
    s[BY] = REF_W / 4
    s[BPX] = 0.0
    s[BPY] = REF_W / 4
    s[BVX] = bvx
    s[BVY] = bvy
    s[DELAY] = INIT_DELAY_FRAMES


@njit(cache=True, inline="always")
def _reset_game(s, obs_l, obs_r, bvx, bvy):
    """Game.reset()"""
    _new_ball(s, bvx, bvy)
    s[LX] = -REF_W / 4
    s[LY] = 1.5
    s[LVX] = 0.0
    s[LVY] = 0.0
    s[LDX] = 0.0
    s[LDY] = 0.0
    s[RX] = REF_W / 4
    s[RY] = 1.5
    s[RVX] = 0.0
    s[RVY] = 0.0
    s[RDX] = 0.0
    s[RDY] = 0.0
    s[LIFE_L] = MAXLIVES
    s[LIFE_R] = MAXLIVES
    _update_obs(s, obs_l, obs_r)


@njit(cache=True, inline="always")
def _game_step(s, obs_l, obs_r, a_l, a_r, bvx, bvy):
    """Game.step(). Returns the point scored from the right agent's view."""
    _set_action(s, LDX, LDY, a_l, -1.0)
    _set_action(s, RDX, RDY, a_r, 1.0)
    _agent_update(s, LX, LY, LVX, LVY, LDX, LDY, -1.0)
    _agent_update(s, RX, RY, RVX, RVY, RDX, RDY, 1.0)

    # DelayScreen.status(): ball is frozen for the first INIT_DELAY_FRAMES
    if s[DELAY] == 0.0:
        s[BVY] += GRAVITY * TIMESTEP
        mag2 = s[BVX] * s[BVX] + s[BVY] * s[BVY]
        if mag2 > MAX_BALL_SPEED * MAX_BALL_SPEED:
            mag = np.sqrt(mag2)
            s[BVX] /= mag
            s[BVY] /= mag
            s[BVX] *= MAX_BALL_SPEED
            s[BVY] *= MAX_BALL_SPEED
        s[BPX] = s[BX]
        s[BPY] = s[BY]
        s[BX] += s[BVX] * TIMESTEP
        s[BY] += s[BVY] * TIMESTEP
    else:
        s[DELAY] -= 1.0

    if _is_colliding(s[BX], s[BY], s[LX], s[LY], AGENT_R):
        _bounce(s, s[LX], s[LY], s[LVX], s[LVY], AGENT_R)
    if _is_colliding(s[BX], s[BY], s[RX], s[RY], AGENT_R):
        _bounce(s, s[RX], s[RY], s[RVX], s[RVY], AGENT_R)
    if _is_colliding(s[BX], s[BY], STUB_X, STUB_Y, STUB_R):
        _bounce(s, STUB_X, STUB_Y, 0.0, 0.0, STUB_R)

    result = -_check_edges(s)

    if result != 0:
        _new_ball(s, bvx, bvy)
        if result < 0:
            s[LIFE_R] -= 1.0
        else:
            s[LIFE_L] -= 1.0
        # NOTE: observations are NOT refreshed on a scoring frame — the
        # reference returns the stale state. Preserved deliberately.
        return result

    _update_obs(s, obs_l, obs_r)
    return 0


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
@njit(cache=True)
def play_game(params_r, type_r, params_l, type_l, w, b, rnn_r, rnn_l,
              bvx_buf, bvy_buf, buf_start, use_buf):
    """multiagent_rollout(): one full game, score from the right agent's view.

    `bvx_buf`/`bvy_buf` supply the ball's initial velocity for the opening
    serve and every subsequent serve. When `use_buf` is False the values are
    drawn internally; when True they are read from the buffers starting at
    `buf_start` (used by the validation harness to feed the reference
    environment's own PCG64 draws through this implementation).

    Returns (score, length, draws_consumed).
    """
    s = np.zeros(STATE_SIZE)
    obs_l = np.zeros(OBS_SIZE)
    obs_r = np.zeros(OBS_SIZE)
    a_l = np.zeros(3)
    a_r = np.zeros(3)
    k = buf_start

    if use_buf:
        bvx = bvx_buf[k]
        bvy = bvy_buf[k]
    else:
        bvx = np.random.uniform(-20.0, 20.0)
        bvy = np.random.uniform(10.0, 25.0)
    k += 1
    _reset_game(s, obs_l, obs_r, bvx, bvy)

    # NOTE: the reference hands the left agent the RIGHT agent's observation on
    # the very first step (`obs_left = obs_right` in multiagent_rollout).
    obs_l_view = obs_r.copy()

    total = 0
    t = 0
    while True:
        act(type_r, params_r, w, b, rnn_r, obs_r, a_r)
        act(type_l, params_l, w, b, rnn_l, obs_l_view, a_l)

        if use_buf:
            nbvx = bvx_buf[k]
            nbvy = bvy_buf[k]
        else:
            nbvx = 0.0
            nbvy = 0.0
        result = _game_step(s, obs_l, obs_r, a_l, a_r, nbvx, nbvy)
        if result != 0:
            if not use_buf:
                # draw the serve exactly where the reference draws it
                s[BVX] = np.random.uniform(-20.0, 20.0)
                s[BVY] = np.random.uniform(10.0, 25.0)
            k += 1
        for i in range(OBS_SIZE):
            obs_l_view[i] = obs_l[i]

        total += result
        t += 1
        if t >= T_LIMIT:
            break
        if s[LIFE_L] <= 0.0 or s[LIFE_R] <= 0.0:
            break

    return total, t, k - buf_start


@njit(cache=True)
def play_game_trace(params_r, type_r, params_l, type_l, w, b, rnn_r, rnn_l,
                    bvx_buf, bvy_buf, trace):
    """As play_game, but records the full state every step into `trace`
    (n_steps, 7): ball x, ball y, ball vx, ball vy, left x, left y, right x.

    Used only by validate_fastvolley.py, which drives both implementations
    from the same serve buffer and compares the traces element by element.
    """
    s = np.zeros(STATE_SIZE)
    obs_l = np.zeros(OBS_SIZE)
    obs_r = np.zeros(OBS_SIZE)
    a_l = np.zeros(3)
    a_r = np.zeros(3)
    k = 0
    _reset_game(s, obs_l, obs_r, bvx_buf[0], bvy_buf[0])
    k += 1
    obs_l_view = obs_r.copy()

    total = 0
    t = 0
    while True:
        act(type_r, params_r, w, b, rnn_r, obs_r, a_r)
        act(type_l, params_l, w, b, rnn_l, obs_l_view, a_l)
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


@njit(cache=True)
def eval_vs_baseline(params, episodes, seed, w, b, reset_rnn):
    """Champion (right) versus the 2015 baseline (left), `episodes` games.

    `reset_rnn=False` reproduces eval_vs_baseline.py, which constructs one
    BaselinePolicy and reuses it across every episode without resetting its
    recurrent state.
    """
    np.random.seed(seed)
    scores = np.empty(episodes, dtype=np.int64)
    lengths = np.empty(episodes, dtype=np.int64)
    rnn_r = np.zeros(7)
    rnn_l = np.zeros(7)
    empty = np.zeros(1)
    for e in range(episodes):
        if reset_rnn:
            for i in range(7):
                rnn_l[i] = 0.0
        sc, ln, _ = play_game(params, POLICY_MLP, params, POLICY_BASELINE,
                              w, b, rnn_r, rnn_l, empty, empty, 0, False)
        scores[e] = sc
        lengths[e] = ln
    return scores, lengths


@njit(cache=True)
def head_to_head(params_a, params_b, games, seed, w, b):
    """`games` games per side. Returns mean point margin for a, and mean length."""
    np.random.seed(seed)
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    total = 0.0
    tlen = 0.0
    for g in range(games):
        sc, ln, _ = play_game(params_a, POLICY_MLP, params_b, POLICY_MLP,
                              w, b, rnn_a, rnn_b, empty, empty, 0, False)
        total += sc
        tlen += ln
        sc, ln, _ = play_game(params_b, POLICY_MLP, params_a, POLICY_MLP,
                              w, b, rnn_a, rnn_b, empty, empty, 0, False)
        total -= sc
        tlen += ln
    return total / (2 * games), tlen / (2 * games)


# --------------------------------------------------------------------------
# The genetic algorithm
# --------------------------------------------------------------------------
@njit(cache=True)
def run_ga(seed, n_tournaments, pop_size, sigma, save_every,
           hof_prob, hof_every, hof_capacity, w, b, init_scale):
    """Ha's tournament-selection self-play GA, with an optional hall of fame.

    Control condition (`hof_prob == 0`) is the reference algorithm:
      draw two distinct individuals, play one game, the loser is overwritten
      by a mutated copy of the winner; on a tie the first is mutated in place.

    Hall-of-fame condition (`hof_prob > 0`): with probability `hof_prob` the
    second contestant is replaced by a uniformly drawn archived champion.
    The archive is immutable, so:
      - archive wins  -> the population member is overwritten by a mutant of
                         the archived genome and inherits its streak counter
                         (exactly Ha's rule, treating the archive entry as if
                         it were in the pool);
      - population wins -> nothing is overwritten, the winner's streak grows.
    A champion is archived every `hof_every` tournaments; once the archive is
    full the oldest entry is replaced (FIFO).

    Returns
      champs   (n_ckpt, PARAM_COUNT)  champion genome at each checkpoint
      streaks  (n_ckpt,)              its winning-streak counter
      meanlen  (n_ckpt,)              mean game length since the last one
      ties     (n_ckpt,)              fraction of tied games since the last one
      hofwins  (n_ckpt,)              fraction of archive games the archive won
    """
    np.random.seed(seed)
    population = np.empty((pop_size, PARAM_COUNT))
    for i in range(pop_size):
        for j in range(PARAM_COUNT):
            population[i, j] = np.random.normal(0.0, 1.0) * init_scale
    winning_streak = np.zeros(pop_size, dtype=np.int64)

    archive = np.zeros((hof_capacity, PARAM_COUNT))
    archive_streak = np.zeros(hof_capacity, dtype=np.int64)
    n_arch = 0
    arch_ptr = 0

    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)
    ties = np.zeros(n_ckpt)
    hofwins = np.zeros(n_ckpt)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)

    len_acc = 0.0
    tie_acc = 0.0
    hof_games = 0
    hof_wins = 0
    ck = 0

    for tournament in range(1, n_tournaments + 1):
        m = np.random.randint(0, pop_size)
        n = np.random.randint(0, pop_size)
        while n == m:
            n = np.random.randint(0, pop_size)

        use_hof = False
        ai = 0
        if hof_prob > 0.0 and n_arch > 0:
            if np.random.random() < hof_prob:
                use_hof = True
                ai = np.random.randint(0, n_arch)

        # right player is n (or the archived champion), left player is m
        if use_hof:
            score, length, _ = play_game(archive[ai], POLICY_MLP,
                                         population[m], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
            hof_games += 1
        else:
            score, length, _ = play_game(population[n], POLICY_MLP,
                                         population[m], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
        len_acc += length

        if score == 0:
            tie_acc += 1.0
            for j in range(PARAM_COUNT):
                population[m, j] += np.random.normal(0.0, 1.0) * sigma
        elif score > 0:
            # the right player won
            if use_hof:
                hof_wins += 1
                for j in range(PARAM_COUNT):
                    mutant[j] = archive[ai, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[m, j] = mutant[j]
                winning_streak[m] = archive_streak[ai]
            else:
                for j in range(PARAM_COUNT):
                    mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[m, j] = mutant[j]
                winning_streak[m] = winning_streak[n]
                winning_streak[n] += 1
        else:
            # the left player (always a population member) won
            if use_hof:
                winning_streak[m] += 1
            else:
                for j in range(PARAM_COUNT):
                    mutant[j] = population[m, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[n, j] = mutant[j]
                winning_streak[n] = winning_streak[m]
                winning_streak[m] += 1

        if hof_prob > 0.0 and tournament % hof_every == 0:
            rh = np.argmax(winning_streak)
            idx = arch_ptr
            if n_arch < hof_capacity:
                idx = n_arch
                n_arch += 1
            else:
                arch_ptr = (arch_ptr + 1) % hof_capacity
            for j in range(PARAM_COUNT):
                archive[idx, j] = population[rh, j]
            archive_streak[idx] = winning_streak[rh]

        if tournament % save_every == 0:
            rh = np.argmax(winning_streak)
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[rh, j]
            streaks[ck] = winning_streak[rh]
            meanlen[ck] = len_acc / save_every
            ties[ck] = tie_acc / save_every
            if hof_games > 0:
                hofwins[ck] = hof_wins / hof_games
            len_acc = 0.0
            tie_acc = 0.0
            hof_games = 0
            hof_wins = 0
            ck += 1

    return champs, streaks, meanlen, ties, hofwins


@njit(cache=True)
def run_ga_with_pops(seed, n_tournaments, pop_size, sigma, save_every,
                     hof_prob, hof_every, hof_capacity, w, b, init_scale,
                     pop_every):
    """As run_ga, but also returns full-population snapshots every
    `pop_every` tournaments. Used for the champion-selection ablation, which
    needs the whole pool, not just the exported champion."""
    np.random.seed(seed)
    population = np.empty((pop_size, PARAM_COUNT))
    for i in range(pop_size):
        for j in range(PARAM_COUNT):
            population[i, j] = np.random.normal(0.0, 1.0) * init_scale
    winning_streak = np.zeros(pop_size, dtype=np.int64)

    archive = np.zeros((hof_capacity, PARAM_COUNT))
    archive_streak = np.zeros(hof_capacity, dtype=np.int64)
    n_arch = 0
    arch_ptr = 0

    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)

    n_pop = n_tournaments // pop_every
    pops = np.zeros((n_pop, pop_size, PARAM_COUNT), dtype=np.float32)
    pop_streaks = np.zeros((n_pop, pop_size), dtype=np.int64)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)
    len_acc = 0.0
    ck = 0
    pk = 0

    for tournament in range(1, n_tournaments + 1):
        m = np.random.randint(0, pop_size)
        n = np.random.randint(0, pop_size)
        while n == m:
            n = np.random.randint(0, pop_size)

        use_hof = False
        ai = 0
        if hof_prob > 0.0 and n_arch > 0:
            if np.random.random() < hof_prob:
                use_hof = True
                ai = np.random.randint(0, n_arch)

        if use_hof:
            score, length, _ = play_game(archive[ai], POLICY_MLP,
                                         population[m], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
        else:
            score, length, _ = play_game(population[n], POLICY_MLP,
                                         population[m], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
        len_acc += length

        if score == 0:
            for j in range(PARAM_COUNT):
                population[m, j] += np.random.normal(0.0, 1.0) * sigma
        elif score > 0:
            if use_hof:
                for j in range(PARAM_COUNT):
                    mutant[j] = archive[ai, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[m, j] = mutant[j]
                winning_streak[m] = archive_streak[ai]
            else:
                for j in range(PARAM_COUNT):
                    mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[m, j] = mutant[j]
                winning_streak[m] = winning_streak[n]
                winning_streak[n] += 1
        else:
            if use_hof:
                winning_streak[m] += 1
            else:
                for j in range(PARAM_COUNT):
                    mutant[j] = population[m, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[n, j] = mutant[j]
                winning_streak[n] = winning_streak[m]
                winning_streak[m] += 1

        if hof_prob > 0.0 and tournament % hof_every == 0:
            rh = np.argmax(winning_streak)
            idx = arch_ptr
            if n_arch < hof_capacity:
                idx = n_arch
                n_arch += 1
            else:
                arch_ptr = (arch_ptr + 1) % hof_capacity
            for j in range(PARAM_COUNT):
                archive[idx, j] = population[rh, j]
            archive_streak[idx] = winning_streak[rh]

        if tournament % save_every == 0:
            rh = np.argmax(winning_streak)
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[rh, j]
            streaks[ck] = winning_streak[rh]
            meanlen[ck] = len_acc / save_every
            len_acc = 0.0
            ck += 1

        if tournament % pop_every == 0:
            for i in range(pop_size):
                for j in range(PARAM_COUNT):
                    pops[pk, i, j] = np.float32(population[i, j])
                pop_streaks[pk, i] = winning_streak[i]
            pk += 1

    return champs, streaks, meanlen, pops, pop_streaks


# --------------------------------------------------------------------------
# Convenience wrappers (plain Python)
# --------------------------------------------------------------------------
def baseline_arrays():
    return BASELINE_WEIGHT.copy(), BASELINE_BIAS.copy()


def evaluate(params, episodes=100, seed=721, reset_rnn=False):
    w, b = baseline_arrays()
    return eval_vs_baseline(np.ascontiguousarray(params, dtype=np.float64),
                            episodes, seed, w, b, reset_rnn)
