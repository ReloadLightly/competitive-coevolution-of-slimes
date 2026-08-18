"""
asymmetric.py — two populations, unequal power.

Everything else in this study is symmetric: one pool playing itself, every agent
with the identical policy class and the identical budget. That is the setting in
which "compete harder" is the only available move, and it cannot say anything
about what happens when one side is simply stronger.

Here two separate populations play only each other, and the strong side is given
roughly twice the *policy capacity* of the weak side — a 12-16-16-3 network with
531 parameters against the study's standard 12-10-10-3 with 273, a ratio of
1.95:1. Capacity is the most literal reading of "raw power" available in a
neuroevolution setup: it is how sophisticated a strategy the side can express at
all, independent of how long it trains. Keeping the weak side at exactly the
standard architecture means its results stay directly comparable with every other
run in the study.

Selection rule (identical for both sides, so the asymmetry is the only
difference):

    draw a from A and b from B, play one full game
    the loser is overwritten by a mutated copy of a randomly drawn distinct peer
      *from its own population*
    on a tie, both are nudged with noise

Genetic material never crosses between populations — the other side supplies
selection pressure, never parents. That is the same discipline the hall-of-fame
analysis showed to be load-bearing.
"""

import numpy as np
from numba import njit

MAX_H = 32           # largest hidden layer this module supports
OBS_SIZE = 12
N_ACT = 3


def param_count(h):
    """Parameters of a 12-h-h-3 tanh network: h^2 + 17h + 3."""
    return h * h + 17 * h + 3


@njit(cache=True, inline="always")
def mlp_forward_var(p, obs, out, h, buf1, buf2):
    """Model.predict() for a 12-h-h-3 tanh network, h chosen at runtime.

    Layout matches estool's flat ordering: w1 b1 w2 b2 w3 b3.
    """
    o1 = 12 * h            # end of w1
    o2 = o1 + h            # end of b1
    o3 = o2 + h * h        # end of w2
    o4 = o3 + h            # end of b2
    o5 = o4 + h * 3        # end of w3
    for j in range(h):
        s = 0.0
        for i in range(12):
            s += obs[i] * p[i * h + j]
        buf1[j] = np.tanh(s + p[o1 + j])
    for j in range(h):
        s = 0.0
        for i in range(h):
            s += buf1[i] * p[o2 + i * h + j]
        buf2[j] = np.tanh(s + p[o3 + j])
    for j in range(3):
        s = 0.0
        for i in range(h):
            s += buf2[i] * p[o4 + i * 3 + j]
        out[j] = np.tanh(s + p[o5 + j])


# --- physics, imported from the validated implementation -------------------
from fastvolley import (LIFE_L, LIFE_R, STATE_SIZE, T_LIMIT, _game_step,
                        _reset_game)


@njit(cache=True)
def play_game_asym(p_r, h_r, p_l, h_l):
    """One full game between two policies of possibly different hidden size.

    Returns (score, length) from the right player's point of view. The physics
    is `fastvolley._game_step`, i.e. the same validated code path every other
    result in this study uses.
    """
    s = np.zeros(STATE_SIZE)
    obs_l = np.zeros(OBS_SIZE)
    obs_r = np.zeros(OBS_SIZE)
    a_l = np.zeros(3)
    a_r = np.zeros(3)
    b1r = np.zeros(MAX_H)
    b2r = np.zeros(MAX_H)
    b1l = np.zeros(MAX_H)
    b2l = np.zeros(MAX_H)

    bvx = np.random.uniform(-20.0, 20.0)
    bvy = np.random.uniform(10.0, 25.0)
    _reset_game(s, obs_l, obs_r, bvx, bvy)
    obs_l_view = obs_r.copy()

    total = 0
    t = 0
    while True:
        mlp_forward_var(p_r, obs_r, a_r, h_r, b1r, b2r)
        mlp_forward_var(p_l, obs_l_view, a_l, h_l, b1l, b2l)
        result = _game_step(s, obs_l, obs_r, a_l, a_r, 0.0, 0.0)
        if result != 0:
            s[4] = np.random.uniform(-20.0, 20.0)   # BVX
            s[5] = np.random.uniform(10.0, 25.0)    # BVY
        for i in range(OBS_SIZE):
            obs_l_view[i] = obs_l[i]
        total += result
        t += 1
        if t >= T_LIMIT:
            break
        if s[LIFE_L] <= 0.0 or s[LIFE_R] <= 0.0:
            break
    return total, t


@njit(cache=True)
def run_coevo_asym(seed, n_games, n_a, n_b, h_a, h_b, d_a, d_b, sigma,
                   save_every, init_scale):
    """Two populations playing only each other.

    A is the 'strong' side by convention (h_a >= h_b), but the code is
    symmetric. `d_a`/`d_b` are the parameter counts implied by `h_a`/`h_b`.

    Returns champions and mean rally length for both sides, plus the running
    win rate of A, which is the headline series: 0.5 means the sides are
    holding each other.
    """
    np.random.seed(seed)
    pop_a = np.zeros((n_a, d_a))
    pop_b = np.zeros((n_b, d_b))
    for i in range(n_a):
        for j in range(d_a):
            pop_a[i, j] = np.random.normal(0.0, 1.0) * init_scale
    for i in range(n_b):
        for j in range(d_b):
            pop_b[i, j] = np.random.normal(0.0, 1.0) * init_scale
    streak_a = np.zeros(n_a, dtype=np.int64)
    streak_b = np.zeros(n_b, dtype=np.int64)

    n_ckpt = n_games // save_every
    champs_a = np.zeros((n_ckpt, d_a))
    champs_b = np.zeros((n_ckpt, d_b))
    meanlen = np.zeros(n_ckpt)
    a_winrate = np.zeros(n_ckpt)
    a_margin = np.zeros(n_ckpt)

    len_acc = 0.0
    a_wins = 0.0
    a_pts = 0.0
    decided = 0.0
    ck = 0

    for game in range(1, n_games + 1):
        ia = np.random.randint(0, n_a)
        ib = np.random.randint(0, n_b)
        # A plays on the right, so the score is from A's point of view
        score, length = play_game_asym(pop_a[ia], h_a, pop_b[ib], h_b)
        len_acc += length
        a_pts += score

        if score > 0:
            a_wins += 1.0
            decided += 1.0
            # B's individual lost: replaced by a mutant of one of ITS OWN peers
            if n_b > 1:
                src = np.random.randint(0, n_b)
                while src == ib:
                    src = np.random.randint(0, n_b)
                for j in range(d_b):
                    pop_b[ib, j] = (pop_b[src, j]
                                    + np.random.normal(0.0, 1.0) * sigma)
                streak_b[ib] = streak_b[src]
            streak_a[ia] += 1
        elif score < 0:
            decided += 1.0
            if n_a > 1:
                src = np.random.randint(0, n_a)
                while src == ia:
                    src = np.random.randint(0, n_a)
                for j in range(d_a):
                    pop_a[ia, j] = (pop_a[src, j]
                                    + np.random.normal(0.0, 1.0) * sigma)
                streak_a[ia] = streak_a[src]
            streak_b[ib] += 1
        else:
            for j in range(d_a):
                pop_a[ia, j] += np.random.normal(0.0, 1.0) * sigma
            for j in range(d_b):
                pop_b[ib, j] += np.random.normal(0.0, 1.0) * sigma

        if game % save_every == 0:
            ra = np.argmax(streak_a)
            rb = np.argmax(streak_b)
            for j in range(d_a):
                champs_a[ck, j] = pop_a[ra, j]
            for j in range(d_b):
                champs_b[ck, j] = pop_b[rb, j]
            meanlen[ck] = len_acc / save_every
            a_winrate[ck] = a_wins / decided if decided > 0 else 0.5
            a_margin[ck] = a_pts / save_every
            len_acc = 0.0
            a_wins = 0.0
            a_pts = 0.0
            decided = 0.0
            ck += 1

    return champs_a, champs_b, meanlen, a_winrate, a_margin


@njit(cache=True)
def eval_var_vs_baseline(params, h, episodes, seed, w, b):
    """A variable-capacity champion against the 2015 baseline RNN."""
    np.random.seed(seed)
    scores = np.empty(episodes, dtype=np.int64)
    lengths = np.empty(episodes, dtype=np.int64)
    rnn_l = np.zeros(7)
    b1 = np.zeros(MAX_H)
    b2 = np.zeros(MAX_H)
    for e in range(episodes):
        s = np.zeros(STATE_SIZE)
        obs_l = np.zeros(OBS_SIZE)
        obs_r = np.zeros(OBS_SIZE)
        a_l = np.zeros(3)
        a_r = np.zeros(3)
        bvx = np.random.uniform(-20.0, 20.0)
        bvy = np.random.uniform(10.0, 25.0)
        _reset_game(s, obs_l, obs_r, bvx, bvy)
        obs_l_view = obs_r.copy()
        total = 0
        t = 0
        while True:
            mlp_forward_var(params, obs_r, a_r, h, b1, b2)
            # inline the baseline RNN: 8 game inputs + 7 recurrent outputs
            inp = np.empty(15)
            for i in range(8):
                inp[i] = obs_l_view[i]
            for i in range(7):
                inp[8 + i] = rnn_l[i]
            for j in range(7):
                acc = 0.0
                for i in range(15):
                    acc += w[j, i] * inp[i]
                rnn_l[j] = np.tanh(acc + b[j])
            a_l[0] = 1.0 if rnn_l[0] > 0.75 else 0.0
            a_l[1] = 1.0 if rnn_l[1] > 0.75 else 0.0
            a_l[2] = 1.0 if rnn_l[2] > 0.75 else 0.0

            result = _game_step(s, obs_l, obs_r, a_l, a_r, 0.0, 0.0)
            if result != 0:
                s[4] = np.random.uniform(-20.0, 20.0)
                s[5] = np.random.uniform(10.0, 25.0)
            for i in range(OBS_SIZE):
                obs_l_view[i] = obs_l[i]
            total += result
            t += 1
            if t >= T_LIMIT:
                break
            if s[LIFE_L] <= 0.0 or s[LIFE_R] <= 0.0:
                break
        scores[e] = total
        lengths[e] = t
    return scores, lengths
