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
def run_coevo_asym(seed, n_games, n_a, n_b, h_a, h_b, d_a, d_b,
                   sigma_a, sigma_b, cross_prob, save_every, init_scale,
                   pop_every):
    """Two populations, each running Ha's loop, a fraction of games crossed.

    A first version of this kernel replaced a loser with a mutant of a *random*
    peer from its own population. That is too weak a rule: winners never
    preferentially propagate, so selection reduces to culling, and neither
    population learned anything in 500,000 games (rally length flat at ~630,
    cross-population win rate pinned at 0.500). It is kept in the decision log
    as a measured negative rather than described as a bug.

    What runs here instead gives each population the rule that is known to work
    — Ha's, verbatim — and crosses a fraction `cross_prob` of its games with the
    other population:

        pick a population X at random; draw m != n from X
        with probability cross_prob the opponent is drawn from the other
          population Y instead of being n
        within-population game -> Ha's rule exactly: the loser is overwritten by
          a mutated copy of the winner
        cross-population game  -> if X's member loses it is overwritten by a
          mutant of n, its own peer, and never by the opponent's genome; if it
          wins, nothing is overwritten and its streak grows

    Genetic material therefore never crosses between populations: the other side
    supplies selection pressure, never parents. That is the same discipline the
    hall-of-fame analysis showed to be load-bearing, and the cross-population
    rule is exactly the `hof-eval` rule with a live co-adapting opponent in
    place of a stale archived one.

    A is the 'strong' side by convention. Returns champions for both sides, the
    mean rally length, A's win rate in cross-population games (0.5 means the
    sides are holding each other), and A's mean point margin in those games.
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

    n_pop = n_games // pop_every
    pops_a = np.zeros((n_pop, n_a, d_a), dtype=np.float32)
    pops_b = np.zeros((n_pop, n_b, d_b), dtype=np.float32)
    pk = 0

    len_acc = 0.0
    a_wins = 0.0
    a_pts = 0.0
    cross_decided = 0.0
    cross_n = 0.0
    ck = 0

    for game in range(1, n_games + 1):
        side_a = np.random.random() < 0.5
        cross = np.random.random() < cross_prob

        if side_a:
            n_x, n_y = n_a, n_b
            d_x, sigma_x = d_a, sigma_a
            h_x, h_y = h_a, h_b
        else:
            n_x, n_y = n_b, n_a
            d_x, sigma_x = d_b, sigma_b
            h_x, h_y = h_b, h_a

        m = np.random.randint(0, n_x)
        n = np.random.randint(0, n_x)
        while n == m and n_x > 1:
            n = np.random.randint(0, n_x)

        if cross:
            iy = np.random.randint(0, n_y)
            # the other population's member plays on the right
            if side_a:
                score, length = play_game_asym(pop_b[iy], h_b, pop_a[m], h_a)
            else:
                score, length = play_game_asym(pop_a[iy], h_a, pop_b[m], h_b)
            len_acc += length
            cross_n += 1.0
            # score > 0 means the RIGHT player (population Y) won
            if score != 0:
                cross_decided += 1.0
                a_won = (score > 0) != side_a   # Y is A when side_a is False
                if a_won:
                    a_wins += 1.0
            a_pts += score if not side_a else -score

            if score > 0:
                # X's member lost: replaced by a mutant of its own peer. The
                # peer did NOT play, so its streak counter is not incremented --
                # crediting it would let the losing population accumulate
                # streaks at random, and the streak counter is what selects the
                # exported champion.
                if side_a:
                    for j in range(d_x):
                        pop_a[m, j] = (pop_a[n, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_a[m] = streak_a[n]
                else:
                    for j in range(d_x):
                        pop_b[m, j] = (pop_b[n, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_b[m] = streak_b[n]
            elif score < 0:
                if side_a:
                    streak_a[m] += 1
                else:
                    streak_b[m] += 1
            else:
                if side_a:
                    for j in range(d_x):
                        pop_a[m, j] += np.random.normal(0.0, 1.0) * sigma_x
                else:
                    for j in range(d_x):
                        pop_b[m, j] += np.random.normal(0.0, 1.0) * sigma_x
        else:
            # within-population game: Ha's rule, verbatim
            if side_a:
                score, length = play_game_asym(pop_a[n], h_a, pop_a[m], h_a)
            else:
                score, length = play_game_asym(pop_b[n], h_b, pop_b[m], h_b)
            len_acc += length
            if score == 0:
                if side_a:
                    for j in range(d_x):
                        pop_a[m, j] += np.random.normal(0.0, 1.0) * sigma_x
                else:
                    for j in range(d_x):
                        pop_b[m, j] += np.random.normal(0.0, 1.0) * sigma_x
            elif score > 0:
                if side_a:
                    for j in range(d_x):
                        pop_a[m, j] = (pop_a[n, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_a[m] = streak_a[n]
                    streak_a[n] += 1
                else:
                    for j in range(d_x):
                        pop_b[m, j] = (pop_b[n, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_b[m] = streak_b[n]
                    streak_b[n] += 1
            else:
                if side_a:
                    for j in range(d_x):
                        pop_a[n, j] = (pop_a[m, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_a[n] = streak_a[m]
                    streak_a[m] += 1
                else:
                    for j in range(d_x):
                        pop_b[n, j] = (pop_b[m, j]
                                       + np.random.normal(0.0, 1.0) * sigma_x)
                    streak_b[n] = streak_b[m]
                    streak_b[m] += 1

        if game % save_every == 0:
            ra = np.argmax(streak_a)
            rb = np.argmax(streak_b)
            for j in range(d_a):
                champs_a[ck, j] = pop_a[ra, j]
            for j in range(d_b):
                champs_b[ck, j] = pop_b[rb, j]
            meanlen[ck] = len_acc / save_every
            a_winrate[ck] = a_wins / cross_decided if cross_decided > 0 else 0.5
            a_margin[ck] = a_pts / cross_n if cross_n > 0 else 0.0
            len_acc = 0.0
            a_wins = 0.0
            a_pts = 0.0
            cross_decided = 0.0
            cross_n = 0.0
            ck += 1

        if game % pop_every == 0 and pk < n_pop:
            for i in range(n_a):
                for j in range(d_a):
                    pops_a[pk, i, j] = np.float32(pop_a[i, j])
            for i in range(n_b):
                for j in range(d_b):
                    pops_b[pk, i, j] = np.float32(pop_b[i, j])
            pk += 1

    return champs_a, champs_b, meanlen, a_winrate, a_margin, pops_a, pops_b


@njit(cache=True)
def eval_population_var(pop, h, episodes, seed, w, b):
    """Every member of a variable-capacity population against the 2015 baseline.

    'Did this side learn' is a question about the population, not about whichever
    individual a proxy happened to export -- see section 2 of the analysis.
    """
    k = pop.shape[0]
    means = np.zeros(k)
    for i in range(k):
        sc, _ = eval_var_vs_baseline(pop[i], h, episodes, seed, w, b)
        s = 0.0
        for e in range(episodes):
            s += sc[e]
        means[i] = s / episodes
    return means


@njit(cache=True)
def internal_rank_var(pop, h, n_opponents, seed):
    """Rank a variable-capacity population by an internal round robin.

    The deployable replacement for the streak proxy (analysis section 2): the
    pool plays itself and the best mean margin is exported. Uses no information
    the algorithm does not already have.
    """
    np.random.seed(seed)
    k = pop.shape[0]
    fitness = np.zeros(k)
    played = np.zeros(k)
    n_games = (k * n_opponents) // 2
    for _ in range(n_games):
        i = np.random.randint(0, k)
        j = np.random.randint(0, k)
        while j == i:
            j = np.random.randint(0, k)
        score, _ = play_game_asym(pop[i], h, pop[j], h)
        fitness[i] += score
        fitness[j] -= score
        played[i] += 1.0
        played[j] += 1.0
    for i in range(k):
        if played[i] > 0.0:
            fitness[i] /= played[i]
    return fitness


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
