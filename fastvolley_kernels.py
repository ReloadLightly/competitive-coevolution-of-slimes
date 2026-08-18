"""
fastvolley_kernels.py — compiled kernels used only by the analysis stage.

Kept out of fastvolley.py so that the training module (and its numba cache)
stays untouched while long runs are in flight.
"""

import numpy as np
from numba import njit

from fastvolley import PARAM_COUNT, POLICY_BASELINE, POLICY_MLP, play_game


@njit(cache=True)
def round_robin(pop, games, seed, w, b):
    """All-play-all, both court sides, `games` games per ordered pair.

    Returns (margin, wins, draws, losses) where margin[i, j] is i's mean point
    margin against j, averaged over both sides so no side bias survives.
    """
    k = pop.shape[0]
    margin = np.zeros((k, k))
    wins = np.zeros((k, k))
    draws = np.zeros((k, k))
    losses = np.zeros((k, k))
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    np.random.seed(seed)
    for i in range(k):
        for j in range(i + 1, k):
            tot = 0.0
            wi = 0.0
            dr = 0.0
            lo = 0.0
            for _ in range(games):
                sc, _, _ = play_game(pop[i], POLICY_MLP, pop[j], POLICY_MLP,
                                     w, b, rnn_a, rnn_b, empty, empty, 0, False)
                tot += sc
                if sc > 0:
                    wi += 1
                elif sc == 0:
                    dr += 1
                else:
                    lo += 1
                sc, _, _ = play_game(pop[j], POLICY_MLP, pop[i], POLICY_MLP,
                                     w, b, rnn_a, rnn_b, empty, empty, 0, False)
                tot -= sc
                if sc < 0:
                    wi += 1
                elif sc == 0:
                    dr += 1
                else:
                    lo += 1
            margin[i, j] = tot / (2 * games)
            margin[j, i] = -margin[i, j]
            wins[i, j] = wi
            draws[i, j] = dr
            losses[i, j] = lo
            wins[j, i] = lo
            draws[j, i] = dr
            losses[j, i] = wi
    return margin, wins, draws, losses


@njit(cache=True)
def eval_population(pop, episodes, seed, w, b):
    """Every individual in a population against the 2015 baseline."""
    k = pop.shape[0]
    means = np.zeros(k)
    wins = np.zeros(k)
    for i in range(k):
        np.random.seed(seed)
        rnn_r = np.zeros(7)
        rnn_l = np.zeros(7)
        empty = np.zeros(1)
        tot = 0.0
        nw = 0.0
        for _ in range(episodes):
            sc, _, _ = play_game(pop[i], POLICY_MLP, pop[i], POLICY_BASELINE,
                                 w, b, rnn_r, rnn_l, empty, empty, 0, False)
            tot += sc
            if sc > 0:
                nw += 1
        means[i] = tot / episodes
        wins[i] = nw / episodes
    return means, wins


@njit(cache=True)
def run_ga_resume(population, winning_streak, seed, n_tournaments, sigma,
                  save_every, w, b):
    """Ha's GA continued from an existing population.

    Used to continue the reference run's committed snapshot inside the
    compiled implementation, so the two implementations can be compared at
    the level of a training trajectory rather than a single game.
    """
    np.random.seed(seed)
    pop_size = population.shape[0]
    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)
    len_acc = 0.0
    ck = 0

    for tournament in range(1, n_tournaments + 1):
        m = np.random.randint(0, pop_size)
        n = np.random.randint(0, pop_size)
        while n == m:
            n = np.random.randint(0, pop_size)
        score, length, _ = play_game(population[n], POLICY_MLP,
                                     population[m], POLICY_MLP, w, b,
                                     rnn_a, rnn_b, empty, empty, 0, False)
        len_acc += length
        if score == 0:
            for j in range(PARAM_COUNT):
                population[m, j] += np.random.normal(0.0, 1.0) * sigma
        elif score > 0:
            for j in range(PARAM_COUNT):
                mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
            for j in range(PARAM_COUNT):
                population[m, j] = mutant[j]
            winning_streak[m] = winning_streak[n]
            winning_streak[n] += 1
        else:
            for j in range(PARAM_COUNT):
                mutant[j] = population[m, j] + np.random.normal(0.0, 1.0) * sigma
            for j in range(PARAM_COUNT):
                population[n, j] = mutant[j]
            winning_streak[n] = winning_streak[m]
            winning_streak[m] += 1

        if tournament % save_every == 0:
            rh = np.argmax(winning_streak)
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[rh, j]
            streaks[ck] = winning_streak[rh]
            meanlen[ck] = len_acc / save_every
            len_acc = 0.0
            ck += 1
    return champs, streaks, meanlen


@njit(cache=True)
def internal_rank(pop, n_opponents, seed, w, b):
    """Rank a population by an internal round robin — no external opponent.

    This is the deployable alternative to Ha's winning-streak proxy: spend a few
    hundred games having the pool play itself, and export the individual with
    the best mean point margin. It uses no information the algorithm does not
    already have, so unlike "best against the 2015 baseline" it is a fix rather
    than an oracle.

    Returns the mean margin of each individual over the games it played.
    """
    np.random.seed(seed)
    k = pop.shape[0]
    fitness = np.zeros(k)
    played = np.zeros(k)
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    n_games = (k * n_opponents) // 2
    for _ in range(n_games):
        i = np.random.randint(0, k)
        j = np.random.randint(0, k)
        while j == i:
            j = np.random.randint(0, k)
        score, _, _ = play_game(pop[i], POLICY_MLP, pop[j], POLICY_MLP, w, b,
                                rnn_a, rnn_b, empty, empty, 0, False)
        fitness[i] += score
        fitness[j] -= score
        played[i] += 1.0
        played[j] += 1.0
    for i in range(k):
        if played[i] > 0.0:
            fitness[i] /= played[i]
    return fitness
