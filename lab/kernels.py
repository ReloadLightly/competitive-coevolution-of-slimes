"""
lab/kernels.py — one compiled GA for every game in the lab.

`run` is Ha's tournament-selection self-play GA exactly as the paper ran it,
with the game chosen by an integer:

  GAME_SLIME    Slime Volleyball through fastvolley.play_game, unchanged
  GAME_DISCMIX  the tunable transitive/cyclic game of lab/games.py

and the archive by `hof_mode`:

  HOF_NONE      the control (fastvolley.run_ga with hof_prob = 0)
  HOF_PARENT    archive as parent (fastvolley.run_ga with hof_prob > 0)
  HOF_TEST      archive as test (algorithms.run_ga_hof_eval)
  HOF_NICHE     archive as test, organised by behaviour instead of time: a
                MAP-Elites-style grid of NICHE_GRID x NICHE_GRID cells over a
                two-number descriptor (the network's mean first two outputs
                on the probe inputs X3, binned within the bounds in gp[5:9]),
                one champion per cell, the newest to land there (WP9)

For GAME_SLIME every mode draws its random numbers in the same order as the
paper's kernel it mirrors, so a lab run reproduces the paper's run bit for
bit; test_repo.py checks this for all three modes and for the population
snapshots. That check is what makes the lab a wrapper of the study rather
than a rewrite of it. HOF_NICHE draws them exactly as HOF_TEST does; only
where a champion is stored differs.
"""

import numpy as np
from numba import njit

from fastvolley import PARAM_COUNT, POLICY_MLP, mlp_forward, play_game

GAME_SLIME = 0
GAME_DISCMIX = 1

HOF_NONE = 0
HOF_PARENT = 1
HOF_TEST = 2
HOF_NICHE = 3
NICHE_GRID = 8

# gp (game parameters) layout for GAME_DISCMIX; see lab/games.py
GP_LAMBDA, GP_ALPHA, GP_BETA, GP_NOISE, GP_TIE = 0, 1, 2, 3, 4
# HOF_NICHE reads the descriptor grid's bounds from four more slots, for any
# game: descriptor 0 is binned over [lo0, hi0], descriptor 1 over [lo1, hi1]
GP_NLO0, GP_NHI0, GP_NLO1, GP_NHI1 = 5, 6, 7, 8


@njit(cache=True)
def discmix_features(p, skill_x, skill_y, style_x):
    """(skill, u0, u1) of one genome.

    skill  how well the network's third output classifies the skill probes
           by a fixed label (+1/-1): mean of label * output, in [-1, 1], 1 only
           when every probe is classified with a saturated output. A property
           of the genome alone, so it orders every population transitively;
    u      the network's mean first two outputs on the style probes: a point
           in [-1, 1]^2 on which the cyclic (disc game) part is played.
    """
    out = np.empty(3)
    sk = 0.0
    for k in range(skill_x.shape[0]):
        mlp_forward(p, skill_x[k], out)
        sk += out[2] * skill_y[k, 0]
    u0 = 0.0
    u1 = 0.0
    for k in range(style_x.shape[0]):
        mlp_forward(p, style_x[k], out)
        u0 += out[0]
        u1 += out[1]
    n = style_x.shape[0]
    return sk / skill_x.shape[0], u0 / n, u1 / n


@njit(cache=True)
def discmix_margin(gp, fr, fl):
    """Expected-margin argument M(right, left): antisymmetric by construction."""
    lam = gp[GP_LAMBDA]
    trans = gp[GP_ALPHA] * (fr[0] - fl[0])
    cyc = gp[GP_BETA] * (fr[1] * fl[2] - fr[2] * fl[1])
    return (1.0 - lam) * trans + lam * cyc


@njit(cache=True)
def discmix_play(gp, skill_x, skill_y, style_x, p_r, p_l):
    """One game: +1 if the right player wins, -1 if it loses, 0 for a tie.

    The outcome is the sign of M + noise, a tie inside +/- GP_TIE. Length is
    reported as 1 (the game has no duration)."""
    a = discmix_features(p_r, skill_x, skill_y, style_x)
    c = discmix_features(p_l, skill_x, skill_y, style_x)
    fr = np.array([a[0], a[1], a[2]])
    fl = np.array([c[0], c[1], c[2]])
    x = discmix_margin(gp, fr, fl) + np.random.normal(0.0, 1.0) * gp[GP_NOISE]
    if x > gp[GP_TIE]:
        return 1, 1
    if x < -gp[GP_TIE]:
        return -1, 1
    return 0, 1


@njit(cache=True)
def niche_cell(p, probes, grid, lo0, hi0, lo1, hi1):
    """Grid cell of a genome's behaviour descriptor.

    The descriptor is the network's mean first two outputs on the probe
    inputs; axis 0 is cut into `grid` equal bins over [lo0, hi0], axis 1 over
    [lo1, hi1], values outside falling into the edge bins."""
    out = np.empty(3)
    d0 = 0.0
    d1 = 0.0
    for k in range(probes.shape[0]):
        mlp_forward(p, probes[k], out)
        d0 += out[0]
        d1 += out[1]
    n = probes.shape[0]
    c0 = int(np.floor((d0 / n - lo0) / (hi0 - lo0) * grid))
    c1 = int(np.floor((d1 / n - lo1) / (hi1 - lo1) * grid))
    c0 = min(max(c0, 0), grid - 1)
    c1 = min(max(c1, 0), grid - 1)
    return c0 * grid + c1


@njit(cache=True)
def play(game, gp, X1, X2, X3, p_r, p_l, w, b, rnn_a, rnn_b, empty):
    """One game between two genomes; score from the right player's view."""
    if game == GAME_SLIME:
        score, length, _ = play_game(p_r, POLICY_MLP, p_l, POLICY_MLP, w, b,
                                     rnn_a, rnn_b, empty, empty, 0, False)
        return score, length
    return discmix_play(gp, X1, X2, X3, p_r, p_l)


@njit(cache=True)
def run(game, gp, X1, X2, X3, seed, n_tournaments, pop_size, sigma, save_every,
        hof_mode, hof_prob, hof_every, hof_capacity, w, b, init_scale,
        pop_every):
    """Ha's GA on `game`, with the archive used as `hof_mode` says.

    Returns champs, streaks, meanlen, ties, hofwins (per checkpoint),
    pops, pop_streaks (per population snapshot; empty if pop_every == 0) and
    arch_size (archived champions at each checkpoint).
    """
    if hof_mode == HOF_NICHE and hof_capacity < NICHE_GRID * NICHE_GRID:
        raise ValueError("HOF_NICHE needs hof_capacity >= NICHE_GRID ** 2")
    if hof_mode == HOF_NICHE and gp.shape[0] <= GP_NHI1:
        raise ValueError("HOF_NICHE needs the grid bounds in gp[5:9]")
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
    # HOF_NICHE: slot of each grid cell in the archive (-1: empty cell)
    cell_slot = -np.ones(NICHE_GRID * NICHE_GRID, dtype=np.int64)

    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)
    ties = np.zeros(n_ckpt)
    hofwins = np.zeros(n_ckpt)
    n_pop = n_tournaments // pop_every if pop_every > 0 else 0
    pops = np.zeros((n_pop, pop_size, PARAM_COUNT), dtype=np.float32)
    pop_streaks = np.zeros((n_pop, pop_size), dtype=np.int64)
    arch_size = np.zeros(n_ckpt, dtype=np.int64)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)

    len_acc = 0.0
    tie_acc = 0.0
    hof_games = 0
    hof_wins = 0
    ck = 0
    pk = 0

    for tournament in range(1, n_tournaments + 1):
        m = np.random.randint(0, pop_size)
        n = np.random.randint(0, pop_size)
        while n == m:
            n = np.random.randint(0, pop_size)

        use_hof = False
        ai = 0
        if hof_mode != HOF_NONE and hof_prob > 0.0 and n_arch > 0:
            if np.random.random() < hof_prob:
                use_hof = True
                ai = np.random.randint(0, n_arch)

        # right player is n (or the archived champion), left player is m
        if use_hof:
            score, length = play(game, gp, X1, X2, X3, archive[ai],
                                 population[m], w, b, rnn_a, rnn_b, empty)
            hof_games += 1
        else:
            score, length = play(game, gp, X1, X2, X3, population[n],
                                 population[m], w, b, rnn_a, rnn_b, empty)
        len_acc += length

        if score == 0:
            tie_acc += 1.0
            for j in range(PARAM_COUNT):
                population[m, j] += np.random.normal(0.0, 1.0) * sigma
        elif score > 0:
            # the right player won
            if use_hof:
                hof_wins += 1
                if hof_mode == HOF_PARENT:
                    # m is replaced by a mutant of the archived genome
                    for j in range(PARAM_COUNT):
                        mutant[j] = archive[ai, j] + np.random.normal(0.0, 1.0) * sigma
                    for j in range(PARAM_COUNT):
                        population[m, j] = mutant[j]
                    winning_streak[m] = archive_streak[ai]
                else:
                    # archive as test: m failed, replaced from the living pool;
                    # n did not play, so its streak does not grow
                    for j in range(PARAM_COUNT):
                        mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
                    for j in range(PARAM_COUNT):
                        population[m, j] = mutant[j]
                    winning_streak[m] = winning_streak[n]
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

        if hof_mode != HOF_NONE and tournament % hof_every == 0:
            rh = np.argmax(winning_streak)
            if hof_mode == HOF_NICHE:
                # one champion per behavioural niche: a new niche takes the
                # next slot, an occupied one is overwritten by the newcomer
                cell = niche_cell(population[rh], X3, NICHE_GRID,
                                  gp[GP_NLO0], gp[GP_NHI0], gp[GP_NLO1],
                                  gp[GP_NHI1])
                if cell_slot[cell] < 0:
                    cell_slot[cell] = n_arch
                    n_arch += 1
                idx = cell_slot[cell]
            else:
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
            arch_size[ck] = n_arch
            len_acc = 0.0
            tie_acc = 0.0
            hof_games = 0
            hof_wins = 0
            ck += 1

        if pop_every > 0 and tournament % pop_every == 0:
            for i in range(pop_size):
                for j in range(PARAM_COUNT):
                    pops[pk, i, j] = np.float32(population[i, j])
                pop_streaks[pk, i] = winning_streak[i]
            pk += 1

    return champs, streaks, meanlen, ties, hofwins, pops, pop_streaks, arch_size
