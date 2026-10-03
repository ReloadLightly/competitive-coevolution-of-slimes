"""
shadow.py — the control GA replayed with shadow counters (WP13).

In Ha's GA the winning-streak counter decides only which individual is
exported. In the control (no archive) it never feeds back into reproduction,
so the same evolutionary history can carry other counters beside it, and each
can be asked which member it would export.

This module replays the control GA exactly: `run_ga_shadow` is a copy of the
control branch of `fastvolley.run_ga_with_pops` (`hof_prob == 0`), drawing the
same random numbers in the same order and calling the same compiled game. On
top it keeps four counters per member, the 2 x 2 of the two bookkeeping
choices in Ha's rule:

                             a tie mutates the genotype   a tie mutates it and
                             and the count is kept        the count restarts
    inherited at birth       INH (Ha's rule)              INH_RESET
    starts at 0 at birth     OWN (wins since birth)       CUR (wins of the
                                                          current genotype)

Ha's rule, for the record: two members m, n play; on a tie m is mutated in
place; otherwise the loser is overwritten by a mutant of the winner, takes
over the winner's count, and the winner's count grows by one. A member that
loses is replaced, so OWN counts the games an individual has won since it
was born, and CUR those it has won since its genotype last changed.

INH equals fastvolley's winning_streak exactly, and champions and
populations are bit-identical to the frozen kernel's for the same seed
(test_repo.py): the counters read the history, they never change it. Each
counter exports the first member with its maximum, as numpy's argmax does in
Ha's rule; slot indices carry no information about skill, so the tie-break
is arbitrary with respect to it.

The frozen files (fastvolley.py and the rest, CLAUDE.md rule 1) are not
touched.
"""

import numpy as np
from numba import njit

import fastvolley as fv
from fastvolley import PARAM_COUNT, POLICY_MLP, play_game

INH, INH_RESET, OWN, CUR = 0, 1, 2, 3
RULES = ("inherited", "inherited-reset", "own", "current")
N_RULES = len(RULES)


@njit(cache=True)
def run_ga_shadow(seed, n_tournaments, pop_size, sigma, save_every, w, b,
                  init_scale, pop_every):
    """The control GA with shadow counters.

    Returns, beside the frozen kernel's five outputs (champs, streaks,
    meanlen, pops, pop_streaks; Ha's rule):
      picks       (n_ckpt, N_RULES, PARAM_COUNT)          each counter's pick
      pick_idx    (n_ckpt, N_RULES) int64                 its slot
      pick_count  (n_ckpt, N_RULES) int64                 its count
      tie_frac    (n_ckpt,)                               share of tied games
      pop_counts  (n_pop, N_RULES, pop_size) int64        every count, at
                                                          every snapshot
      pop_born    (n_pop, pop_size) int64                 tournament at which
                                                          the member was born
                                                          (0: initial)
      pop_changed (n_pop, pop_size) int64                 tournament at which
                                                          its genotype last
                                                          changed
      pop_games   (n_pop, pop_size) int64                 games it has played
                                                          since birth
    """
    np.random.seed(seed)
    population = np.empty((pop_size, PARAM_COUNT))
    for i in range(pop_size):
        for j in range(PARAM_COUNT):
            population[i, j] = np.random.normal(0.0, 1.0) * init_scale
    winning_streak = np.zeros(pop_size, dtype=np.int64)

    cnt = np.zeros((N_RULES, pop_size), dtype=np.int64)
    born = np.zeros(pop_size, dtype=np.int64)
    changed = np.zeros(pop_size, dtype=np.int64)
    games = np.zeros(pop_size, dtype=np.int64)

    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)
    picks = np.zeros((n_ckpt, N_RULES, PARAM_COUNT))
    pick_idx = np.zeros((n_ckpt, N_RULES), dtype=np.int64)
    pick_count = np.zeros((n_ckpt, N_RULES), dtype=np.int64)
    tie_frac = np.zeros(n_ckpt)

    n_pop = n_tournaments // pop_every
    pops = np.zeros((n_pop, pop_size, PARAM_COUNT), dtype=np.float32)
    pop_streaks = np.zeros((n_pop, pop_size), dtype=np.int64)
    pop_counts = np.zeros((n_pop, N_RULES, pop_size), dtype=np.int64)
    pop_born = np.zeros((n_pop, pop_size), dtype=np.int64)
    pop_changed = np.zeros((n_pop, pop_size), dtype=np.int64)
    pop_games = np.zeros((n_pop, pop_size), dtype=np.int64)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)
    len_acc = 0.0
    tie_acc = 0.0
    ck = 0
    pk = 0

    for tournament in range(1, n_tournaments + 1):
        m = np.random.randint(0, pop_size)
        n = np.random.randint(0, pop_size)
        while n == m:
            n = np.random.randint(0, pop_size)

        # right player is n, left player is m (as in the frozen kernel)
        score, length, _ = play_game(population[n], POLICY_MLP,
                                     population[m], POLICY_MLP, w, b,
                                     rnn_a, rnn_b, empty, empty, 0, False)
        len_acc += length
        games[m] += 1
        games[n] += 1

        if score == 0:
            tie_acc += 1.0
            for j in range(PARAM_COUNT):
                population[m, j] += np.random.normal(0.0, 1.0) * sigma
            # m's genotype changed in place: INH and OWN keep the count,
            # INH_RESET and CUR restart it
            cnt[INH_RESET, m] = 0
            cnt[CUR, m] = 0
            changed[m] = tournament
        elif score > 0:
            for j in range(PARAM_COUNT):
                mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
            for j in range(PARAM_COUNT):
                population[m, j] = mutant[j]
            winning_streak[m] = winning_streak[n]
            winning_streak[n] += 1
            # m is reborn as a mutant of the winner n
            cnt[INH, m] = cnt[INH, n]
            cnt[INH_RESET, m] = cnt[INH_RESET, n]
            cnt[OWN, m] = 0
            cnt[CUR, m] = 0
            for r in range(N_RULES):
                cnt[r, n] += 1
            born[m] = tournament
            changed[m] = tournament
            games[m] = 0
        else:
            for j in range(PARAM_COUNT):
                mutant[j] = population[m, j] + np.random.normal(0.0, 1.0) * sigma
            for j in range(PARAM_COUNT):
                population[n, j] = mutant[j]
            winning_streak[n] = winning_streak[m]
            winning_streak[m] += 1
            cnt[INH, n] = cnt[INH, m]
            cnt[INH_RESET, n] = cnt[INH_RESET, m]
            cnt[OWN, n] = 0
            cnt[CUR, n] = 0
            for r in range(N_RULES):
                cnt[r, m] += 1
            born[n] = tournament
            changed[n] = tournament
            games[n] = 0

        if tournament % save_every == 0:
            rh = np.argmax(winning_streak)
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[rh, j]
            streaks[ck] = winning_streak[rh]
            meanlen[ck] = len_acc / save_every
            tie_frac[ck] = tie_acc / save_every
            for r in range(N_RULES):
                i = np.argmax(cnt[r])
                pick_idx[ck, r] = i
                pick_count[ck, r] = cnt[r, i]
                for j in range(PARAM_COUNT):
                    picks[ck, r, j] = population[i, j]
            len_acc = 0.0
            tie_acc = 0.0
            ck += 1

        if tournament % pop_every == 0:
            for i in range(pop_size):
                for j in range(PARAM_COUNT):
                    pops[pk, i, j] = np.float32(population[i, j])
                pop_streaks[pk, i] = winning_streak[i]
                for r in range(N_RULES):
                    pop_counts[pk, r, i] = cnt[r, i]
                pop_born[pk, i] = born[i]
                pop_changed[pk, i] = changed[i]
                pop_games[pk, i] = games[i]
            pk += 1

    return (champs, streaks, meanlen, pops, pop_streaks, picks, pick_idx,
            pick_count, tie_frac, pop_counts, pop_born, pop_changed, pop_games)


def replay(seed, n_tournaments, pop_size=128, sigma=0.10, save_every=5_000,
           init_scale=0.5, pop_every=50_000):
    """Run the shadow kernel with the control's parameters; returns a dict."""
    w, b = fv.baseline_arrays()
    out = run_ga_shadow(seed, n_tournaments, pop_size, sigma, save_every, w, b,
                        init_scale, pop_every)
    keys = ("champs", "streaks", "meanlen", "pops", "pop_streaks", "picks",
            "pick_idx", "pick_count", "tie_frac", "pop_counts", "pop_born",
            "pop_changed", "pop_games")
    return dict(zip(keys, out))
