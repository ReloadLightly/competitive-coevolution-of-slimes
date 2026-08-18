"""
algorithms.py — two further neuroevolutionary approaches, for comparison with
Ha's 2020 tournament-selection GA.

The rest of this study holds the search algorithm fixed and varies its knobs.
That answers questions about the selection *signal* but not about the
selection *machinery*, so this module adds two genuinely different families.
All three share the identical policy class (the fixed 273-parameter
`slimevolleylite` MLP) and the identical environment, so the only thing that
differs is how a population is turned into the next population.

1. `run_ga2015` — a generational GA in the style of Ha's *original* 2015
   experiment: a population of 100, each agent playing several random peers per
   generation, the top 20% retained, the rest refilled by crossover and
   mutation. This is the algorithm that actually produced the baseline policy
   this study measures against, and it differs from the 2020 GA in three ways
   at once: it is generational rather than steady-state, it ranks by an
   explicitly *computed* average fitness rather than a winning-streak proxy,
   and it has crossover.

   Two documented deviations from Ha (2015): the policy is the fixed
   feed-forward MLP rather than a recurrent net (so that the comparison isolates
   the search algorithm), and each agent plays ten random peers per generation
   rather than eight, which makes a generation cost exactly 500 games and lines
   the checkpoint grid up with the rest of the study.

2. `run_es` — self-play evolution strategy in the OpenAI-ES style: one mean
   vector, mirrored Gaussian perturbations, fitness from games *among the
   perturbations themselves*, and a rank-normalised gradient estimate. This
   family is interesting here for a specific reason: it has no champion
   selection problem at all. The reported individual is the distribution mean,
   not a member of the population, so whatever volatility the winning-streak
   proxy contributes elsewhere is absent by construction.
"""

import numpy as np
from numba import njit

from fastvolley import PARAM_COUNT, POLICY_MLP, play_game


@njit(cache=True)
def run_ga2015(seed, n_games, pop_size, n_opponents, n_elite, sigma,
               save_every_games, w, b, init_scale):
    """Generational GA with computed fitness, elitism and uniform crossover.

    Each generation plays `pop_size * n_opponents / 2` games between randomly
    drawn pairs; every game's point margin is credited to both participants.
    Fitness is the mean margin over the games an individual happened to play.

    Returns (champs, fitness_of_champ, meanlen, games_per_ckpt) with one row per
    checkpoint, the champion being the highest-fitness individual of the
    generation in which the checkpoint falls.
    """
    np.random.seed(seed)
    population = np.empty((pop_size, PARAM_COUNT))
    for i in range(pop_size):
        for j in range(PARAM_COUNT):
            population[i, j] = np.random.normal(0.0, 1.0) * init_scale

    games_per_gen = (pop_size * n_opponents) // 2
    n_gen = n_games // games_per_gen
    gens_per_ckpt = save_every_games // games_per_gen
    n_ckpt = n_gen // gens_per_ckpt

    champs = np.zeros((n_ckpt, PARAM_COUNT))
    champ_fit = np.zeros(n_ckpt)
    meanlen = np.zeros(n_ckpt)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    newpop = np.empty((pop_size, PARAM_COUNT))
    fitness = np.empty(pop_size)
    played = np.empty(pop_size)
    len_acc = 0.0
    len_n = 0
    ck = 0

    for gen in range(1, n_gen + 1):
        for i in range(pop_size):
            fitness[i] = 0.0
            played[i] = 0.0

        for _ in range(games_per_gen):
            i = np.random.randint(0, pop_size)
            j = np.random.randint(0, pop_size)
            while j == i:
                j = np.random.randint(0, pop_size)
            score, length, _ = play_game(population[i], POLICY_MLP,
                                         population[j], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
            # i played on the right, so the score is from i's point of view
            fitness[i] += score
            fitness[j] -= score
            played[i] += 1.0
            played[j] += 1.0
            len_acc += length
            len_n += 1

        for i in range(pop_size):
            if played[i] > 0.0:
                fitness[i] /= played[i]

        order = np.argsort(-fitness)

        # elitism: the top n_elite survive untouched
        for e in range(n_elite):
            src = order[e]
            for j in range(PARAM_COUNT):
                newpop[e, j] = population[src, j]
        # the rest: uniform crossover of two distinct elites, then mutation
        for c in range(n_elite, pop_size):
            pa = order[np.random.randint(0, n_elite)]
            pb = order[np.random.randint(0, n_elite)]
            while pb == pa and n_elite > 1:
                pb = order[np.random.randint(0, n_elite)]
            for j in range(PARAM_COUNT):
                if np.random.random() < 0.5:
                    g = population[pa, j]
                else:
                    g = population[pb, j]
                newpop[c, j] = g + np.random.normal(0.0, 1.0) * sigma
        for i in range(pop_size):
            for j in range(PARAM_COUNT):
                population[i, j] = newpop[i, j]

        if gen % gens_per_ckpt == 0 and ck < n_ckpt:
            best = order[0]
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[0, j]  # elite 0 == the best genome
            champ_fit[ck] = fitness[best]
            meanlen[ck] = len_acc / len_n if len_n > 0 else 0.0
            len_acc = 0.0
            len_n = 0
            ck += 1

    return champs, champ_fit, meanlen, games_per_gen


@njit(cache=True)
def run_es(seed, n_games, n_candidates, n_opponents, sigma, alpha,
           save_every_games, w, b, init_scale):
    """Self-play evolution strategy with mirrored sampling and rank shaping.

    One iteration:
      * draw n_candidates/2 perturbations eps and form the mirrored pairs
        theta +/- sigma*eps;
      * play n_candidates * n_opponents / 2 games between randomly drawn
        candidates, crediting each game's margin to both;
      * shape the fitnesses into centred ranks in [-0.5, 0.5];
      * theta += alpha / (n_candidates * sigma) * sum_i u_i * (candidate_i - theta)/sigma

    The reported champion is theta itself.
    """
    np.random.seed(seed)
    theta = np.empty(PARAM_COUNT)
    for j in range(PARAM_COUNT):
        theta[j] = np.random.normal(0.0, 1.0) * init_scale

    half = n_candidates // 2
    games_per_iter = (n_candidates * n_opponents) // 2
    n_iter = n_games // games_per_iter
    iters_per_ckpt = save_every_games // games_per_iter
    n_ckpt = n_iter // iters_per_ckpt

    champs = np.zeros((n_ckpt, PARAM_COUNT))
    spread = np.zeros(n_ckpt)
    meanlen = np.zeros(n_ckpt)

    eps = np.empty((half, PARAM_COUNT))
    cand = np.empty((n_candidates, PARAM_COUNT))
    fitness = np.empty(n_candidates)
    played = np.empty(n_candidates)
    ranks = np.empty(n_candidates)
    grad = np.empty(PARAM_COUNT)
    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    len_acc = 0.0
    len_n = 0
    ck = 0

    for it in range(1, n_iter + 1):
        for h in range(half):
            for j in range(PARAM_COUNT):
                eps[h, j] = np.random.normal(0.0, 1.0)
        for h in range(half):
            for j in range(PARAM_COUNT):
                cand[h, j] = theta[j] + sigma * eps[h, j]
                cand[half + h, j] = theta[j] - sigma * eps[h, j]

        for i in range(n_candidates):
            fitness[i] = 0.0
            played[i] = 0.0
        for _ in range(games_per_iter):
            i = np.random.randint(0, n_candidates)
            j = np.random.randint(0, n_candidates)
            while j == i:
                j = np.random.randint(0, n_candidates)
            score, length, _ = play_game(cand[i], POLICY_MLP,
                                         cand[j], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
            fitness[i] += score
            fitness[j] -= score
            played[i] += 1.0
            played[j] += 1.0
            len_acc += length
            len_n += 1
        for i in range(n_candidates):
            if played[i] > 0.0:
                fitness[i] /= played[i]

        # centred ranks in [-0.5, 0.5]: robust to the scale of a relative signal
        order = np.argsort(fitness)
        for r in range(n_candidates):
            ranks[order[r]] = r / (n_candidates - 1.0) - 0.5

        for j in range(PARAM_COUNT):
            grad[j] = 0.0
        for h in range(half):
            u = ranks[h] - ranks[half + h]      # mirrored antithetic estimate
            for j in range(PARAM_COUNT):
                grad[j] += u * eps[h, j]
        scale = alpha / (n_candidates * sigma)
        for j in range(PARAM_COUNT):
            theta[j] += scale * grad[j]

        if it % iters_per_ckpt == 0 and ck < n_ckpt:
            for j in range(PARAM_COUNT):
                champs[ck, j] = theta[j]
            spread[ck] = np.std(fitness)
            meanlen[ck] = len_acc / len_n if len_n > 0 else 0.0
            len_acc = 0.0
            len_n = 0
            ck += 1

    return champs, spread, meanlen, games_per_iter


@njit(cache=True)
def run_ga_hof_eval(seed, n_tournaments, pop_size, sigma, save_every,
                    hof_prob, hof_every, hof_capacity, w, b, init_scale):
    """Ha's GA with an archive used as a TEST, never as a parent.

    The first archive design in this study (`fastvolley.run_ga` with
    hof_prob > 0) applied Ha's replacement rule verbatim to archive games: when
    the archived genome won, the population member was overwritten by a mutant
    *of the archived genome*. That injects old genetic material back into the
    pool on roughly p * P(archive wins) of all games, which is a standing
    regression pressure, and it abolished learning outright.

    That is not how a hall of fame is used in the literature (Rosin & Belew,
    1997): the archive supplies opponents against which fitness is measured,
    and archive members are never parents. This kernel implements that reading:

      * draw m != n from the population, as usual;
      * with probability hof_prob the OPPONENT of m is an archived champion
        instead of n;
      * archive game, m loses -> m is overwritten by a mutant of n, exactly as
        if n had won. Failing a test the living pool is expected to pass costs
        m its slot, and the replacement genes come from the pool;
      * archive game, m wins -> nothing is overwritten, m's streak grows;
      * ordinary game -> unchanged.

    Genetic material therefore never leaves the living population, and the
    archive's only role is to decide who is culled.
    """
    np.random.seed(seed)
    population = np.empty((pop_size, PARAM_COUNT))
    for i in range(pop_size):
        for j in range(PARAM_COUNT):
            population[i, j] = np.random.normal(0.0, 1.0) * init_scale
    winning_streak = np.zeros(pop_size, dtype=np.int64)

    archive = np.zeros((hof_capacity, PARAM_COUNT))
    n_arch = 0
    arch_ptr = 0

    n_ckpt = n_tournaments // save_every
    champs = np.zeros((n_ckpt, PARAM_COUNT))
    streaks = np.zeros(n_ckpt, dtype=np.int64)
    meanlen = np.zeros(n_ckpt)
    archwin = np.zeros(n_ckpt)

    rnn_a = np.zeros(7)
    rnn_b = np.zeros(7)
    empty = np.zeros(1)
    mutant = np.empty(PARAM_COUNT)
    len_acc = 0.0
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

        if use_hof:
            # archive plays on the right, the population member on the left
            score, length, _ = play_game(archive[ai], POLICY_MLP,
                                         population[m], POLICY_MLP, w, b,
                                         rnn_a, rnn_b, empty, empty, 0, False)
            hof_games += 1
            len_acc += length
            if score > 0:
                # m failed the test: replaced from the LIVING pool, not the archive
                hof_wins += 1
                for j in range(PARAM_COUNT):
                    mutant[j] = population[n, j] + np.random.normal(0.0, 1.0) * sigma
                for j in range(PARAM_COUNT):
                    population[m, j] = mutant[j]
                winning_streak[m] = winning_streak[n]
                winning_streak[n] += 1
            elif score < 0:
                winning_streak[m] += 1
            else:
                for j in range(PARAM_COUNT):
                    population[m, j] += np.random.normal(0.0, 1.0) * sigma
        else:
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

        if tournament % hof_every == 0:
            rh = np.argmax(winning_streak)
            idx = arch_ptr
            if n_arch < hof_capacity:
                idx = n_arch
                n_arch += 1
            else:
                arch_ptr = (arch_ptr + 1) % hof_capacity
            for j in range(PARAM_COUNT):
                archive[idx, j] = population[rh, j]

        if tournament % save_every == 0:
            rh = np.argmax(winning_streak)
            for j in range(PARAM_COUNT):
                champs[ck, j] = population[rh, j]
            streaks[ck] = winning_streak[rh]
            meanlen[ck] = len_acc / save_every
            archwin[ck] = hof_wins / hof_games if hof_games > 0 else 0.0
            len_acc = 0.0
            hof_games = 0
            hof_wins = 0
            ck += 1

    return champs, streaks, meanlen, archwin
