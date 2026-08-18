# Methods

## 1. Task and environment

We study competitive coevolution in Slime Volleyball, the two-player,
zero-sum, symmetric game introduced as a browser demo by Ha (2015) and later
released as the Gym environment `slimevolleygym` (Ha, 2020). The game is
attractive for this study for three reasons: it is genuinely adversarial, it
has a *frozen external opponent of known strength* (the 2015 champion policy
shipped with the environment), and a full self-play run is small enough that
every claim can be checked end to end.

**Observations.** Each agent receives a 12-dimensional vector — its own
position and velocity, the ball's position and velocity, and the opponent's
position and velocity — expressed in that agent's own frame (the *x* axis is
negated for the left player) and divided by 10.

**Actions.** Three binary buttons: forward, backward, jump. Both may be
pressed; forward and backward cancel.

**Episode.** Each side starts with five lives. An episode ends when a side
loses all five, or after 3,000 timesteps, whichever comes first. The episode
score is points won minus points lost, so it lies in $[-5, +5]$; a $0$ means
the two agents held each other to a draw for the full 3,000 steps.

**Policy class.** Every evolved agent is the same fixed feed-forward network,
`slimevolleylite`: 12 inputs, two hidden layers of 10 units, 3 outputs, $\tanh$
throughout, with a button pressed when its output exceeds 0. The genotype is
the flat weight vector, $d = 273$ parameters. Nothing else evolves: no
topology change, no crossover, no learned representation. This is deliberate.
The question here is what an *improvement signal* can do on its own, so
everything except the signal is held fixed.

**External yardstick.** The 2015 baseline is a small recurrent network that
was itself evolved in the original browser demo — a population of 100 agents,
each playing eight random opponents for 20 seconds of gameplay, left running
for about a day (Ha, 2015). As shipped in `slimevolleygym` it has a 7×15 weight
matrix and 7 biases, i.e. **112 parameters**; the class docstring says 120,
which is worth noting only because it is the kind of number that gets copied
forward without being counted. Its 15 inputs are 8 game observations plus its
own 7 outputs fed back — so the 2015 champion never sees its opponent's
position or velocity at all. It is never used as a training signal anywhere in
this work; it appears only after training, as a frozen measuring instrument,
and its recurrent state is carried across evaluation episodes exactly as the
reference evaluation code does.

## 2. The algorithm

The control condition is a faithful replication of Ha's 2020
tournament-selection self-play GA (`training_scripts/train_ga_selfplay.py`),
which is a deliberately minimal coevolutionary algorithm:

```
population P ← 128 genomes, each ~ N(0, 0.5²) elementwise
winning_streak[i] ← 0

repeat 500,000 times:
    draw m ≠ n uniformly from P
    score ← play one full game: n on the right, m on the left

    if score = 0:                                  # tie
        P[m] ← P[m] + σ·N(0, I)
    else if score > 0:                              # n won
        P[m] ← P[n] + σ·N(0, I)
        winning_streak[m] ← winning_streak[n];  winning_streak[n] += 1
    else:                                           # m won
        P[n] ← P[m] + σ·N(0, I)
        winning_streak[n] ← winning_streak[m];  winning_streak[m] += 1

champion ← P[argmax winning_streak]
```

with $\sigma = 0.1$. Three properties of this algorithm matter for everything
that follows.

*Fitness is implicit and relative.* There is no fitness function. An
individual survives by beating whichever peer it happened to be drawn against.
Nothing in the loop refers to an absolute standard of play, which is exactly
why a population of uniformly terrible agents can still generate a useful
gradient: victories are always available against someone.

*There is no memory.* The only state carried forward is the current
population. A skill that no current opponent punishes is not selected for and
can be lost — the textbook argument for a hall-of-fame archive
(Rosin & Belew, 1997; Risi, Tang, Ha & Miikkulainen, 2025, ch. 7.2).

*The exported champion is a proxy.* Ha selects the individual with the longest
winning lineage, "without actually computing who is best to save time". The
counter is inherited by the loser on every replacement, so it measures the age
of a lineage, not the quality of an individual. Section 3 of the analysis
tests what that shortcut costs.

## 3. Interventions

Each condition changes exactly one thing about the loop above.

**Hall of fame (`hof-0.25`, `hof-0.50`).** Every 1,000 tournaments the current
champion is copied into a FIFO archive of capacity 64. With probability
$p \in \{0.25, 0.5\}$ the second contestant is drawn from the archive instead
of from the population. Archived genomes are immutable, so the update rule
specialises as follows: if the archived opponent wins, the population member is
overwritten by a mutant of the archived genome and inherits its streak counter
— exactly Ha's rule, treating the archive entry as if it were in the pool; if
the population member wins, nothing is overwritten and its streak counter
grows. This is the smallest change that gives selection a memory.

**Mutation scale (`sigma-0.05`, `sigma-0.20`).** $\sigma$ halved and doubled.
This asks whether the instability of self-play here is really a coevolutionary
phenomenon or simply a step-size that is too large for the landscape.

**Population size (`pop-32`, `pop-512`).** The pool shrunk and grown fourfold
at a fixed game budget. Larger populations dilute selection pressure per
individual but hold more diversity; this is the collective-system axis of the
design.

## 4. A compiled environment, validated against the reference

The reference environment runs at roughly 10 games/second on one core, which
puts a 500,000-game run at about twelve core-hours. That is affordable once
and not affordable for a seeded, multi-condition design — which is precisely
why the first version of this study had a single run.

We therefore transcribed the environment, the MLP policy and the baseline RNN
into `fastvolley.py` and compiled them with numba. The port preserves the
reference control flow statement for statement, including the parts that look
like bugs but are load-bearing: the left player receives the *right* player's
observation on the first step of every episode; observations are not refreshed
on a frame in which a point is scored; and the ground test in the collision
handler returns before the ceiling and fence tests run.

Three deviations are deliberate and documented: the random-number stream (gym's
PCG64 versus numba's Mersenne Twister), the way the tournament pair is drawn
(`np.random.choice(pop, 2, replace=False)` versus rejection sampling of two
distinct integers), and the floating-point details of the network forward pass
(NumPy dispatches to BLAS and a SIMD `tanh`; the compiled version uses a plain
loop and libm, which differ in the last one or two units in the last place).

The last of these cannot change a trajectory unless it flips the *sign* of a
network output, because the game only ever reads `action[i] > 0`. The
validation harness (`validate_fastvolley.py`) therefore drives both
implementations from an identical stream of serve velocities and compares them
step by step — ball position and velocity, both agents' positions, the frame at
which every rally ends, the final score and the episode length — across four
policy populations chosen to exercise different parts of the state space:
freshly initialised genomes (short rallies, many scoring frames), a trained
champion against a fresh genome, champion against earlier champion, and the
champion against the 2015 baseline (3,000-step rallies with thousands of
collisions). Results are in Appendix A.

We additionally check the port at the level of a *training trajectory* rather
than a single game (`resume_fast.py`): the reference run's committed population
snapshot is continued in both implementations, and the resulting learning
curves are compared.

## 5. Evaluation protocol

The protocol below was fixed before any run in the matrix was launched and is
recorded in `results/matrix/protocol.json`.

**Checkpoints.** Every run exports a champion every 5,000 tournaments, giving
100 checkpoints per run.

**Sweep.** Every checkpoint plays 200 episodes against the 2015 baseline at
evaluation seed 20260901. This produces the learning curves.

**Held-out re-scoring.** The *peak* checkpoint of a run is by construction a
selected maximum, so its sweep score is optimistically biased. Both the final
and the peak champion of every run are therefore re-scored over 1,000 episodes
at seed 20260902, disjoint from the sweep seed. Every headline number in the
results is a held-out number.

**Metrics.** All are computed on the sweep curve; the window is the last
100,000 games, i.e. the last 20 of the 100 checkpoints.

| Metric | Definition | Question |
|---|---|---|
| `final` | score of the $t=500{,}000$ champion | how good is the run's endpoint (no selection)? |
| `peak` | best checkpoint score | how good does it ever get? |
| `above_parity` | fraction of the 100 checkpoints scoring $> 0$ | how often is it better than the 2015 expert? |
| `late_mean` | mean checkpoint score over the window | how good is it once trained? |
| `volatility` | mean $\lvert\Delta\rvert$ between consecutive checkpoints in the window | how much does it swing? |
| `drawdown` | mean gap between best-so-far and current, in the window | how much of its best does it hold? |
| `t_parity` | first checkpoint above parity | how quickly does internal progress transfer? |

`final` is the primary endpoint; it is the only one with no checkpoint
selection in it. The others are reported as descriptive, and we do not correct
for multiplicity across them — with 3–6 runs per condition the honest reading
of any single secondary comparison is "consistent with" rather than
"demonstrates".

**Statistics.** Sample sizes are 3–6 runs per condition, which rules out
anything asymptotic. Condition comparisons use the exact two-sided
Mann–Whitney $U$ test, enumerating all $\binom{n+m}{n}$ label assignments, with
Cliff's $\delta$ as the effect size and percentile bootstrap intervals for
means (`stats_utils.py`, written out rather than imported so every number can
be audited). With $n = m = 6$ the smallest attainable two-sided $p$-value is
$0.002$; with $n = 3$, $m = 6$ it is $0.024$.

**Coevolution-specific measurements.** Three questions are invisible to the
external yardstick and are measured directly (`coevolution_analysis.py`):

1. *Intransitivity.* Within each run, checkpoints spaced 50,000 games apart
   play a round robin, 50 games per pair, both court sides. We fit
   Bradley–Terry ratings on the Elo scale and count *cyclic triads* — triples
   where A beats B beats C beats A — among pairs whose mean margin exceeds a
   0.25-point deadband. A transitive tournament with rising Elo means the
   swings in the external curve are noise in the champion proxy; cycles mean
   genuine coevolutionary cycling.
2. *Cross-run strength.* Every run's final champion plays every other run's
   final champion. This ranks conditions with no frozen opponent involved, so
   it cannot be gamed by a policy that happens to specialise against the 2015
   baseline.
3. *Champion-selection cost.* The control runs snapshot their full population
   every 50,000 games. Every member is scored against the baseline, and the
   individual the streak counter exports is compared with the best individual
   in the same pool.

## 6. Compute

The experiment matrix is 27 runs $\times$ 500,000 games $=$ 13.5 million games,
plus roughly 1.1 million evaluation episodes. It ran on four cores of a single
cloud container: three cores for the matrix and its evaluation, one core
carrying the reference run on the unmodified `slimevolleygym` environment to
500,000 games. Wall-clock for the matrix is a few hours; the same design on the
reference environment would have taken roughly ten days on the same machine.
