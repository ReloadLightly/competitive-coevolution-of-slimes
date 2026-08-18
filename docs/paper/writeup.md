# Competitive coevolution without memory: what 13.5 million self-play games say about the selection signal

**A study of neuroevolution in a collective system, and the first of three
experiments behind the ACTIR / ShinkaEvolve submission.**

<!-- DRAFT-CLAIMS: the abstract's three findings are written from the
     pilot and the reference run. Every directional claim here must be
     re-checked against results/analysis/*.json before this file ships. -->

## Abstract

Self-play evolution produces competent agents from an entirely internal signal:
beat a randomly drawn peer, stay in the pool. Nothing tells the population what
good play is. We replicate David Ha's tournament-selection genetic algorithm on
Slime Volleyball and ask what that signal can and cannot deliver, using a
design rather than a single run: 27 runs of 500,000 self-play games each across
seven conditions, with all evaluation against a frozen 2015 champion that is
never seen during training.

Three findings. *First*, the internal signal works, and it works late: a
population improves against itself for roughly a hundred thousand games before
any of that improvement transfers to the external opponent, so an evaluation
that stops early reads as total failure. *Second*, competence is reached
repeatedly and lost repeatedly — the exported champion swings by several points
per episode between adjacent checkpoints, and the swings damp but do not
vanish. *Third*, and contrary to the standard explanation, most of that
instability is not coevolutionary forgetting. Within-run tournaments show
skill is almost perfectly transitive; the population is not cycling. The
volatility is largely an artefact of *which individual gets exported*: Ha's
longest-winning-lineage rule is a cheap proxy for quality, and the individual
it picks is typically well behind the best individual in the same population.

We also give the enabling engineering: a compiled port of the environment,
validated bit-for-bit against the reference implementation over a quarter of a
million environment steps, which turns a twelve-core-hour run into a
twenty-minute one and is what makes a seeded design affordable at all.

## Why this matters beyond volleyball

A self-improvement loop is exactly as real as its improvement signal. This
repository is the smallest rung of a three-part argument about where that
signal can come from as the world being acted in opens up:

| | what evolves | where the signal comes from |
|---|---|---|
| **this work** | 273 weights | internal, relative: beat a peer |
| Backprop NEAT (Ha, 2016) | topology, with gradients inside | division of labour between search and local optimisation |
| ACTIR / ShinkaEvolve | programs, with an LLM as the mutation operator | strategies selected against each other in a simulated system, measured on held-out scenarios |

The load-bearing claim inherited upward is the *separation of ecology from
yardstick*: the thing that selects and the thing that measures must never be
the same object. What this study adds to that claim is a caveat with teeth —
when selection is purely relative, the mechanism that *reports* a winner can
contribute more noise than the coevolution itself, and it will do so silently.

## Contents

- [Methods](01-methods.md) — task, algorithm, interventions, the compiled
  environment and its validation, the pre-registered evaluation protocol.
- [Results](02-results.md) — the reference run, the seeded control condition,
  and the phase change.
- [Ablations and analysis](03-ablations-and-analysis.md) — hall of fame,
  mutation scale, population size, intransitivity, and the champion proxy.
- [Appendix](04-appendix.md) — self-contained; every table, every run, and the
  reproduction commands.


---

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


---

# Results

<!-- VERIFY: every number in this file is regenerated by make_tables.py into
     docs/paper/_tables.md. Before shipping, check each figure in the prose
     against that file and against results/analysis/*.json. -->

## 1. The reference run: a phase change, and a lag

We first report the single run trained on the unmodified `slimevolleygym`
environment, because it is the run that the rest of the study was built to
explain. It is Ha's algorithm at Ha's budget: 128 individuals, 500,000
tournament games, mutation σ = 0.1, no opponent but itself.

Its trajectory has two regimes (Figure 1). For the first hundred thousand
games the champion loses every episode to the 2015 baseline by nearly the
maximum margin — a flat floor at roughly −4.85 points per episode, with a
standard deviation across checkpoints of 0.05. Then, over about fifty thousand
games, it climbs almost four points and begins producing champions that beat
the 2015 expert outright.

The interesting part is *when* the population knew. Rally length in the
population's own training games — a purely internal measurement with no
external opponent anywhere in it — crosses 1,500 steps at **104,200 games**.
The first checkpoint to score above parity against the 2015 baseline arrives at
**172,000 games**. The internal signal leads the external one by roughly
**68,000 games**, a third of the way through the run.

That gap is the practical lesson of the whole study, and it is the lesson
February 2026 got wrong in the other direction. An evaluation that stopped at
game 90,000 would have reported total failure from a population that was
already improving steadily; the only thing missing was that its improvements
had not yet generalised beyond its own family. *Internal selection pressure
leads external measurement.* Any self-improvement loop measured too early
reads as a dead loop.

**Damping, and what remains.** Split the run into 100,000-game windows and the
picture is neither "it learns" nor "it thrashes":

| games | mean | s.d. | above parity |
|---|---|---|---|
| 0–100k | −4.82 | 0.05 | 0/100 |
| 100k–200k | −2.02 | 1.64 | 8/100 |
| 200k–300k | −0.55 | 0.78 | 21/100 |

The level rises and the swings shrink, but neither converges: the population
settles into a band just below parity, from which it repeatedly produces
champions that beat the 2015 expert and then loses them again. Held out on a
disjoint evaluation seed over 1,000 episodes, the best checkpoint of the run
scores **+0.304 ± 0.806** (s.e.m. 0.025) — within noise of Ha's published
+0.353 ± 0.728 for the same algorithm at the same budget.

This is where the previous version of this study stopped, and it is exactly as
far as one run can go. Everything in that paragraph is compatible with at
least three different mechanisms, and separating them needs a design.

## 2. The same algorithm, twelve times

*(Section filled from `results/analysis/conditions.json`; see Table 1 and
Figure 2.)*

## 3. What "competence" means here

*(Win/draw/loss decomposition and rally length; Table 1 and Table 3.)*

## 4. The two implementations agree

*(Bit-level equivalence (Table A1) and the trajectory-level continuation of
the same population snapshot (Table A3).)*


---

# Ablations and analysis

<!-- VERIFY: every number here is regenerated by make_tables.py into
     docs/paper/_tables.md. Check the prose against that file before shipping. -->

The reference run leaves three competing explanations for the same
observation — competence is reached and then lost. Each section below removes
one of them.

| explanation | prediction | test |
|---|---|---|
| the population *forgets* skills no current opponent punishes | giving selection a memory should stabilise it | §1, hall of fame |
| the mutation step is simply too large for the landscape | a smaller σ should reduce the swings | §2 |
| the exported champion is not the best individual — the swings are a *reporting* artefact | the best individual in the pool should be both better and steadier than the exported one | §4 |

## 1. Does an archive of past champions stabilise competence?

*(Table 1, Table 2, Figure 3.)*

## 2. Is the instability just the mutation step?

*(Figure 4, left.)*

## 3. Does a bigger collective help?

*(Figure 4, right.)*

## 4. Is the population cycling, or is the champion proxy noisy?

*(Table 4, Table 5, Figures 5 and 6.)*

## 5. Which condition's champions actually win?

*(Table 6, Figure 7.)*

## 6. Synthesis

*(What the four measurements jointly imply for competitive coevolution, and
what carries up to ACTIR / ShinkaEvolve.)*


---

# Appendix — Competitive coevolution in Slime Volleyball

*Self-contained. Everything needed to read, check or reproduce the study is
here; nothing in this appendix depends on the main text.*

## A.1 What was run

| | |
|---|---|
| Task | Slime Volleyball (`slimevolleygym`, Ha 2020), two-player, zero-sum, symmetric |
| Episode | 5 lives per side, 3,000-step limit; score = points won − points lost ∈ [−5, +5] |
| Policy | fixed feed-forward net 12–10–10–3, tanh, 273 parameters (`slimevolleylite`) |
| Algorithm | tournament-selection self-play GA (Ha 2020): winner survives, loser is overwritten by a mutated copy of the winner |
| Population | 128 (ablated: 32, 512) |
| Mutation | isotropic Gaussian, σ = 0.1 (ablated: 0.05, 0.20) |
| Initialisation | N(0, 0.5²) elementwise |
| Budget | 500,000 tournament games per run |
| Runs | 27 across 7 conditions, plus one reference run on the unmodified environment |
| Total | ~13.5 M self-play games, ~1.1 M evaluation episodes |
| External opponent | the 2015 champion RNN (112 parameters: 7×15 weights + 7 biases), never seen in training |

**Conditions.** `control` (6 seeds), `hof-0.25` and `hof-0.50` (6 and 3 seeds),
`sigma-0.05` and `sigma-0.20` (3 each), `pop-32` and `pop-512` (3 each). Seeds
are 101–106 and 101–103.

**Hall-of-fame rule.** Every 1,000 tournaments the current champion enters a
FIFO archive of capacity 64. With probability *p* the second contestant is
drawn from the archive. Archive entries are immutable: if the archived genome
wins, the population member is overwritten by a mutant of it and inherits its
streak counter (Ha's rule, applied as though the archive entry were in the
pool); if the population member wins, nothing is overwritten.

## A.2 Evaluation protocol

Fixed before any run was launched; recorded in `results/matrix/protocol.json`.

| | |
|---|---|
| Checkpoints | champion exported every 5,000 games → 100 per run |
| Sweep | 200 episodes per checkpoint vs the 2015 baseline, seed 20260901 |
| Held-out re-scoring | final **and** peak champion, 1,000 episodes, seed 20260902 |
| Window for stability metrics | last 100,000 games = last 20 checkpoints |
| Primary endpoint | `final`, the score of the t = 500,000 champion (no checkpoint selection) |
| Tests | exact two-sided Mann–Whitney U (full enumeration), Cliff's δ, percentile bootstrap |

The peak checkpoint is a selected maximum, so its sweep score is
optimistically biased by construction. That is why every headline number is a
*held-out* re-score on a disjoint evaluation seed with five times the episodes.

The 2015 baseline's recurrent state is not reset between evaluation episodes,
matching the reference evaluation code (`eval_vs_baseline.py`), which
constructs one `BaselinePolicy` and reuses it. The effect is small — the state
washes out within a few steps of a serve — but it is part of the protocol and
is applied identically everywhere.

## A.3 Metric definitions

All computed on the 200-episode sweep curve; *window* = last 20 checkpoints.

- **final** — score of the t = 500,000 champion.
- **peak** — best checkpoint score in the run.
- **above parity** — fraction of the 100 checkpoints scoring > 0.
- **late_mean** — mean checkpoint score over the window.
- **volatility** — mean |Δ| between consecutive checkpoints in the window.
- **drawdown** — mean (best-so-far − current) over the window.
- **t_parity** — games at the first checkpoint scoring > 0.
- **cyclic triads** — among checkpoint triples in a within-run round robin,
  the fraction where A beats B beats C beats A. Pairs whose mean margin is
  within ±0.25 points are undecided and their triads are skipped.

## A.4 The compiled environment

The reference environment runs ~9 games/s per core; a 500,000-game run is
about twelve core-hours, which is why the earlier version of this study had one
run. `fastvolley.py` is a numba-compiled transcription of the physics, the MLP
policy and the baseline RNN, preserving the reference control flow statement
for statement — including three behaviours that look like bugs and are
load-bearing:

1. the left player is handed the **right** player's observation on the first
   step of every episode (`obs_left = obs_right` in `multiagent_rollout`);
2. observations are **not** refreshed on a frame in which a point is scored, so
   the returned observation is one step stale;
3. the ground test in `checkEdges` returns immediately, so the ceiling and
   fence tests are skipped on a scoring frame.

Three deviations are deliberate: the RNG family (gym's PCG64 vs numba's
Mersenne Twister), the tournament pair draw (`np.random.choice(pop, 2,
replace=False)` vs rejection sampling of two distinct integers), and
floating-point details of the forward pass (NumPy dispatches `matmul` to BLAS
and uses a SIMD `tanh`; the compiled version uses a plain loop and libm). The
third differs only in the last one or two units in the last place and cannot
change a trajectory unless it flips the *sign* of a network output, since the
game only ever reads `action[i] > 0`.

Validation drives both implementations from an identical stream of serve
velocities and compares them step by step (`validate_fastvolley.py`); results
in Table A1.

## A.5 Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-fast.txt

# 1. the port is the benchmark: bit-level check against slimevolleygym
.venv/bin/python validate_fastvolley.py --games 50

# 2. the matrix: 27 runs x 500,000 games (~4.5 h on 3 cores)
.venv/bin/python run_experiments.py --workers 3

# 3. metrics, held-out re-scoring, condition comparisons
.venv/bin/python analyze_matrix.py --holdout

# 4. the three coevolution-specific measurements
.venv/bin/python coevolution_analysis.py --within --across --proxy

# 5. the reference run, re-scored under the same protocol
.venv/bin/python eval_reference.py

# 6. tables and figures
.venv/bin/python make_tables.py
.venv/bin/python make_figures.py
```

The reference run itself (unmodified `slimevolleygym`, ~12 core-hours for
500,000 games) is continued from its committed population snapshot with:

```bash
.venv/bin/python train_ga_selfplay.py --resume --snapshot-freq 2500
```

## A.6 Data inventory

| Path | Contents |
|---|---|
| `results/ga_selfplay/` | the reference run: champion checkpoints every 1,000 games, population snapshot, training history |
| `results/matrix/*.npz` | one file per run: 100 champion genomes, streak counters, training rally lengths, sweep evaluation, and (control runs) full population snapshots every 50,000 games |
| `results/matrix/protocol.json` | the pre-registered protocol |
| `results/analysis/per_run.json` | metrics for every run |
| `results/analysis/conditions.json` | per-condition aggregates and the tests against control |
| `results/analysis/within_run.json` | within-run round robins: Elo, margins, cyclic triads |
| `results/analysis/across_runs.json` | the all-runs tournament of final champions |
| `results/analysis/champion_proxy.json` | population-level scores, diversity, and the exported champion's rank |
| `results/analysis/reference_curve.json` | the reference run re-scored under the matrix protocol |
| `results/validation.json` | the bit-level comparison against `slimevolleygym` |

## A.7 Tables

All tables below are generated by `make_tables.py` from the files in
`results/analysis/`; none is typed by hand. Re-running it after new results
land rewrites them in place, so `git diff docs/paper` shows exactly which
numbers moved.

### Table 1 — conditions, held-out scores and stability

<!-- table:1 -->
*(not yet generated)*
<!-- /table:1 -->

### Table 2 — each intervention against the control

<!-- table:2 -->
*(not yet generated)*
<!-- /table:2 -->

### Table 3 — the reference run on the unmodified environment

<!-- table:3 -->
| checkpoint | games | score vs 2015 baseline | won / drawn / lost | mean rally |
|---|---|---|---|---|
| final | 301,000 | -0.339 ± 0.978 (s.e.m. 0.031) | 16% / 44% / 40% | 3000 steps |
| peak | 234,000 | +0.304 ± 0.806 (s.e.m. 0.025) | 32% / 58% / 10% | 3000 steps |
| Ha (2020), same algorithm and budget | 500,000 | +0.353 ± 0.728 | — | — |

29 of 301 checkpoints score above parity on the 200-episode sweep. Held-out rows are 1,000 episodes at the disjoint evaluation seed. Internal transition (training rally length above 1,500 steps): not reached games; first checkpoint above parity: not reached games; lag — games.
<!-- /table:3 -->

### Table 4 — intransitivity inside a run

<!-- table:4 -->
*(not yet generated)*
<!-- /table:4 -->

### Table 5 — what the champion-selection proxy costs

<!-- table:5 -->
*(not yet generated)*
<!-- /table:5 -->

### Table 6 — cross-run tournament of final champions

<!-- table:6 -->
*(not yet generated)*
<!-- /table:6 -->

### Table 7 — the damping claim, across seeds

<!-- table:7 -->
*(not yet generated)*
<!-- /table:7 -->

### Table A1 — the compiled environment against the reference

<!-- table:a1 -->
| scenario | paired games | identical score | identical length | identical trajectory | max abs deviation | env steps compared |
|---|---|---|---|---|---|---|
| random vs random | 50 | 50/50 | 50/50 | 50/50 | 0 | 31,971 |
| champion vs random | 50 | 50/50 | 50/50 | 50/50 | 0 | 39,900 |
| champion vs champion | 50 | 50/50 | 50/50 | 50/50 | 0 | 45,017 |
| champion vs 2015 baseline | 50 | 50/50 | 50/50 | 50/50 | 0 | 148,909 |

All 200 paired games agree bit for bit over 265,797 environment steps. Throughput on one core: 8.7 games/s reference, 274 games/s compiled (31×).
<!-- /table:a1 -->

### Table A2 — every run

<!-- table:a2 -->
*(not yet generated)*
<!-- /table:a2 -->

### Table A3 — the same population continued in both implementations

<!-- table:a3 -->
*(not yet generated)*
<!-- /table:a3 -->

## A.8 Limitations

- **One environment.** Every claim here is about Slime Volleyball. Slime
  Volleyball is symmetric, zero-sum and fully observed, which is the friendliest
  possible setting for purely relative selection.
- **3–6 runs per condition.** Enough to distinguish an effect from seed noise
  when the effect is large; not enough for a small one. With *n* = *m* = 6 the
  smallest attainable two-sided *p* is 0.002, and with *n* = 3 it is 0.024.
- **One algorithm family.** The control is Ha's *simplified 2020* GA
  (feed-forward, no crossover, tournament selection), not the original 2015
  training procedure, which used recurrent networks, generational selection
  against multiple opponents, top-20% retention and crossover.
- **One archive design.** `hof-*` tests one specific hall of fame — FIFO,
  capacity 64, uniform sampling. A different archive (quality-diversity,
  curated, or larger) is a different experiment.
- **Multiplicity.** Only `final` is a pre-registered primary endpoint. The
  other metrics are descriptive and are not corrected for multiple comparisons.
