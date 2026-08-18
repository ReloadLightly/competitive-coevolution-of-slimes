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
| Budget | 500,000 games per run, for every condition and every family |
| External opponent | the 2015 champion RNN (112 parameters: 7×15 weights + 7 biases), never seen in training |

**Conditions.** Exact seed counts per condition are in Table A2; the design is:

| condition | what changes | archive |
|---|---|---|
| `control` | nothing — Ha's 2020 GA | — |
| `hof-eval` | archive supplies opponents; genes stay in the living pool | capacity 512 (whole run), p = 0.25 |
| `hof-0.25` | archive supplies opponents **and** parents | capacity 64 (last 64k games), p = 0.25 |
| `hof-0.50` | as `hof-0.25` at twice the dose | capacity 64, p = 0.50 |
| `hof-full` | as `hof-0.25` with an archive spanning the whole run | capacity 512, p = 0.25 |
| `sigma-0.05`, `sigma-0.20` | mutation scale halved / doubled | — |
| `pop-32`, `pop-512` | population shrunk / grown fourfold | — |
| `ga2015` | generational GA: computed fitness, elitism, uniform crossover | — |
| `es` | self-play evolution strategy; reports the distribution mean | — |

**The two archive rules.** Every 1,000 games the current champion is copied into
a FIFO archive; with probability *p* the population member's opponent is drawn
from the archive instead of from the pool. The two conditions differ in what
happens next.

*Archive as parent* (`hof-0.25`, `hof-0.50`, `hof-full`) applies Ha's
replacement rule verbatim: if the archived genome wins, the population member is
overwritten by a mutant **of the archived genome** and inherits its streak
counter. This injects old genetic material back into the pool and, as §3 of the
analysis reports, abolishes learning. It is retained in the study as a measured
negative result rather than deleted.

*Archive as test* (`hof-eval`) follows Rosin & Belew (1997): if the archived
genome wins, the population member is overwritten by a mutant **of the living
population member it was originally paired with**. Failing a test the pool is
expected to pass costs the member its slot, but no archive genome is ever a
parent, so genetic material never leaves the living population.

**Algorithm families.** `ga2015`: population 100, each agent plays ten random
peers per generation (500 games per generation, 1,000 generations), fitness is
the mean point margin over those games, the top 20 survive unchanged and the
remaining 80 are uniform crossovers of two elites plus Gaussian mutation; the
exported champion is the highest-fitness individual, which unlike the control is
a *computed* ranking. Two documented deviations from Ha (2015): the policy is the
fixed feed-forward MLP rather than a recurrent net, so that the comparison
isolates the search algorithm; and ten peers rather than eight, so a generation
costs exactly 500 games and the checkpoint grid matches the rest of the study.
`es`: 50 mirrored candidates per iteration (25 antithetic pairs), each playing
four random peers (100 games per iteration, 5,000 iterations), σ = 0.1,
learning rate 0.03, centred-rank fitness shaping; the reported champion is the
distribution mean, so this family has no champion-selection proxy at all.

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
- **t_internal** — games at the first checkpoint whose *training* games average
  more than 1,500 steps. Measured with no external opponent involved; the gap
  between this and `t_parity` is the internal-to-external lag.
- **reached** — whether `t_internal` exists at all, i.e. whether the population
  ever learned to rally. Stability metrics are meaningless without it: a run
  that never learned has zero volatility and zero drawdown and therefore scores
  as maximally stable. Every stability comparison is reported both over all runs
  and over the `reached` subset.
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
| condition | runs | final (held out) | peak (held out) | mean, last 100k | volatility | drawdown | above parity | median first parity |
|---|---|---|---|---|---|---|---|---|
| control (Ha 2020 GA) | 6 | -0.15 ± 0.27 | +0.32 ± 0.06 | -0.32 ± 0.23 | 0.67 | 0.60 | 26% | 192k (6/6) |
| archive as test, full span | 6 | -1.10 ± 0.78 | -0.44 ± 0.71 | -1.49 ± 0.68 | 0.66 | 0.59 | 8% | 365k (4/6) |
| archive as parent, p=0.25 | 6 | -4.10 ± 0.74 | -3.94 ± 0.90 | -4.14 ± 0.70 | 0.12 | 0.24 | 2% | 365k (1/6) |
| archive as parent, p=0.50 | 1 | -4.84 ± — | -4.84 ± — | -4.85 ± — | 0.01 | 0.06 | 0% | — (0/1) |
| generational GA (Ha 2015) | 6 | -2.00 ± 0.82 | -0.68 ± 0.83 | -1.61 ± 0.70 | 0.63 | 0.70 | 3% | 345k (5/6) |
| self-play ES | 6 | -2.08 ± 0.94 | -1.56 ± 0.99 | -2.46 ± 0.93 | 0.21 | 0.18 | 7% | 398k (2/6) |

Scores are points per episode against the 2015 baseline, mean ± s.e.m. across runs. `final` and `peak` are re-scored on the held-out evaluation seed over 1,000 episodes; the other columns come from the 200-episode sweep.
<!-- /table:1 -->

### Table 2 — each intervention against the control

<!-- table:2 -->
| condition | metric | difference | Cliff's δ | exact p |
|---|---|---|---|---|
| archive as test, full span | `final_holdout` | -0.953 | -0.28 | 0.485 |
| archive as test, full span | `late_mean` | -1.178 | -0.67 | 0.065 |
| archive as test, full span | `volatility` | -0.015 | +0.00 | 1.000 |
| archive as test, full span | `drawdown` | -0.014 | +0.00 | 1.000 |
| archive as test, full span | `above_parity` | -0.187 | -0.75 | 0.028 |
| archive as parent, p=0.25 | `final_holdout` | -3.951 | -0.94 | 0.004 |
| archive as parent, p=0.25 | `late_mean` | -3.828 | -0.94 | 0.004 |
| archive as parent, p=0.25 | `volatility` | -0.557 | -0.83 | 0.015 |
| archive as parent, p=0.25 | `drawdown` | -0.366 | -0.67 | 0.065 |
| archive as parent, p=0.25 | `above_parity` | -0.245 | -0.94 | 0.004 |
| generational GA (Ha 2015) | `final_holdout` | -1.852 | -0.67 | 0.065 |
| generational GA (Ha 2015) | `late_mean` | -1.299 | -0.78 | 0.026 |
| generational GA (Ha 2015) | `volatility` | -0.046 | +0.00 | 1.000 |
| generational GA (Ha 2015) | `drawdown` | +0.103 | +0.00 | 1.000 |
| generational GA (Ha 2015) | `above_parity` | -0.230 | -0.83 | 0.015 |
| self-play ES | `final_holdout` | -1.933 | -0.33 | 0.394 |
| self-play ES | `late_mean` | -2.145 | -0.33 | 0.394 |
| self-play ES | `volatility` | -0.464 | -0.72 | 0.041 |
| self-play ES | `drawdown` | -0.420 | -0.83 | 0.015 |
| self-play ES | `above_parity` | -0.197 | -0.83 | 0.013 |

Exact two-sided Mann–Whitney U over all label assignments. Difference is condition minus control in points per episode (`above_parity` is a fraction). Only `final_holdout` is the pre-registered primary endpoint; the rest are descriptive and uncorrected for multiplicity.
<!-- /table:2 -->

### Table 3 — the reference run on the unmodified environment

<!-- table:3 -->
| checkpoint | games | score vs 2015 baseline | won / drawn / lost | mean rally |
|---|---|---|---|---|
| final | 500,000 | +0.036 ± 0.734 (s.e.m. 0.023) | 19% / 64% / 17% | 3000 steps |
| peak | 439,000 | +0.496 ± 0.856 (s.e.m. 0.027) | 42% / 52% / 6% | 3000 steps |
| Ha (2020), same algorithm and budget | 500,000 | +0.353 ± 0.728 | — | — |

112 of 500 checkpoints score above parity on the 200-episode sweep. Held-out rows are 1,000 episodes at the disjoint evaluation seed. Internal transition (training rally length above 1,500 steps): 104,200 games; first checkpoint above parity: 172,000 games; lag 67,800 games.
<!-- /table:3 -->

### Table 4 — intransitivity inside a run

<!-- table:4 -->
| condition | runs | ρ(Elo, training time) | cyclic triads | undecided pairs |
|---|---|---|---|---|
| control (Ha 2020 GA) | 5 | +0.72 | 1/421 (0.2%) | 10.0 |
| archive as parent, p=0.25 | 5 | +0.28 | 17/127 (13.4%) | 23.4 |

Checkpoints 50,000 games apart play a round robin, 50 games per pair over both court sides. A pair whose mean margin is inside ±0.25 points counts as undecided and its triads are skipped. A cyclic triad is A beats B beats C beats A.
<!-- /table:4 -->

### Table 5 — what the champion-selection proxy costs

<!-- table:5 -->
| games | exported champion | best in the same pool | gap | exported rank | ρ(streak, score) | above parity in pool | mean pairwise genotype distance |
|---|---|---|---|---|---|---|---|
| 50,000 | -4.74 | -4.46 | 0.28 | 58 / 128 | +0.12 | 0 | 12.92 |
| 100,000 | -4.32 | -2.65 | 1.67 | 72 / 128 | -0.00 | 1 | 11.03 |
| 150,000 | -3.40 | -2.53 | 0.87 | 45 / 128 | +0.09 | 3 | 10.32 |
| 200,000 | -2.48 | -1.78 | 0.70 | 63 / 128 | +0.03 | 5 | 9.88 |
| 250,000 | -1.57 | -0.62 | 0.94 | 46 / 128 | +0.09 | 20 | 9.18 |
| 300,000 | -1.38 | -0.50 | 0.87 | 51 / 128 | +0.13 | 25 | 10.54 |
| 350,000 | -1.67 | -0.46 | 1.21 | 72 / 128 | +0.08 | 30 | 10.33 |
| 400,000 | -1.41 | -0.41 | 1.00 | 63 / 128 | +0.11 | 46 | 11.81 |
| 450,000 | -0.58 | +0.60 | 1.18 | 71 / 128 | -0.02 | 45 | 12.23 |
| 500,000 | -0.29 | +0.66 | 0.94 | 73 / 128 | +0.06 | 58 | 11.36 |

Control runs only (5 seeds), averaged across seeds. Every member of the snapshotted population is scored against the 2015 baseline; 'exported' is the individual Ha's longest-winning-lineage rule selects.
<!-- /table:5 -->

### Table 6 — cross-run tournament of final champions

<!-- table:6 -->
| condition | runs | median Elo | best run | worst run |
|---|---|---|---|---|
| control (Ha 2020 GA) | 5 | +373 | +785 | +238 |
| archive as parent, p=0.25 | 5 | -421 | -355 | -483 |

Bradley–Terry ratings on the Elo scale from an all-play-all tournament of the 10 final champions, 50 games per pair over both court sides. Cyclic triads across the whole tournament: 0/83 (0.0%).
<!-- /table:6 -->

### Table 7 — the damping claim, across seeds

<!-- table:7 -->
| games | mean across seeds | within-run s.d. | spread across seeds | checkpoints above parity |
|---|---|---|---|---|
| 0–100,000 | -4.32 | 0.74 | 0.57 | 1/120 |
| 100,000–200,000 | -2.76 | 0.83 | 1.78 | 14/120 |
| 200,000–300,000 | -1.46 | 0.69 | 1.75 | 34/120 |
| 300,000–400,000 | -1.01 | 0.51 | 1.71 | 45/120 |
| 400,000–500,000 | -0.32 | 0.72 | 0.52 | 64/120 |

Control condition, 6 seeds. 'Within-run s.d.' is the spread of checkpoint scores inside a window, averaged over seeds — the quantity the single-run version of this study claimed was damping. 'Spread across seeds' is the s.d. of the per-seed window means.
<!-- /table:7 -->

### Table 8 — when the phase change happens

<!-- table:8 -->
| condition | runs | reached long rallies | internal transition (median, range) | first parity (median, range) | lag (median) |
|---|---|---|---|---|---|
| control (Ha 2020 GA) | 6 | 6/6 | 120k (55k–415k) | 192k (85k–440k) (6/6) | 58k |
| archive as test, full span | 6 | 6/6 | 350k (155k–495k) | 365k (210k–445k) (4/6) | 58k |
| archive as parent, p=0.25 | 6 | 1/6 | 80k (80k–80k) | 365k (365k–365k) (1/6) | 285k |
| archive as parent, p=0.50 | 1 | 0/1 | never | never (0/1) | — |
| generational GA (Ha 2015) | 6 | 5/6 | 205k (160k–420k) | 345k (290k–480k) (5/6) | 105k |
| self-play ES | 6 | 4/6 | 372k (320k–490k) | 398k (385k–410k) (2/6) | 70k |
| *reference run (1 run, real environment)* | 1 | 1/1 | *104k* | *172k* | *68k* |

'Internal transition' is the first checkpoint at which the population's own training games average more than 1,500 steps — measured with no external opponent involved. 'First parity' is the first checkpoint scoring above 0 against the 2015 baseline. The lag between them is how far internal progress runs ahead of anything an external evaluation can see.
<!-- /table:8 -->

### Table 9 — promotion rules compared

<!-- table:9 -->
| promotion rule | games spent ranking | mean score of the exported individual | volatility of the reported series | gap to best closed | ρ(rule statistic, true skill) |
|---|---|---|---|---|---|
| winning streak (Ha's rule) | 0 | -1.95 | 0.84 | — | +0.04 |
| *pick the population median* | *0* | *-1.83* | *0.57* | *—* | *—* |
| internal round robin, 4 peers each | 256 | -1.77 | 0.72 | 22% | +0.32 |
| internal round robin, 8 peers each | 512 | -1.55 | 0.77 | 50% | +0.41 |
| internal round robin, 16 peers each | 1,024 | -1.49 | 0.70 | 57% | +0.49 |
| internal round robin, 32 peers each | 2,048 | -1.47 | 0.73 | 60% | +0.53 |
| internal round robin, 64 peers each | 4,096 | -1.43 | 0.68 | 65% | +0.57 |
| *best in pool (oracle, not deployable)* | *—* | *-1.14* | *0.61* | *100%* | *+1.00* |

Control runs only (6 seeds), across all population snapshots. Every population member is scored against the 2015 baseline over 60 episodes to establish true skill; the promotion rules then compete to pick the best member using only what they are entitled to see. 'Volatility' is the mean absolute change in the exported individual's score between consecutive snapshots. For scale, 4,096 ranking games is 0.8% of a 500,000-game run.
<!-- /table:9 -->

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
| condition | seed | final (sweep) | final (held out) | peak (held out) | mean last 100k | volatility | drawdown | above parity | internal transition | first parity | train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 101 | +0.14 | +0.26 | +0.39 | +0.10 | 0.23 | 0.29 | 21% | 235k | 335k | 31.0 |
| control | 102 | -1.25 | -1.35 | +0.36 | -0.43 | 0.94 | 0.95 | 27% | 60k | 135k | 42.7 |
| control | 103 | -0.29 | -0.31 | +0.04 | -1.40 | 1.24 | 0.72 | 3% | 415k | 440k | 17.3 |
| control | 104 | +0.50 | +0.41 | +0.32 | -0.23 | 0.78 | 0.77 | 32% | 55k | 85k | 43.9 |
| control | 105 | -0.33 | -0.23 | +0.36 | +0.02 | 0.35 | 0.42 | 25% | 160k | 250k | 38.8 |
| control | 106 | +0.32 | +0.32 | +0.49 | +0.06 | 0.48 | 0.46 | 50% | 80k | 120k | 50.0 |
| hof-eval | 101 | -2.17 | -2.06 | +0.34 | -1.12 | 1.33 | 1.18 | 4% | 330k | 445k | 33.2 |
| hof-eval | 102 | -0.36 | -0.45 | -0.05 | -2.07 | 0.42 | 0.35 | 0% | 435k | — | 20.8 |
| hof-eval | 103 | -0.27 | -0.21 | +0.30 | -0.46 | 0.93 | 0.82 | 17% | 265k | 300k | 39.8 |
| hof-eval | 104 | +0.49 | +0.43 | +0.43 | -0.06 | 0.57 | 0.52 | 22% | 155k | 210k | 37.7 |
| hof-eval | 105 | -4.50 | -4.54 | -3.99 | -4.59 | 0.11 | 0.10 | 0% | 495k | — | 12.2 |
| hof-eval | 106 | +0.24 | +0.21 | +0.36 | -0.66 | 0.59 | 0.55 | 3% | 370k | 430k | 17.7 |
| hof-0.25 | 101 | -4.85 | -4.83 | -4.83 | -4.84 | 0.02 | 0.06 | 0% | — | — | 15.2 |
| hof-0.25 | 102 | -4.82 | -4.84 | -4.83 | -4.85 | 0.01 | 0.08 | 0% | — | — | 11.2 |
| hof-0.25 | 103 | -4.85 | -4.86 | -4.84 | -4.85 | 0.01 | 0.06 | 0% | — | — | 10.9 |
| hof-0.25 | 104 | -4.85 | -4.85 | -4.85 | -4.85 | 0.01 | 0.03 | 0% | — | — | 10.1 |
| hof-0.25 | 105 | -0.25 | -0.41 | +0.55 | -0.64 | 0.64 | 1.13 | 11% | 80k | 365k | 33.5 |
| hof-0.25 | 106 | -4.82 | -4.83 | -4.85 | -4.84 | 0.01 | 0.05 | 0% | — | — | 12.4 |
| hof-0.50 | 101 | -4.88 | -4.84 | -4.84 | -4.85 | 0.01 | 0.06 | 0% | — | — | 15.7 |
| ga2015 | 101 | -3.88 | -3.81 | +0.04 | -1.94 | 0.82 | 0.57 | 1% | 420k | 480k | 13.8 |
| ga2015 | 102 | -0.77 | -0.62 | +0.32 | -0.26 | 0.43 | 0.56 | 6% | 160k | 345k | 24.0 |
| ga2015 | 103 | -4.84 | -4.83 | -4.83 | -4.84 | 0.02 | 0.05 | 0% | — | — | 9.3 |
| ga2015 | 104 | -0.20 | -0.09 | +0.04 | -1.49 | 1.10 | 1.64 | 3% | 205k | 300k | 18.7 |
| ga2015 | 105 | -0.18 | -0.26 | +0.26 | -0.44 | 0.54 | 0.70 | 7% | 185k | 290k | 23.7 |
| ga2015 | 106 | -2.31 | -2.40 | +0.12 | -0.72 | 0.84 | 0.70 | 3% | 235k | 465k | 22.9 |
| es | 101 | -2.43 | -2.46 | -0.95 | -2.53 | 0.59 | 0.64 | 0% | 410k | — | 15.3 |
| es | 102 | -4.84 | -4.83 | -4.82 | -4.82 | 0.03 | 0.04 | 0% | — | — | 9.2 |
| es | 103 | -1.68 | -1.68 | -0.12 | -3.15 | 0.38 | 0.21 | 0% | 490k | — | 11.0 |
| es | 104 | +0.37 | +0.41 | +0.45 | +0.29 | 0.07 | 0.09 | 24% | 320k | 385k | 20.5 |
| es | 105 | -4.38 | -4.44 | -4.44 | -4.74 | 0.04 | 0.03 | 0% | — | — | 10.6 |
| es | 106 | +0.55 | +0.50 | +0.50 | +0.18 | 0.15 | 0.08 | 16% | 335k | 410k | 19.6 |
<!-- /table:a2 -->

### Table A3 — the same population continued in both implementations

<!-- table:a3 -->
| continuation | games added | final score | checkpoints above parity |
|---|---|---|---|
| compiled, resume_s900 | 176,000 | -2.82 | 19/35 |
| compiled, resume_s901 | 176,000 | +0.41 | 21/35 |
| reference environment | 176,000 | +0.01 | 78/176 |

All continuations start from the identical committed population snapshot at tournament 324,000. Independent continuations of one population diverge because the algorithm is stochastic; the question is whether the compiled ones land in the same band as the reference one.
<!-- /table:a3 -->

## A.8 Adding another algorithm later — including NEAT

Nothing in this study is closed to a later comparison, provided a new
algorithm is measured against the *frozen* protocol rather than a re-derived
one. Three things make that possible and are committed to the repository:

1. **The environment is fixed and verified.** `fastvolley.py` is validated bit
   for bit against `slimevolleygym` (Table A1), and `test_repo.py` re-checks
   that agreement plus the three February failure modes on every run. A run
   done months later is playing exactly the same game.
2. **The evaluation protocol is a file, not a habit.**
   `results/matrix/protocol.json` records the sweep seed and episode count, the
   disjoint held-out seed and episode count, the checkpoint spacing and the
   metric window. A later run scored with `analyze_matrix.py` lands on the same
   axes with no renegotiation.
3. **Every champion genome is committed.** `results/matrix/*.npz` holds all 100
   champion genomes of every run. That means a future algorithm can be compared
   two ways: on the external yardstick, and — more informative — head to head in
   the all-runs tournament (`coevolution_analysis.py --across`), where its
   champions actually play this session's champions.

**The contract for a new algorithm.** A training function must accept a seed, a
game budget and its own hyperparameters, and return champion genomes at the
protocol's checkpoint spacing plus the mean training rally length per
checkpoint. `algorithms.py` has three worked examples; wiring one in is a
dict entry in `CONDITIONS` and a branch in `one_run`.

**What NEAT specifically would need.** Everything above is reusable; the part
that does not yet exist is a *variable-topology* network evaluator. The
compiled game takes a flat 273-parameter vector and runs a fixed 12–10–10–3
MLP, so NEAT would need:

- a genome representation with node and connection lists, innovation numbers,
  and enable/disable flags;
- a compiled forward pass over an arbitrary graph (one activation sweep per
  timestep, which also gets recurrence for free — the 2015 champion was
  recurrent, so this is a feature rather than a complication);
- crossover aligned on innovation numbers, and speciation with fitness sharing,
  without which topological innovations are out-competed before they are tuned;
- an adapter so a NEAT champion can be entered into the cross-run tournament
  against the fixed-topology champions.

That is a substantial piece of work — on the order of the compiled environment
itself — which is why this study reports fixed-topology families only and says
so rather than gesturing at NEAT results it does not have. It is worth noting
that the February 2026 NEAT attempt documented in
`docs/postmortem-february.md` failed for three reasons that had nothing to do
with NEAT (an all-buttons action bug, a silently-ignored opponent assignment,
and an inverted observation sign convention). All three are now covered by
`test_repo.py`, so a second attempt would start from a base where those
failures cannot recur silently.

## A.9 Limitations

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
