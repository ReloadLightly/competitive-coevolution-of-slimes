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
| External opponent | the 2015 champion RNN (120 parameters), never seen in training |

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

## A.7 Limitations

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
