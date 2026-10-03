# Preregistration: why the streak counter fails (WP13)

Written 2026-10-03, before any run of this experiment's fresh seeds: when this
file, `shadow.py`, `run_counter.py`, `counter_analysis.py` and
`results/counter/protocol.json` were committed, `results/counter/` held the
protocol record and the exploratory replays of the six original control runs
(`explore/`, described in §1), and no run with seeds 601–612. The commit that
adds this file is the preregistration; its push timestamp on GitHub is the
proof. Any later change to these files gets a dated entry in
`results/matrix/decisions.md`, and what it touches becomes exploratory.

## 1. Why, and what was known

C4: Ha's rule exports an individual near the median of its population, and
most of the champion curve's losses are the exported individual's. The paper
explains this with a mechanism it never tested: the counter "is inherited on
every replacement, so it records how long a lineage has survived, not how good
its current member is".

Two facts from the stored runs (decisions.md, 2026-10-03): at 299 of 300
stored population snapshots every member's counter lies within 10% of the
maximum, the median member a median 10 wins behind (0.5%), so the count is
almost all shared ancestry; and late in training 29–53% of games are ties, on
which one player is mutated in place and keeps its count.

In the control the counter never feeds back into reproduction, so other
counters can be kept on the identical evolutionary history. `shadow.py`
replays the control GA bit for bit (`test_repo.py`; the six exploratory
replays are identical to their stored files) and keeps four counters, the
2 × 2 of Ha's two bookkeeping choices:

| | a tie mutates the genotype and the count is kept | ... and the count restarts |
|---|---|---|
| **inherited** at birth | `inherited` (Ha's rule) | `inherited-reset` |
| **starts at 0** at birth | `own` (wins since birth) | `current` (wins of the current genotype) |

**Exploration (seeds 101–106, the paper's six control runs, replayed; not
evidence for the hypotheses below).** `results/counter/explore/analysis.json`.
All six replays are identical to their stored files (champions,
populations, Ha's counter, learning curve). At the population snapshots, held
out:

| rule | level | rank in population | ρ(counter, 60-episode score) |
|---|---|---|---|
| `inherited` (Ha's rule) | −1.92 | 64 | +0.05 |
| `inherited-reset` | −1.69 | 54 | +0.10 |
| `own` | −1.60 | 44 | +0.17 |
| `current` | −1.54 | 44 | +0.17 |
| `tournament-16` (1,024 games) | −1.51 | 44 | — |
| median member | −1.82 | 65 | — |
| best member (oracle) | −1.27 | 1 | — |

`current` beat Ha's counter in 5 of 6 runs (+0.38 on average); against the
16-peer tournament it differed by −0.03 (higher in 3 of 6). The 2 × 2 main
effects on level were +0.24 for not inheriting and +0.14 for restarting at a
tie, with a negative interaction (−0.18): either change alone recovers much
of the gain. On the reported curve (100 checkpoints) the mean absolute change
between checkpoints fell from 0.52 (`inherited`) to 0.33 (`current`), lower
in 6 of 6 runs, and the summed declines from 23.5 to 14.0. The median
maximum count was 2,285 for Ha's counter and 6 for `current`; late in
training 43% of games were ties. One run (s103) never learned to rally, and
there every rule ties near −3.9.

The hypotheses below follow from the mechanism; they were drafted after the
curve-level numbers above had been seen and before the snapshot-level ones.
The exploration sets only the expectation in the power statement.

## 2. Design

| | |
|---|---|
| runs | 12 fresh control runs, seeds 601–612, used by no earlier experiment (matrix 101–106, NEAT 101–112, replication and niche 201–212, WP8 301–312, export 401–412 and 501–512, NEAT exploration 901–904) |
| algorithm | the paper's control: Ha's GA, population 128, σ = 0.10, 500,000 games, initial scale 0.5; a champion every 5,000 games, the whole population every 50,000 (`run_experiments.CONDITIONS["control"]`), run by `shadow.run_ga_shadow`, which is bit-identical to the frozen kernel |
| pairing | the counters never feed back into reproduction, so every rule is applied to the same history and the same snapshots: each run is its own paired comparison |
| command | `python run_counter.py --seeds <seed>`, one seed per job of `.github/workflows/counter.yml`, each run file committed when it lands |
| analysis | `python counter_analysis.py`, once, after all 12 run files exist |

**Rules**, applied at every population snapshot (10 per run): the four
counters (each exports the first member with its maximum count, as numpy's
argmax does in Ha's rule); `tournament-16`, WP10's preregistered rule with its
seeds (comparison only); `median` and `best`, the members with the median and
the best 60-episode score (description; `best` is an oracle).

**Outcomes.**
- **Level**: for each rule, the mean over the 10 snapshots of its pick's
  held-out score against the 2015 baseline (1,000 episodes on the selection
  seed), as in WP10.
- **Curve volatility**: for each counter, the mean absolute change between
  consecutive checkpoints of its reported curve: its pick at every one of the
  100 checkpoints, scored on the learning-curve sweep (200 episodes on the
  sweep seed), as every champion curve in the paper is.
- Described without a decision: declines (summed falls between snapshots),
  the pick's rank in its population by the 60-episode score, Spearman ρ
  between each counter and that score, the curve's mean and summed declines,
  the tie rate.

## 3. Hypotheses, tests and decision rules

Each test is a one-sided exact paired sign-flip test over the 12 runs (all
2¹² sign patterns, `stats_utils.signflip_greater`); the four form one family,
corrected with Holm's step-down procedure at family-wise α = 0.05. Positive
differences favour the hypothesis.

| | hypothesis | per-run difference |
|---|---|---|
| **H13a** | counting only the current genotype's wins exports better members than Ha's counter | level(`current`) − level(`inherited`) |
| **H13b** | and reports a curve that swings less between checkpoints | volatility(`inherited`) − volatility(`current`) |
| **H13c** | not inheriting the count at birth helps (main effect in the 2 × 2) | ½ [level(`own`) − level(`inherited`) + level(`current`) − level(`inherited-reset`)] |
| **H13d** | restarting the count when a tie mutates the genotype helps (main effect) | ½ [level(`inherited-reset`) − level(`inherited`) + level(`current`) − level(`own`)] |

Described without a decision: `current` against `tournament-16` (mean
difference, runs higher, two-sided sign-flip p). A difference that is not
detected is reported as "not detected", never as equivalence.

**Stopping rule.** Exactly these 12 runs. A run that crashes is restarted with
the same seed (the runner is deterministic and skips finished runs). No run
is excluded.

**Power.** The exploration's six runs all favoured `current` on the curve.
If at least 10 of 12 runs favour a hypothesis by similar amounts, its exact
p is at most about 0.02; fewer, or very unequal differences, may not be
detected, and the text will then say "not detected".

## 4. What the results change

- **H13a holds**: C4 gains a remedy that costs no games: README and paper say
  that counting only the current genotype's wins (no inheritance, a restart
  when the genotype changes) exports better champions than Ha's counter
  (preregistered, fresh seeds), and report its comparison with the 1,024-game
  tournament as described above. **Fails**: they say the bookkeeping change
  did not detectably export better members.
- **H13b holds**: part of the reported curve's swings is the counter's
  bookkeeping, and the text says so with the size. **Fails**: no reduction
  was detected.
- **H13c holds**: the paper's mechanism (the count is inherited, so it
  measures a lineage) is confirmed as a cause. **Fails**: the paper's
  sentence is weakened to a hypothesis that was tested and not confirmed.
- **H13d holds**: keeping the count through in-place mutation is a second
  cause. **Fails**: not detected.
