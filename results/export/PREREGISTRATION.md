# Preregistration: alternative export rules (WP10)

Written 2026-10-03, before any run of this experiment: when this file,
`run_export.py` and `export_analysis.py` were committed, `results/export/`
held no run file. `export_analysis.py` was exercised only on synthetic stand-in
files (random populations), never on real runs. Any later change to these
files gets a dated entry in `results/matrix/decisions.md`, and what it touches
becomes exploratory.

## 1. Why, and what was known

C4: Ha's winning-streak rule exports an individual near the median of its own
population (rank about 60 of 128), and most of the champion curve's losses are
the exported individual's, not the population's. The paper's re-export
analysis (`reexport.py`, post hoc, the 6 original control runs) ranked each
stored population by a short internal round robin instead. With 16 peers per
member (1,024 games per snapshot) the exported individual's mean score against
the 2015 baseline rose from −1.95 to −1.49 (about 57% of the gap to the
population's best member), higher in all 6 runs; the series' mean absolute
change fell from 0.84 to 0.70, lower in 4 of 6 runs. That analysis chose its
budgets after seeing the data and used 60-episode scores.

This experiment tests the same rule prospectively: fresh runs, the rule, its
budget, the outcomes and the tests fixed here. It also asks whether the fix
survives cyclic skill (WP8's discmix game), where a member that beats its own
population need not beat outsiders.

## 2. Design

| | |
|---|---|
| Slime Volleyball | 12 fresh `control` runs, seeds 401–412: `run_experiments.one_run` unchanged (the paper's parameters, 500,000 games, population snapshots every 50,000 games, so 10 per run) |
| discmix | 48 fresh control runs, `discmix-<λ>-control` at λ = 0, 0.25, 0.5, 0.75, seeds 501–512: `run_lab.one_run` unchanged (WP8's design) |
| seeds | used by no earlier experiment (matrix 101–106, NEAT 101–112, replication and niche 201–212, WP8 301–312, NEAT exploration 901–904) |
| pairing | in the control GA the exported individual never feeds back into training, so every rule is applied to the same stored snapshots: each run is its own paired comparison |
| command | `python run_export.py` (or one condition and seed per invocation) |
| analysis | `python export_analysis.py`, once, after all 60 run files exist |

**Rules**, applied at every snapshot:

| rule | which member is exported |
|---|---|
| `streak` | the largest winning-streak counter (Ha's rule; the paper's champions) |
| **`tournament-16`** | **the preregistered rule**: the best mean point margin in an internal round robin, each member against 16 random peers on average, (128 · 16) / 2 = 1,024 games, about a fifth of the 5,000 games between checkpoints. Slime Volleyball: `fastvolley_kernels.internal_rank`, the post hoc analysis's rule. discmix: the same scheme, each game's outcome drawn as `lab.kernels.discmix_play` draws it |
| `tournament-4`, `tournament-64` | the same with 4 and 64 peers (256 and 4,096 games): the budget, described only |
| `random` | one member drawn at random: a null, described only |
| `best` | the member best on the outcome itself: an oracle, not deployable, described only as the upper bound. In Slime Volleyball it is chosen on 60 episodes per member on a separate seed and then re-scored like every other rule |

**Outcomes.**
- Slime Volleyball: the exported member's mean score against the 2015 baseline on the held-out seed (1,000 episodes), as every champion in the paper is re-scored.
- discmix: *outsider strength*, the exported member's exact mean expected score against every member of the other 11 runs' populations at the same λ and snapshot. Strength against its own population is not used: the tournament estimates exactly that, so a gain there would be circular.
- Per run, for each rule's series of 10 exported members: **level**, the mean over snapshots; **declines**, the summed falls between consecutive snapshots (the measure of C4).

## 3. Hypotheses, tests and decision rules

| | hypothesis | test | holds if |
|---|---|---|---|
| **H10a** | Slime Volleyball: `tournament-16` exports better individuals than `streak` | per-run level difference (tournament-16 − streak); one-sided paired sign-flip permutation test, exact over the 2¹² sign patterns | rejected under Holm with H10b |
| **H10b** | Slime Volleyball: `tournament-16`'s series loses less competence | per-run declines difference (streak − tournament-16); same test | rejected under Holm with H10a |
| **H10c** | discmix: `tournament-16` exports better individuals than `streak`, judged by outsiders | per-run outsider level difference, all 48 runs; one-sided paired sign-flip test (100,000 random sign patterns, seed 20261014) | rejected under Holm with H10d |
| **H10d** | discmix: that advantage shrinks as skill becomes cyclic | Spearman ρ(λ, per-run advantage) < 0; one-sided permutation test (20,000 permutations of the advantages over runs, seed 20261014) | rejected under Holm with H10c |

α = 0.05 within each family (H10a, H10b; H10c, H10d).

Reported without a decision: every rule's level, declines and rank in its
population; the share of the streak-to-best gap that each tournament budget
recovers; per-λ advantages; the final exported members of `streak`,
`tournament-16` and `best` against the zoo GA (WP7's stronger yardstick).

**Stopping rule.** Exactly these 60 runs; none added, repeated or excluded.

**Power.** 12 paired runs. The post hoc analysis saw the gain in 6 of 6
runs. If at least 10 of the 12 runs gain by similar amounts, H10a's exact p
is at most about 0.02, inside Holm's 0.025 for the smaller p; fewer, or very
unequal gains, may not be detected, and the text will then say "not
detected", not "no gain".

## 4. What the results change

- **H10a holds**: C4 gains a prospective, deployable remedy. The README and
  paper say that a 1,024-game internal tournament exports a measurably
  better champion than the streak rule (preregistered, fresh seeds), and the
  post hoc re-export analysis is cited as its origin, not as its evidence.
  **H10a fails**: the re-export result stays post hoc, and the README and
  paper say the prospective test did not confirm it.
- **H10b holds**: "the losses of competence are the export rule's" gains a
  remedy that removes part of them. **Fails**: the text says the deployable
  rule did not reduce them detectably.
- **H10c holds**: the remedy carries over to cyclic skill when judged by
  outsiders. **Fails**: it is reported as shown only for transitive skill.
- **H10d holds**: the remedy weakens as skill becomes cyclic, and the
  Discussion's hypothesis about other evolutionary loops (including
  LLM-driven program evolution) gets that caveat. **Fails**: no such
  weakening was detected.
