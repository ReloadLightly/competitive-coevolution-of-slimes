# Preregistration: confirmatory replication of C1–C5b

Written 2026-10-02, before any replication run was started: when this file
and `replication.py` were committed, `results/replication/` held no run file.
The commit that adds this file is the preregistration; its push timestamp on
GitHub is the proof. Any later change to this file or to `replication.py` gets
a dated entry in `results/matrix/decisions.md`, and every hypothesis it
touches is then reported as exploratory, not confirmatory.

## 1. Why

The paper's claims rest on six seeds per main condition, and its design was
revised while it ran (`decisions.md`). Both are listed as limitations. This
replication tests the claims that rest on the single-population GA (C1–C5b)
on fresh seeds, with every test and decision rule fixed in advance.

Not part of this replication, and never to be described as replicated by it:
C6 (algorithm families: would need the generational GA and the ES rerun) and
C7 (unequal power, exploratory).

## 2. Design

| | |
|---|---|
| conditions | `control` (Ha's GA), `hof-0.25` (archive as parent), `hof-eval-v2` (archive as test) |
| parameters | exactly those in `run_experiments.CONDITIONS` at the commit of this file; the science code is the frozen code of tag `paper-v1` |
| seeds | 201–212 for every condition (fresh; the paper used 101–106) |
| runs | 36, each 500,000 self-play games |
| command | `python run_experiments.py --outdir results/replication --only hof-eval-v2,control,hof-0.25 --seeds 201,202,203,204,205,206,207,208,209,210,211,212` |
| launcher | `scripts/replicate.sh`: the command above, plus a commit and push of each run file as it lands |
| analysis | `python replication.py`, once, after all 36 run files exist |

**Stopping rule.** Exactly these 36 runs; no further seeds are added whatever
the results. A run that crashes is restarted with the same seed (the runner is
deterministic and skips finished runs). No run is excluded from the analysis.

**Measurements** are the paper's own, imported, not re-implemented:
`analyze_matrix.metrics` (learned to rally, transitions, above parity, late
mean, archive win rates), its held-out re-scoring (1,000 episodes on the
disjoint selection seed), the within-run round robin of
`coevolution_analysis` (checkpoints every 50,000 games, 50 games per pair) and
the population-snapshot scoring of `reexport` (60 episodes per individual).

## 3. Hypotheses, tests and decision rules

All tests are exact. "Sign test" is the one-sided exact binomial test with
zeros dropped. The five directional tests form one family, corrected with
Holm's step-down procedure at family-wise α = 0.05.

| claim | hypothesis | test / criterion | in the Holm family |
|---|---|---|---|
| **C1** internal improvement precedes external transfer | in control runs that both learn to rally and reach parity, the internal transition comes first (lag > 0) | sign test on the lags | H1 |
| **C2** the phase change is robust, its timing is not | (a) at least 10 of 12 control runs learn to rally; (b) the latest internal transition is at least 3× the earliest | thresholds | — |
| **C3** the population is not cycling | (a) ρ(within-run Elo, training time) > 0 across control runs; (b) under 1% of decided checkpoint triads are cyclic, pooled over control runs | (a) sign test on the per-run ρ; (b) threshold | H3a |
| **C4** the export rule is the noise source | (a) the exported individual's mean rank in its own pool of 128 is outside the top quarter (> 32); (b) streak counter and skill are uncorrelated: the 90% bootstrap CI of the mean per-run ρ(streak, score) lies inside (−0.2, +0.2); (c) the exported individual's summed decline between consecutive snapshots exceeds the best member's | (a) sign test on mean rank − 32; (b) equivalence criterion; (c) sign test on the paired difference | H4a, H4c |
| **C5a** archive as parent destroys learning | fewer `hof-0.25` runs than control runs learn to rally | one-sided Fisher exact test | H5 |
| **C5b** archive as test neither harms nor helps detectably | `hof-eval-v2` vs control on final (held out), peak (held out), share of checkpoints above parity, late mean | two-sided exact Mann–Whitney on each; the claim stands if all four p ≥ 0.05 | — (a null claim) |

A claim **replicates** when all of its parts hold (tests: rejected under
Holm). The verdicts are computed by `replication.py` and written to
`results/replication/analysis.json`.

**C5b and power.** "No detectable effect" is a statement about power. At 12
runs per arm the exact two-sided test has about 80% power only for effects of
Cliff's |δ| ≈ 0.6 or larger (simulated for a normal shift: |δ| = 0.33 →
27%, 0.52 → 61%, 0.68 → 89%). A replicated C5b therefore rules out large
effects only, and is to be worded that way. Its descriptive part is reported
alongside without a decision: the late archive win rate is at most 0.20 in
every `hof-eval-v2` run that learned and at least 0.40 in every run that did
not.

## 4. What the verdicts change

- A claim that replicates keeps its wording; README and paper gain the
  replication as a separate section with generated numbers.
- A claim that does not replicate is weakened in README and paper (CLAUDE.md
  rule 5), and both the original and the replication results are reported.
- Pooled estimates over all 18 seeds may be reported as secondary,
  descriptive results; they decide nothing.

## 5. The same analysis applied to the original seeds

For reference (not part of the decision), `replication.py --runs
results/matrix --seeds 101-106` applies this analysis to the paper's six seeds
per condition. Its measurements reproduce the committed analysis files bit for
bit (`reexport.json`, `within_run.json` and `holdout.json`: 0 differing
values), so the replication measures exactly what the paper measured. With six
runs the threshold of C2 (a) scales to 5 of 6.

Runs per condition: control 6, archive as parent 6, archive as test 6. Tests in the Holm family count as holding when rejected at family-wise α = 0.05.

| claim | criterion | result | holds |
|---|---|---|---|
| C1 | internal transition before parity (lag > 0) | 6/6 runs, p = 0.016 | **no** |
| C2 | control runs that learn to rally | 6/6 | yes |
| C2 | latest / earliest internal transition ≥ 3 | 7.5× | yes |
| C3 | ρ(Elo, time) > 0 within runs | 6/6 runs (mean +0.74), p = 0.016 | **no** |
| C3 | cyclic share of decided triads < 1% | 1/499 (0.2%) | yes |
| C4 | exported individual outside its pool's top quarter | 6/6 runs (mean rank 64), p = 0.016 | **no** |
| C4 | ρ(streak, skill) inside ±0.2 (90% CI) | +0.04 [+0.03, +0.06] | yes |
| C4 | exported declines more than the best member | 6/6 runs (8.6 vs 1.1), p = 0.016 | **no** |
| C5a | archive as parent learns less often than control | 1/6 vs 6/6, p = 0.0076 | yes |
| C5b | archive as test vs control: all four p ≥ 0.05 | final δ -0.28 p 0.48, peak δ +0.00 p 1.00, above δ -0.33 p 0.37, late δ -0.22 p 0.59 | yes |

Verdicts: C1 **not replicated**, C2 replicated, C3 **not replicated**, C4 **not replicated**, C5a replicated, C5b replicated.
Archive as test (description, no decision): late archive win rate 0.16, 0.12, 0.14, 0.16, 0.11 in runs that learned; 0.50 in runs that did not (5/6 learned).

The four rows marked **no** at six seeds are not evidence against the claims.
With six runs the smallest possible one-sided sign-test p is 1/64 = 0.016.
Holm rejects C5a first (p = 0.0076 against 0.05/5 = 0.01); the next threshold
is 0.05/4 = 0.0125, which no sign test on six runs can meet, so those rows
cannot hold at n = 6 whatever the data. That floor is why the replication uses
12 seeds per condition, where 12 of 12 gives p = 1/4096 and 11 of 12 gives
p = 0.003.
