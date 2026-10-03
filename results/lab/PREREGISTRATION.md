# Preregistration: does skill structure change C3, C4 and C5b? (WP8)

Written 2026-10-03, before any run of this experiment: when this file,
`run_lab.py` and `lab_analysis.py` were committed, `results/lab/` held no run
file. The game's constants were fixed earlier the same day; the only runs of
the game before this file were pilot runs at λ = 0 (seed 11) that looked at
skill trajectories and nothing else (`docs/lab.md`). Any later change to this
file, `run_lab.py`, `lab_analysis.py` or `lab/` gets a dated entry in
`results/matrix/decisions.md`, and the hypotheses it touches become
exploratory.

## 1. Why

The paper's Limitations: Slime Volleyball turned out almost purely
transitive, and "in a genuinely cyclic game the archive and the export rule
may behave differently". The discmix game (`lab/games.py`, `docs/lab.md`)
sets how cyclic skill is with one number, λ, and gives exact expected scores,
so the paper's three population-level claims can be asked again where the
answer is not forced by transitivity:

- **C3** the population is not cycling;
- **C4** the streak export rule picks an individual near the middle of its
  pool, and the exported champion loses ground the pool does not;
- **C5b** the archive used as a test neither harms nor helps detectably.

## 2. Design

| | |
|---|---|
| game | `discmix` at the commit of this file (α 4.62, β 3.32, noise 0.3, tie 0.05, 256 skill probes, 16 style probes) |
| λ | 0, 0.25, 0.5, 0.75 (1.0 is left out: with no transitive part there is no skill to learn) |
| modes | `control` (Ha's GA) and `test` (archive as test: p = 0.25, one archived champion per 1,000 games, capacity 512, as `hof-eval-v2`) |
| seeds | 301–312 in every cell |
| runs | 4 λ × 2 modes × 12 seeds = 96, each 500,000 games, population 128, σ 0.1, initial scale 0.5 |
| algorithm | `lab.kernels.run`, which reproduces the paper's kernels bit for bit on Slime Volleyball (`test_repo.py`) |
| command | `python run_lab.py` (or one condition and seed per invocation; the result per run is the same) |
| analysis | `python lab_analysis.py`, once, after all 96 run files exist |

**Stopping rule.** Exactly these 96 runs; no run is added, repeated or
excluded.

## 3. Measurements (all exact, from expected scores)

Per run: the cyclic share of triads among its 100 exported champions and
ρ(strength, time) (C3); at each of 10 population snapshots, the exported
individual's rank by true strength (mean expected score against the rest of
its pool), ρ(streak, strength), and the exported and best members' scores
against a fixed external panel of 64 genomes from the initial distribution,
never seen in training (C4); the final champion's mean expected score against
the final champions of every other run at the same λ (C5b).

## 4. Hypotheses, tests and decision rules

| | hypothesis | test | holds if |
|---|---|---|---|
| **H8a** (C3) | within-run cycling rises with λ | per mode, Spearman ρ(λ, cyclic share) over 48 runs, one-sided permutation test (20,000 permutations, seed 20261006) | p < 0.05 in both modes |
| **H8b** (C4) | in control runs the exported individual sits outside its pool's top quarter at every λ | per λ, sign test on mean rank − 32 over 12 runs; Holm over the four λ | rejected at all four λ |
| **H8c** (C5b) | the archive-as-test effect on cross-run strength grows with λ | trend T = Σ (λ − mean λ) · δ_λ, δ_λ = Cliff's δ (test vs control); one-sided permutation test, mode labels permuted within λ | p < 0.05 |

H8a and H8c answer separate questions and are tested at α = 0.05 each; H8b is
corrected over its four λ. At λ = 0 the cyclic share is zero by construction
(a check of the game, not a hypothesis).

The direction of H8c is the literature's expectation: an archive is meant
for cyclic games (Rosin & Belew 1997; Ficici & Pollack). The paper found no
detectable effect in a transitive game; H8c asks whether the effect appears
as cycling grows. Per-λ δ with two-sided exact Mann–Whitney p is reported for
every λ without a decision.

**Power.** 12 runs per cell: per-λ comparisons detect only large effects
(80% power at Cliff's |δ| ≈ 0.6, as in the replication). The trend test
pools the four λ.

## 5. What the verdicts change

These are results of the lab, not claims of the paper. They are reported in
their own README section with generated numbers, labelled with this
preregistration. If H8b fails at some λ, the paper's Discussion (which
generalises C4 to other evolutionary loops) must say that the export-rule
finding did not survive a cyclic game. If H8a or H8c fail, the
Limitations bullet on cyclic games is updated with what was found.
