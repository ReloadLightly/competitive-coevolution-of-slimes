# Preregistration: an archive organised by behaviour (WP9, part 1)

Written 2026-10-03, before any run of this experiment: when this file,
`run_qd.py` and `qd_analysis.py` were committed, `results/qd/` held no run
file. The only niche-archive runs before it were smoke tests of a few
thousand games in a scratch directory, which checked the file format and
were deleted. Any later change to this file, `run_qd.py`, `qd_analysis.py`
or the niche mode of `lab/kernels.py` gets a dated entry in
`results/matrix/decisions.md`, and the hypotheses it touches become
exploratory.

## 1. Why

The paper tested one archive design per reading: a time-ordered archive of
past champions, used as a parent (C5a, destroys learning) or as a test (C5b,
no detectable effect). WP8 added that the time-ordered test archive does not
help more as skill becomes cyclic (H8c failed). The literature's case for an
archive is diversity: a population should be tested against the variety of
strategies it has met, not against its recent past. This experiment keeps
the archive's role (a test, never a parent) and changes only what it holds:
one champion per behavioural niche.

## 2. Design

| | |
|---|---|
| archive | `lab.kernels.HOF_NICHE`: an 8 × 8 grid over a two-number behaviour descriptor (the network's mean first two outputs on fixed probe inputs); every 1,000 games the current champion (largest streak) enters its cell, replacing the cell's previous champion; opponents are drawn uniformly over occupied cells; used with p = 0.25, exactly as `hof-eval-v2` uses its archive |
| descriptor inputs | Slime Volleyball: 256 observations of the 2015 baseline playing itself (`lab.games.slime_probes`, seed 20261007); discmix: the game's 16 style probes, so the descriptor is the style point u on which the cycle is played |
| grid bounds | fixed before any niche run: per axis, the 1st to 99th percentile of the descriptor over all exported champions of existing runs, rounded outward to 0.05 (Slime Volleyball: the replication's control and archive-as-test runs, forward ∈ [−0.35, 0.95], backward ∈ [−0.95, 0.35]; discmix: all WP8 runs, u₀ ∈ [−0.80, 0.80], u₁ ∈ [−0.75, 0.80]) |
| Slime Volleyball | `slime-niche`, seeds 201–212, 500,000 games, population 128, σ 0.1, evaluated exactly as every matrix run; compared with the replication's `control` and `hof-eval-v2` runs on the same seeds |
| discmix | `discmix-<λ>-niche` at λ = 0, 0.25, 0.5, 0.75, seeds 301–312, the WP8 design otherwise; compared with WP8's `control` and `test` runs |
| runs | 12 + 48 = 60 new runs |
| command | `python run_qd.py` (or one condition and seed per invocation) |
| analysis | `python qd_analysis.py`, once, after all 60 run files exist; it recomputes the comparison runs from their raw files |

**Stopping rule.** Exactly these 60 runs; none added, repeated or excluded.

## 3. Hypotheses, tests and decision rules

| | hypothesis | test | holds if |
|---|---|---|---|
| **H9a** | in Slime Volleyball, an archive organised by behaviour, used as a test, neither harms nor helps detectably (C5b's statement for a different archive) | niche vs control on final (held out), peak (held out), share of checkpoints above parity, late mean; two-sided exact Mann–Whitney each | all four p ≥ 0.05 |
| **H9b** | in discmix, the niche archive's effect on cross-run strength (vs the control) grows with λ | trend T = Σ (λ − mean λ) · δ_λ, δ_λ = Cliff's δ (niche vs control); one-sided permutation test, labels permuted within λ (20,000 permutations, seed 20261008) | rejected under Holm with H9c |
| **H9c** | at λ = 0.75, the most cyclic setting, the niche archive's final champions are stronger than the time-ordered archive's | cross-run strength, niche vs WP8's `test`; one-sided exact Mann–Whitney | rejected under Holm with H9b |

Cross-run strength in discmix is a final champion's mean exact expected score
against the final champions of every other run at the same λ (control, test
and niche: 36 runs per λ).

Reported without a decision: learned-to-rally counts, niche vs the
time-ordered archive in Slime Volleyball, per-λ effects in discmix, archive
coverage (occupied cells at the end) and within-run cycling under each
archive.

**Power.** 12 runs per arm: 80% power only for Cliff's |δ| ≈ 0.6. H9a, like
C5b, can rule out large effects only and is to be worded that way.

## 4. What the verdicts change

These are results of the lab. They get their own README section with
generated numbers, labelled with this preregistration. If H9a fails (the
niche archive harms or helps detectably in Slime Volleyball), the paper's
C5b paragraph and its Limitations bullet on archive designs say so. If H9b
or H9c holds, the Limitations bullet on cyclic games says that a
behaviour-organised archive helped where the time-ordered one did not; if
both fail, it says that neither archive design helped.
