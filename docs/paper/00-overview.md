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
