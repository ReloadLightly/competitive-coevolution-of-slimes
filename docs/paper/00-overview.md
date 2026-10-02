# The champion you export is not the champion you evolved

**Competitive coevolution in Slime Volleyball, measured across 59 runs —
and the first of three experiments behind the ACTIR / ShinkaEvolve submission.**

## Abstract

Self-play evolution produces competent agents from an entirely internal signal:
beat a randomly drawn peer, stay in the pool. Nothing tells the population what
good play is. We replicate David Ha's tournament-selection genetic algorithm on
Slime Volleyball and ask what that signal can and cannot deliver, using a design
rather than a single run: 59 independent runs of 500,000 self-play games each across
thirteen conditions, with every evaluation against a frozen 2015
champion that is never seen during training.

The internal signal works, and it works late. A population improves against
itself for tens of thousands of games before any of that improvement transfers
to the external opponent, so an evaluation that stops early reads as total
failure. The phase change at which transfer begins is robust — every control run
reaches the same band — but its *timing* is not reproducible at all, ranging over
more than a sevenfold spread across seeds. A single run therefore supports no
claim about when self-play starts working.

Our main finding concerns the instability that follows. Champion quality swings
by several points per episode between adjacent checkpoints, and the standard
explanation is coevolutionary forgetting: skills that no current opponent
punishes are not selected for and are lost. That explanation does not survive
measurement. Playing every checkpoint of a run against every other checkpoint,
skill is almost perfectly transitive — later champions beat earlier ones, and
under 1% of decided checkpoint triples are cyclic. The population is not going
in circles.

What is going in circles is the *reporting*. Ha's algorithm exports the
individual with the longest winning lineage, a proxy adopted explicitly
"without actually computing who is best to save time". Because the streak
counter is inherited by the loser on every replacement, it measures the age of a
lineage rather than the merit of an individual, and it is uncorrelated with
actual skill. Scored against the whole population, the exported champion sits
near the *median* of its own 128 members and about a point per episode below the
best of them; at the end of a run it is below parity while dozens of its peers
are above it. A substantial part of what has been read as coevolutionary
instability is measurement noise injected at the last step, and it is invisible
because a champion curve looks the same either way.

A separate condition splits the population in two and has the halves play only
each other, with one side given twice the policy capacity. Doubling capacity did
not reliably decide the contest — the outcome is dominated by spontaneous
symmetry breaking, which occurs just as readily in a symmetric control where the
two sides differ only in their random seed. What the condition does show, in 18
runs, is that a bilateral contest is a qualitatively different thing from a
shared ecology: exactly one run ended with both sides holding a competent
individual. The side that falls behind loses every game, and a contest you always
lose carries no gradient, so it stops improving while its opponent continues.

We also report a negative result with a mechanism. Our first hall-of-fame
implementation applied the replacement rule to archive games, making a winning
archived genome the *parent* of the member it beat. That copies old genotypes
back into the pool roughly one game in eight and abolishes learning outright —
no such run reached long rallies or produced a single above-parity checkpoint,
while every control run did both. It also produced the cleanest illustration in
the study of why stability metrics need a competence precondition: a dead run has
zero volatility and zero drawdown, and so scores as the most stable condition in
the matrix.

Supporting all of this is the engineering that made a seeded design affordable: a
compiled port of the environment, validated bit for bit against the reference
implementation over a quarter of a million environment steps, which turns a
twelve-core-hour run into a half-hour one.

## Why this matters beyond volleyball

A self-improvement loop is exactly as real as its improvement signal. This
repository is the smallest rung of a three-part argument about where that signal
can come from as the world being acted in opens up:

| | what evolves | where the signal comes from |
|---|---|---|
| **this work** | 273 weights | internal, relative: beat a peer |
| Backprop NEAT (Ha, 2016) | topology, with gradients inside | division of labour between search and local optimisation |
| ACTIR / ShinkaEvolve | programs, with an LLM as the mutation operator | strategies selected against each other in a simulated system, measured on held-out scenarios |

The load-bearing claim inherited upward is the *separation of ecology from
yardstick*: the thing that selects and the thing that measures must never be the
same object. What this study adds is a caveat with teeth. When selection is
purely relative, there is a third object in the loop — the mechanism that decides
which candidate to *report* — and here it contributed more noise than the
coevolution did. Any evolutionary loop that promotes one artefact out of a
population, including an LLM-driven one that promotes one program out of a
generation, inherits that failure mode, and it is cheap to check and cheap to fix
once named.

## Contents

- [Methods](01-methods.md) — task, algorithms, interventions, the compiled
  environment and its validation, the pre-registered evaluation protocol.
- [Results](02-results.md) — the reference run, seed variance in the phase
  change, and what competence looks like concretely.
- [Ablations and analysis](03-ablations-and-analysis.md) — transitivity, the
  export rule, archives as parent and as test, mutation scale, population size,
  and two further algorithm families.
- [Appendix](04-appendix.md) — self-contained; every table, every run, the
  decision log, and the reproduction commands.
