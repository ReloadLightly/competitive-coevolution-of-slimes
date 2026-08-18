# Decision log for the experiment matrix

Anything that changed after the protocol was fixed is recorded here, with what
was known at the time.

## 2026-08-18 14:41 UTC — protocol fixed, matrix launched

27 runs: control 6 seeds, hof-0.25 6, hof-0.50 3, sigma-0.05 3, sigma-0.20 3,
pop-32 3, pop-512 3. Protocol in `protocol.json`. Budget was set from an
estimated ~30 min per run.

## 2026-08-18 15:00 UTC — seed count increased

Runs turned out to take ~17.3 min rather than ~30, leaving spare capacity.
The seed count was therefore raised to 12 for `control` and `hof-0.25` and to 6
for the four side conditions.

State of knowledge at the time of this decision: exactly one run of the matrix
had finished (`control_s103`: final -0.29, peak +0.17). No hall-of-fame run had
completed, no condition had been compared with another, and
`analyze_matrix.py` had not been run on any real data. The change is therefore
a budget decision, not a data-dependent one.

Seeds 101-106 (all conditions) come from the first wave; 107-112 (control,
hof-0.25) and 104-106 (side conditions) come from the second. All are reported
together; none is dropped.

## 2026-08-18 15:07 UTC — second wave narrowed

Wave-1 runs turned out to vary between ~17 and ~35 minutes: a run whose
population reaches long-rally play early spends the rest of its budget on
3,000-step games and is correspondingly slower in wall-clock. The projected
finish for the full second wave no longer fitted the session.

The second wave was therefore narrowed to the headline comparison only —
`control` and `hof-0.25`, seeds 107-112, taking both to 12 seeds. The four side
conditions stay at their pre-registered 3 seeds. State of knowledge: one run
finished (`control_s103`), no hall-of-fame run finished, no comparison run.

## 2026-08-18 15:14 UTC — `hof-full` added

Reviewing the archive design before any hall-of-fame run had started: with a
champion archived every 1,000 games and a FIFO capacity of 64, the archive of
`hof-0.25` and `hof-0.50` spans only the most recent 64,000 games of a 500,000
game run. That is a recency buffer, not a hall of fame in the sense the
literature means, and it tests a weaker hypothesis than intended.

Rather than discard the runs already in flight, a condition `hof-full` was
added: same dose as `hof-0.25` (p = 0.25) but capacity 512, which at one entry
per 1,000 games holds every champion the run ever produced. The two together
separate "play recent past selves" from "play every past self".

`hof-0.25` and `hof-0.50` keep capacity 64 for every seed, first wave and
second, so the condition stays internally consistent. State of knowledge: two
control runs finished (s101 final +0.14, s103 final -0.29); no hall-of-fame run
had started.

Queue order after the first wave: `hof-full` (6 seeds) first, since a new
condition carries more information than extra seeds of an existing one, then
`control` and `hof-0.25` seeds 107-112.

## 2026-08-18 16:40 UTC — the first archive design was wrong; `hof-eval` added

The container was restarted at ~16:15 (nothing was lost beyond in-flight runs;
all completed results had been pushed). On restart, three `hof-0.25` runs
finished, and they finished suspiciously fast — 11 to 15 minutes against 17 to
44 for the control. The reason is that their games stayed short:

| condition | training rally length, first -> last checkpoint | checkpoints above parity |
|---|---|---|
| control (5 seeds) | 630 -> 2985 | 3 to 32 of 100 |
| hof-0.25 (3 seeds) | 632 -> 630..727 | 0 of 100, every seed |

So the intervention did not fail to stabilise competence; it abolished learning.

Diagnosed mechanism: `fastvolley.run_ga` applies Ha's replacement rule verbatim
to archive games, so when an archived genome wins, the population member is
overwritten by a mutant *of the archived genome*. With p = 0.25 and a measured
archive win rate of ~0.5, about 12.5% of all games therefore copy an older
genotype back into the pool — a standing regression pressure that cancels
progress. That is not how a hall of fame is used in the literature
(Rosin & Belew, 1997), where the archive supplies opponents for fitness
evaluation and archive members are never parents.

Rather than delete the runs, both readings are now tested:

* `hof-0.25`, `hof-0.50`, `hof-full` keep the original rule and are reported as
  what they are — an archive that is also a gene source.
* `hof-eval` (new, 4 seeds, archive capacity 512 so it spans the whole run)
  implements the literature's reading: an archive game that the population
  member loses costs it its slot, but the replacement genes come from the
  living pool. Genetic material never leaves the population.

`hof-eval` is queued ahead of the new algorithm families, because without it
the study's headline question is answered only by a broken design.

## 2026-08-18 19:20 UTC — `hof-0.50` dropped from the remaining queue

The queue was flattened from six sequential per-condition invocations into one
ordered pass, because each invocation ended with fewer jobs than workers and left
cores idle at every boundary.

`hof-0.50` was dropped from the remainder at the same time. One seed of it had
completed (final -4.88, 0/100 above parity), matching `hof-0.25` (5 seeds) and
`hof-full`. A fourth dose of an archive design already shown to abolish learning
adds no information. The single completed seed is retained and reported.

## 2026-08-18 19:35 UTC — population sweep replaced by an unequal-power condition

The within-population size sweep (`pop-32`, `pop-512`) was dropped in favour of
a condition the study otherwise cannot speak to at all: **two populations of
unequal power playing only each other**.

Every other condition in this study is symmetric — one pool playing itself, all
agents with the identical policy class and budget — which is exactly the setting
where "compete harder" is the only available move. The new condition gives the
strong side roughly twice the policy capacity of the weak side (12-16-16-3, 531
parameters, against the standard 12-10-10-3, 273; ratio 1.95:1) and asks whether
the weaker side can hold parity. A symmetric two-population control (both sides
273 parameters) separates the effect of the asymmetry from the effect of
two-population coevolution as such.

The variable-capacity forward pass was checked against the fixed 273-parameter
kernel at h=10 and is bit-identical, so the weak side is running exactly the
policy used everywhere else in the study.

Three seeds per condition. `pop-32` / `pop-512` remain defined in
`run_experiments.py` and can be run later; they are simply not part of this
session's results.
