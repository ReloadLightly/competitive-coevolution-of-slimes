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

## 2026-08-18 21:35 UTC — the asymmetric kernel was rewritten after a null run

The first two-population kernel replaced a loser with a mutant of a *random*
peer from its own population. That is too weak a rule: winners never
preferentially propagate, so selection reduces to culling. Measured result, over
a full 500,000-game symmetric control run: training rally length flat at ~630
from start to finish, both sides at -4.84 against the 2015 baseline, and the
cross-population win rate pinned at 0.500. Neither population learned anything.

Recorded here rather than quietly fixed, because it is a real measurement about
selection rules: *culling losers is not enough; something has to propagate
winners.*

The kernel now gives each population Ha's rule verbatim for its
within-population games and crosses a fraction of games with the other
population, where a loss costs the member its slot but the replacement genes
come from its own pool. That is the `hof-eval` rule with a live co-adapting
opponent in place of a stale archived one, and both of those components are
known from this study to support learning.

Budget was also raised: two populations sharing one game budget each get less
within-population selection than a single pool. At CROSS = 0.25 and 750,000
games, each side is active in ~375,000 games, ~281,000 within its own pool,
against the control's 500,000 — close enough that a failure to transition is
informative rather than an artefact of starvation. The three earlier
asymmetric result files were deleted; nothing else in the matrix is affected.

## 2026-08-18 23:00 UTC — asymmetric condition re-implemented and rerun

The first properly-learning asymmetric runs produced an interpretable headline
(the contest is winner-take-all, and a 1.95:1 capacity advantage does not decide
it) but rested on measurements that this study's own section 2 shows are
unreliable. Rather than report that with a caveat, the condition was
re-implemented and rerun from scratch. Three changes:

1. **A streak-accounting bug is fixed.** When a population member lost a
   cross-population game it was replaced by a mutant of a randomly drawn peer,
   and *that peer's streak counter was incremented* although it had not played.
   In the losing population — which loses nearly every cross game — this
   inflated streak counters at random, and the streak counter is what selects
   the exported champion. The peer now inherits nothing it did not earn.

2. **Both populations are snapshotted every 50,000 games** and every member is
   scored against the 2015 baseline. "Did this side learn" is now a question
   about the pool, not about whichever individual a proxy exported.

3. **Three promotion rules are recorded side by side** — winning streak,
   internal round robin, and best-in-pool — so no conclusion depends on a
   single champion series.

Seeds were raised from 3 to 6 per condition, because the outcome is close to a
coin flip and three seeds cannot separate a capacity effect from
symmetry-breaking noise. The game budget was reduced from 750,000 to 500,000 to
pay for them; at CROSS = 0.25 each side is still active in ~250,000 games,
~187,000 of them within its own pool, against a control median transition of
120,000. Jobs are interleaved by seed so that an interrupted session yields
fewer seeds of every condition rather than no seeds of some.

The earlier asymmetric result files were deleted. Nothing else in the matrix is
affected.

## 2026-08-19 01:45 UTC — final inventory

59 runs completed: 41 single-population and 18 two-population, plus the
reference run on the unmodified environment, which reached its full 500,000
games. Seed counts per condition are in Table A2 and vary by design (see the
entries above): 6 for the conditions that carry the argument, 3 for the
mutation-scale sweep, 1 for `hof-0.50` and `pop-32`, which were dropped from the
queue in favour of the unequal-power condition and are reported with the single
seed they have rather than deleted.

`pop-512` never ran. The population sweep was replaced by the unequal-power
condition; `pop-32` has one run because it had already started when the queue
was stopped. Both remain defined in `run_experiments.py` and can be run later.

## 2026-10-02 — streak bug in `run_ga_hof_eval`; `hof-eval` superseded by `hof-eval-v2`

`algorithms.run_ga_hof_eval` (the archive-as-test kernel) carried the same
streak-accounting bug that was fixed for the asymmetric kernel on 2026-08-18
(commit `f362bf0`). When an archived champion beat population member `m`, `m`
was overwritten by a mutant of a living peer `n` and inherited `n`'s streak —
and then `n`'s streak was incremented, although `n` had not played. The fix
removes that increment; nothing else in the kernel changes.

The streak is not cosmetic in this kernel. `argmax(winning_streak)` picks the
individual copied into the archive every 1,000 games, so the bug changed the
archive's contents and therefore the opponents of a quarter of all games: the
whole trajectory, not only the exported champion. The six `hof-eval` runs can
therefore not be repaired by re-exporting; they are rerun as a new condition,
`hof-eval-v2`, with identical parameters (p = 0.25, archive capacity 512,
seeds 101-106). The `hof-eval` files stay on disk under their names, are
listed under `superseded` in `protocol.json`, and are excluded from every
table, figure and claim from now on.

State of knowledge at the time of this decision (before any `hof-eval-v2` run
started), from the superseded `hof-eval` runs:

| `hof-eval` (superseded), 6 seeds | value |
|---|---|
| learned to rally | 6/6 |
| best champion, held out (mean; median) | -0.44; +0.32 |
| end-of-run champion, held out (mean) | -1.10 |
| checkpoints above parity (mean) vs control | 7.7% vs 26% (exact MWU p = 0.028) |
| archive win rate, first checkpoints -> last 50,000 games | ~0.50 -> 0.03-0.14, every seed |
| share of games with no selection event, last 50,000 games | 21-24% |
| largest exported streak counter | 3,645-4,201 |

These are the numbers behind claim C5b ("archive as test is useless; its win
rate decays from 0.50 to under 0.10"). C5b is withdrawn until `hof-eval-v2`
is analysed, and will be rewritten from the new runs alone, whatever they show.
The comparison will be recorded in a follow-up entry.

## 2026-10-02 — reproducing the analysis files; proxy episode count corrected

To give every analysis file a provenance record (`results/analysis/provenance.json`,
see `provenance.py`), each one is being regenerated by the script that
produced it and compared with the committed version before it is replaced.

* `reference_curve.json` (eval_reference.py, defaults) reproduces **byte for
  byte**.
* `champion_proxy.json` does not reproduce with the script's default of 40
  episodes per individual. It reproduces exactly with 30: checked on two
  snapshots of `control_s103` (exported, best and median scores all equal to
  four decimals at 30, all different at 40 and 60). The committed file was
  therefore produced with `--proxy-episodes 30`, which no document recorded.
  `PROXY_EPISODES` is set to 30, the value the published numbers rest on, so
  the documented command reproduces them. No number changes.
* `reexport.json` records 60 episodes per individual in its own summary and is
  rerun with `--episodes 60`.

## 2026-10-02 — `hof-eval-v2` analysed: C5b rewritten from the new runs

All six `hof-eval-v2` seeds finished (78.6 min on four workers) and the
analysis was rerun over every file on disk with the superseded `hof-eval`
runs excluded. Before (superseded `hof-eval`, streak bug) and after
(`hof-eval-v2`), six seeds each; control for reference:

| | `hof-eval` (bug) | `hof-eval-v2` | control |
|---|---|---|---|
| learned to rally | 6/6 | 5/6 | 6/6 |
| at least one checkpoint above parity | 4/6 | 4/6 | 6/6 |
| internal transition, median (range) | 350k (155k–495k) | 150k (115k–190k) | 120k (55k–415k) |
| checkpoints above parity, mean (median) | 7.7% (3.5%) | 17.7% (21%) | 26% (26%) |
| best champion, held out, mean (median) | -0.44 (+0.32) | -0.55 (+0.38) | +0.32 (+0.36) |
| end-of-run champion, held out, mean (median) | -1.10 (-0.33) | -1.01 (-0.07) | -0.15 (+0.02) |
| vs control, above parity: Cliff's δ, exact p | -0.75, 0.028 | -0.33, 0.370 | |
| vs control, mean last 100k: δ, p | -0.67, 0.065 | -0.22, 0.589 | |
| vs control, best champion: δ, p | -0.33, 0.372 | 0.00, 1.000 | |
| archive win rate, first 50k games | 0.48–0.51 | 0.46–0.50 | |
| archive win rate, last 50k games | 0.03–0.14 (all seeds) | 0.11–0.16 (the 5 that learned); 0.50 (the one that did not) | |
| games without a selection event, last 50k | 21–24% | 21–22% (learned); 12% (not) | |

What changes in the claim. The old C5b said the archive used as a test
"stops being destructive but still does not help", made runs measurably
worse than the control, and that its win rate decays "from ~0.50 to under
0.10 in every seed". With the bug fixed:

* The archive as a test is **not distinguishable from the control** on any
  outcome at six seeds (all p ≥ 0.37). One seed in six never learned to
  rally, against none of six in the control; that is a difference of one run.
  The corrected archive therefore shows **no detectable benefit and no
  detectable cost**; the earlier "measurably worse" was largely the bug.
* The bug slowed learning: the buggy runs transitioned at a median of 350k
  games, the corrected ones at 150k, in line with the control.
* The decay of the archive's win rate holds qualitatively — from about 0.5 to
  0.11–0.16 in every seed that learned, so about a fifth of late games
  produce no selection event — but **not "under 0.10"**. In the seed that
  never learned the archive keeps winning half its games: the decay tracks
  whether the population improves, which is what the diagnostic is meant to
  detect, but it is a description of these runs, not a validated predictor.

C6 rechecked with `hof-eval-v2` as the fourth family: unchanged. Only the
control learned to rally and produced an above-parity champion in every
seed; the s.d. of end-of-run champions across seeds is 3.0×, 3.5× and 3.1×
the control's for the generational GA, the ES and the corrected archive; the
highest end-of-run champion of the study is still a self-play ES seed.
