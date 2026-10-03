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

## 2026-10-02 — analysis defaults set to the values the committed files used

So that the reproduction commands in the paper's appendix need no arguments,
three analysis scripts now default to the settings their committed outputs
were produced with (found when reproducing them earlier today): `reexport.py`
60 episodes per individual (default was 100), `validate_fastvolley.py` 50
games per scenario (was 60), `resume_fast.py` 2 continuations (was 3). No
output changes; the provenance records already carry these parameters.

## 2026-10-02 — next phase: section 4 of CLAUDE.md lifted; confirmatory replication

PR #8 (WP1–WP4) was merged into `main` the same evening; that state is tagged
`paper-v1`. Roland then lifted the "out of scope" list in CLAUDE.md section 4
("whatever is a blocker such as what is written in paragraph 4, i overrule
it"): more seeds, new conditions, new environments and algorithms are now
allowed where a work package calls for them. CLAUDE.md section 3b lists the
new work packages (WP6–WP9), ordered by the paper's Limitations section.

What was known when the replication was designed: everything in the paper.
The replication's hypotheses, tests and decision rules were derived from the
claims as published (C1–C5b) and from the original seeds' values, which are
reported next to the hypotheses in `results/replication/PREREGISTRATION.md`.
No replication run existed when that file and `replication.py` were committed.
The replication writes into its own directory, so no existing table, number or
figure changes when its runs land.

## 2026-10-02 — WP7: stronger yardsticks from the slimevolleygym zoo

A new analysis, no change to any run or existing number. Ha's two published
feed-forward policies (zoo GA, 273 parameters; zoo CMA-ES, 743 parameters;
slimevolleygym commit 8ac22434, Apache-2.0, copied verbatim into
`results/zoo/`) were added as fixed opponents, and every final champion of
the 41 single-population runs played 200 games against each
(`yardsticks.py`, seed 20261004). The compiled games were first checked bit
for bit against the reference environment for both network sizes (40 games,
120,000 steps, all identical). Result: the 2015 baseline ranks champions much
as the stronger opponents do, but no final champion beats the zoo GA (README
"Against stronger opponents", paper Appendix A.6, table z).

Found on the way: `make_tables.py` skipped a table marker pair with nothing
between it, and `--check` passed on the empty block. The injection pattern now
accepts an empty block, so a new marker is filled and an unfilled one fails
the check; no existing table changed.

## 2026-10-03 — replication runs move to GitHub Actions (execution only)

The replication was launched in the cloud session's container on 2026-10-02
at 22:55 UTC and relaunched at 23:49 after the container was restarted; the
container was reclaimed again while the session was paused (Roland's usage
limit), at some time after 00:07. One run had finished and was committed
(`hof-eval-v2_s201`, which never learned to rally; 12.6 minutes). Every
other run in progress was lost; none had produced a file, so nothing was
discarded or selected.

The remaining 35 runs execute on GitHub's runners
(`.github/workflows/replication.yml`), one run per job, each committed when
it finishes. Per run, the computation is the preregistered one: the same code,
parameters, seeds, output directory and command (`run_experiments.py --outdir
results/replication --only <condition> --seeds <seed>`), in the environment
pinned by `requirements-lock.txt`. The machine differs from the one that
produced `hof-eval-v2_s201`; floating-point results of the compiled kernels
may then differ in the last bits between machines (e.g. through the C
library's tanh), so a run is reproducible on the machine type it ran on, not
necessarily bit for bit across machine types. Each job log records its CPU and
library versions. This changes where runs execute, not what they compute, and
no run is repeated or replaced: a run whose file exists is skipped.

Check, same day: `control_s206`, the first run GitHub's runner finished, was
rerun in the session container with the identical command. The two files are
identical in all 25 data arrays (every champion, every evaluation score,
every population snapshot; only the wall-clock times differ: 4.7 minutes on
the runner, 9.3 here). For this run, the change of machine changed nothing.

## 2026-10-03 — WP6: the preregistered replication, analysed

`python replication.py` was run once, after all 36 run files existed
(`results/replication/analysis.json`, provenance scope `replication`).
Verdicts, by the rules of `results/replication/PREREGISTRATION.md`: **C1, C2,
C3, C4, C5a and C5b all replicated.**

| claim | original six seeds | replication, seeds 201–212 |
|---|---|---|
| C1 lag > 0 | 6/6 | 10/10 eligible runs, p = 0.001 |
| C2 learned to rally | 6/6 | 10/12 (threshold 10/12) |
| C2 timing spread | 7.5× | 5.8× |
| C3 ρ(Elo, time) | +0.74 (6/6 > 0) | +0.79 (12/12 > 0) |
| C3 cyclic triads | 0.2% | 0.5% |
| C4 exported rank | 64 of 128 | 56 of 128 (12/12 outside top quarter) |
| C4 ρ(streak, skill) | +0.04 | +0.04 [+0.02, +0.06] |
| C4 declines exported / best | 8.6 / 1.1 | 21.2 / 7.6 (10/12 runs, p = 0.019) |
| C5a learned, parent vs control | 1/6 vs 6/6 | 1/12 vs 10/12 |
| C5b archive as test vs control | all p ≥ 0.37 | all p ≥ 0.44 |

Weaker than the six seeds suggested, and to be said where the claims are:
two of twelve control runs never learned to rally (C2 holds by its 10/12
threshold, but "in every seed" was a six-seed statement); and the
description attached to C5b without a decision fails: the late archive win
rate fell to 0.06–0.16 in the ten archive-as-test runs that learned, but one
of the two runs that did not learn also ends at 0.21, so a low archive win
rate does not by itself indicate learning.

## 2026-10-03 — WP8: the discmix experiment, analysed

`python lab_analysis.py` was run once, after all 96 run files existed
(`results/lab/analysis.json`, provenance scope `lab`). Verdicts by the rules
of `results/lab/PREREGISTRATION.md`:

- **H8a holds**: within-run cycling rises with λ (cyclic share of champion
  triads 0 / 1.4 / 5.7 / 13.5% in control; ρ = +0.97, p < 0.001 in both
  modes).
- **H8b holds**: at every λ the exported individual sits outside its pool's
  top quarter (mean rank 52–63 of 128; Holm-corrected sign tests all
  rejected); ρ(streak, strength) +0.07 to +0.13.
- **H8c does not hold**: the archive-as-test effect does not grow with λ
  (trend p = 0.116). Per λ (no decision preregistered): δ = −0.44, −0.29,
  −0.47, +0.06 at λ = 0, 0.25, 0.5, 0.75 (p = 0.068, 0.242, 0.052, 0.843).

Found when reading the results, and reported with them: the external panel
(64 initial-distribution genomes) is beaten almost always by the evolved
pools at λ ≤ 0.5, so the summed declines there are about zero for every
member and say nothing; only at λ = 0.75 do they discriminate (exported 1.78,
best member 0.08). The decline was a descriptive column, not a test.

As the preregistration requires for a failed H8c, the paper's Limitations
bullet on cyclic games now reports what was found; H8b's result is added to
the Discussion's paragraph on other loops.

## 2026-10-03 — WP9, part 1: niche archive; WP8's protocol.json restored

`results/lab/protocol.json` was missing: the WP8 workflow committed only the
run files, not the protocol record `run_lab.py` writes when it starts. It was
regenerated from `run_lab.py`'s constants; every one of the 96 WP8 run files
agrees with it on each parameter it records (λ, mode, seed, budget,
checkpoint and snapshot spacing, the game's α, β, noise and tie band).

`lab/kernels.py` gained a fourth archive mode, `HOF_NICHE` (an archive used
as a test, organised by a behaviour grid instead of by time), and `run()`
now also returns the archive size per checkpoint. The existing modes are
unchanged: `test_repo.py` still finds the lab bit-identical to the paper's
three kernels on Slime Volleyball, and WP8's `discmix-0.50-test_s305`,
rerun with the changed kernel, is identical in all 14 data arrays.
`run_lab.py` only unpacks the extra return value.

The niche experiment is preregistered in `results/qd/PREREGISTRATION.md`
(60 runs, three hypotheses). Its grid bounds were set from the descriptor
spread of champions in runs that already existed (the replication's control
and archive-as-test runs; all WP8 runs), not from any niche run.

## 2026-10-03 — WP9, part 2: NEAT as a fifth family, preregistered

`lab/neat.py` adds NEAT to the lab as a new module; the paper's frozen files
are untouched. It plays the study's compiled game through the same physics:
a NEAT genome encoding a 12-10-10-3 network computes that network's outputs
bit for bit, scores identically against the 2015 baseline on the same
serves, and plays identical games step by step (`test_repo.py`). Self-play
evaluation and export are the generational GA's (`run_ga2015`); the budget
is the study's 500,000 games.

Two short pilots (seeds 7 and 999, at most 100,000 games, scratch
directories, never scored against anything) changed the implementation
before the preregistration was written:

- with weights moved in steps of 0.1, a fixed compatibility threshold of 3.0
  was never reached and every genome stayed in one species. The threshold
  now adapts towards 8 species, the mechanism NEAT's coevolution study uses
  (Stanley & Miikkulainen 2004, Appendix A, target 10);
- crossover could join two loops into a feedforward cycle (one parent closed
  a loop 13→14, the other 14→13, and the child took each gene's "recurrent"
  flag at random). The child now takes the flag from the fitter parent;
  `test_repo.py` builds that case, and the old rule fails it (58 of 200
  children cyclic).

Design, the one test (H9d: NEAT vs the generational GA, end-of-run champion
held out, two-sided) and the descriptive measures are fixed in
`results/neat/PREREGISTRATION.md`; the analysis is `neat_analysis.py`
(provenance scope `neat`). Both NEAT references were verified and added to
`paper/refs.bib` (`stanley2002evolving` via the publisher's page,
`stanley2004competitive` via arXiv); the 2004 paper leaves
`UNVERIFIED_REFS.md`.

## 2026-10-03 — WP9, part 1: the niche archive, analysed

All 60 run files of `results/qd/PREREGISTRATION.md` arrived from
`.github/workflows/qd.yml`; each was checked against `results/qd/protocol.json`
(seed, condition, λ, grid bounds, archive probability, archive size within
1–64), with no mismatch. `python qd_analysis.py` was then run once
(`results/qd/analysis.json`, provenance scope `qd`). Verdicts by the
preregistered rules:

- **H9a holds**: in Slime Volleyball the behaviour-organised archive, used as
  a test, has no detectable effect against the replication's control on the
  same seeds (final δ −0.11 p 0.67, peak δ −0.24 p 0.35, above parity
  δ −0.23 p 0.35, late mean δ −0.26 p 0.29; 10/12 learned in both arms).
  Against the time-ordered archive every p ≥ 0.51. The archive was not
  empty: 37–53 of 64 cells occupied at the end.
- **H9b does not hold**: the effect against the control does not grow
  detectably with λ (one-sided permutation p = 0.072; Holm needed p < 0.025).
- **H9c does not hold**: at λ = 0.75 the niche archive's final champions are
  not detectably stronger than the time-ordered archive's (one-sided
  p = 0.34).

As the preregistration requires for H9a holding and H9b, H9c failing: the
paper's C5b paragraph is unchanged; its Limitations bullet on cyclic games
now says that neither archive design helped, and the scope bullet that the
archive as a test was tried in two designs. README: new section "An archive
organised by behaviour" (tables qd, qdm).

## 2026-10-03 — WP9, part 2: NEAT failed; exploratory follow-up (not preregistered)

`python neat_analysis.py` was run once after all 12 preregistered NEAT runs
existed (each checked against `results/neat/protocol.json`). **H9d finds a
detectable difference, against NEAT**: NEAT learned to rally in 0 of 12 runs;
its final champions score −4.84 ± 0.01 against the 2015 baseline, against
−2.00 for the generational GA (Cliff's δ −0.92, two-sided exact
p = 0.0008), and rank last in the cross-run tournament (median Elo −406;
the lowest other family +143). The champions grew 23–39 hidden nodes and
144–170 enabled connections. The verdict stands as preregistered.

A result this extreme could be an implementation defect, so the following
checks were made afterwards, knowing the result:

- the game path, again: a minimal NEAT genome computes tanh(Wx + b) exactly;
  a pack of 30 grown genomes computes each one as a pack of one does; and
  NEAT-vs-NEAT games of four encoded MLPs equal the paper's MLP-vs-MLP games
  game for game, on both sides of the net. The fitness sign (the right-hand
  player's score credited to it, the left-hand player's debited) matches
  `algorithms.run_ga2015`.
- three exploratory variants, first run in a scratch directory (seeds
  901–903, 901–904 for `as-ga`), all at the full budget:
  - `mlp-start` (NEAT as preregistered, but starting from the study's
    network): 0/3 learned;
  - `no-reset` (no weight replacement): 0/3 learned;
  - `as-ga` (NEAT's loop reduced towards the generational GA: the study's
    network, no structural mutation, one species, no stagnation removal,
    always crossover; one elite instead of twenty): 1/4 learned and reached
    parity (seed 902: rallies above 1,500 steps at 245,000 games, final
    +0.05); seed 901 ended with rallies of 1,494 steps.

So the game, the evaluation and the loop can learn; what stops it is NEAT's
own reproduction machinery in this setting (speciation, offspring shared
out by species, one elite per species, structural mutation), not the
minimal start and not the weight resets alone. To make these numbers
citable they are rerun from committed code: `neat_explore.py`, all three
variants on seeds 901–904 (12 runs, `.github/workflows/neat-explore.yml`),
into `results/neat/explore/`, summarised by `neat_explore.py --summary`
(provenance scope `neat_explore`). They are exploratory throughout and are
labelled so. `lab.neat.run` gained an optional `init` argument for the
variants that start from the study's network; the default path is
unchanged (the first champion of `neat_s101` reproduces exactly).

### 2026-10-03 — the exploratory NEAT runs, from committed code

All 12 runs of `neat_explore.py` arrived (`results/neat/explore/`,
summary `summary.json`): `mlp-start` 0/4 learned, `no-reset` 0/4, `as-ga`
1/4 learned and reached parity (seed 902, rallies above 1,500 steps at
245,000 games, final +0.05). Ten of the eleven runs that also ran in the
scratch directory are identical there and on GitHub's runners. The
eleventh, `no-reset` seed 903, differs because the scratch run was wrong:
the scratch script switched `mlp-start` on by replacing a module function
inside its worker processes, and a worker reused after an `mlp-start` job
started this `no-reset` run from the study's network too (it ended with 50
hidden nodes, 20 of them from the start; the committed run ends with 34).
The committed runs pass the starting genome as an argument, so they cannot
be contaminated this way. The conclusion is unchanged; the numbers quoted
are the committed ones.

## 2026-10-03 — WP10: alternative export rules, preregistered

Roland asked to test alternative export rules. What was known: the paper's
post hoc re-export analysis (`reexport.py`, the 6 original control runs)
found that exporting the winner of a short internal round robin (16 peers per
member, 1,024 games) instead of the longest streak raised the exported
member's mean score from −1.95 to −1.49 (all 6 runs higher), with budgets
chosen after seeing the data. No other run had been analysed for this
question; in particular the replication's and WP8's population snapshots
were not.

`results/export/PREREGISTRATION.md`, `run_export.py` and `export_analysis.py`
fix a prospective test before any run: 12 fresh Slime Volleyball control
runs (seeds 401–412) and 48 fresh discmix control runs (seeds 501–512), the
streak rule against the 16-peer tournament applied to the same stored
populations, four hypotheses in two Holm families (H10a level and H10b
declines in Slime Volleyball; H10c outsider level and H10d its trend over λ
in discmix). The runners are the existing ones, unchanged (`run_experiments.
one_run`, `run_lab.one_run`). New: `stats_utils.signflip_greater` (exact
paired sign-flip test, checked against brute force) and the discmix
tournament, which draws each game as `lab.kernels.discmix_play` does
(`test_repo.py` checks both). The analysis was exercised on synthetic
stand-in files only.

## 2026-10-03 — WP10: alternative export rules, analysed

All 60 run files of `results/export/PREREGISTRATION.md` arrived from
`.github/workflows/export.yml`; each was checked against
`results/export/protocol.json` (seed, condition, λ, snapshot spacing,
budget, archive probability 0), with no mismatch. `python
export_analysis.py` was then run once (`results/export/analysis.json`,
provenance scope `export`). Verdicts by the preregistered rules:

- **H10a holds**: in Slime Volleyball the 16-peer internal tournament
  exports better members than the streak rule (level −2.16 against −2.33,
  9 of 12 runs higher, exact one-sided p = 0.010; Holm needed 0.025). The
  gain (+0.17) is smaller than the post hoc estimate for the same budget
  (+0.46): it closes 43% of the gap to the best member, not 57%.
- **H10b does not hold**: summed declines 1.37 against 1.98, fewer in 8 of
  12 runs, p = 0.16.
- **H10c holds**: in discmix the tournament's members are stronger against
  other runs' populations, 42 of 48 runs, p < 0.001.
- **H10d holds**: that advantage shrinks as skill becomes cyclic, +0.11 at
  λ = 0 to +0.03 at λ = 0.75 (ρ = −0.43, p = 0.001).

Noted when reading the results, not tested: 10 of the 12 Slime Volleyball
runs learned to rally (as in the replication); in the two that did not,
every member is equally weak and the rules tie.

As the preregistration requires: C4 gains a prospective, deployable remedy
(the README's new finding 4b; the paper's export-rule section, where the post
hoc analysis is now the origin, not the evidence); the text says the remedy
did not detectably reduce declines; and the Discussion's hypothesis about
other loops gains the caveat that the remedy weakens as skill becomes cyclic.
New claim C8 in CLAUDE.md.

## 2026-10-03 — WP11: one narrative for README and paper

No data, analysis or claim changed. The README was reorganised from one
section per work package into one argument (findings renumbered 1–8, the
export-rule test becoming finding 5) by a script that moved blocks and
checked that every generated number and table survived; the paper gained a
results roadmap, a subsection that collects the lab experiments (cyclic
skill, the behaviour-organised archive, NEAT) and an appendix with their
tables, while the Limitations now state limits rather than repeat results.
Two wording corrections came with it: the paper's first contribution no
longer says an internal round robin "recovers most of the loss" (the
preregistered test recovered 43% of the gap), and undefined claim labels
(C6, C7) in the paper were replaced by section references.

## 2026-10-03 — WP12: a review of repository and paper

Roland asked for a thorough review ("do everything in our might to address
issues"). No raw result, preregistered verdict or frozen file changed. What
the review found, and what changed:

**The export check compared the exported member with an inflated best
member (C4).** `coevolution_analysis.py --proxy` scores every member of a
population snapshot on 30 episodes and compares the exported member with
the member that scored best on those same episodes; `reexport.py` does the
same on 60 episodes, and so does the declines criterion of the replication.
The maximum of 128 noisy scores is biased upward (a winner's curse), and a
maximum fluctuates less than one member's score, so both the gap and the
declines comparison leaned towards the claim. What was known: every champion
in the paper is re-scored on held-out episodes; this comparison alone had no
held-out step. `proxy_heldout.py` (new, provenance scope `proxy_heldout`)
keeps every choice (exported: largest streak; best and median: by the
30-episode scores) and re-scores the chosen members on the held-out seed
(1,000 episodes, `SELECT_SEED`), plus an independent 30-episode retest
(seed 20261015) for the reliability of the per-member scores. Before → after,
the six original control runs:

| quantity | before (selecting episodes) | after (held out) |
|---|---|---|
| gap, best minus exported | 0.90 | 0.63 (best member's inflation 0.27) |
| summed declines, exported / best / median | 8.6 / 1.1 / 0.9 (60-episode re-export scores) | 7.8 / 2.0 / 1.9 |
| ratio, exported over best | about 8 | 3.8; exported fell further in 6 of 6 runs |
| exported vs median member, level | — | −1.92 vs −1.78 |
| retest correlation of the 30-episode scores | — | median r = 0.94 (snapshots whose scores vary) |
| last snapshot, exported member | −0.08 (30 episodes) | −0.15 |

The replication's twelve control runs, held out: the exported member lost
19.5 points between snapshots, the best member 7.7, more in 9 of 12 runs
(one-sided sign test p = 0.073; the preregistered measure gave 21.2 vs 7.6,
10 of 12, p = 0.019). The preregistered verdict stands as preregistered (the
analysis ran once and is not redone); the README and paper report the
held-out reading next to it as the more conservative one. Export-test runs:
held-out gap 0.37, exported fell further in 10 of 12. C4's qualitative
claims hold; its sizes are now the held-out ones (README finding 4, paper
§3.4 and Discussion, Table 5's score and gap columns, Figures 6 and 9, the
caption of Table 9, whose oracle row stays inflated and is labelled so).

**C6 is weakened.** "The families differ in reliability, not ceiling"
rested on the original control's 6 of 6 runs learning and its end-of-run
spread (s.d. 0.66, against 2.00–2.29 for the generational GA, ES and archive
as test). The replication (preregistered for C2, not for C6) ran 12 fresh
control runs: 10 of 12 learned, as often as the archive as test on the same
seeds, with s.d. 1.92. This is a re-reading of data that already existed,
not a test. C6 now reads: the families reach a similar ceiling; six seeds
could not rank their reliability (README finding 7, paper abstract,
contributions, §3.7, Limitations).

**C8's comparison with the post hoc analysis.** The text compared shares of
the gap (43% prospective vs 57% post hoc), whose denominators differ (the
post hoc oracle is inflated, the prospective one held out). It now compares
gains for the same 1,024-game budget: +0.17 prospective vs +0.46 post hoc.

**Smaller corrections** (commit ad837c0 and this one): `lab/games.calibrate`
and `docs/lab.md` said α and β were calibrated at the runs' initial scale
0.5; they were calibrated at 1.0 (at 0.5 the two parts' spreads are 0.89 and
0.85, still about equal; no result changes). `yardsticks.py`: the 2015
baseline has 112 parameters, not 120. "Highest champion of the study" became
"of the matrix" (the macro is computed from the matrix). README quickstart
and paper Appendix C list the commands of every follow-up. Related work
engages with Seals & Tauritz (2026), abstract read on the publisher's page.
The paper gains Figure 12 (the preregistered export-rule test). Figure
legends moved off the data.

**Noted, not changed.** H10c's 48 paired differences are not fully
independent (each run's outsiders are the other runs' populations at the
same λ); the margin is large (42 of 48, p < 0.001), but the p-value is
optimistic (`paper/REVIEW_NOTES.md`, item 21). Cliff (1993) and the numba
paper still could not be verified and stay uncited.

## 2026-10-03 — an external review, checked against the code; next step WP13

Roland shared a review of main at `d71514a` (written outside this
repository) and asked which step strengthens the scientific core next. Each
factual point was checked against the code and the run files before
anything changed:

- **Correct: the export test's gain is against one yardstick.** Against the
  zoo GA the final champions of the streak rule and the 16-peer tournament
  scored alike (−1.26 and −1.25; the tournament's lower in 7 of 12 runs).
  The README's table carried the column, the prose did not say it. C8 now
  says its gain is against the 2015 baseline and was not seen against the
  zoo GA (a description: one snapshot, 200 games each).
- **Correct: the discmix export tests treat dependent runs as
  independent.** `lab.kernels` draws the initial population from the seed
  alone, so the four runs that share a seed (one per λ) start from the same
  population, and runs at one λ are scored against each other's
  populations. `export_robustness.py` (not preregistered) repeats H10c with
  the seed as the unit (positive for 12 of 12 seeds, exact p = 0.00024) and
  H10d with λ permuted only within seeds (p = 0.0015). The preregistered
  verdicts stand; the shared outsider panels remain a dependence no test
  here removes.
- **Correct: C3 was worded beyond its measurement.** Transitivity was
  measured on the sequence of exported champions, not on the population.
  The text now says "the champions are not cycling" and states the
  inference: the exported champion is close to a random member of its
  population (C4), so the sequence samples the population as it moves.
  Cycles among members of one population at one moment were not measured.
- **Correct: the niche archive is not MAP-Elites proper.** The latest
  streak-chosen champion to enter a cell replaces the occupant (no
  competition within a cell), the archive serves only as opponents, and it
  has 64 slots where the time-ordered archive it was compared with has 512.
  The Slime Volleyball niche runs stored no population snapshots. README
  and paper now say so, and the Limitations list the archive that keeps each
  niche's best occupant as untested.
- **Correct: the NEAT explanation named a bundle.** The exploratory `as-ga`
  variant removed structural mutation, speciation, species retirement and
  the crossover settings together, so it cannot say which mattered. The
  retirement rule judges a species by its best self-play fitness ever, a
  score against opponents that keep changing; that is a plausible mismatch
  with coevolution, untested. The review's description of NEAT's own
  coevolution study was verified on arXiv 1107.0037: two populations, each
  network against the four best species' champions and eight hall-of-fame
  opponents, 24 games; that study also handled stagnation differently (the
  lowest-performing species older than 30 generations may not reproduce).
  README and paper now say which mechanisms were removed together and how
  this setup differs from the published one.

**Where the review's order was not followed, and why.** The review put a
population-export experiment (exporting a mixture of members instead of
one) first. In Slime Volleyball skill was transitive at every resolution
measured, and against one opponent distribution a mixture's mean score
cannot exceed its best member's, so that experiment's Slime arm would mostly
re-measure transitivity; it is planned after WP13 (WP14), starting with the
measurement C3 lacks (payoffs among members of one population). First
instead: the review's fourth item, the counter's causal ingredients. The
paper's headline (C4) rests on a mechanism the paper states but never
tested ("inherited on every replacement"). Two facts from the stored runs,
noted while checking the review, make the test pressing: at 299 of the 300
stored population snapshots (all 30 control runs that store them), every
member's inherited counter lies within 10% of the maximum (in the other, 126
of 128), and the median member trails the maximum by a median of 10 wins,
0.5% of the count (s101 at the first snapshot: maximum 422, median 412). The
count is almost all shared ancestry, and the argmax is decided by the last
few wins. [Corrected the same day: a first version of this entry said "at
all 180 snapshots" of 18 runs, from a median that had been read as a
minimum.] And late in
training 29–53% of games end in a tie (sigma-0.05 runs, last 20
checkpoints), on which one player is mutated in place and keeps its count.
Because the counter never feeds back into reproduction in the control, a
replay of the same history can carry alternative counters. WP13 in
CLAUDE.md sets out the design: exploratory replays of seeds 101–106, then a
preregistration, then 12 fresh runs.
