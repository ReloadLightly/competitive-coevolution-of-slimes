# The lab: one GA kernel, several games

The paper studies one algorithm in one game. Its Limitations section names the
most important gap: Slime Volleyball turned out to be almost purely
transitive, and in a game where skill is cyclic the archive and the export
rule may behave differently. The `lab/` package is the machinery for asking
that question: the study's algorithm, unchanged, applied to games whose skill
structure can be set.

## Layout

| module | what it holds |
|---|---|
| `lab/kernels.py` | `run`: Ha's tournament-selection GA, compiled, with the game chosen by an integer and the archive by a mode (none, as parent, as test, as a niche-organised test), plus population snapshots |
| `lab/games.py` | game definitions as data for `run`: `slime` and `discmix`; exact expected scores for `discmix` |

The paper's science code (`fastvolley.py`, `fastvolley_kernels.py`,
`algorithms.py`, ...) is not modified; the lab calls it.

## Why the lab can be trusted

`lab.kernels.run` on Slime Volleyball reproduces the paper's kernels bit for
bit: `fastvolley.run_ga` (control and archive as parent),
`algorithms.run_ga_hof_eval` (archive as test) and
`fastvolley.run_ga_with_pops` (snapshots). It draws its random numbers in the
same order, and the game call for Slime Volleyball is `fastvolley.play_game`
itself. `test_repo.py` checks all four on every push, so a change to the lab
that would make it a different algorithm fails CI.

## The games

### `slime`

Slime Volleyball, exactly as in the paper.

### `discmix`: transitive and cyclic skill, mixed by one number

Each genome is the same 273-parameter 12-10-10-3 tanh network as in Slime
Volleyball, initialised and mutated the same way. Two fixed sets of probe
inputs define what a network *is* in this game:

- **skill**: how well its third output classifies 256 skill probes by the
  sign of a fixed random teacher network (mean of label × output, in
  [−1, 1]). Skill belongs to the genome alone, so it orders any population
  transitively.
- **u**: its mean first two outputs on 16 style probes, a point in
  [−1, 1]².

A game between a right player r and a left player l is won by r when
M + ε > τ, lost when M + ε < −τ and tied otherwise, with ε ~ N(0, 0.3²),
τ = 0.05, and

    M = (1 − λ) · α · (skill_r − skill_l)  +  λ · β · (u_r × u_l)

where × is the 2-D cross product. λ = 0 is purely transitive; λ = 1 is the
disc game of Balduzzi et al. (2019), purely cyclic. In random populations the
share of cyclic triads rises with λ: 0, 1.9, 12, 23 and 26% at λ = 0, 0.25,
0.5, 0.75 and 1.

Because M is a known function of two genomes, the expected score of any
genome against any other is known exactly. True skill, true intransitivity
and the true rank of an exported champion in its pool are measured without
sampling noise, which no physical game allows.

How the constants were fixed (all on 2026-10-03, before any `discmix`
experiment):

- α = 4.62 and β = 3.32 give both parts unit spread over random pairs of an
  initial population (`lab.games.calibrate()`), so λ = 0.5 weighs them
  equally at the start.
- The skill task, the noise and the tie band come from pilot runs at λ = 0
  only (seed 11), and are the first setting tried in which skill keeps
  rising over a 500,000-game run. Two earlier tasks failed: regressing onto a
  teacher's outputs stalled at once, and a linear classification rule
  saturated within 50,000 games. No pilot looked at cycles, archives or the
  export rule.

A 500,000-game `discmix` run takes about 3–4 minutes on one core.

## Archive modes

| mode | the archive | used as | in the paper |
|---|---|---|---|
| `HOF_NONE` | none | — | `control` |
| `HOF_PARENT` | the last champions, in time order | parent | `hof-0.25`, `hof-0.50`, `hof-full` |
| `HOF_TEST` | the last champions, in time order | test | `hof-eval-v2` |
| `HOF_NICHE` | one champion per cell of an 8 × 8 behaviour grid, the newest to land there | test | WP9 (`run_qd.py`) |

The niche archive's descriptor is the network's mean first two outputs on
fixed probe inputs (`X3`): real game states for Slime Volleyball
(`lab.games.slime_probes`), the style probes for discmix, where the
descriptor is exactly the point on which the cycle is played. The grid's
bounds are passed in `gp[5:9]`.

## NEAT

`lab/neat.py` is NEAT (Stanley & Miikkulainen 2002) in the study's
self-play setting. Genomes start minimal (every input wired to every
output) and grow hidden nodes and connections; crossover aligns genes by
innovation number; speciation with fitness sharing protects new structure;
recurrent connections are allowed. Reproduction is plain Python on dict
genomes; the forward pass and the games are compiled.

- **The same game.** A NEAT genome encoding a 12-10-10-3 network
  (`neat.from_mlp`) computes that network's outputs bit for bit, scores
  identically against the 2015 baseline on the same serves, and plays
  identical games step by step (`test_repo.py`). So NEAT champions and the
  paper's champions can play each other in one tournament.
- **The same evaluation.** Self-play fitness is the generational GA's (500
  games per generation between random pairs of 100 genomes, mean point
  margin), the budget is 500,000 games, and the generation's fittest genome
  is exported every 5,000 games.
- **Settings.** `neat.PARAMS`. The mechanisms and most rates follow the
  parameter appendix of NEAT's coevolution study (Stanley & Miikkulainen
  2004, arXiv:1107.0037), including its compatibility threshold that adapts
  towards a target number of species (here 8): with weights moved in steps
  of 0.1 a fixed threshold of 3.0 was never reached and every genome stayed
  in one species. Population, weight scale and units are the study's (100
  genomes, σ 0.1, initial scale 0.5, tanh); the preregistration lists which
  of the remaining values were chosen here.
- **A bug found in a pilot.** Taking a matching gene's "recurrent" flag from
  either parent can join two loops into a feedforward cycle (one parent
  closed a loop 13→14, the other 14→13). The child now takes the flag from
  the fitter parent, whose structure it inherits; `test_repo.py` builds that
  case and checks every child compiles.

## Adding a game

1. Give it an integer id in `lab/kernels.py` and a branch in `play`, which
   returns (score from the right player's view, length).
2. Describe its data in `lab/games.make`.
3. If it has a reference implementation, check the compiled game against it
   step by step (as `validate_fastvolley.py` and `yardsticks.py --validate`
   do); otherwise test its defining properties (as `test_repo.py` does for
   `discmix`).

## Status

- Lab kernel, `slime` and `discmix`: done and tested.
- WP8, the discmix experiment ([preregistration](../results/lab/PREREGISTRATION.md),
  96 runs on GitHub's runners, `.github/workflows/lab.yml`): analysed once by
  `lab_analysis.py`. Cycling rises with λ (H8a holds); the export rule picks a
  middling individual at every λ (H8b holds); an archive used as a test does
  not help more as skill becomes cyclic (H8c does not hold). Numbers: README,
  "When skill is cyclic: the lab".
