# Competitive coevolution of slimes

**What a population of self-playing agents learns, what it forgets, and which
of the two you actually measure.**

**Paper (draft, not yet submitted):** [*What a self-playing population learns,
what it forgets, and which of the two you measure*](paper/main.pdf) — LaTeX
sources in [`paper/`](paper/); every number in it is generated from the run files.

Self-play neuroevolution on [David Ha's Slime Volleyball](https://otoro.net/slimevolley/),
run as a designed experiment rather than a demo: <!-- n:n_runs_total -->59<!-- /n --> independent runs of
<!-- n:budget -->500,000<!-- /n --> self-play games each, in a compiled environment that reproduces the
original bit for bit, with every agent scored against a frozen opponent it never
met in training. The headline finding is not the one we expected. Competence is
reached and lost repeatedly, and most of that instability is not coevolution: it
is injected at the last step, by the rule that decides which individual to call
the champion. Ha's rule exports an individual that ranks on average <!-- n:proxy_rank_mean -->60<!-- /n --> of
<!-- n:proxy_pop -->128<!-- /n --> in its own population, its lineage counter is uncorrelated with its skill
(ρ = <!-- n:proxy_rho -->+0.06<!-- /n -->), and the population itself is not cycling. Internal improvement
precedes any external transfer by tens of thousands of games; the phase change is
robust but its timing varies <!-- n:ctrl_t_internal_ratio -->7.5<!-- /n -->-fold across seeds; an archive of past
champions abolishes learning when used as a parent and does nothing detectable
when used as a test. Those claims held in a preregistered replication on <!-- n:rep_total -->36<!-- /n --> fresh
runs (<!-- n:rep_verdicts -->6 of 6<!-- /n --> replicated).

<table>
<tr>
<td align="center" width="20%"><a href="results/figures/fig6_champion_proxy.png"><img src="results/figures/fig6_champion_proxy.png" alt="the export rule"></a><br><sub>The exported champion vs its own population</sub></td>
<td align="center" width="20%"><a href="results/figures/fig1_reference_trajectory.png"><img src="results/figures/fig1_reference_trajectory.png" alt="reference run"></a><br><sub>Internal improvement before external transfer</sub></td>
<td align="center" width="20%"><a href="results/figures/fig2_control_seeds.png"><img src="results/figures/fig2_control_seeds.png" alt="six seeds"></a><br><sub>One phase change, six timings</sub></td>
<td align="center" width="20%"><a href="results/figures/fig5_coevolution.png"><img src="results/figures/fig5_coevolution.png" alt="transitivity"></a><br><sub>No cycling: skill is transitive</sub></td>
<td align="center" width="20%"><a href="results/figures/fig3_hall_of_fame.png"><img src="results/figures/fig3_hall_of_fame.png" alt="archives"></a><br><sub>Archives as parent and as test</sub></td>
</tr>
</table>

---

## What this study found

**1. The internal signal leads the external one by tens of thousands of games.**
A population improves against *itself* long before any of that improvement
transfers to an opponent it has never met. In the reference run, the population's
own rallies pass <!-- n:long_rally -->1,500<!-- /n --> steps at <!-- n:ref_t_internal -->104,200<!-- /n --> games; the first champion that beats the
2015 expert appears at <!-- n:ref_t_parity -->172,000<!-- /n -->. An evaluation stopped in between reports total
failure from a population that is already working.

![reference trajectory](results/figures/fig1_reference_trajectory.png)

*The reference run over <!-- n:budget -->500,000<!-- /n --> self-play games. Top: rally length when the
population plays itself. Bottom: score against the 2015 baseline, which is never
seen during training. The two curves rise tens of thousands of games apart.*

**2. The phase change is real; its timing does not replicate.** Across control
seeds the internal transition happens anywhere from <!-- n:ctrl_t_internal_min -->55,000<!-- /n --> to <!-- n:ctrl_t_internal_max -->415,000<!-- /n --> games — a
<!-- n:ctrl_t_internal_ratio -->7.5<!-- /n -->-fold spread. The single-run version of this repository reported a
single transition time. That was one seed.

![six control seeds](results/figures/fig2_control_seeds.png)

*The same algorithm under six seeds. The phase change appears in all of them and
at a different time in each, from <!-- n:ctrl_t_internal_min -->55,000<!-- /n --> to <!-- n:ctrl_t_internal_max -->415,000<!-- /n --> games.*

**3. The population is not cycling.** Playing every checkpoint of a run against
every other checkpoint, skill is essentially transitive: ρ(Elo, training time) =
<!-- n:rho_within_ctrl -->+0.74<!-- /n --> in the control, and <!-- n:cyclic_ctrl_pct -->0.2<!-- /n -->% of its decided checkpoint triples are
cyclic (at most <!-- n:cyclic_learning_max_pct -->0.2<!-- /n -->% in any condition whose runs all learned). The textbook
explanation for the swings — the population forgets skills no current opponent
punishes — does not fit this data.

![within-run tournament and intransitivity](results/figures/fig5_coevolution.png)

*Every checkpoint of a run played against every other. Left: the margin matrix,
later against earlier. Centre: Elo within the run against training time. Right:
the fraction of decided checkpoint triples that are cyclic.*

Those checkpoints are 50,000 games apart, so a cycle that opens and closes in
less time would be invisible. Every run stores all 100 of its champions, so the
control runs were also played at 5,000-game spacing (<!-- n:fine_games -->20<!-- /n --> games per pair, table
below). Under the rule above (a pair counts as decided when its mean margin
leaves ±0.25), <!-- n:fine_cyclic_deadband_pct -->1.0<!-- /n -->% of decided triads are now cyclic, and <!-- n:fine_short_deadband_pct -->4.5<!-- /n -->% of those spanning
at most 50,000 games. But at 20 games per pair that rule lets sampling noise
decide pairs between near-equal policies, and noise makes cycles: when a pair
counts as decided only if an exact sign test on its wins and losses says so,
<!-- n:fine_cyclic_sign -->35<!-- /n --> of <!-- n:fine_triads_sign -->273,376<!-- /n --> decided triads are cyclic (<!-- n:fine_cyclic_sign_pct -->0.01<!-- /n -->%). At the finer resolution the
population is still not cycling. What does move at that resolution is the
exported champion: where adjacent champions differ significantly, the later one
wins only <!-- n:fine_next_wins -->88/144<!-- /n --> times (<!-- n:fine_next_wins_pct -->61<!-- /n -->%), the export-rule noise of finding 4 seen
from the other side.

<!-- table:t -->
| run | ρ(Elo, time) | cyclic, ±0.25 rule | cyclic, sign test | within 50k games, sign test | next beats previous |
|---|---|---|---|---|---|
| control_s101 | +0.95 | 0.69% (774/112,190) | 0.00% (0/51,268) | 0.00% (0/272) | 10/14 |
| control_s102 | +0.76 | 0.55% (541/98,864) | 0.02% (11/51,840) | 0.23% (1/430) | 18/35 |
| control_s103 | +0.81 | 3.82% (3,675/96,104) | 0.00% (1/21,569) | 0.00% (0/198) | 7/13 |
| control_s104 | +0.68 | 0.50% (431/86,322) | 0.04% (16/43,215) | 0.23% (1/440) | 21/34 |
| control_s105 | +0.91 | 0.22% (241/108,654) | 0.00% (1/58,693) | 0.00% (0/454) | 16/24 |
| control_s106 | +0.85 | 0.30% (253/85,284) | 0.01% (6/46,791) | 0.86% (3/349) | 16/24 |
| *all control runs* | — | 1.01% (5,915/587,418) | 0.01% (35/273,376) | 0.23% (5/2,143) | 88/144 |

Every one of the 100 champions of each control run (one per 5,000 games) played every other, 20 games per pair. A triad counts when all three of its pairs are decided: by the paper's rule (mean margin outside ±0.25) or by an exact sign test on wins against losses (p < 0.05). 'Next beats previous': adjacent champions whose difference the sign test decides, and how often the later one wins.
<!-- /table:t -->

**4. The champion-export rule is the noise source.** Ha selects the individual
with the longest winning lineage, "without actually computing who is best to save
time". That counter is inherited by the loser on every replacement, so it
measures the age of a lineage, not merit. Measured against the whole population:
the exported individual ranks, on average, number <!-- n:proxy_rank_mean -->60<!-- /n --> of <!-- n:proxy_pop -->128<!-- /n --> in its own
pool, the correlation between its streak counter and its actual skill is
indistinguishable from zero (ρ = <!-- n:proxy_rho -->+0.06<!-- /n -->), and it scores <!-- n:proxy_gap -->0.90<!-- /n --> points per episode
worse than the best individual in the same pool. At the last snapshot the
exported champion averages <!-- n:proxy_end_exported -->-0.08<!-- /n --> while <!-- n:proxy_end_above -->67<!-- /n --> of its peers are above parity. The losses of
competence are the export rule's too: summed over the control runs, the exported
champion's score fell by <!-- n:decline_exported -->8.6<!-- /n --> points between consecutive population
snapshots, the best member of the same pools by <!-- n:decline_best -->1.1<!-- /n -->. The population almost
never gets worse; the individual we report does.

![what the export rule costs](results/figures/fig6_champion_proxy.png)

*Left: the exported champion against the best individual in the same population.
Right: Spearman correlation between the streak counter and actual skill, over
training.*

![the re-export experiment](results/figures/fig9_reexport.png)

*Re-exporting the same runs under a ranking rule that spends games measuring who
is best. Left to right: level of the exported individual, volatility of the
resulting curve, and correlation between rule and true skill, as a function of
the ranking budget.*

**5. An archive of past champions does not help when skill is transitive — and
as a parent it destroys learning.** We tested both readings of the archive. Used
as a *parent* (a winning archived genome becomes the parent of the member it
beat) it abolishes learning outright: <!-- n:hofparent_reached -->1/6<!-- /n --> runs ever got off the floor, against
<!-- n:ctrl_reached -->6/6<!-- /n --> for the control. Used as a *test* in the literature's sense (archive
supplies opponents, never genes) it is neither destructive nor helpful: at
<!-- n:seeds_main -->6<!-- /n --> seeds it is indistinguishable from the control on every outcome
(checkpoints above parity <!-- n:hoftest_above_pct -->18<!-- /n -->% against <!-- n:ctrl_above_pct -->26<!-- /n -->%, exact p = <!-- n:hoftest_vs_ctrl_above_p -->0.370<!-- /n -->;
best champion δ = <!-- n:hoftest_vs_ctrl_peak_delta -->+0.00<!-- /n -->), and <!-- n:hoftest_n_failed -->1<!-- /n --> of its seeds never learned. The
archive's win rate against the current population shows why it has little to
do. It starts near <!-- n:hoftest_win_early_max -->0.50<!-- /n --> and, in every run whose population learned, ends
at <!-- n:hoftest_win_late_learned_min -->0.11<!-- /n -->–<!-- n:hoftest_win_late_learned_max -->0.16<!-- /n -->: skill here is transitive, so a past champion is simply a
weaker player, and <!-- n:hoftest_skip_late_learned_min -->21<!-- /n -->–<!-- n:hoftest_skip_late_learned_max -->22<!-- /n -->% of late-training games produce no selection
event at all. In the run that never learned, the archive's win rate stayed at
<!-- n:hoftest_win_late_failed -->0.50<!-- /n -->. Logging an archive's win rate is a cheap way to see
whether it still has anything to teach; here it tracked progress, but that is a
description of <!-- n:seeds_main -->6<!-- /n --> runs, not a validated predictor. An earlier version of this
condition carried a streak-accounting bug that made it look measurably worse
than the control; the corrected rerun and the before/after comparison are in
`results/matrix/decisions.md`.

![hall of fame, both readings](results/figures/fig3_hall_of_fame.png)

*The archive used as a test (opponents only) and as a parent (genes flow back).
As a parent it abolishes learning: <!-- n:hofparent_reached -->1/6<!-- /n --> runs left the floor.*

![archive decay](results/figures/fig10_archive_decay.png)

*Left: the archive's win rate against the current population, falling to
<!-- n:hoftest_win_late_learned_min -->0.11<!-- /n -->–<!-- n:hoftest_win_late_learned_max -->0.16<!-- /n --> in every archive-as-test run whose population learned, and staying
near <!-- n:hoftest_win_late_failed -->0.50<!-- /n --> in the one that did not. Right: the share of games that therefore
produce no selection event.*

**6. Exploratory: a bilateral contest produces one winner and one collapsed
side.** Splitting one pool of agents into two populations of <!-- n:pop -->128<!-- /n --> that play only
each other changes the outcome qualitatively: of <!-- n:asym_runs -->18<!-- /n --> runs, exactly **<!-- n:asym_mutual -->1<!-- /n -->** ended
with both pools holding an individual above parity, and <!-- n:asym_runaway -->13<!-- /n --> ended with one side
winning more than <!-- n:runaway_pct -->90<!-- /n -->% of the cross-population games. <!-- n:asym_runaway_sym -->4<!-- /n --> of those runaways
occurred in the *symmetric* control, where the two sides differ only in their
random seed — so in this block runaway dominance looks like the default of the
structure, not a consequence of imbalance. Doubling one side's policy capacity
(<!-- n:asym_param_large -->531<!-- /n --> vs <!-- n:param_count -->273<!-- /n --> parameters) did **not** reliably decide the contest. Correcting the
mutation scale for genome size may have: at a common per-parameter σ the larger
side led in <!-- n:asym_cap_lead -->2/6<!-- /n --> runs, with the step norm matched in <!-- n:asym_norm_lead -->5/6<!-- /n --> (exact Mann–Whitney
p = <!-- n:asym_norm_vs_cap_p -->0.065<!-- /n --> on the final win rates, six seeds per arm). This block was designed after the main matrix had run and was
re-implemented twice (see `results/matrix/decisions.md`); it is reported as
exploratory and suggests hypotheses rather than confirming any.

![unequal sides](results/figures/fig11_asymmetric.png)

*Two populations of <!-- n:pop -->128<!-- /n --> with unequal policy capacity, selecting only against
each other. Top: the larger-network side's win rate in cross-population games.
Bottom: the best individual each pool contains, scored against the 2015
baseline — red the larger side, blue the smaller.*

**7. Reliability is where the methods differ — not ceiling.** Four families were
run at <!-- n:budget -->500,000<!-- /n --> games each: the 2020 GA, a generational GA in the style of Ha's
2015 experiment, a self-play evolution strategy, and the corrected archive. They
reach a *similar* ceiling — the single highest endpoint in the study,
<!-- n:best_endpoint -->+0.50<!-- /n -->, is a <!-- n:best_endpoint_label -->self-play ES<!-- /n --> seed, above every control seed (best <!-- n:ctrl_best_endpoint -->+0.41<!-- /n -->).
They differ enormously in their floor. Only the plain 2020 GA learned to rally
in every seed and beat the 2015 expert in every seed (<!-- n:ctrl_reached -->6/6<!-- /n --> and <!-- n:ctrl_parity -->6/6<!-- /n -->; generational
GA <!-- n:ga_reached -->5/6<!-- /n --> and <!-- n:ga_parity -->5/6<!-- /n -->, self-play ES <!-- n:es_reached -->4/6<!-- /n --> and <!-- n:es_parity -->2/6<!-- /n -->, archive as test <!-- n:hoftest_reached -->5/6<!-- /n --> and <!-- n:hoftest_parity -->4/6<!-- /n -->);
the spread of end-of-run scores across seeds is <!-- n:family_sd_ratio_min -->3.0<!-- /n -->–<!-- n:family_sd_ratio_max -->3.5<!-- /n --> times the control's for
the other three. A study running one seed per family could easily have crowned
the ES. This is the clearest thing in the repository about why a design beats a
run.

![algorithm families](results/figures/fig8_algorithm_families.png)

*Three of the four families at an identical budget of <!-- n:budget -->500,000<!-- /n --> games (the
archive variant is in the hall-of-fame figure above). The ceilings are close
together. The floors are not.*

<!-- table:r -->
| condition | runs | learned to rally | best champion (held out) | end-of-run champion | checkpoints above parity |
|---|---|---|---|---|---|
| control (Ha 2020 GA) | 6 | 6/6 | +0.32 ± 0.06 | -0.15 ± 0.27 | 26% |
| archive as test, full span | 6 | 5/6 | -0.55 ± 0.86 | -1.01 ± 0.83 | 18% |
| archive as parent, p=0.25 | 6 | 1/6 | -3.94 ± 0.90 | -4.10 ± 0.74 | 2% |
| archive as parent, p=0.50 | 1 | 0/1 | -4.84 ± — | -4.84 ± — | 0% |
| archive as parent, full span | 3 | 0/3 | -4.83 ± 0.01 | -4.84 ± 0.00 | 0% |
| generational GA (Ha 2015) | 6 | 5/6 | -0.68 ± 0.83 | -2.00 ± 0.82 | 3% |
| self-play ES | 6 | 4/6 | -1.56 ± 0.99 | -2.08 ± 0.94 | 7% |
| sigma = 0.05 | 3 | 3/3 | +0.34 ± 0.07 | -0.17 ± 0.23 | 26% |
| sigma = 0.20 | 3 | 2/3 | -1.44 ± 1.70 | -1.66 ± 1.60 | 11% |
| population 32 | 1 | 0/1 | -4.84 ± — | -4.86 ± — | 0% |
| *reference run, unmodified environment* | 1 | 1/1 | *+0.50 ± 0.03* | *+0.04 ± 0.02* | *22%* |
| *Ha (2020), same algorithm and budget* | 1 | — | *+0.35 ± 0.02* | — | — |

Points per episode against the 2015 champion policy, which is never seen during training. Held-out columns are 1,000 episodes on an evaluation seed disjoint from the one used to pick the checkpoint. 'Learned to rally' counts runs whose population ever held 1,500-step rallies against itself.
<!-- /table:r -->

![cross-run tournament](results/figures/fig7_cross_run_elo.png)

*Final champions from every run of every condition, played against each other in
one tournament and scored by Elo.*

### Against stronger opponents

The 2015 baseline never sees its opponent, so "above parity" against it is a low
bar. Ha's slimevolleygym also ships two trained policies: a self-play GA champion
of the same 273-parameter class, and a 743-parameter CMA-ES policy trained
against the baseline (<!-- n:zoo_ga_vs_base -->+0.35<!-- /n --> and <!-- n:zoo_cma_vs_base -->+1.15<!-- /n --> against it). Every final champion played
<!-- n:zoo_games -->200<!-- /n --> games against each, on the compiled environment, which reproduces
slimevolleygym bit for bit for these networks too (<!-- n:zoo_valid_games -->40<!-- /n --> games, <!-- n:zoo_valid_steps -->120,000<!-- /n --> steps compared).

The baseline ranks the champions much as the stronger opponents do (Spearman
ρ = <!-- n:zoo_rho_base_ga -->+0.87<!-- /n --> against the zoo GA and <!-- n:zoo_rho_base_cma -->+0.89<!-- /n --> against the zoo CMA-ES, over <!-- n:zoo_n -->41<!-- /n --> final
champions), so the comparisons above do not hinge on it. But it flatters them:
<!-- n:zoo_beat_ga -->0<!-- /n --> of the <!-- n:zoo_n -->41<!-- /n --> beat Ha's published GA champion (the best manages <!-- n:zoo_best_vs_ga -->-0.04<!-- /n -->, a draw
in all but name), and of the <!-- n:zoo_above_base -->10<!-- /n --> that beat the baseline, <!-- n:zoo_above_beat_cma -->5<!-- /n --> also beat the CMA-ES
policy. The zoo GA is one exported champion Ha chose to publish, not an
unselected endpoint like these. And the CMA-ES policy, trained only against the
baseline, is the stronger of the two by the baseline's measure yet loses to the
zoo GA head to head (<!-- n:zoo_ga_vs_cma -->+0.30<!-- /n --> for the GA): specialising against the yardstick,
in one line.

<!-- table:z -->
| condition | runs | vs 2015 baseline | vs zoo GA | vs zoo CMA-ES | beat zoo GA | beat zoo CMA-ES |
|---|---|---|---|---|---|---|
| control (Ha 2020 GA) | 6 | -0.15 | -0.98 | -0.10 | 0/6 | 2/6 |
| archive as test, full span | 6 | -1.01 | -1.72 | -1.25 | 0/6 | 1/6 |
| archive as parent, p=0.25 | 6 | -4.10 | -4.34 | -3.95 | 0/6 | 1/6 |
| archive as parent, p=0.50 | 1 | -4.84 | -4.99 | -4.87 | 0/1 | 0/1 |
| archive as parent, full span | 3 | -4.84 | -4.98 | -4.92 | 0/3 | 0/3 |
| generational GA (Ha 2015) | 6 | -2.00 | -3.83 | -1.47 | 0/6 | 1/6 |
| self-play ES | 6 | -2.08 | -3.34 | -2.41 | 0/6 | 1/6 |
| sigma = 0.05 | 3 | -0.17 | -0.62 | +0.10 | 0/3 | 2/3 |
| sigma = 0.20 | 3 | -1.66 | -2.79 | -2.08 | 0/3 | 0/3 |
| population 32 | 1 | -4.86 | -4.99 | -4.88 | 0/1 | 0/1 |

Final (t = 500,000) champion of every single-population run, mean points per episode. Baseline column: held out, 1,000 episodes; zoo columns: 200 games per champion, half on each side. 'Beat' counts runs whose champion scores above 0. For scale, against the 2015 baseline the zoo GA scores +0.35 and the zoo CMA-ES +1.15; head to head the zoo GA scores +0.30 against the zoo CMA-ES.
<!-- /table:z -->

### A preregistered replication

The findings above rest on six seeds per condition, and the design changed
while it ran. So the claims about Ha's GA and the archive (findings 1–5) were
tested again on fresh seeds, with every test and decision rule committed
before the first run ([preregistration](results/replication/PREREGISTRATION.md)):
<!-- n:rep_runs -->12<!-- /n --> new runs each of the control, the archive as parent and the archive as test,
<!-- n:rep_total -->36<!-- /n --> in all, analysed once by [`replication.py`](replication.py). **<!-- n:rep_verdicts -->6 of 6<!-- /n --> claims
replicated.** The internal transition came first in <!-- n:rep_lag_first -->10/10<!-- /n --> runs; checkpoint skill
was transitive (ρ = <!-- n:rep_rho -->+0.79<!-- /n -->, <!-- n:rep_cyclic_pct -->0.5<!-- /n -->% cyclic triads); the exported individual ranked
on average <!-- n:rep_rank -->56<!-- /n --> of 128 in its pool, its streak was unrelated to its skill (ρ = <!-- n:rep_rho_streak -->+0.04<!-- /n -->),
and it lost <!-- n:rep_decl_exp -->21.2<!-- /n --> points between snapshots where the pool's best member lost <!-- n:rep_decl_best -->7.6<!-- /n -->; the
archive as parent learned in <!-- n:rep_parent_learned -->1/12<!-- /n --> runs; the archive as test was again
indistinguishable from the control (every p ≥ <!-- n:rep_archive_pmin -->0.44<!-- /n -->).

Two things came out weaker than six seeds suggested. Not every control run
learns to rally: <!-- n:rep_ctrl_learned -->10/12<!-- /n --> did, which meets the preregistered threshold but ends
"in every seed". And a falling archive win rate is not by itself a sign of
learning: it fell to <!-- n:rep_arch_learned_min -->0.06<!-- /n -->–<!-- n:rep_arch_learned_max -->0.16<!-- /n --> in the archive-as-test runs that learned, but
also to <!-- n:rep_arch_failed_min -->0.21<!-- /n --> in one that never did. Findings 6 and 7 (algorithm families,
unequal power) were not part of the replication.

<!-- table:rep -->
Runs per condition: control 12, archive as parent 12, archive as test 12. Tests in the Holm family count as holding when rejected at family-wise α = 0.05.

| claim | criterion | result | holds |
|---|---|---|---|
| C1 | internal transition before parity (lag > 0) | 10/10 runs, p = 9.8e-04 | yes |
| C2 | control runs that learn to rally | 10/12 | yes |
| C2 | latest / earliest internal transition ≥ 3 | 5.8× | yes |
| C3 | ρ(Elo, time) > 0 within runs | 12/12 runs (mean +0.79), p = 2.4e-04 | yes |
| C3 | cyclic share of decided triads < 1% | 4/849 (0.5%) | yes |
| C4 | exported individual outside its pool's top quarter | 12/12 runs (mean rank 56), p = 2.4e-04 | yes |
| C4 | ρ(streak, skill) inside ±0.2 (90% CI) | +0.04 [+0.02, +0.06] | yes |
| C4 | exported declines more than the best member | 10/12 runs (21.2 vs 7.6), p = 0.019 | yes |
| C5a | archive as parent learns less often than control | 1/12 vs 10/12, p = 3.2e-04 | yes |
| C5b | archive as test vs control: all four p ≥ 0.05 | final δ +0.01 p 0.98, peak δ -0.19 p 0.44, above δ -0.15 p 0.56, late δ -0.08 p 0.76 | yes |

Verdicts: C1 replicated, C2 replicated, C3 replicated, C4 replicated, C5a replicated, C5b replicated.
Archive as test (description, no decision): late archive win rate 0.13, 0.16, 0.06, 0.10, 0.13, 0.12, 0.14, 0.14, 0.09, 0.10 in runs that learned; 0.40, 0.21 in runs that did not (10/12 learned).
<!-- /table:rep -->

---

## Why the numbers can be trusted

**The protocol was fixed before the runs.** `results/matrix/protocol.json`
records the checkpoint spacing, the sweep evaluation seed and episode count, and
a *disjoint* held-out seed used to re-score selected champions. Peak-checkpoint
scores are selected maxima and therefore biased, so every headline number is a
<!-- n:select_episodes -->1,000<!-- /n -->-episode re-score on the held-out seed.

**Every decision made after launch is logged.** `results/matrix/decisions.md`
records each change to the design, what was known at the time, and why — including
the archive bug above, which was found after three runs had completed and is
reported rather than quietly fixed, and a later streak-accounting bug in the
archive-as-test kernel, whose runs were rerun under a new name while the
originals stay on disk, marked superseded.

**The fast environment is the benchmark, not an approximation.** A <!-- n:budget -->500,000<!-- /n -->-game
run costs about <!-- n:val_ref_core_hours -->13<!-- /n --> core-hours on the reference `slimevolleygym`, which is why
the first version of this repository had exactly one run. `fastvolley.py` is a
numba-compiled transcription, and it is validated bit for bit: driven from an
identical stream of serve velocities, <!-- n:val_games -->200<!-- /n --> paired games match on every ball
position, every agent position, every rally end, every score and every episode
length, across <!-- n:val_steps -->304,812<!-- /n --> environment steps — at <!-- n:val_speedup -->26<!-- /n -->× the throughput. Three
deviations (RNG family, pair-sampling call, BLAS vs libm in the forward pass) are
documented; the third cannot change a trajectory because the game only ever reads
`action[i] > 0`.

**Stability is only reported conditional on competence.** A run that never
learned anything has zero volatility and zero drawdown, and so scores perfectly
on every stability metric. The archive-as-parent runs are exactly that case, and
they are the reason every stability comparison is also reported over the subset of
runs that actually learned.

**Statistics match the sample size.** <!-- n:runs_min -->1<!-- /n -->–<!-- n:runs_max -->6<!-- /n --> runs per condition rules out anything
asymptotic, so comparisons use the exact Mann–Whitney U test with full
enumeration, Cliff's δ, and percentile bootstrap intervals — written out in
`stats_utils.py` rather than imported, so every number can be audited.

---

## The write-up

| | |
|---|---|
| [Overview](docs/paper/00-overview.md) | abstract and what the study contributes |
| [Methods](docs/paper/01-methods.md) | task, algorithms, interventions, the compiled environment, the protocol |
| [Results](docs/paper/02-results.md) | the reference run, seed variance, what competence means here |
| [Ablations and analysis](docs/paper/03-ablations-and-analysis.md) | transitivity, the export rule, archives, mutation scale, population size, algorithm families |
| [Appendix](docs/paper/04-appendix.md) | self-contained: every table, every run, reproduction commands |
| [February postmortem](docs/postmortem-february.md) | three verified mechanisms that made an earlier attempt uninterpretable |
| [Lineage](docs/trajectory.md) | Ha 2015 → Backprop NEAT 2016 → slimevolleygym 2020 → ShinkaEvolve 2025 |

Every table in the write-up and in this README is generated by `make_tables.py`
from the files in `results/analysis/`, every number in the prose by
`prose_numbers.py`, and every figure by `make_figures.py`. Each analysis file
records a hash of every raw file it was computed from, and
`make_tables.py --check`, run on every push, fails if a table or a number no
longer follows from the files on disk. None is typed by hand, so
`git diff docs/paper` shows exactly which numbers moved
when new results land.

---

## The algorithms compared

All share the identical policy — a fixed <!-- n:obs_size -->12<!-- /n -->–<!-- n:hidden -->10<!-- /n -->–<!-- n:hidden -->10<!-- /n -->–<!-- n:n_act -->3<!-- /n --> tanh network, <!-- n:param_count -->273<!-- /n -->
parameters — and the identical environment. Only the machinery differs.

| family | how a population becomes the next population |
|---|---|
| **Ha 2020 GA** (control) | steady-state: draw two individuals, play one game, the loser is overwritten by a mutated copy of the winner. Champion = longest winning lineage. |
| **Ha 2015 GA** | generational: population <!-- n:ga_pop -->100<!-- /n -->, each agent plays <!-- n:ga_opponents -->10<!-- /n --> random peers, top <!-- n:ga_elite_pct -->20<!-- /n -->% retained, remainder refilled by uniform crossover + mutation. Champion = highest *computed* fitness. |
| **Self-play ES** | OpenAI-ES: one mean vector, <!-- n:es_candidates -->50<!-- /n --> mirrored perturbations per iteration, fitness from games among the perturbations, rank-shaped gradient. Reports the distribution mean — so it has no champion-selection problem at all. |
| **Archive variants** | the control plus a hall of fame, tested both as a parent (wrong) and as a test (right). |
| **Knob sweeps** | mutation scale σ ∈ {<!-- n:sigma_small -->0.05<!-- /n -->, <!-- n:sigma_ctrl -->0.10<!-- /n -->, <!-- n:sigma_big -->0.20<!-- /n -->}; population ∈ {<!-- n:pop_small -->32<!-- /n -->, <!-- n:pop -->128<!-- /n -->} (<!-- n:pop_big -->512<!-- /n --> was defined but never run). |

![mutation scale and population size](results/figures/fig4_ablations.png)

*The two knob sweeps against the control. Left: mutation scale. Right:
population size — only population <!-- n:pop_small -->32<!-- /n --> ran, with <!-- n:pop_small_runs -->1<!-- /n --> seed, before the sweep
was replaced by the unequal-power block.*

Topology-evolving methods (NEAT) are deliberately out of scope; `docs/paper/04-appendix.md`
§A.8 specifies exactly what a later NEAT run would need and how it would stay
comparable to these results.

---

## Quickstart

```bash
python3 -m venv .venv          # Python 3.11
.venv/bin/pip install -r requirements.txt -r requirements-fast.txt
# or, for every transitive package pinned too:  pip install -r requirements-lock.txt

# the repository self-test: the three February failures, a bit-level
# comparison of the compiled environment against slimevolleygym, and the
# archive streak regression
.venv/bin/python test_repo.py

# the port is the benchmark: 200 paired games, step by step
.venv/bin/python validate_fastvolley.py --games 50

# the matrix (hours, not days: ~30 min per 500,000-game run per core)
.venv/bin/python run_experiments.py --workers 3

# metrics, held-out re-scoring, and the three coevolution-specific measurements
.venv/bin/python analyze_matrix.py --holdout
.venv/bin/python coevolution_analysis.py --within --across --proxy

# tables, figures, the single-page HTML write-up and the paper
.venv/bin/python make_tables.py && .venv/bin/python make_figures.py
.venv/bin/pip install -r requirements-docs.txt && .venv/bin/python build_paper.py --md
paper/build.sh                 # needs a TeX distribution with latexmk
```

The reference run on the unmodified environment is continued from its committed
population snapshot with
`.venv/bin/python train_ga_selfplay.py --resume --snapshot-freq 1000`.

---

## Repository layout

| Path | What it is |
|---|---|
| `fastvolley.py` | compiled port of the physics, the MLP policy and the 2015 baseline RNN, plus the control GA |
| `validate_fastvolley.py` | the bit-level comparison against `slimevolleygym` |
| `algorithms.py` | the generational GA, the self-play ES, and the corrected hall of fame |
| `run_experiments.py` | the matrix: conditions, seeds, and the pre-registered protocol |
| `analyze_matrix.py` / `stats_utils.py` | metrics, exact tests, bootstrap intervals |
| `coevolution_analysis.py` | within-run Elo and intransitivity, cross-run tournament, the champion-proxy check |
| `make_tables.py` / `make_figures.py` / `build_paper.py` | everything in the write-up, generated |
| `train_ga_selfplay.py` / `eval_vs_baseline.py` | the reference implementations, unmodified |
| `test_repo.py` | the repository self-test |
| `scripts/` | the shell scripts that drove the runs in ephemeral containers (queueing, watchdog, autocommit) |
| `results/matrix/` | one file per run: <!-- n:n_ckpt -->100<!-- /n --> champion genomes, streaks, rally lengths, evaluations |
| `results/ga_selfplay/` | the reference run: checkpoints, population snapshot, training history |
| `archive/february/` | the earlier failed attempt, unmodified, as evidence |

---

## Limitations

- **One environment.** Slime Volleyball is symmetric, zero-sum and fully
  observed — the friendliest possible setting for purely relative selection.
- **<!-- n:runs_min -->1<!-- /n -->–<!-- n:runs_max -->6<!-- /n --> runs per condition.** <!-- n:seeds_main -->6<!-- /n --> for every condition that carries a claim:
  enough to separate a large effect from seed noise, not a small one. Findings
  1–5 held in a [preregistered replication](#a-preregistered-replication) on
  <!-- n:rep_runs -->12<!-- /n --> fresh runs per condition; findings 6 and 7 were not replicated.
- **Fixed topology.** Nothing here evolves structure; see §A.8 for what NEAT
  would need.
- **One archive design per reading.** A quality-diversity or curated archive is a
  different experiment.

## Credits

- Environment, baseline policy and the original GA: **David Ha (hardmaru)** —
  [slimevolleygym](https://github.com/hardmaru/slimevolleygym) (Apache-2.0),
  [Neural Slime Volleyball (2015)](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/).
- Competitive coevolution background: Rosin & Belew (1997); Risi, Tang, Ha &
  Miikkulainen, [*Neuroevolution*](https://neuroevolutionbook.com) (MIT Press,
  2025), ch. 7.2.
- This repository: MIT (see LICENSE).
