# Preregistration: NEAT as a fifth family (WP9, part 2)

Written 2026-10-03, before any NEAT run: when this file, `run_neat.py`,
`neat_analysis.py` and `lab/neat.py` were committed, `results/neat/` held no
run file. The only NEAT runs before it were short pilots (seed 7 and 999, at
most 100,000 games, in scratch directories) that checked the mechanics:
speciation (which led to the adaptive compatibility threshold) and a
crossover bug (a feedforward cycle; fixed, with a test). No pilot was scored
against anything. Any later change to these files gets a dated entry in
`results/matrix/decisions.md`, and what it touches becomes exploratory.

## 1. Why

The paper compares four fixed-topology families (C6) and lists "fixed
topology" as a limitation; `docs/paper/04-appendix.md` §A.8 specifies what
NEAT would need. NEAT (Stanley & Miikkulainen 2002; used for competitive coevolution in 2004) evolves the network's
structure along with its weights, so it asks whether evolving structure
changes what this self-play setting can reach.

## 2. Design

| | |
|---|---|
| algorithm | `lab/neat.py`: NEAT with innovation numbers, speciation with explicit fitness sharing, crossover, add-node and add-connection mutations, recurrent connections allowed. As in NEAT's coevolution study (Stanley & Miikkulainen 2004, Appendix A): a compatibility threshold that adapts towards a target number of species (here 8, step 0.05, start 3.0, c1 = c2 = 1), the champion of every species with more than five members copied unchanged, 80% of genomes weight-mutated with 90% of weights perturbed, a 75% chance that a gene disabled in either parent stays disabled, 25% of offspring from mutation alone. The study's own: 100 genomes, σ 0.1, initial weight scale 0.5, tanh units. Chosen here, not taken from a paper: c3 0.4, add-node 0.03, add-connection 0.05, interspecies mating 0.001, the best 20% of a species reproduce, a species that has not improved for 15 generations is removed unless it is one of the best two |
| self-play evaluation | the generational GA's (`algorithms.run_ga2015`): 100 genomes, 500 games per generation between random pairs, fitness the mean point margin, 1,000 generations = 500,000 games; the generation's fittest genome is exported every 10 generations (5,000 games) |
| game and yardstick | the study's: the compiled physics, each champion scored against the 2015 baseline on the sweep seed (200 episodes), final and best champions re-scored on the held-out seed (1,000 episodes) |
| validation | a NEAT genome encoding a 12-10-10-3 network computes its outputs bit for bit, scores identically against the 2015 baseline on the same serves, and plays identical games step by step (`test_repo.py`) |
| seeds | 101–112 (12 runs; the families of C6 have 6 each) |
| command | `python run_neat.py` (or one seed per invocation) |
| analysis | `python neat_analysis.py`, once, after all 12 run files exist |

**Stopping rule.** Exactly these 12 runs; none added, repeated or excluded.

## 3. What is tested and what is described

| | |
|---|---|
| **H9d** | NEAT against the generational GA, the family with the same selection scheme and a fixed topology: end-of-run champion (held out), two-sided exact Mann–Whitney. Reported as "a detectable difference" if p < 0.05, otherwise as "no detectable difference" with the power caveat (12 vs 6 runs). No direction is predicted. |
| described | NEAT in the family table of C6 (runs that learned to rally, runs that reached parity, end-of-run and best champions, share of checkpoints above parity, the spread of end-of-run scores relative to the control); the cross-run tournament (NEAT's final champions against the final champion of every single-population matrix run, Bradley–Terry ratings); the structure NEAT grew (hidden nodes, connections, species) |

## 4. What the results change

NEAT is reported as a fifth family in its own README section with generated
numbers, labelled with this preregistration. C6 ("the families differ in
reliability, not ceiling") is a claim about four families; it is not
extended to NEAT unless the numbers support it, and its wording does not
change. The paper's Limitations bullet "fixed topology" is updated with what
NEAT did.
