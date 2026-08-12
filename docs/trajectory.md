# The lineage — where this repository sits

**One sentence: this repository is the ground floor of a ladder that runs
from evolving *weights* in a closed game (2015) through evolving *topologies*
(2016) to evolving *programs* in open worlds (2025) — and it exists to make
that first rung solid, replicable and honest.**

## David Ha's arc

| Year | Artifact | What evolves | Links |
|---|---|---|---|
| 2015 | **Neural Slime Volleyball** — recurrent nets learn volleyball purely by playing each other ("arms race" self-play); the champion became the famous baseline policy | weights | [writeup](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/) · [play it](https://otoro.net/slimevolley/) · [watch evolution live in the browser](https://otoro.net/slimevolley/training.html) ([post](https://blog.otoro.net/2015/05/13/neural-slime-volleyball-evolution-demo/)) |
| 2016 | **Backprop NEAT** — NEAT evolves the network *topology* while gradient descent tunes the weights | topologies (+ gradients) | [post](https://blog.otoro.net/2016/05/07/backprop-neat/) · [playground](https://otoro.net/ml/neat-playground/) |
| 2020 | **slimevolleygym** — the 2015 game as a proper gym environment, with PPO / CMA-ES / GA-self-play baselines; the substrate under this repository | (benchmark) | [repo](https://github.com/hardmaru/slimevolleygym) |
| 2025 | **ShinkaEvolve** (Sakana AI, which Ha co-founded) — evolution over *programs*, with an LLM as the mutation operator | programs | [blog](https://sakana.ai/shinka-evolve/) · [code](https://github.com/SakanaAI/ShinkaEvolve) · [paper](https://arxiv.org/abs/2509.19349) |

Read down the third column: **weights → topologies → programs.** Each rung
loosens what evolution must hold fixed. And the very first rung was already
*competitive coevolution* — agents improving only by playing each other, with
no external teacher.

## This repository's place

Slime Volleyball is competitive coevolution in its purest, most verifiable
form: a symmetric zero-sum game, a tiny genotype, an implicit relative
fitness ("win and stay in the pool"), and a published reference result to
replicate (0.353 ± 0.728 vs the 2015 baseline after 500k tournament games).
That makes it the ideal **control condition** for any larger claim about
evolved decision-making: before arguing that evolutionary search can navigate
open worlds, demonstrate — cleanly — what it does in a closed one.

This repo is part of a broader research program on neuroevolutionary
approaches to decision-making in social systems, where the same ladder
continues upward: from evolved *behavior* (this repo), through evolved
*structure* (a backprop-NEAT study), toward evolved *strategies as programs*
(LLM-driven program evolution in the ShinkaEvolve tradition, applied to
international-relations strategy spaces). The rung above always inherits its
methodology from the rung below — which is exactly why this one had to be
built first, and built right.

*See also: [Risi, Tang, Ha & Miikkulainen, "Neuroevolution" (MIT Press,
2025)](https://neuroevolutionbook.com) — chapter 7.2 treats competitive
coevolution; this repository is a minimal working example of that chapter.*
