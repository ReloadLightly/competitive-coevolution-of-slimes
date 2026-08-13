# Neural Slime Volleyball — Summer

Self-play neuroevolution for [David Ha's Slime Volleyball](https://otoro.net/slimevolley/), done right this time.

In February 2026 I tried to evolve Slime Volleyball agents with a custom NEAT
implementation and learned nothing from the runs — not why agents failed to
improve, not even what they were doing. This repository finishes that work the
honest way: a replication of **David Ha's self-play genetic algorithm** on the
real [slimevolleygym](https://github.com/hardmaru/slimevolleygym) environment,
a verified autopsy of the February failure, and the lineage connecting a 2015
browser demo to LLM-driven program evolution in 2025.

## Results

Trained by **self-play only** — the 2015 baseline policy was never seen during
training and serves purely as an external yardstick.

| Games played | Score vs 2015 baseline | Episodes won / drawn / lost |
|---|---|---|
| 5,000 | −4.93 ± 0.26 | 0 / 0 / 400 |
| 111,000 | −4.02 ± 1.16 | 0 / 0 / 50 |
| 145,000 | −2.58 ± 1.63 | 2 / 2 / 36 |
| **175,000 (best champion)** | **+0.23 ± 0.77** | **123 / 231 / 46** |
| *Ha's reference (500,000 games)* | *+0.353 ± 0.728* | *—* |

Score is points won minus points lost per episode (range −5 … +5), over 400
episodes. **The evolved champion beats the 2015 expert on points** — reaching
Ha's quality regime at ~35% of his training budget, and never once having seen
that opponent during training.

![learning curve](results/figures/learning_curve.png)

**Three things this shows.**

*A phase change.* ~100,000 games of flat −4.9 floor, then a steep climb. The
population was improving against *itself* the whole time (winning streaks and
rally lengths grew steadily from the start), but the external yardstick only
registers it once those skills generalize beyond the family. Internal selection
pressure leads; external measurement lags — here by roughly 100,000 games. An
evaluation stopped at game 90,000 would have reported total failure.

*What "improved" means concretely.* The champion stopped getting crushed and
started going the distance: every one of its 400 evaluation episodes ran the
full 3,000 steps, and 88% ended drawn or won.

*Coevolution is unstable — and this run measures it.* Champion quality
oscillates violently between neighbouring checkpoints: +0.40 at game 175,000,
−4.05 at 180,000, back to −0.75 at 182,000. Only 3 of 182 checkpoints score
above parity. Two mechanisms, both textbook and both left in deliberately by
replicating Ha exactly: (1) selection is relative, so a population can *forget*
skills that no current opponent punishes — the classic argument for a
hall-of-fame archive of past opponents ([Neuroevolution](https://neuroevolutionbook.com),
ch. 7.2), which this algorithm has none of; (2) the exported "champion" is the
individual with the longest winning lineage, a cheap proxy Ha adopted
explicitly "without actually computing who is best to save time" — and a noisy
one. The honest reading: **self-play produced baseline-beating play, but does
not hold it.** Stability is a separate problem from competence, and this
repository's data isolates it.

| After 5,000 games | The best champion (175,000 games) |
|---|---|
| ![early](results/figures/early_match.gif) | ![final](results/figures/final_match.gif) |
| loses 0–5 in 543 steps | 0–0 after 1,600 steps of sustained rallying |

**Honest run notes.** Training reached **182,000 of Ha's 500,000 tournament
games** before the cloud sandbox became the limiting factor: the container is
reclaimed after inactivity, so training advanced only while a session was live
(9 restarts, each resumed from the latest population snapshot with the RNG
reseeded deterministically from `seed + resume point` — a documented deviation
from a single continuous run). Evaluation sweeps used 40 episodes per
checkpoint and 400 for the headline rows; all evaluation used seed 721, and
the baseline policy was never used as a training signal.

Two open questions this run does **not** answer, both left explicitly to
`resume_to_500k.sh` (which continues this exact run from its last snapshot,
~9 h single-core): whether the remaining 318,000 games raise the *ceiling* or
merely reshuffle the same oscillation, and whether a hall-of-fame variant
would convert intermittent baseline-beating play into stable baseline-beating
play. The second is the more interesting experiment, and this repository's
checkpoint archive is exactly the material for it.

## The method, in plain language

- The **genotype** is the flat weight vector (273 numbers) of a tiny fixed
  feed-forward network. Nothing else evolves — no topology change, no crossover.
- A **population** of 128 individuals plays tournaments: two are drawn at
  random and play one full game.
- **Selection**: the winner survives untouched; the loser is overwritten by a
  copy of the winner plus Gaussian noise — the mutation operator.
- **Fitness is implicit and relative**: win games, stay in the pool. No reward
  shaping, no curriculum, no expert opponent. Everyone starts equally terrible,
  so victories are always achievable — which is exactly why self-play works
  where February's "train a newborn against the 2015 expert" framing could not.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# smoke test (~20 s)
.venv/bin/python train_ga_selfplay.py --tournaments 600 --save-freq 200 --outdir results/smoke

# train from scratch (Ha's full scale, ~10 h single core)
.venv/bin/python train_ga_selfplay.py

# continue this repository's run from its last snapshot
bash resume_to_500k.sh

# measure any champion against the 2015 baseline
.venv/bin/python eval_vs_baseline.py results/ga_selfplay/ga_00175000.json --episodes 1000

# watch a match
.venv/bin/python render_gif.py results/ga_selfplay/ga_00175000.json --out match.gif
```

## Repository layout

| Path | What it is |
|---|---|
| `train_ga_selfplay.py` | Faithful port of Ha's tournament-selection GA (deviations documented in the docstring) |
| `eval_vs_baseline.py` | External yardstick: champion vs the 2015 baseline policy |
| `render_gif.py` | Headless match rendering |
| `plot_curve.py` | The learning-curve figure |
| `watchdog.sh` / `resume_to_500k.sh` | Crash-safe continuation of a long run |
| `docs/postmortem-february.md` | The autopsy: three verified mechanisms that made February's runs uninterpretable |
| `docs/trajectory.md` | The lineage: Ha 2015 → backprop NEAT 2016 → slimevolleygym 2020 → ShinkaEvolve 2025 |
| `archive/february/` | The February 2026 scripts, unmodified, as evidence |
| `results/` | 182 champion checkpoints, learning curves, evaluation data, figures |

## Credits

- Environment, baseline policy and original GA: **David Ha (hardmaru)** —
  [slimevolleygym](https://github.com/hardmaru/slimevolleygym) (Apache-2.0),
  [Neural Slime Volleyball (2015)](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/).
- This repository: MIT (see LICENSE).
