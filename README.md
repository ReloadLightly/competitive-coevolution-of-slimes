# Neural Slime Volleyball — Summer

Self-play neuroevolution for [David Ha's Slime Volleyball](https://otoro.net/slimevolley/), done right this time.

In February 2026 we tried to evolve Slime Volleyball agents with a custom NEAT
implementation and understood nothing from the runs. This repository finishes
that work the honest way: an **exact replication of David Ha's self-play
genetic algorithm** on the real [slimevolleygym](https://github.com/hardmaru/slimevolleygym)
environment, plus a verified autopsy of why February failed, plus the lineage
that connects a 2015 browser toy to LLM-driven program evolution in 2025.

**Status: 🟡 training in progress** (started 2026-08-12; ~3 h for the full
500,000-tournament run). Checkpoints, curves and GIFs land here as they arrive.

## The method, in plain language

- The **genotype** is the flat weight vector (273 numbers) of a tiny fixed
  feed-forward network. Nothing else evolves.
- A **population** of 128 random individuals plays tournaments: two are drawn
  at random, they play one full game.
- **Selection**: the winner survives untouched; the loser is overwritten by a
  copy of the winner plus Gaussian noise — the mutation operator.
- **Fitness is implicit and relative**: win games, stay in the pool. There is
  no reward shaping, no curriculum, no expert opponent. Everyone starts
  equally terrible, so victories are always achievable — which is exactly why
  self-play works where February's "train a newborn against the 2015 expert"
  framing could not.
- The 2015 baseline policy is used **only as an external yardstick** after
  training, never during it. Reference to beat: **0.353 ± 0.728** mean score
  over 1000 episodes (Ha's GA self-play result).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# smoke test (~20 s)
.venv/bin/python train_ga_selfplay.py --tournaments 600 --save-freq 200 --outdir results/smoke

# the real thing (~3 h on one CPU core)
.venv/bin/python train_ga_selfplay.py

# measure the champion against the 2015 baseline
.venv/bin/python eval_vs_baseline.py results/ga_selfplay/ga_00500000.json --episodes 1000
```

## Repository layout

| Path | What it is |
|---|---|
| `train_ga_selfplay.py` | Faithful port of Ha's tournament-selection GA (every deviation documented in the docstring) |
| `eval_vs_baseline.py` | External yardstick: champion vs the 2015 baseline policy |
| `docs/postmortem-february.md` | The autopsy: three verified mechanisms that made February's runs uninterpretable |
| `docs/trajectory.md` | The lineage: Ha 2015 → backprop NEAT 2016 → slimevolleygym 2020 → ShinkaEvolve 2025, and where this repo sits |
| `archive/february/` | The February 2026 scripts, unmodified, as evidence |
| `results/` | Checkpoints, learning curves, evaluation data (landing during the day) |

## Results

*(pending — the run is in progress; this section will carry the final score,
learning curve and match GIFs)*

## Credits

- Environment, baseline policy and original GA: **David Ha (hardmaru)** —
  [slimevolleygym](https://github.com/hardmaru/slimevolleygym) (Apache-2.0),
  [Neural Slime Volleyball (2015)](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/).
- This repository: MIT (see LICENSE).
