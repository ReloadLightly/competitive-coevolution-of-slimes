# Why February taught us nothing — the autopsy

In February 2026 we tried to solve Slime Volleyball with a custom NEAT
implementation and two training scripts (`archive/february/`), and came away
having understood *nothing* — not why agents didn't improve, not even what the
agents were actually doing. This document explains why, with every claim
verified on 2026-08-12 against the slimevolleygym source and live tests
(Python 3.11, gym 0.26.2, slimevolleygym @ 8ac2243).

**The one-sentence spine: the experiment that ran was not the experiment we
designed — three independent mechanisms each silently redirected selection
pressure, and any one of them alone would have been fatal.**

*(The February NEAT core itself — the `neat/` package the scripts import —
was not preserved, so this autopsy covers the experiment layer. If the folder
resurfaces, a fourth section gets written.)*

## Mechanism 1: the all-buttons bug — a population with zero behavioral variance

`train_coevo.py` converted network outputs to button presses with

```python
return [1 if o > 0 else 0 for o in outputs]
```

The network's outputs came from a steepened **sigmoid**, whose range is
(0, 1) — *every* output is greater than 0, always. So every genome, at every
step, pressed all three buttons: forward + backward (which cancel) + jump.
The entire population produced the **identical behavior** — a slime jumping
in place forever. Selection had literally nothing to select on; 100% of
"evolution" was drift on noise.

The proof is in February's own follow-up: `train_baseline.py`'s header lists
as Fix #1 *"Action threshold 0.5 (steepened sigmoid outputs [0,1])"* — the
fix documents the disease. And the smoking gun was visible all along: the
debug output printed per-button press counts, which read L = R = J = steps.

## Mechanism 2: the silent wrapper bug — the coevolution that never happened

`train_coevo.py` built the environment with `gym.make("SlimeVolley-v0")` and
installed evolved opponents by assignment:

```python
env.policy = GenomePolicy(opponent_genome)   # looks right, does nothing
```

`gym.make` returns the environment wrapped in `TimeLimit`/`OrderEnforcing`
wrappers. Attribute assignment lands **on the wrapper**; the real environment
underneath keeps its default `BaselinePolicy` — the 2015 expert. Verified
live in this repo's stack:

```
env2 = gym.make("SlimeVolley-v0")
env2.policy = Dummy()
type(env2.unwrapped.policy).__name__   # -> 'BaselinePolicy'  (unchanged!)
```

Consequence: **every single "coevolution" game was actually played against
the built-in 2015 expert.** The left and right populations never met. Weeks
of "competitive coevolution" curves described an experiment that never ran.
(Cruel detail: February's `test_setup.py` used `SlimeVolleyEnv()` directly,
where the assignment *would* have worked — the trainers used `gym.make`.)

This is why every script in this repository constructs `SlimeVolleyEnv()`
directly and never touches `gym.make`.

## Mechanism 3: fitness pointed the wrong way

Three compounding reward-design errors:

1. **Sign-inverted ball shaping.** Both scripts rewarded `obs[4] > 0` with a
   comment saying "ball on opponent's side is good". Verified empirically:
   the observation is given in the agent's own frame, where **positive x is
   the agent's OWN half** (slimevolleygym mirrors observations so both sides
   see the same geometry). The shaping paid the agent for keeping the ball
   on its *own* side — rewarding exactly the losing pattern.
2. **Survival bonus.** `+0.0001` per step alive rewards passivity; combined
   with (1), standing still near the ball was a local optimum.
3. **Half-length episodes.** Games were truncated at 1500 steps, but the
   environment's real limit is `t_limit = 3000` with 5 lives per side. Win
   bonuses (±5) were routinely decided on unfinished games, adding pure
   noise on top of the inverted shaping.

## The frame error above all: a newborn versus the Olympic champion

Even with all three bugs fixed, the February *design* — evolve from scratch
against the built-in expert — fights the problem's structure. David Ha's own
tutorial says it best: training a randomly initialized network against the
2015 expert is *"like an infant learning to play volleyball against an
Olympic gold medalist"* — the score is maximally negative regardless of small
improvements, so the learning signal is flat. (His PPO baseline needed
1.3–3M timesteps before its first positive score.)

The field's native answer — Ha's answer in 2015, and this repository's —
is **self-play**: everyone starts equally bad, victories are always
achievable, and the opponent difficulty scales automatically with the
population. The expert is demoted to what it should have been all along:
an external yardstick, never a training signal.

## What would have caught all of this in five minutes

The checklist this repository enforces, in plain terms:

1. **Behavioral probe before training**: print the action histogram of a few
   random genomes. (Would have caught Mechanism 1 instantly.)
2. **Wiring test**: assert the opponent policy actually receives `predict()`
   calls during a game. (Catches Mechanism 2.)
3. **Sign test for every shaping term**: perturb the state, check the reward
   moves the way the comment claims. (Catches Mechanism 3.)
4. **Full-length episodes only.**
5. **Strongest known baseline first**: replicate a published, known-good
   result (Ha's GA self-play, 0.353 ± 0.728) before innovating on top.
6. **One external yardstick, never trained on.**

Item 5 is this repository's centerpiece; items 1–4 and 6 are built into its
scripts.
