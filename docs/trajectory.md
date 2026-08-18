# The lineage — where this repository sits

**One sentence: a self-improvement loop is exactly as real as its improvement
signal — and this repository is the first of three experiments asking where
that signal can come from as the world opens up.**

Slime volleyball answers the question in its easiest setting: a closed,
symmetric, zero-sum game where the signal can be purely internal ("beat a
peer") and still produce competence that transfers to an opponent evolution
never met. Everything above this rung inherits that finding and loosens one
more constraint.

## David Ha's arc

| Year | Artifact | What evolves | Links |
|---|---|---|---|
| 2015 | **Neural Slime Volleyball** — recurrent nets learn volleyball purely by playing each other; the champion became the famous baseline policy | weights | [writeup](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/) · [play it](https://otoro.net/slimevolley/) · [watch evolution live](https://otoro.net/slimevolley/training.html) |
| 2016 | **Backprop NEAT** — NEAT evolves network *topology* while gradient descent tunes the weights | topologies (+ gradients) | [post](https://blog.otoro.net/2016/05/07/backprop-neat/) · [playground](https://otoro.net/ml/neat-playground/) |
| 2020 | **slimevolleygym** — the 2015 game as a gym environment with PPO / CMA-ES / GA-self-play baselines; the substrate under this repository | (benchmark) | [repo](https://github.com/hardmaru/slimevolleygym) |
| 2025 | **ShinkaEvolve** (Sakana AI) — evolution over *programs*, with an LLM as the mutation operator | programs | [blog](https://sakana.ai/shinka-evolve/) · [code](https://github.com/SakanaAI/ShinkaEvolve) · [paper](https://arxiv.org/abs/2509.19349) |

Read the third column downward: **weights → topologies → programs.** Each rung
loosens what evolution must hold fixed.

## Three experiments, one argument

The portfolio this repository opens is not a sequence of unrelated demos. Each
project contributes exactly one principle, and the third is the *composition*
of the first two.

**1 · Neural slime volleyball — the selection-signal principle.**
With everything fixed except 273 weights, an internal *relative* signal (beat a
randomly drawn peer, stay in the pool) produces competence that transfers to an
opponent never seen in training. No reward shaping, no expert teacher, no
hand-designed fitness. The measured lesson: the ecology selects, the frozen
yardstick only measures — and the two must never be the same thing. Also
measured here: transfer *lags* internal progress by ~100,000 games, so an
evaluation that starts too early reads as total failure (which is what February
mistook for "the method doesn't work"). *Leaves open:* the representation was
fixed by hand. What if the structure itself must be discovered?

**2 · Backprop NEAT — the division-of-labor principle.**
Evolution proposes *structure*; an inner optimizer refines *parameters*.
Evolution retreats to what it is uniquely good at — open-ended structural
search — and delegates local polishing to a stronger inner loop. *Leaves open:*
both rungs still search a space someone specified in advance. What happens when
the space itself is open — when candidate solutions are programs, and the world
they act in is not a game with a scoreboard?

**3 · ACTIR / ShinkaEvolve — the composition.**
Foreign-policy strategies as *programs*, evolved by an LLM acting in gradient
descent's old seat (principle 2), selected by how they fare against each other
inside a simulated international system rather than scored by an external
oracle (principle 1), and measured against held-out scenarios the evolution
never touches (principle 1's yardstick separation). Both load-bearing parts of
the boss battle arrive with provenance: each was demonstrated, measured, and
falsifiable at a scale where failure was still attributable.

That is the argument the trilogy makes: as the world opens from game to
structure to open-ended strategy space, the improvement signal must migrate
from *given* to *internal* — and each migration has to be demonstrated
somewhere small enough to verify before it is trusted somewhere large enough
to matter.

*See also: [Risi, Tang, Ha & Miikkulainen, "Neuroevolution" (MIT Press,
2025)](https://neuroevolutionbook.com), ch. 7.2 on competitive coevolution —
this repository is a minimal working example of that chapter.*

**Update (August 2026).** Two of the extension slots this document originally
listed as absent have since been filled, and the results changed the argument
above rather than decorating it:

- *Hall-of-fame opponent sampling* is now run in two readings — archive as parent
  and archive as test — with the first abolishing learning outright and the second
  neither helping nor hurting.
- *A test for intransitivity in the champion lineage* is now run, and it comes
  back almost perfectly transitive. Since a hall of fame is the standard remedy
  for intransitivity, that result *predicts* the null above.

What replaced them as the interesting finding is a third object that this
document did not anticipate: the rule that decides which individual to export.
It contributes more variance to a champion curve than the coevolution does, and
it is uninformative about actual skill. That matters directly for rung 3, where
an LLM-driven loop must promote one program out of each generation — see §8 of
[the analysis](paper/03-ablations-and-analysis.md).

Still absent, and therefore still available as extension slots:
quality-diversity archives, indirect encodings, and topology evolution (NEAT) —
for which [appendix §A.8](paper/04-appendix.md) specifies the interface and what
would keep a later run comparable to these.
