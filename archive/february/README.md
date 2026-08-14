# The February 2026 attempt — preserved unmodified

These four files are the failed first attempt, exactly as they were. They are
kept as evidence, not as working code: **do not run them.**

| File | What it was meant to do |
|---|---|
| `train_coevo.py` | Two NEAT populations evolving against each other |
| `train_baseline.py` | A single NEAT population evolving against the built-in 2015 expert |
| `test_setup.py` | Dependency check |
| `requirements.txt` | The original, unpinned dependency list |

Three defects in this code — verified against the environment source in August
2026 — made every run uninterpretable: an action threshold that made every
genome press all three buttons on every step, an opponent assignment that was
silently swallowed by a gym wrapper, and a reward-shaping term with an inverted
sign. `docs/postmortem-february.md` has the full autopsy, and `test_repo.py`
checks that none of the three can recur in the current code.

The NEAT implementation these scripts import (`neat/genome.py`,
`neat/population.py`, `neat/config.py`) was not preserved, so the autopsy covers
the experiment layer rather than the search algorithm itself.
