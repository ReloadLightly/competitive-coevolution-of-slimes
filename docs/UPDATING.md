# Keeping this repository current

Everything reported in `docs/paper/` and in the README is generated from the
files under `results/`. There is no step where a number is copied by hand, so
refreshing the write-up after new results is three commands:

```bash
.venv/bin/python analyze_matrix.py --holdout      # metrics + held-out re-scoring
.venv/bin/python make_tables.py                   # rewrites every table in place
.venv/bin/python make_figures.py                  # rewrites every figure
.venv/bin/python build_paper.py --md              # single-page HTML + writeup.md
```

`make_tables.py --check` exits non-zero if any table in the write-up is stale
relative to the data, which makes it usable as a pre-commit or CI check.

## Adding a run

`run_experiments.py` skips any run whose `.npz` already exists, so it is safe to
re-run at any time and safe to interrupt. To add seeds to an existing condition:

```bash
.venv/bin/python run_experiments.py --workers 3 --only control --seeds 107,108
```

## Adding a condition

Add an entry to `CONDITIONS` in `run_experiments.py` and, if it needs new
machinery, a training kernel in `algorithms.py`. The contract a kernel must
satisfy — and what a topology-evolving method such as NEAT would additionally
need — is specified in [appendix §A.8](paper/04-appendix.md).

## Continuing the reference run

The run in `results/ga_selfplay/` is trained on the unmodified `slimevolleygym`
environment and is continued from its committed population snapshot:

```bash
.venv/bin/python train_ga_selfplay.py --resume --snapshot-freq 1000
```

Roughly twelve core-hours for a full 500,000 games; safe to stop and restart at
any point, since progress is snapshotted continuously. Each restart reseeds the
RNG deterministically from `seed + resume point`, which is why the run is
documented as a chain of restarts rather than one exactly resumable trajectory.

## Conventions worth preserving

- **The protocol is a file.** `results/matrix/protocol.json` holds the evaluation
  seeds, episode counts, checkpoint spacing and metric window. Changing an
  evaluation seed invalidates comparability with every existing run; add a new
  seed rather than editing an old one.
- **Decisions get logged.** `results/matrix/decisions.md` records every change
  made after the first run was launched, together with what was known at the
  time. A design change that is not in that file is not reproducible.
- **Stability is reported conditional on competence.** A run that never learned
  has zero volatility and zero drawdown; see `analyze_matrix.py` and the
  `reached` flag.
