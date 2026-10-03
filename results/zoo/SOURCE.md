# Opponents from Ha's slimevolleygym model zoo

Copied verbatim from https://github.com/hardmaru/slimevolleygym at commit
`8ac22434fc5a587a3395311f7854aeda17d10303`, the commit `requirements.txt`
pins. Both files were added upstream in commit `9d7bc67` ("initial commit",
2020-06-09) and are unchanged since. Licensed under the Apache License 2.0
(`LICENSE` in this directory, copied from the same commit); the files are
redistributed unmodified.

| file here | upstream path | network | trained by | sha256 |
|---|---|---|---|---|
| `ga.json` | `zoo/ga_sp/ga.json` | 12-10-10-3 tanh, 273 parameters | Ha's self-play GA (`training_scripts/train_ga_selfplay.py`) | `b8e35984df5b93236fae7c68a793efb13d13f33d9a8be1d28ba15c41eb35807c` |
| `slimevolley.cma.64.96.best.json` | `zoo/cmaes/slimevolley.cma.64.96.best.json` | 12-20-20-3 tanh, 743 parameters | CMA-ES against the 2015 baseline (estool) | `3523cca6459ea6a979b98b19de8466eba8d5d1318ce6fa71f8f7cd084f38ce9b` |

Each file is `[parameters, fitness]` in estool's flat layout (w1 b1 w2 b2 w3
b3), loaded upstream by `slimevolleygym.mlp.makeSlimePolicyLite` and
`makeSlimePolicy` respectively. `yardsticks.py` uses them as fixed opponents.
