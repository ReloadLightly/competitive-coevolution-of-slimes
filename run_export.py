"""
run_export.py — fresh control runs for the export-rule experiment (WP10).

Design and analysis are fixed in results/export/PREREGISTRATION.md before any
run exists. Nothing new is trained here: the runs are the paper's own
`control` condition (run_experiments.one_run, population snapshots every
50,000 games) and WP8's discmix control (run_lab.one_run), on seeds no
earlier experiment used. The export rules are applied afterwards, to the
stored population snapshots, by export_analysis.py: in the control GA the
exported individual never feeds back into training, so every rule can be
compared on the same populations.

    python run_export.py                          # everything, one worker per core
    python run_export.py --only control --seeds 401
    python run_export.py --only discmix-0.75-control
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import run_experiments as RE
import run_lab as RL

OUT = "results/export"
SLIME_SEEDS = list(range(401, 413))
LAB_SEEDS = list(range(501, 513))
CONDITIONS = {"control": SLIME_SEEDS}
CONDITIONS.update({f"discmix-{lam:.2f}-control": LAB_SEEDS for lam in RL.LAMBDAS})


def one_run(job):
    cond, seed = job
    if cond == "control":
        return RE.one_run((cond, seed, OUT, RE.TOURNAMENTS))
    return RL.one_run((cond, seed, OUT))


def write_protocol(path):
    """The design, recorded once; refuse to change an existing record."""
    new = json.loads(json.dumps({
        "conditions": CONDITIONS,
        "slime": {"tournaments": RE.TOURNAMENTS, "save_every": RE.SAVE_EVERY,
                  "pop_every": RE.POP_EVERY, "sweep_episodes": RE.SWEEP_EPISODES,
                  "sweep_seed": RE.SWEEP_SEED, "control": RE.CONDITIONS["control"]},
        "discmix": {"tournaments": RL.TOURNAMENTS, "save_every": RL.SAVE_EVERY,
                    "pop_every": RL.POP_EVERY, "pop": RL.POP, "sigma": RL.SIGMA,
                    "lambdas": list(RL.LAMBDAS)}}))
    if os.path.exists(path) and json.load(open(path)) != new:
        raise SystemExit(f"{path} differs from this design; refusing to overwrite it")
    with open(path, "w") as f:
        json.dump(new, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated conditions")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    write_protocol(os.path.join(OUT, "protocol.json"))
    conds = args.only.split(",") if args.only else list(CONDITIONS)
    jobs = []
    for c in conds:
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else CONDITIONS[c]
        jobs += [(c, s) for s in seeds if s in CONDITIONS[c]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
