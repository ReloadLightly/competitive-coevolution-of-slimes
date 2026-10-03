"""
neat_explore.py — exploratory follow-up to the preregistered NEAT result (WP9).

NOT preregistered. Written after neat_analysis.py had shown that NEAT never
learned to rally (0 of 12 runs), to tell a defect of this implementation
apart from a property of NEAT in this setting. Each variant changes one
thing, or, for `as-ga`, removes what makes NEAT NEAT:

  mlp-start  NEAT as preregistered, but every genome starts as the study's
             12-10-10-3 network (random weights, scale 0.5) instead of the
             minimal network
  no-reset   NEAT as preregistered, but a mutated weight is always perturbed,
             never replaced by a fresh random value
  as-ga      NEAT's loop reduced towards the generational GA: the study's
             network, no structural mutation, one species, no stagnation
             removal, every offspring by crossover and weight perturbation
             (one elite instead of the generational GA's twenty)

Seeds 901-904, 500,000 games, each run evaluated exactly as run_neat.py
(200 sweep episodes per champion). The game, the forward pass and the
packing were checked before this (test_repo.py; NEAT-vs-NEAT games equal the
paper's MLP-vs-MLP games), so these runs probe the reproduction loop.

    python neat_explore.py --variant as-ga --seeds 901,902
    python neat_explore.py --summary      # results/neat/explore/summary.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import analyze_matrix as am
import fastvolley as fv
import provenance as pv
import run_experiments as RE
from lab import neat as N

OUT = "results/neat/explore"
SEEDS = [901, 902, 903, 904]
VARIANTS = {
    "mlp-start": {},
    "no-reset": {"p_weight_perturb": 1.0},
    "as-ga": {"p_add_node": 0.0, "p_add_conn": 0.0, "p_weight_mutate": 1.0,
              "p_weight_perturb": 1.0, "p_crossover": 1.0, "p_interspecies": 0.0,
              "threshold": 1e9, "target_species": 1, "stagnation": 10 ** 9},
}
MLP_START = ("mlp-start", "as-ga")


def _mlp_genome(scale):
    def init(rng, inno):
        return N.from_mlp(rng.normal(size=fv.PARAM_COUNT) * scale, inno)
    return init


def one_run(job):
    variant, seed = job
    path = os.path.join(OUT, f"{variant}_s{seed}.npz")
    if os.path.exists(path):
        return f"{variant}_s{seed}: already done"
    P = dict(N.PARAMS, **VARIANTS[variant])
    init = _mlp_genome(P["init_scale"]) if variant in MLP_START else None
    w, b = fv.baseline_arrays()
    t0 = time.time()
    out = N.run(seed, RE.TOURNAMENTS, RE.SAVE_EVERY, w, b, P=P, init=init)
    train_sec = time.time() - t0
    champs = out["champs"]
    n = len(champs)
    Pk = N.pack([N.compile_genome(g) for g in champs])
    mean_score, std_score = np.zeros(n), np.zeros(n)
    win, tie, loss, mean_len = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        sc, ln = N.eval_vs_baseline(Pk, i, RE.SWEEP_EPISODES, RE.SWEEP_SEED, w, b)
        mean_score[i], std_score[i] = sc.mean(), sc.std()
        win[i], tie[i], loss[i] = (sc > 0).mean(), (sc == 0).mean(), (sc < 0).mean()
        mean_len[i] = ln.mean()
    arrays = [N.genome_arrays(g) for g in champs]
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            champ_nodes=np.concatenate([a[0] for a in arrays]),
            champ_node_off=np.cumsum([0] + [len(a[0]) for a in arrays]),
            champ_conns=np.concatenate([a[1] for a in arrays]),
            champ_conn_off=np.cumsum([0] + [len(a[1]) for a in arrays]),
            champ_fit=np.array(out["champ_fit"]), n_species=np.array(out["n_species"]),
            hidden=np.array(out["hidden"]), connections=np.array(out["connections"]),
            threshold=np.array(out["threshold"]), params=np.array([json.dumps(P)]),
            variant=np.array([variant]),
            streaks=np.zeros(n, dtype=np.int64), train_meanlen=np.array(out["meanlen"]),
            tie_rate=np.zeros(n), hof_winrate=np.zeros(n),
            mean_score=mean_score, std_score=std_score, win_rate=win,
            tie_rate_eval=tie, loss_rate=loss, eval_meanlen=mean_len,
            tournaments=np.array([RE.TOURNAMENTS]), save_every=np.array([RE.SAVE_EVERY]),
            sigma=np.array([P["sigma"]]), pop=np.array([P["pop"]]),
            hof_prob=np.array([0.0]), seed=np.array([seed]), algo=np.array(["neat"]),
            sweep_episodes=np.array([RE.SWEEP_EPISODES]),
            sweep_seed=np.array([RE.SWEEP_SEED]), train_sec=np.array([train_sec]),
            pops=np.zeros((0, 0, 0), dtype=np.float32),
            pop_streaks=np.zeros((0, 0), dtype=np.int64), pop_every=np.array([0]))
    os.replace(tmp, path)
    return (f"{variant}_s{seed}: train {train_sec / 60:.1f} min, final "
            f"{mean_score[-1]:+.2f}, best {mean_score.max():+.2f}, "
            f"rallies {out['meanlen'][-1]:.0f}")


def summary(out=os.path.join(OUT, "summary.json")):
    """Per run and per variant, from the run files (sweep scores, no
    held-out re-scoring: nothing here is a test)."""
    runs = {}
    for v in VARIANTS:
        for s in SEEDS:
            path = os.path.join(OUT, f"{v}_s{s}.npz")
            m = am.metrics(am.load_run(path))
            z = np.load(path)
            runs[f"{v}_s{s}"] = {
                "variant": v, "seed": s, "reached": m["reached"],
                "t_internal": m["t_internal"], "t_parity": m["t_parity"],
                "final": m["final"], "peak": m["peak"],
                "train_meanlen_final": m["train_meanlen_final"],
                "hidden_final": int(z["hidden"][-1])}
    by = {v: {"runs": len(SEEDS),
              "learned": sum(r["reached"] for r in runs.values() if r["variant"] == v),
              "reached_parity": sum(r["t_parity"] is not None
                                    for r in runs.values() if r["variant"] == v),
              "best_final": max(r["final"] for r in runs.values() if r["variant"] == v),
              "max_train_meanlen": max(r["train_meanlen_final"]
                                       for r in runs.values() if r["variant"] == v)}
          for v in VARIANTS}
    res = {"exploratory": True, "variants": by, "per_run": runs}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    pv.record(out, "neat_explore")
    print(json.dumps(by, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, help="one of " + ", ".join(VARIANTS))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.summary:
        return summary()
    variants = [args.variant] if args.variant else list(VARIANTS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = [(v, s) for v in variants for s in seeds]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one_run, jobs):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


if __name__ == "__main__":
    main()
