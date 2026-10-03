"""
provenance.py — which raw files every analysis output was computed from.

Every script that writes into results/ calls `record(output, scope)` right
after writing. results/analysis/provenance.json then holds, for each output,
the sha256 of the output itself and of every raw input it read, and the name
of the scope rule the inputs were drawn from.

`verify()` recomputes all of it from disk. It reports a problem when

  * an input recorded for an output is missing or has changed,
  * the scope rule now selects files the output was not computed from
    (a run was added, or rerun, and the analysis was not),
  * an output was edited after it was written.

make_tables.py --check fails on any of these, so a table can only pass if it
follows from the raw files currently on disk.

    python provenance.py          # print the verification report
"""

import fcntl
import glob
import hashlib
import json
import os
import sys

MATRIX = "results/matrix"
PROVENANCE = "results/analysis/provenance.json"


def superseded(matrix=MATRIX):
    """Conditions kept on disk but excluded from analysis (see decisions.md)."""
    p = os.path.join(matrix, "protocol.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("superseded", {})


def condition_of(path):
    return os.path.basename(path)[:-4].rsplit("_s", 1)[0]


def matrix_runs(matrix=MATRIX, pattern="*_s*.npz"):
    """Every run file that counts: all matrix runs except superseded ones."""
    sup = superseded(matrix)
    return sorted(p for p in glob.glob(os.path.join(matrix, pattern))
                  if condition_of(p) not in sup)


# Each scope is a rule for "the files this kind of output must be computed
# from". The rule is re-evaluated at verification time, so a file that
# appears later makes every output in its scope stale.
SCOPES = {
    # every run file on disk, one- and two-population, INCLUDING superseded
    # ones: no analysis reads those, but they are raw results that must never
    # be deleted (CLAUDE.md, rule 2), so their absence is caught too
    "matrix": lambda: sorted(glob.glob(os.path.join(MATRIX, "*_s*.npz"))),
    # single-population runs: the two-population conditions have their own
    # analysis and their genomes differ in size
    "single": lambda: [p for p in matrix_runs()
                       if not os.path.basename(p).startswith("asym")],
    # the only condition that keeps full population snapshots
    "control": lambda: matrix_runs(pattern="control_s*.npz"),
    # the reference run on the unmodified environment
    "reference": lambda: (sorted(glob.glob("results/ga_selfplay/ga_*.json"))
                          + ["results/ga_selfplay/history.jsonl"]),
    # the committed population snapshot the compiled continuations start from
    "resume": lambda: ["results/analysis/resume_base.npz"],
    # single-population runs scored against the slimevolleygym zoo policies
    "yardsticks": lambda: ([p for p in matrix_runs()
                            if not os.path.basename(p).startswith("asym")]
                           + sorted(glob.glob("results/zoo/*.json"))),
    # NEAT as a fifth family (WP9) and the single-population runs it is
    # compared with (their analysed values come from per_run.json)
    "neat": lambda: (sorted(glob.glob("results/neat/*_s*.npz"))
                     + [p for p in matrix_runs()
                        if not os.path.basename(p).startswith("asym")]),
    # exploratory NEAT variants (WP9, not preregistered)
    "neat_explore": lambda: sorted(glob.glob("results/neat/explore/*_s*.npz")),
    # the niche-archive experiment (WP9) and the runs it is compared with
    "qd": lambda: (sorted(glob.glob("results/qd/*_s*.npz"))
                   + sorted(glob.glob("results/replication/control_s*.npz"))
                   + sorted(glob.glob("results/replication/hof-eval-v2_s*.npz"))
                   + sorted(glob.glob("results/lab/*_s*.npz"))),
    # the discmix experiment of the lab (WP8)
    "lab": lambda: sorted(glob.glob("results/lab/*_s*.npz")),
    # the preregistered confirmatory replication (fresh seeds, own directory)
    "replication": lambda: sorted(glob.glob("results/replication/*_s*.npz")),
    # generated from code and fixed seeds only
    "none": lambda: [],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load():
    return json.load(open(PROVENANCE)) if os.path.exists(PROVENANCE) else {}


def record(output, scope, script=None, params=None):
    """Record that `output` was just computed from the files in `scope`."""
    files = SCOPES[scope]()
    entry = {
        "script": script or os.path.basename(sys.argv[0]),
        "scope": scope,
        "params": params or {},
        "output_sha256": sha256(output),
        "inputs": {p: sha256(p) for p in files},
    }
    os.makedirs(os.path.dirname(PROVENANCE), exist_ok=True)
    # analysis scripts may finish concurrently: read-modify-write under a lock
    with open(PROVENANCE + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prov = _load()
        prov[output] = entry
        with open(PROVENANCE + ".tmp", "w") as f:
            json.dump(prov, f, indent=1, sort_keys=True)
        os.replace(PROVENANCE + ".tmp", PROVENANCE)


def verify(outputs=None):
    """Return a list of problems; empty means every output follows from disk.

    `outputs` restricts the check to those files, and each of them must have
    a provenance entry.
    """
    prov = _load()
    problems = []
    names = sorted(prov) if outputs is None else list(outputs)
    for out in names:
        e = prov.get(out)
        if e is None:
            problems.append(f"{out}: no provenance record (rerun the script "
                            f"that produces it)")
            continue
        if not os.path.exists(out):
            problems.append(f"{out}: missing")
            continue
        if sha256(out) != e["output_sha256"]:
            problems.append(f"{out}: changed after it was computed")
        recorded = e["inputs"]
        current = set(SCOPES[e["scope"]]())
        for p in sorted(set(recorded) - current):
            problems.append(f"{out}: input {p} is "
                            + ("missing" if not os.path.exists(p)
                               else "no longer in scope"))
        for p in sorted(current - set(recorded)):
            problems.append(f"{out}: {p} is on disk but was not analysed")
        for p in sorted(set(recorded) & current):
            if sha256(p) != recorded[p]:
                problems.append(f"{out}: input {p} changed since analysis")
    return problems


if __name__ == "__main__":
    probs = verify()
    prov = _load()
    for out, e in sorted(prov.items()):
        print(f"{out}: {len(e['inputs'])} inputs, scope '{e['scope']}', "
              f"by {e['script']}")
    print("\n".join(probs) if probs else "all outputs follow from the files "
          "on disk")
    sys.exit(1 if probs else 0)
