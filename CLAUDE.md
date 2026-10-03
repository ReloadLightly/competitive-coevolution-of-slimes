# CLAUDE.md — competitive-coevolution-of-slimes

Claude Code reads this file at the start of every session. It is the project's
memory: previous chat sessions are NOT available to you, so everything you need
is here, in `docs/`, in `results/matrix/decisions.md` and in the git history.

Owner: Roland (GitHub ReloadLightly). He reads, he does not watch terminals.
Work autonomously through the work packages below, commit and push as you go,
and only stop where a STOP is written.

---

## 1. What this repository is

Self-play neuroevolution on David Ha's Slime Volleyball, run as a designed
experiment (500,000 self-play games per run, 6 seeds per main condition,
bit-exact compiled environment in `fastvolley.py` / `fastvolley_kernels.py`).

Findings currently claimed (README + `docs/paper/`):

| ID | Claim | Status |
|---|---|---|
| C1 | Internal improvement precedes external transfer (reference run: rallies pass 1,500 steps at 104,200 games; first champion beating the 2015 baseline at 172,000) | keep; replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered) |
| C2 | The phase change is robust but its timing is not (55,000–415,000 games across 6 control seeds) | keep; replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered), but 10/12 control runs learned, not every seed |
| C3 | The population is not cycling: ρ(Elo, time) = +0.74 in the control (the earlier +0.72 did not match the data), <1% cyclic checkpoint triples | keep; replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered) |
| C4 | Ha's winning-streak export rule is the main noise source: the exported champion ranks near the median of its own 128; streak and skill are uncorrelated | keep; replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered) |
| C5a | Hall of fame as PARENT destroys learning (1/6 vs 6/6) | keep; replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered) |
| C5b | Hall of fame as TEST (`hof-eval-v2`): neither harms nor helps detectably (all p ≥ 0.37 vs control, 5/6 learned); archive win rate 0.5 → 0.11–0.16 in runs that learned, a description rather than a validated diagnostic | rewritten in WP3 from `hof-eval-v2` (decisions.md 2026-10-02); replicated 2026-10-03 (WP6, 12 fresh seeds, preregistered); the archive-win-rate description did not hold (0.21 in a run that never learned); an archive organised by behaviour (WP9, preregistered) has no detectable effect either |
| C6 | The algorithm families differ in reliability, not ceiling | keep; rechecked in WP3 with `hof-eval-v2` as the fourth family, unchanged |
| C7 | Unequal power (asym block, 18 runs): mutual improvement 1/18, runaway 13/18; 2:1 capacity not decisive; norm-matched σ flips dominance 2/6 → 5/6 (p≈0.065) | exploratory only; never state as confirmed |

The paper's broader hook (Discussion only, never a Result): any evolutionary
loop that promotes ONE artifact out of a population — including LLM-driven
program evolution — inherits the export-rule failure mode of C4.

## 2. Standing rules (apply in every session)

1. **The paper's science code is frozen**: `fastvolley.py`,
   `fastvolley_kernels.py`, `asymmetric.py`, `algorithms.py`,
   `train_ga_selfplay.py`. Refactors, renames and "cleanups" of these files are
   forbidden; commit `376b658` (tag `paper-v1`) is the state the paper was
   built from. New environments, algorithms and archives (WP8, WP9) go into
   new modules that leave these files and every result they produced
   untouched.
2. **Never delete or overwrite raw results** (`results/**/*.npz`, `*.jsonl`,
   `protocol.json`). Superseded runs stay on disk under their old names and are
   marked as superseded in `results/matrix/decisions.md`.
3. **Every number in README and paper is generated**, never typed by hand.
   `make_tables.py` injects tables; `make_figures.py` makes figures. If a
   number you need has no generator, write one.
4. **Every change to the protocol after data exists** gets a dated entry in
   `results/matrix/decisions.md` stating what was known at the time.
5. **No claim upgrades.** If a rerun weakens a claim, the text weakens. If a
   result is exploratory (C7), it is labelled exploratory everywhere.
6. **No invented citations.** Every reference must be verified against a DOI,
   arXiv ID or publisher page before it enters the bibliography. Anything you
   could not verify goes into `paper/UNVERIFIED_REFS.md`, not the paper.
7. **Commit small, push often**, with messages that say what changed and why.
   Long runs commit results as each seed finishes.
8. Numbers in prose: 2–3 significant digits. Full precision lives in JSON.

## 3. Work packages (October 2026)

WP1–WP4 are done (PR #8, merged 2026-10-02 as `376b658`, tag `paper-v1`). They are
kept below as the record of what was asked. Work continues at WP5.

### WP1 — Bring main up to date

The full asymmetric block (79 npz in `results/matrix/`, vs 43 on main) and the
fixed `asymmetric.py` live on branch `claude/neural-slime-actir-rerun-rwdcmz`.
Main is 20 commits behind it; a test merge on 2026-10-02 was conflict-free.

1. Merge that branch into `main`.
2. Reconcile the two READMEs. `README (1).md` was uploaded by Roland through the
   web UI on 2026-08-20 and is NOT rendered by GitHub. Diff it against the
   merged `README.md`. Keep one `README.md`: Roland's newer wording wins unless
   it contradicts the data or the merged asym results, in which case the data
   wins and you list each conflict in the PR description. Delete `README (1).md`.
3. Move the operational shell scripts (`autocommit.sh queue.sh resume_to_500k.sh
   run_chunk.sh status.sh stop_after_pop.sh swap_to_asym.sh tiers.sh
   watchdog.sh`) into `scripts/` and fix any paths that reference them.
4. Run `python test_repo.py` and `python make_tables.py --check`. Both must pass.

Acceptance: main has 79+ matrix npz files, one README, tests green.

### WP2 — Make the consistency check impossible to fool

`make_tables.py --check` currently passes when raw data for a table is missing,
because the table is silently skipped. That is how main ended up showing an
asymmetry table with no data behind it.

1. Make `--check` FAIL if any table present in `docs/paper/_tables.md` or the
   README cannot be regenerated from raw data.
2. Make the per-run counts in `results/analysis/*.json` agree with the files on
   disk (previously 41 npz vs 39 runs in the analysis JSON). Regenerate.
3. Add a GitHub Actions workflow that runs `test_repo.py` and
   `make_tables.py --check` on every push (CPU only, no training).

Acceptance: deleting any single npz makes `--check` fail; CI green on main.

### WP3 — Fix the streak bug in `run_ga_hof_eval` and rerun `hof-eval`

Bug: in `algorithms.py`, `run_ga_hof_eval`, archive branch (around line 322):

```python
if score > 0:   # m lost to the archived champion
    ...population[m] = mutant of population[n]...
    winning_streak[m] = winning_streak[n]
    winning_streak[n] += 1      # BUG: n never played this game
```

This is the same bug fixed for the asymmetric kernel in commit `f362bf0`.
Apply the same fix: `m` inherits `n`'s streak, `n` is not incremented.

The streak counter is NOT cosmetic here: `argmax(winning_streak)` decides which
individual is copied into the archive every `hof_every` games, so the bug
changes the archive and therefore the whole trajectory. The old `hof-eval`
results cannot be repaired by re-export — they must be rerun.

1. Fix the bug. Add a unit test that fails on the old code (an archive loss
   must leave every other individual's streak unchanged).
2. Register a NEW condition `hof-eval-v2` (same parameters as `hof-eval`,
   seeds 101–106) in `run_experiments.py` / `protocol.json`. Old `hof-eval`
   files stay and are marked superseded in `decisions.md`.
3. Run the 6 seeds (~20–35 min each). Run as many in parallel as there are
   cores; make the runner skip seeds whose output already exists, so an
   interrupted session can simply be restarted. Commit each finished seed.
4. Rerun the analysis pipeline. Replace `hof-eval` with `hof-eval-v2`
   everywhere in tables, figures, README and paper.
5. Rewrite C5b and recheck C6 using ONLY the new numbers. Whatever comes out
   is the result — including "the archive-as-test diagnostic does not hold".
   Record the before/after numbers in `decisions.md`.

Acceptance: 6 `hof-eval-v2` npz files, test for the bug, C5b text matches data.

### WP4 — Paper (LaTeX, arXiv-ready draft)

`docs/paper/` holds ~42,000 words. That is a lab notebook, not a paper.
Write `paper/main.tex` (+ `paper/refs.bib`, figures in `paper/figures/`).

- Target: 9–10 pages main text + appendix, arXiv category cs.NE.
- Title (working): *What a self-playing population learns, what it forgets,
  and which of the two you measure.*
- Structure: Introduction (the question + C4 as the headline) · Setup
  (environment, algorithm, measurement, conditions table) · Results (C1–C6,
  one subsection each, every one with its figure) · Unequal power (C7,
  explicitly exploratory) · Discussion (export rules in any evolutionary loop,
  incl. LLM program evolution — as hypothesis) · Limitations · Reproducibility.
- Abstract: first sentence states the main finding in plain words.
- Every number in the .tex comes from a generated `paper/numbers.tex`
  (`\newcommand` macros written by a script), never typed.
- Related work, to be VERIFIED before citing (rule 6): Ha's 2015 slime
  volleyball / estool; Rosin & Belew 1997 (hall of fame, competitive
  coevolution); Cliff & Miller 1995 (CIAO plots / Red Queen tracking);
  Ficici & Pollack on coevolutionary memory/Pareto coevolution; Popovici,
  Bucci, Wiegand & de Jong, "Coevolutionary Principles"; Balduzzi et al. 2019
  (open-ended learning in symmetric zero-sum games); Czarnecki et al. 2020
  (spinning tops); arXiv 2602.16805 (simple baselines vs code evolution) for
  the Discussion hook.
- Build `paper/main.pdf` and commit it. README gets a "Paper" link at the top.

STOP after WP4: write `paper/REVIEW_NOTES.md` listing (a) every claim whose
wording changed in WP3, (b) every unverified reference, (c) anything you think
a reviewer will attack. Roland reviews and submits; you do not submit anything.

### WP5 — Repository finish

- GitHub "About" text and topics (neuroevolution, coevolution, self-play,
  reinforcement-learning, artificial-life). The session's tools cannot set
  them; write the text into the PR for Roland to paste.
- `pyproject.toml` or a pinned `requirements*.txt` that installs cleanly;
  verify the README quickstart on a fresh clone.
- README top: one-paragraph abstract, the paper link, the five key figures.

## 3b. Next phase: from one study to a lab (decided 2026-10-02)

Roland's goal: turn the repository into a neuroevolution / evolutionary
computation lab. The work packages follow the paper's Limitations section,
cheapest and most valuable first. Section 4 below was lifted by Roland on
2026-10-02 (see `results/matrix/decisions.md`); the standing rules in section
2 still apply to everything.

### WP6 — Preregistered confirmatory replication (answers "few seeds" and "post hoc design")

Status (2026-10-03): done. 6 of 6 claims replicated; see README "A preregistered replication" and decisions.md.

1. Write `results/replication/PREREGISTRATION.md` and the analysis script
   `replication.py` BEFORE any replication run starts; commit and push them.
   The push timestamp is the proof of preregistration.
2. Run `control`, `hof-0.25` and `hof-eval-v2` with fresh seeds 201–212
   (36 runs) into `results/replication/`, with the frozen code and the
   paper's parameters. Commit each run as it finishes.
3. Run `python replication.py` once all 36 exist. Its verdicts are the result,
   whatever they are. Record them in `decisions.md`.
4. README and paper: add the replication as its own section, generated
   numbers only. A claim that does not replicate is weakened (rule 5); a claim
   that replicates keeps its wording and gains the replication as evidence.
   C6 and C7 are not part of this replication and must not be described as
   replicated.

### WP7 — Analyses on data already on disk (no training)

Status (2026-10-03): done (`yardsticks.py`, `transitivity_fine.py`; tables z and t).

1. Within-run transitivity at 5,000-game spacing (every run already stores
   100 champions), closing "cycles shorter than 50,000 games are invisible".
2. Stronger external yardsticks: Ha's slimevolleygym zoo (the self-play GA
   champion and the CMA-ES policy, Apache-2.0; verify the license and record
   the source commit). Score every final champion against them.
3. Each as its own generated table; the paper's existing numbers stay as they
   are unless a result contradicts them (then rule 5).

### WP8 — A second environment and a lab interface

Status (2026-10-03): done. `lab/`, `docs/lab.md`, `run_lab.py`, `lab_analysis.py`; H8a and H8b hold, H8c does not (decisions.md). A physical second game is still open.

1. A small interface for environments and algorithms in a new package (e.g.
   `lab/`), with Slime Volleyball wrapped, not rewritten.
2. A second compiled game whose skill structure has a tunable cyclic
   component, validated the way `fastvolley.py` was (bit-level or documented
   deviations).
3. Rerun the control, archive-as-test and export-rule analyses there: do C3,
   C4 and C5b change when skill is cyclic? Preregister first, as in WP6.

### WP9 — Structure and archives (NEAT, quality diversity)

Status (2026-10-03): done. Item 2, the quality-diversity (niche) archive: H9a holds, H9b and H9c do not. Item 1, NEAT: never learned to rally (0/12), H9d finds it detectably worse than the generational GA; exploratory follow-up (`neat_explore.py`, not preregistered) locates the failure in NEAT's reproduction machinery, not in the game path. See decisions.md.

1. NEAT as specified in `docs/paper/04-appendix.md` §A.8, comparable to the
   existing families.
2. A quality-diversity archive (e.g. MAP-Elites over behaviour descriptors)
   as a further archive design.

Each WP ends with: tests and `make_tables.py --check` green, decisions.md
updated, README/paper text matching the data, a PR for Roland.

## 4. Formerly out of scope — lifted by Roland on 2026-10-02

Until 2026-10-02 the following were out of scope. Roland lifted these limits
("whatever is a blocker such as what is written in paragraph 4, i overrule
it"), so they are now allowed where a work package calls for them:

- New conditions, more seeds, longer runs.
- Re-running the asymmetric block or extending C7 (it stays exploratory until
  a preregistered replication says otherwise).
- New environments, networks or algorithms, in new modules (rule 1).
- Alternative export rules, as new conditions; existing conditions and
  their results are never changed (rule 2).
