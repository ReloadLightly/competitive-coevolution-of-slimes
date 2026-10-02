# Review notes for Roland (STOP after WP4)

Nothing has been submitted. This file lists what changed, what could not be
verified, and what a reviewer is likely to attack. Section (a) is filled in
from the `hof-eval-v2` analysis.

## (a) Claims whose wording changed in WP3

Source of every number below: `results/matrix/decisions.md`, entry
"2026-10-02 — hof-eval-v2 analysed". The superseded `hof-eval` runs (streak
bug) are excluded everywhere; the corrected runs are `hof-eval-v2`.

**C5b — archive used as a test (weakened).**

| | before (README, write-up §3, CLAUDE.md) | after (README finding 5, write-up §3, paper §3.6) |
|---|---|---|
| headline | "A hall of fame is a tax, not insurance — when skill is transitive" | "An archive of past champions does not help when skill is transitive — and as a parent it destroys learning" |
| effect vs control | "stops being destructive but still does not help"; write-up: "it makes the run *worse*" (above-parity δ = −0.75, p = 0.028) | "neither destructive nor helpful": indistinguishable from the control on every outcome at six seeds (all p ≥ 0.37); 5/6 learned to rally vs 6/6 |
| learning speed | (buggy runs transitioned at a median 350k games; not stated) | median 150k, in line with the control's 120k |
| archive win rate | "decays from ~0.50 to under 0.10 over a run, in every seed" | from 0.46–0.50 to 0.11–0.16 in every run that learned; stays at 0.50 in the one run that did not |
| games without selection | "roughly 22% of late-training games" | 21–22% in the runs that learned |
| diagnostic | "The diagnostic generalises: … if it decays to zero, the archive has become a free win and its budget is being subtracted from the live arms race" | "a cheap check …; here it tracked progress, but that is a description of six runs, not a validated predictor" |
| fig10 | caption "falling … to under 0.10 in every seed"; right panel "What the archive costs" | caption as above; right panel "Games without a selection event" |

The bug made the archive look worse than it is: the "tax" reading (a
measurable cost to learning) was largely the bug and is withdrawn.

**C6 — families differ in reliability, not ceiling (unchanged in substance).**
Rechecked with `hof-eval-v2` as the fourth family: only the control learned to
rally and beat the 2015 policy in every seed; the archive as test is 5/6 and
4/6; the spread of end-of-run scores is 3.0–3.5× the control's (was "about
three times") for the other three families; the best single endpoint is still
a self-play ES seed.

**Also changed this session, outside WP3:**

- **C3**: ρ(Elo, time) is +0.74 over the control seeds, not +0.72 (WP1).
- **C4** gained a direct measurement instead of an inference: between
  consecutive 50,000-game population snapshots, the exported champion's score
  fell by 8.6 points in total over the six control runs, the best member of
  the same pools by 1.1, the median member by 0.9. The paper's Discussion now
  quotes this ratio (about 8×) instead of "contributed more of the measured
  volatility than the coevolutionary dynamics did".
- **Compiled environment**: the validation was rerun on today's checkpoints:
  still 200/200 games bit-identical, now over 304,812 steps; measured speed-up
  26× (was 31×), so a reference run is about 13 core-hours (README said
  "about twelve"; an intermediate generated value was 16, from the old
  throughput measurement).
- **README (WP1)**: seed counts are 1–6 per condition, not 3–12; fig11 shows
  unequal capacity, not unequal size; fig8 shows three families, not four;
  only population 32 ran (one seed).

## (b) References that could not be verified

Not cited in `main.tex`; details and the reason in `UNVERIFIED_REFS.md`.

- **Cliff (1993)**, the source of Cliff's δ. The paper names and uses the
  statistic without a citation (Section 2, Measurements). Worth verifying (Psychological Bulletin
  114(3), believed DOI 10.1037/0033-2909.114.3.494) and adding.
- **Watson & Pollack (2001)**, coevolutionary dynamics in a minimal substrate
  (GECCO 2001, no DOI). Natural citation for intransitivity/disengagement.
- **Lam, Pitrou & Seibert (2015)**, the numba paper. The paper names numba
  without a citation.
- **Stanley & Miikkulainen**, dominance tournament: not checked.

Verified but only partially:

- **Seals & Tauritz (2026)**, *Tripping Over the Past* (EvoApplications 2026):
  title, authors, venue and DOI verified; **the content was not read** (no
  access through this session's network). It is cited only for its topic. It
  is the closest prior work to C5b and should be read before submission: if it
  already reports that a hall of fame inflates or deflates measured progress
  in a transitive setting, the related-work paragraph and C5b's framing need
  to engage with it.
- **Novikov et al. (2025)**, AlphaEvolve: first author and arXiv ID verified;
  the full author list is abbreviated as "and others".

How every reference was checked: `REFERENCES.md`. The session's network
blocked doi.org, Crossref, arXiv and most publishers; checks went through the
alphaXiv connector, publisher-domain-restricted web search, and git.

## (c) What a reviewer will attack

1. **"The export-rule result is specific to Ha's streak counter."** True, and
   the paper should say so more loudly than it does: the streak counter is an
   unusually bad proxy because it is inherited by the loser. The general claim
   (any promotion rule has its own error term) rests on one rule in one game;
   the LLM-evolution paragraph is explicitly a hypothesis. Expect a request for
   a second promotion rule in a second domain.
2. **Seeds.** Six per main condition, one or three for side conditions. The
   strongest claims (C4: rank, gap, ρ) pool over snapshots and seeds and are
   robust; condition comparisons at n = 6 have a floor of p = 0.002 and most
   secondary comparisons are uncorrected for multiplicity.
3. **Post hoc design changes.** The design changed while running (seed counts,
   hof-full added, hof-0.50 and the population sweep cut, the asymmetric block
   rebuilt twice, hof-eval rerun after a bug). All of it is in `decisions.md`
   with what was known at the time, but the final design is not the
   pre-registered one. The paper says so in Limitations; a reviewer may still
   ask for a clean confirmatory replication of C4/C5b.
4. **Transitivity is measured at 50,000-game spacing.** Cycles shorter than
   that are invisible to the within-run tournament. The claim "the population
   is not cycling" should be read as "not at the resolution measured"; a finer
   tournament on one or two runs would close this cheaply.
5. **The 2015 baseline is a single, weak, recurrent opponent** that does not
   see its opponent. "Above parity" against it is a low bar, and a policy can
   specialise against it. The cross-run tournament (Appendix) is the check;
   the families section of the main text now quotes it (every control
   champion rates above zero, each other family has a champion far below),
   with the full tournament in Appendix A.5. It is still a tournament among
   these runs' own champions, not an outside opponent.
6. **Self-play ES was tuned only at pilot scale.** C6 says so; a reviewer will
   still discount the ES row.
7. **The unequal-power block** is exploratory, was redesigned twice, and its
   headline comparison has p = 0.065 at six seeds per arm. It is labelled
   exploratory everywhere; a reviewer may ask to move it to an appendix.
8. **The reference run vs the compiled port.** The compiled environment is
   validated bit for bit per game, but training trajectories differ by RNG
   family; Table A3 shows the continuations land in the same band, not on the
   same trajectory.
9. **Analysis parameters were not all recorded.** The champion-proxy check was
   run with 30 episodes per individual while the script defaulted to 40; this
   was found when reproducing the analysis and is logged in `decisions.md`
   (2026-10-02). No published number changed, but it shows that before this
   session the analysis was not fully reproducible from the documented
   commands.
10. **The mutation-step subsection (§3.5) rests on three seeds per setting**
   and one seed for population 32. "The mutation step is not the cause" is
   the notebook's own reading (`docs/paper/03-ablations-and-analysis.md` §4);
   the paper gives the run counts next to every number, but a reviewer may
   read the heading as stronger than three seeds allow.
11. **"Neither harms nor helps detectably" (C5b) is a statement about power
   as much as about the archive.** At six seeds a side, a moderate effect in
   either direction would not be detected. One archive-as-test seed never
   learned (the control: none); that is one run.
12. **The decline measurement behind C4** uses the population snapshots
   (every 50,000 games, 60 episodes per individual), not the 5,000-game
   checkpoint curve where most of the visible swings are. It shows the pool
   does not lose ground between snapshots while the exported champion does; it
   does not decompose the checkpoint-level swings.
13. **The write-up in `docs/paper/` is a lab notebook** and still contains
   hand-typed numbers outside the sections rewritten in this session (§3 of
   the analysis, the re-export lessons, the appendix design table). Every
   table there is generated, and README and `paper/main.tex` are fully
   generated; the notebook prose is not. Converting it, or marking it as
   superseded by the paper, is your call.

## (d) The paper now covers the whole README

After WP4 the paper was extended so that every README section and every
result in the repository appears in it, each number generated
(`prose_numbers.py` → `paper/numbers.tex`, tables from `make_tables.py`):

- **Main text** (to the top of page 11): §3.1 says what competence means
  (win/draw/loss of the final champion); §3.2 the damping of the
  trajectory; new §3.5 the mutation-scale and population sweeps; the
  families section quotes the cross-run tournament; Acknowledgements credit
  Ha's environment, baseline and GA (Apache-2.0).
- **Appendix A** (supplementary results): the reference run in detail, the
  export rule over training and the alternative promotion rules, the sweeps,
  the archive conditions, the cross-run tournament, the unequal-power block
  (labelled exploratory), and the per-condition tables for every run.
- **Appendix B**: the compiled environment, its three documented
  deviations, the per-game validation and the compiled continuations of the
  reference run.
- **Appendix C**: the command list that regenerates every number, table
  and figure, plus provenance, CI and the decision log.

The PDF is 20 pages: about 10 of main text, 2 of references, 8 of appendix.
Not carried over from the README: the quickstart, the documents index and
the February postmortem, which are repository material rather than results.

## Decisions left to you

- **Author line and affiliation.** `main.tex` lists "Roland Loechli" with no
  affiliation or contact.
- **Disclosure of AI assistance.** Code, analysis and text were drafted with an
  AI coding assistant in this and earlier sessions. arXiv and most venues
  expect authors to take responsibility for such content and many ask for a
  disclosure; `main.tex` has none yet.
- **arXiv category**: cs.NE as primary per CLAUDE.md; cs.LG or cs.MA as
  cross-lists would fit.
