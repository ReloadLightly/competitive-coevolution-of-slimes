# Review notes for Roland (STOP after WP4)

Nothing has been submitted. This file lists what changed, what could not be
verified, and what a reviewer is likely to attack. Section (a) is filled in
from the `hof-eval-v2` analysis.

## (a) Claims whose wording changed in WP3

_Filled in after the hof-eval-v2 analysis._

## (b) References that could not be verified

Not cited in `main.tex`; details and the reason in `UNVERIFIED_REFS.md`.

- **Cliff (1993)**, the source of Cliff's δ. The paper names and uses the
  statistic without a citation. Worth verifying (Psychological Bulletin
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
   the paper could lean on it more.
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

## Decisions left to you

- **Author line and affiliation.** `main.tex` lists "Roland Loechli" with no
  affiliation or contact.
- **Disclosure of AI assistance.** Code, analysis and text were drafted with an
  AI coding assistant in this and earlier sessions. arXiv and most venues
  expect authors to take responsibility for such content and many ask for a
  disclosure; `main.tex` has none yet.
- **arXiv category**: cs.NE as primary per CLAUDE.md; cs.LG or cs.MA as
  cross-lists would fit.
