# How each reference was verified

Rule 6 of `CLAUDE.md`: every reference is checked against a DOI, an arXiv ID
or a publisher page before it enters `refs.bib`. Checked on 2026-10-02.

The session's network policy blocked direct access to doi.org, Crossref,
arXiv, dblp, OpenAlex, Semantic Scholar, Springer and MIT Press, so checks
went through three channels that were open: the alphaXiv connector (arXiv
full text and metadata), a web search index restricted to the publisher's
own domain (the hit is the publisher's page for the work, at the DOI-bearing
URL), and `git clone` from GitHub. A field that none of these confirmed was
left out of the entry rather than guessed.

| key | verified against | channel |
|---|---|---|
| `ha2015slime` | blog.otoro.net/2015/03/28/neural-slime-volleyball/ (author's page) | search, domain-restricted |
| `ha2020slimevolleygym` | the repository's own `README.md` citation block (author, title, 2020); first commit 2020-06-09 | git clone |
| `ha2017estool` | github.com/hardmaru/estool | search, domain-restricted |
| `risi2026neuroevolution` | mitpress.mit.edu/9780262054768/neuroevolution/ (title, authors, ISBN; print date 27 Oct 2026) | search, domain-restricted |
| `stanley2002evolving` | direct.mit.edu/evco/article/10/2/99/1123 — EC 10(2):99–127, 2002 (checked 2026-10-03). The DOI 10.1162/106365602320169811 appeared only in secondary listings and is left out | search, domain-restricted |
| `stanley2004competitive` | arXiv:1107.0037, page 1 header: JAIR 21 (2004) 63–100, title and authors; Appendix A read for NEAT's settings (checked 2026-10-03) | alphaXiv |
| `rosin1997new` | direct.mit.edu/evco/article-abstract/5/1/1/790 — EC 5(1):1–29, 1997. Some bibliographies give 1996 (the tech report); the journal issue is 1997 | search, domain-restricted |
| `cliff1995tracking` | link.springer.com/chapter/10.1007/3-540-59496-5_300 — LNCS 929, pp. 200–218 | search, domain-restricted |
| `ficici2001pareto` | link.springer.com/chapter/10.1007/3-540-44811-x_34 — LNCS 2159, pp. 316–325. (The DOI is `_34`; `_35`, which an earlier draft guessed, is a different chapter) | search, domain-restricted |
| `ficici2003memory` | link.springer.com/chapter/10.1007/3-540-45105-6_35 — LNCS 2723, pp. 286–297 | search, domain-restricted |
| `cartlidge2004combating` | ieeexplore.ieee.org/document/6790490 — EC 12(2):193–222, 2004 | search, domain-restricted |
| `miconi2009why` | link.springer.com/chapter/10.1007/978-3-642-01181-8_5 — LNCS 5481, pp. 49–60 | search, domain-restricted |
| `popovici2012coevolutionary` | link.springer.com/referenceworkentry/10.1007/978-3-540-92910-9_31 — pp. 987–1033 | search, domain-restricted |
| `simione2020longterm` | direct.mit.edu/artl/article/26/4/409/97302 — Artificial Life 26(4):409–430 | search, domain-restricted |
| `nolfi2025global` | frontiersin.org, DOI 10.3389/frobt.2024.1470886 (published 21 Jan 2025) | search |
| `seals2026tripping` | link.springer.com/chapter/10.1007/978-3-032-23607-4_32 (title, authors, EvoApplications 2026). Pages not confirmed; **content not read** — cited for its topic only | search, domain-restricted |
| `balduzzi2019openended` | arXiv:1901.08106 (title, authors); PMLR 97:434–443 | alphaXiv; search |
| `czarnecki2020spinning` | arXiv:2004.09468 (title, authors); proceedings.neurips.cc (NeurIPS 2020) | alphaXiv; search, domain-restricted |
| `salimans2017evolution` | arXiv:1703.03864 | search (arxiv.org) |
| `henderson2018matters` | arXiv:1709.06560; ojs.aaai.org, DOI 10.1609/aaai.v32i1.11694 | search |
| `agarwal2021precipice` | arXiv:2108.13264; proceedings.neurips.cc (NeurIPS 2021) | search |
| `mann1947test` | projecteuclid.org/euclid.aoms/1177730491 — AMS 18(1):50–60 | search |
| `bradley1952rank` | academic.oup.com/biomet/article-abstract/39/3-4/324/326091 — Biometrika 39(3–4):324–345 | search, domain-restricted |
| `novikov2025alphaevolve` | arXiv:2506.13131 (title, first author; full author list not transcribed) | search (arxiv.org) |
| `lange2025shinkaevolve` | arXiv:2509.19349 (title, three authors) | search (arxiv.org) |
| `gideoni2026simple` | arXiv:2602.16805 (title, authors, content) | alphaXiv |

Quoted material from Ha's repository, checked in `TRAINING.md` of
`hardmaru/slimevolleygym`: the lineage proxy "without actually computing who
is best to save time" (line 80), and the published GA self-play score
0.353 ± 0.728 over 1,000 episodes (line 84).
