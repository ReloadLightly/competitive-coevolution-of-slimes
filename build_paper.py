"""
build_paper.py — assemble the writeup into one self-contained HTML page.

Concatenates docs/paper/0*.md in order, converts to HTML, inlines every
referenced figure as a data URI, and wraps the result in a stylesheet whose
palette is taken from the game itself: `slimevolleygym.slimevolley`'s night
colours are the ball's orange (217, 79, 0), the left agent's blue
(35, 93, 188), the ground's grey (116, 114, 117) and the background
(11, 16, 19). The dashed parity line — score 0 against the 2015 champion, the
thing the whole study is measured against — is the section divider.

    python build_paper.py                    # -> docs/paper/paper.html
    python build_paper.py --md               # also -> docs/paper/writeup.md
"""

import argparse
import base64
import os
import re

import markdown

PAPER = "docs/paper"
SECTIONS = ["00-overview.md", "01-methods.md", "02-results.md",
            "03-ablations-and-analysis.md", "04-appendix.md"]

CSS = """
:root {
  /* Light: paper ground, ink from the game's background colour. */
  --ground: #FCFCFB;
  --surface: #F4F4F2;
  --surface-2: #EBEBE8;
  --ink: #14191C;
  --ink-2: #414749;
  --ink-3: #747275;          /* the game's ground colour */
  --rule: #DCDCD8;
  --accent: #1F55AB;         /* the left agent, darkened for contrast on paper */
  --accent-soft: #E7EDF8;
  --ball: #C24700;           /* the ball, darkened for contrast on paper */
  --ball-soft: #FBEBE1;
  --good: #1F6B45;
  --shadow: 0 1px 2px rgba(20, 25, 28, .06), 0 8px 24px rgba(20, 25, 28, .05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0B1013;       /* the game's night background, verbatim */
    --surface: #141A1E;
    --surface-2: #1D252A;
    --ink: #E9E7E3;
    --ink-2: #B4B2AE;
    --ink-3: #8A888B;
    --rule: #262F35;
    --accent: #74A4F2;
    --accent-soft: #16233A;
    --ball: #FF7F33;
    --ball-soft: #2B1A0F;
    --good: #63C295;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
  }
}
:root[data-theme="dark"] {
  --ground: #0B1013;
  --surface: #141A1E;
  --surface-2: #1D252A;
  --ink: #E9E7E3;
  --ink-2: #B4B2AE;
  --ink-3: #8A888B;
  --rule: #262F35;
  --accent: #74A4F2;
  --accent-soft: #16233A;
  --ball: #FF7F33;
  --ball-soft: #2B1A0F;
  --good: #63C295;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
}

* { box-sizing: border-box; }

body {
  background: var(--ground);
  color: var(--ink);
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.62;
  margin: 0;
  padding: 0 1.25rem 6rem;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 47rem; margin: 0 auto; }

/* ---- masthead ---------------------------------------------------------- */
.masthead {
  padding: 4.5rem 0 2rem;
  display: flex; flex-direction: column; gap: 1.1rem;
}
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--ink-3);
}
h1 {
  font-family: "Newsreader", Georgia, serif;
  font-weight: 500; font-size: clamp(2.1rem, 5.2vw, 3.1rem);
  line-height: 1.1; letter-spacing: -.015em; margin: 0;
  text-wrap: balance;
}
.standfirst {
  font-family: "Newsreader", Georgia, serif;
  font-size: 1.16rem; line-height: 1.5; color: var(--ink-2);
  font-style: italic; margin: 0; max-width: 34rem;
}

/* the dashed parity line: score 0 against the 2015 champion */
.parity {
  border: 0; height: 0; margin: 3rem 0;
  border-top: 1.5px dashed var(--ball); opacity: .55;
}

/* ---- headings ---------------------------------------------------------- */
h2 {
  font-family: "Newsreader", Georgia, serif;
  font-weight: 500; font-size: 1.72rem; line-height: 1.2;
  letter-spacing: -.01em; margin: 3.4rem 0 1rem; text-wrap: balance;
  padding-top: 1.6rem; border-top: 1px solid var(--rule);
}
h2:first-of-type { border-top: 0; padding-top: 0; }
h3 {
  font-family: "Newsreader", Georgia, serif;
  font-weight: 600; font-size: 1.17rem; line-height: 1.3;
  margin: 2.3rem 0 .6rem; text-wrap: balance;
}
h4 {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-weight: 600; font-size: .78rem; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-3); margin: 2rem 0 .5rem;
}
p { margin: 0 0 1.05rem; }
strong { font-weight: 600; }
em { font-style: italic; }

a { color: var(--accent); text-decoration: none;
    border-bottom: 1px solid color-mix(in srgb, var(--accent) 35%, transparent); }
a:hover { border-bottom-color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px; }

ul, ol { margin: 0 0 1.05rem; padding-left: 1.3rem; }
li { margin-bottom: .38rem; }
li::marker { color: var(--ink-3); }

blockquote {
  margin: 1.6rem 0; padding: .1rem 0 .1rem 1.3rem;
  border-left: 2px solid var(--ball); color: var(--ink-2); font-style: italic;
}

/* ---- data: tables and code -------------------------------------------- */
.tablewrap {
  overflow-x: auto; margin: 1.4rem 0 1.1rem;
  border: 1px solid var(--rule); border-radius: 3px; background: var(--surface);
}
table {
  border-collapse: collapse; width: 100%;
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .745rem; font-variant-numeric: tabular-nums; line-height: 1.45;
}
thead th {
  text-align: left; font-weight: 600; color: var(--ink-2);
  background: var(--surface-2); padding: .55rem .7rem;
  border-bottom: 1px solid var(--rule); white-space: nowrap;
  position: sticky; top: 0;
}
td { padding: .42rem .7rem; border-bottom: 1px solid var(--rule); white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td:first-child, thead th:first-child { white-space: normal; min-width: 8rem; }

pre {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  padding: .95rem 1.1rem; overflow-x: auto; margin: 1.3rem 0;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .77rem; line-height: 1.6;
}
code {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .84em; background: var(--surface-2); padding: .1em .35em;
  border-radius: 2px;
}
pre code { background: none; padding: 0; font-size: 1em; }

/* ---- figures ----------------------------------------------------------- */
figure { margin: 2rem 0; }
figure img {
  width: 100%; height: auto; display: block; border-radius: 3px;
  border: 1px solid var(--rule); background: #fff;
}
figcaption {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .72rem; line-height: 1.55; color: var(--ink-2);
  margin-top: .6rem; padding-left: .1rem;
}
figcaption b { color: var(--ink); font-weight: 600; }

/* ---- contents ---------------------------------------------------------- */
.toc {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
  padding: 1.1rem 1.3rem; margin: 2rem 0 0;
}
.toc h4 { margin-top: 0; }
.toc ol { margin: 0; padding-left: 1.15rem;
          font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .78rem; }
.toc li { margin-bottom: .3rem; }

.note {
  background: var(--accent-soft); border-left: 2px solid var(--accent);
  padding: .85rem 1.1rem; margin: 1.6rem 0; font-size: .93rem;
  color: var(--ink-2); border-radius: 0 3px 3px 0;
}

footer {
  margin-top: 4rem; padding-top: 1.4rem; border-top: 1px solid var(--rule);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .72rem; color: var(--ink-3); line-height: 1.7;
}

@media (max-width: 640px) {
  body { font-size: 16px; padding: 0 1rem 4rem; }
  .masthead { padding-top: 2.75rem; }
  table { font-size: .7rem; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

HEAD = """<title>Losing on Purpose</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&\
family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&\
family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>%s</style>
""" % CSS


def inline_images(html, base):
    """Replace every <img> src with a data URI so the page is self-contained.

    Matches src wherever it sits in the tag: python-markdown emits
    `<img alt="..." src="..." />`, so a src-first pattern silently matches
    nothing and the page ships with broken images.
    """
    def repl(m):
        src = m.group(2)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        path = os.path.normpath(os.path.join(base, src))
        if not os.path.exists(path):
            path = os.path.normpath(src)
        if not os.path.exists(path):
            print(f"  missing figure: {src}")
            return m.group(0)
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'{m.group(1)}src="data:{mime};base64,{b64}"'
    return re.sub(r'(<img[^>]*?)src="([^"]+)"', repl, html)


def wrap_tables(html):
    """Give every table its own horizontally scrollable container."""
    return re.sub(r"<table>(.*?)</table>",
                  lambda m: f'<div class="tablewrap"><table>{m.group(1)}</table></div>',
                  html, flags=re.DOTALL)


def figure_captions(html):
    """Turn '<p><img ...></p><p><em>Figure N. ...</em></p>' into a real figure."""
    pat = re.compile(r"<p>(<img [^>]+>)</p>\s*<p><em>(Figure [^<]*)</em></p>",
                     re.DOTALL)
    return pat.sub(lambda m: f"<figure>{m.group(1)}"
                             f"<figcaption>{caption_bold(m.group(2))}</figcaption>"
                             f"</figure>", html)


def caption_bold(text):
    return re.sub(r"^(Figure \d+\.)", r"<b>\1</b>", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PAPER, "paper.html"))
    ap.add_argument("--md", action="store_true",
                    help="also write the concatenated markdown")
    args = ap.parse_args()

    parts = []
    for name in SECTIONS:
        path = os.path.join(PAPER, name)
        if not os.path.exists(path):
            print(f"  skipping missing {name}")
            continue
        parts.append(open(path).read())
    src = "\n\n<hr class='parity'/>\n\n".join(parts)

    if args.md:
        with open(os.path.join(PAPER, "writeup.md"), "w") as f:
            f.write("\n\n---\n\n".join(parts))

    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    body = figure_captions(body)
    body = wrap_tables(body)
    body = inline_images(body, PAPER)

    # the first h1 becomes the masthead
    m = re.search(r"<h1>(.*?)</h1>\s*<p><strong>(.*?)</strong></p>", body,
                  re.DOTALL)
    if m:
        head_html = (
            '<header class="masthead">'
            '<div class="eyebrow">Neural Slime Volleyball &middot; '
            'self-play neuroevolution</div>'
            f'<h1>{m.group(1)}</h1>'
            f'<p class="standfirst">{m.group(2)}</p>'
            '</header><hr class="parity"/>')
        body = body[:m.start()] + head_html + body[m.end():]

    html = (HEAD + '<div class="wrap">' + body +
            '<footer>Generated from the repository at build time: every table '
            'is written by <code>make_tables.py</code> and every figure by '
            '<code>make_figures.py</code>, from the files in '
            '<code>results/</code>. Environment, baseline policy and the '
            'original GA are David Ha\'s '
            '<a href="https://github.com/hardmaru/slimevolleygym">slimevolleygym</a> '
            '(Apache-2.0).</footer></div>')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"{len(html)/1024:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
