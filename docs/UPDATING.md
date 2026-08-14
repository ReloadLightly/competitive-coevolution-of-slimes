# Publish kit v3 — update your GitHub repo

The folder `neural-slime-volleyball-summer/` in this zip **is** the repository:
full git history, all code, the writeup, results, figures, GIFs, 206 champion
checkpoints and the population snapshot. The GitHub remote is already set.

## How to update your repo (2 minutes)

1. **Delete** the old unzipped folder from the previous kit — don't merge them.
2. Unzip this file somewhere (Downloads is fine).
3. **Windows:** double-click `publish.bat`
   **macOS / Linux:** open a terminal in the folder and run `bash publish.sh`
4. If git asks you to sign in, sign in as **ReloadLightly** and approve.

When it finishes it prints "Done!" and everything is live at
https://github.com/ReloadLightly/neural-slime-volleyball-summer

That is the whole procedure. Every future kit works exactly the same way:
delete old folder → unzip new one → double-click → done.

### If publish.bat fails

Open a terminal inside the `neural-slime-volleyball-summer` folder and run:

```
git push -u origin main
```

The error message it prints will say what's wrong (usually just a sign-in).

---

## Optional — finish the full 500,000-game run on your own machine

The cloud run stopped at **286,700 of 500,000 games** because that sandbox
reclaims its container whenever the session goes idle. Your machine has no
such limit, and the population snapshot is included, so this continues *this
exact run* rather than starting a new one.

**macOS / Linux**, from inside the repo folder:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_ga_selfplay.py --resume --snapshot-freq 2500
```

**Windows**, from inside the repo folder:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python train_ga_selfplay.py --resume --snapshot-freq 2500
```

Roughly about 6 hours for the remaining 213,000 games. Safe to stop and restart at
any time — progress is snapshotted every 2,500 games, and re-running the same
command continues from there. Leave it overnight.

When it's done, either bring the numbers back to Claude, or regenerate
everything yourself:

```
.venv/bin/python eval_vs_baseline.py results/ga_selfplay/ga_00500000.json --episodes 1000
.venv/bin/python eval_vs_baseline.py results/ga_selfplay --all --episodes 40
.venv/bin/python plot_curve.py results/ga_selfplay/eval_curve.jsonl
.venv/bin/python render_gif.py results/ga_selfplay/ga_00500000.json --out results/figures/final_match.gif
git add -A && git commit -m "Full 500k run complete" && git push
```

## Optional — the champion-vs-champion test (no training needed, ~1 hour)

Answers whether the population really *forgets*, or whether the champion-picking
rule is just noisy:

```
.venv/bin/python cross_generation.py --every 20000 --games 20
```

It prints a ranking and an upset count, and writes
`results/cross_generation.json`. A clean "later beats earlier" ranking points to
proxy noise; results that go in circles point to genuine coevolutionary cycling.
