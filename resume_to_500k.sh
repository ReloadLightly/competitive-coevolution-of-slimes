#!/bin/bash
# resume_to_500k.sh — continue this repository's self-play run from its last
# population snapshot until Ha's full 500,000 tournament games are played.
#
# Safe to stop and restart at any time: progress is snapshotted every 2,500
# games, so re-running this script always picks up where it left off.
# Expect roughly 9 hours on one CPU core from the 182k snapshot.
#
#   bash resume_to_500k.sh
#
# When it finishes, regenerate the results:
#   .venv/bin/python eval_vs_baseline.py results/ga_selfplay/ga_00500000.json --episodes 1000
#   .venv/bin/python eval_vs_baseline.py results/ga_selfplay --all --episodes 100
#   .venv/bin/python plot_curve.py results/ga_selfplay/eval_curve.jsonl
#   .venv/bin/python render_gif.py results/ga_selfplay/ga_00500000.json --out results/figures/final_match.gif

set -e
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

echo "resuming from $(ls results/ga_selfplay/ga_*.json | tail -1)"
echo "target: 500,000 tournaments — this takes hours; Ctrl-C is safe."
echo

"$PY" -W ignore train_ga_selfplay.py --resume --snapshot-freq 2500

echo
echo "done — 500,000 tournaments played."
