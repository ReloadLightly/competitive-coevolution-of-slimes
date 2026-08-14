#!/bin/bash
# run_chunk.sh — run one foreground chunk of training, then exit cleanly.
#
# Written for ephemeral cloud containers that are reclaimed as soon as the
# session goes idle: a chunk must therefore BLOCK (keeping the session busy)
# rather than launch training in the background and return.
#
#   bash run_chunk.sh [seconds]     # default 540 (9 minutes)
#
# Exits immediately (status 0, prints COMPLETE) once 500,000 games are done.

cd "$(dirname "$0")"
SECS=${1:-540}
HIST=results/ga_selfplay/history.jsonl

last_t() {
  grep -oE '"tournament": [0-9]+' "$HIST" 2>/dev/null | tail -1 | grep -oE '[0-9]+'
}

T=$(last_t); T=${T:-0}
if [ "$T" -ge 500000 ]; then
  echo "COMPLETE t=$T"
  exit 0
fi

pkill -f train_ga_selfplay 2>/dev/null   # never two trainers on one snapshot
sleep 1

.venv/bin/python -W ignore train_ga_selfplay.py \
  --resume --snapshot-freq 1000 --max-seconds "$SECS" >> results/train_full.log 2>&1

T=$(last_t); T=${T:-0}
git add -A >/dev/null 2>&1
git commit -q -m "Training chunk: through tournament $T" >/dev/null 2>&1
if [ "$T" -ge 500000 ]; then
  echo "COMPLETE t=$T"
else
  echo "CHUNK DONE t=$T  ($((500000 - T)) games remaining)"
fi
