#!/bin/bash
# watchdog.sh — keep the 500k self-play run alive across container restarts.
#
# The cloud sandbox this runs in can restart at any time, killing background
# processes. This script is idempotent: run it as often as you like; it only
# acts when the trainer is dead, hung, or finished.
#
# Liveness is checked via PID file + /proc cmdline (a plain `pgrep -f` can
# match the calling shell itself — learned the hard way) plus a freshness
# check on history.jsonl (catches hangs, not just deaths).

cd "$(dirname "$0")"
PIDFILE=results/train.pid
LOG=results/train_full.log
HIST=results/ga_selfplay/history.jsonl

alive() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid=$(cat "$PIDFILE")
  [ -r "/proc/$pid/cmdline" ] &&
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "train_ga_selfplay"
}

fresh() {
  [ -f "$HIST" ] || return 1
  [ $(( $(date +%s) - $(stat -c %Y "$HIST") )) -lt 180 ]
}

last_t() {
  grep -oE '"tournament": [0-9]+' "$HIST" 2>/dev/null |
    tail -1 | grep -oE '[0-9]+'
}

T=$(last_t)
T=${T:-0}

if [ "$T" -ge 500000 ]; then
  echo "COMPLETE t=$T"
  exit 0
fi

if alive && fresh; then
  echo "OK t=$T"
  exit 0
fi

if alive && ! fresh; then
  echo "HUNG at t=$T — killing pid $(cat "$PIDFILE")"
  kill -9 "$(cat "$PIDFILE")" 2>/dev/null
  sleep 2
fi

nohup .venv/bin/python -W ignore train_ga_selfplay.py \
  --resume --snapshot-freq 2500 >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 8
if alive; then
  echo "RESTARTED from t=$T (pid $(cat "$PIDFILE"))"
else
  echo "RESTART FAILED — check $LOG"
  exit 1
fi
