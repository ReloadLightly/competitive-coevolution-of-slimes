#!/bin/bash
# status.sh — one-line view of everything in flight.
cd "$(dirname "$0")"
echo "== $(date -u +%H:%M) =="
echo "matrix runs done: $(ls results/matrix/*.npz 2>/dev/null | wc -l)"
tail -n 3 results/matrix.log | grep "min\]" || true
T=$(grep -oE '"tournament": [0-9]+' results/ga_selfplay/history.jsonl 2>/dev/null | tail -1 | grep -oE '[0-9]+')
echo "reference run: ${T:-?} / 500000"
grep -h "seed 9" results/resume_fast.log 2>/dev/null | tail -3
echo "load: $(uptime | sed 's/.*load average/load/')"
