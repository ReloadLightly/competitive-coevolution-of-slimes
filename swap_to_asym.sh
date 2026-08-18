#!/bin/bash
# Stops the main queue once everything before the population sweep is done, then
# runs the asymmetric-power condition in its place. The queue process already
# holds its job list, so editing queue.sh alone would not take effect.
cd "$(dirname "$0")"
for i in $(seq 1 500); do
  if [ -f results/matrix/hof-full_s103.npz ]; then
    Q=$(pgrep -f 'run_experiments[.]py --workers 3 --only hof-eval' | head -1)
    [ -n "$Q" ] && kill "$Q" 2>/dev/null
    sleep 3
    for p in $(pgrep -f 'multiprocessing[-]fork'); do kill "$p" 2>/dev/null; done
    sleep 2
    echo "QUEUE STOPPED before the population sweep; starting asymmetric runs" >> results/matrix.log
    .venv/bin/python -W ignore run_asymmetric.py --workers 3 >> results/matrix.log 2>&1
    echo "ASYMMETRIC COMPLETE" >> results/matrix.log
    exit 0
  fi
  sleep 60
done
