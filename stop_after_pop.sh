#!/bin/bash
# Stops the run queue once the population sweep is complete, so the dropped
# hof-0.50 seeds never start. The queue process already holds its job list, so
# editing queue.sh alone would not have taken effect.
cd "$(dirname "$0")"
for i in $(seq 1 400); do
  if [ -f results/matrix/pop-512_s103.npz ] && [ -f results/matrix/pop-32_s103.npz ]; then
    Q=$(pgrep -f 'run_experiments[.]py --workers 3 --only hof-eval' | head -1)
    [ -n "$Q" ] && kill "$Q" 2>/dev/null
    sleep 3
    for p in $(pgrep -f 'multiprocessing[-]fork'); do kill "$p" 2>/dev/null; done
    echo "QUEUE STOPPED after the population sweep (hof-0.50 dropped)" >> results/matrix.log
    exit 0
  fi
  sleep 60
done
