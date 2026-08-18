#!/bin/bash
# queue.sh — one continuous, priority-ordered job list. Sequential per-tier
# invocations left workers idle whenever a tier ended with fewer jobs than
# workers; a single ordered list keeps all three busy to the end. Runs whose
# .npz already exists are skipped, so this is safe to re-run.
cd "$(dirname "$0")"
echo "QUEUE: single ordered pass over all remaining work" >> results/matrix.log
.venv/bin/python -W ignore run_experiments.py --workers 3 --only \
  hof-eval,ga2015,es,sigma-0.05,sigma-0.20,hof-full,pop-32,pop-512 \
  >> results/matrix.log 2>&1
echo "QUEUE COMPLETE" >> results/matrix.log
