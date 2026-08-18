#!/bin/bash
# tiers.sh — run the matrix in priority order, so the scientific core lands
# first and the precision-only runs come last.
cd "$(dirname "$0")"
P=.venv/bin/python
run () { $P -W ignore run_experiments.py --workers 3 "$@" >> results/matrix.log 2>&1; }
echo "TIER 1a: finish control + hof-0.25" >> results/matrix.log
run --only control,hof-0.25
echo "TIER 1b: corrected hall of fame (archive as test, not parent)" >> results/matrix.log
run --only hof-eval --seeds 101,102,103,104
echo "TIER 1c: new algorithm families" >> results/matrix.log
run --only ga2015,es --seeds 101,102,103,104
echo "TIER 2: hof-full (flawed design, wider archive span)" >> results/matrix.log
run --only hof-full --seeds 101,102,103
echo "TIER 3: mutation scale (closes the step-size explanation)" >> results/matrix.log
run --only sigma-0.05,sigma-0.20
echo "TIER 4: top up the three key conditions to 6 seeds" >> results/matrix.log
run --only hof-eval,ga2015,es --seeds 105,106
echo "TIER 5: population size (the collective axis)" >> results/matrix.log
run --only pop-32,pop-512
echo "TIER 6: hof dose response" >> results/matrix.log
run --only hof-0.50
echo "ALL TIERS COMPLETE" >> results/matrix.log
