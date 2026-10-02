#!/bin/bash
# replicate.sh — run the preregistered replication (WP6) and commit every run
# the moment it lands, so a reclaimed container loses at most the runs in
# progress. Restartable: run_experiments.py skips runs whose .npz exists.
#
#   scripts/replicate.sh [branch]
cd "$(dirname "$0")/.."
BRANCH=${1:-$(git branch --show-current)}
OUT=results/replication
SEEDS=201,202,203,204,205,206,207,208,209,210,211,212
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WquqFTQNEufr8FYayKSSzh"

commit_new() {
  for f in "$OUT"/*_s*.npz; do
    [ -e "$f" ] || continue
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue
    git add "$f" "$OUT/protocol.json" &&
    git commit -q -m "Replication: $(basename "$f" .npz) finished" \
        -m "Preregistered in $OUT/PREREGISTRATION.md; $(ls "$OUT"/*_s*.npz | wc -l) of 36 runs on disk." \
        -m "$TRAILER" -- "$f" "$OUT/protocol.json" || true
  done
  git push -q origin "$BRANCH" 2>/dev/null || true
}

.venv/bin/python -W ignore run_experiments.py --outdir "$OUT" \
    --only hof-eval-v2,control,hof-0.25 --seeds "$SEEDS" \
    --workers "$(nproc)" >> "$OUT/run.log" 2>&1 &
RUNNER=$!
while kill -0 "$RUNNER" 2>/dev/null; do
  sleep 120
  commit_new
done
wait "$RUNNER"; echo "runner exit $?" >> "$OUT/run.log"
commit_new
