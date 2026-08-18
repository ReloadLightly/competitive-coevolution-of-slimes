#!/bin/bash
# autocommit.sh — the container can be reclaimed at any moment; anything not
# pushed is gone. Commits and pushes new results every 8 minutes.
cd "$(dirname "$0")"
while true; do
  sleep 480
  git add -A results/matrix results/analysis results/ga_selfplay results/validation.json 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    N=$(ls results/matrix/*.npz 2>/dev/null | wc -l)
    T=$(grep -oE '"tournament": [0-9]+' results/ga_selfplay/history.jsonl 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    git -c user.email=roland.loechli@googlemail.com -c user.name="Roland Loechli" \
        commit -q -m "Results snapshot: ${N} matrix runs, reference run at ${T:-?}" 2>/dev/null || true
    git push -q origin claude/neural-slime-actir-rerun-rwdcmz 2>/dev/null || true
  fi
done
