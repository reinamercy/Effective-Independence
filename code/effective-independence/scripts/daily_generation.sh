#!/usr/bin/env bash
# Daily generation run, intended to be driven by cron.
#
# Purpose: 108 cells remain, all on gemini-3.5-flash, which is capped at 20
# requests/day on the free tier. Full coverage therefore needs roughly six
# productive days rather than one long run. This script does exactly one
# attempt per invocation and records the result, so a daily cron entry walks
# the matrix to 360/360 without anyone re-running it by hand.
#
# Cron-safe: uses absolute paths derived from its own location, does not rely
# on a login shell environment, and appends rather than overwrites its log.
# Idempotent: run_full_generation.py is cache-based, so extra invocations are
# harmless and simply report no gain.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

PY="$PROJECT_DIR/.venv/bin/python"
LOG="$PROJECT_DIR/logs/daily_generation.log"

coverage() {
  "$PY" - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/error_matrix.csv")))
print(sum(1 for r in rows if r["correct"] in ("True", "False")))
EOF
}

before=$(coverage)

if [ "$before" -ge 360 ]; then
  echo "$(date '+%Y-%m-%d %H:%M') already COMPLETE at ${before}/360, nothing to do" >> "$LOG"
  exit 0
fi

"$PY" scripts/run_full_generation.py > /tmp/daily_gen_out.txt 2>&1
rc=$?
after=$(coverage)
gained=$((after - before))

echo "$(date '+%Y-%m-%d %H:%M') ${before} -> ${after} (+${gained}), missing $((360 - after)), exit=${rc}" >> "$LOG"

if [ "$after" -ge 360 ]; then
  echo "$(date '+%Y-%m-%d %H:%M') *** FULL COVERAGE REACHED (360/360) ***" >> "$LOG"
  echo "$(date '+%Y-%m-%d %H:%M') Next: re-run run_analysis.py and run_budget_experiment.py," >> "$LOG"
  echo "$(date '+%Y-%m-%d %H:%M')       then refresh the numbers in paper/main.tex." >> "$LOG"
fi
