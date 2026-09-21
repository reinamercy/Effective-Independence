#!/usr/bin/env bash
# Retry generation on a fixed interval until the daily quota resets and cells
# start landing again, or until we reach full coverage.
#
# Why this exists: the free-tier cap is 20 requests/day/model on
# gemini-3.5-flash, and all remaining cells depend on that one model. Once the
# day's allowance is spent, further runs return immediately without producing
# anything. Rather than re-running by hand, this polls on an interval so the
# first run after the (unconfirmed) reset picks cells up automatically.
#
# Safe to interrupt at any point: run_full_generation.py is cache-based and
# idempotent, so nothing is lost and nothing is duplicated.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="./.venv/bin/python"
LOG="logs/retry_loop.log"
INTERVAL_SECONDS=${INTERVAL_SECONDS:-3600}   # 1 hour between attempts
MAX_ATTEMPTS=${MAX_ATTEMPTS:-18}             # ~18h of cover

coverage() {
  "$PY" - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/error_matrix.csv")))
print(sum(1 for r in rows if r["correct"] in ("True", "False")))
EOF
}

echo "=== retry loop started $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
start_cov=$(coverage)
echo "starting coverage: ${start_cov}/360" >> "$LOG"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before=$(coverage)

  if [ "$before" -ge 360 ]; then
    echo "$(date '+%H:%M:%S') COMPLETE at ${before}/360, stopping." >> "$LOG"
    break
  fi

  "$PY" scripts/run_full_generation.py > /tmp/retry_gen_out.txt 2>&1
  after=$(coverage)
  gained=$((after - before))

  echo "$(date '+%H:%M:%S') attempt ${attempt}: ${before} -> ${after} (+${gained})" >> "$LOG"

  if [ "$after" -ge 360 ]; then
    echo "$(date '+%H:%M:%S') COMPLETE at 360/360." >> "$LOG"
    break
  fi

  # Quota has clearly reset if a run produces a meaningful batch; keep going
  # immediately instead of idling through the rest of the allowance.
  if [ "$gained" -ge 5 ]; then
    echo "$(date '+%H:%M:%S') quota appears reset (+${gained}); retrying right away" >> "$LOG"
    sleep 60
    continue
  fi

  sleep "$INTERVAL_SECONDS"
done

final=$(coverage)
echo "=== retry loop finished $(date '+%Y-%m-%d %H:%M:%S') at ${final}/360 ===" >> "$LOG"
