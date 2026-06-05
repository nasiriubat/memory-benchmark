#!/usr/bin/env bash
# Start a new benchmark campaign: set RUN_ID and reset progress tracking.
#
# Usage:
#   source scripts/init_run.sh          # sets RUN_ID in your shell
#   scripts/init_run.sh               # prints export line to copy
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-benchmark_$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p results
echo "$RUN_ID" > results/.current_run_id

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -c "from benchmark.progress import init_run; init_run('${RUN_ID}')"
fi

export RUN_ID
echo "RUN_ID=${RUN_ID}"
echo "Progress file: results/run_status.json"
