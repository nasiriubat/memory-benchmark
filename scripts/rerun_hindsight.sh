#!/usr/bin/env bash
# Clear bad hindsight results and re-run (no `make` required).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f results/.current_run_id ]]; then
  RUN_ID="${RUN_ID:-$(tr -d '[:space:]' < results/.current_run_id)}"
fi
if [[ -z "${RUN_ID:-}" ]]; then
  echo "Set RUN_ID first, e.g. export RUN_ID=benchmark_20260604T154903Z" >&2
  exit 1
fi
export RUN_ID
echo "RUN_ID=${RUN_ID}"

bash scripts/clear_system.sh hindsight
bash scripts/run_one.sh hindsight
