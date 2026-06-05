#!/usr/bin/env bash
# Remove results for ONE memory system (keeps other systems and RUN_ID campaign).
#
# Usage:
#   scripts/clear_system.sh hindsight
#   RUN_ID=benchmark_20260604T154903Z scripts/clear_system.sh hindsight
set -euo pipefail

cd "$(dirname "$0")/.."

SYSTEM="${1:-}"
if [[ -z "${SYSTEM}" ]]; then
  echo "Usage: scripts/clear_system.sh <memory_system>" >&2
  echo "  e.g. hindsight  (keeps stateless and other systems)" >&2
  exit 1
fi

if [[ -f results/.current_run_id ]]; then
  RUN_ID="${RUN_ID:-$(tr -d '[:space:]' < results/.current_run_id)}"
fi
RUN_ID="${RUN_ID:-}"

echo ">> Clearing results for memory_system=${SYSTEM}"
if [[ -n "${RUN_ID}" ]]; then
  echo ">> RUN_ID=${RUN_ID}"
  rm -rf "results/by_memory_system/${SYSTEM}/${RUN_ID}"
  rm -rf "results/raw/${RUN_ID}/${SYSTEM}"
else
  echo ">> No RUN_ID set — removing ALL run folders for ${SYSTEM}"
  rm -rf "results/by_memory_system/${SYSTEM}"
  find results/raw -mindepth 2 -maxdepth 2 -type d -name "${SYSTEM}" -exec rm -rf {} + 2>/dev/null || true
fi

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python - <<PY
from pathlib import Path
from benchmark.progress import load_status, save_status, MEMORY_SYSTEMS_ORDER

root = Path("results")
status = load_status(root)
if "${SYSTEM}" in status.get("systems", {}):
    status["systems"]["${SYSTEM}"] = {
        "status": "pending",
        "tasks_completed": 0,
        "run_folder": None,
        "finished_at": None,
    }
if status.get("current_system") == "${SYSTEM}":
    status["current_system"] = None
save_status(status, root)
print(">> Updated results/run_status.json")
PY
fi

echo ">> Done. Re-run: export RUN_ID=${RUN_ID:-<your-id>}; make ${SYSTEM}"
