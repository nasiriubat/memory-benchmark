#!/usr/bin/env bash
# Run exactly ONE memory system at a time.
#
# This stops any currently running memory system (and its self-hosted
# backends) BEFORE starting the requested one, so two memory systems never run
# concurrently and the machine is not overloaded.
#
# Usage:
#   scripts/run_one.sh <memory_system> [extra args passed to the runner]
#
# Example:
#   scripts/run_one.sh stateless
#   scripts/run_one.sh hindsight --models openai_frontier,google_frontier
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/_docker.sh
source "$(dirname "$0")/_docker.sh"

SYSTEM="${1:-}"
if [[ -z "${SYSTEM}" ]]; then
  echo "Usage: scripts/run_one.sh <memory_system> [extra runner args]" >&2
  echo "  memory systems: stateless hindsight acontext mem0 zep supermemory" >&2
  exit 1
fi
shift || true

# Shared RUN_ID across all systems in one benchmark campaign.
if [[ -z "${RUN_ID:-}" && -f results/.current_run_id ]]; then
  RUN_ID="$(tr -d '[:space:]' < results/.current_run_id)"
  export RUN_ID
fi
RUN_ID="${RUN_ID:-benchmark_$(date +%Y%m%dT%H%M%SZ)}"
export RUN_ID

_progress_py() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -c "$1" 2>/dev/null || true
  fi
}

echo ">> RUN_ID=${RUN_ID}"
echo ">> Stopping memory-agent-benchmark services only (project: ${COMPOSE_PROJECT_NAME})..."
dc down --remove-orphans || true

echo ">> Starting memory system: ${SYSTEM}"
runner_args=(--memory-system "${SYSTEM}" --run-id "${RUN_ID}")
if [[ "$#" -gt 0 ]]; then
  runner_args+=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  runner_args+=(--models "${MODELS}")
  echo ">> MODELS=${MODELS}"
fi
set +e
dc --profile "${SYSTEM}" run --rm "${SYSTEM}" \
  python scripts/run_memory_system.py "${runner_args[@]}"
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  _progress_py "from benchmark.progress import set_done; set_done('${RUN_ID}', '${SYSTEM}')"
  echo ">> Finished ${SYSTEM} (RUN_ID=${RUN_ID})"
else
  _progress_py "from benchmark.progress import set_failed; set_failed('${RUN_ID}', '${SYSTEM}', 'docker exit ${rc}')"
  echo ">> FAILED ${SYSTEM} (exit ${rc})" >&2
  exit "$rc"
fi
