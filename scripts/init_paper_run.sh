#!/usr/bin/env bash
# Start a paper-scale campaign: clear old results, set profile env, new RUN_ID.
#
# Usage:
#   bash scripts/init_paper_run.sh
#   source scripts/init_paper_run.sh   # also exports RUN_ID in your shell
set -euo pipefail

cd "$(dirname "$0")/.."

_set_env_var() {
  local key="$1"
  local val="$2"
  if [[ ! -f .env ]]; then
    echo ">> No .env found — copy .env.example to .env first." >&2
    exit 1
  fi
  if grep -q "^${key}=" .env; then
    if [[ "$(uname -s)" == Darwin ]]; then
      sed -i '' "s|^${key}=.*|${key}=${val}|" .env
    else
      sed -i "s|^${key}=.*|${key}=${val}|" .env
    fi
  else
    echo "${key}=${val}" >> .env
  fi
}

echo ">> Clearing previous benchmark results..."
bash scripts/clear_results.sh

echo ">> Setting paper profile in .env..."
_set_env_var BENCHMARK_PROFILE paper
_set_env_var DEFAULT_NUM_RUNS 5
_set_env_var MAX_TASKS_PER_DOMAIN 50
_set_env_var RUN_ID ""

export BENCHMARK_PROFILE=paper
export DEFAULT_NUM_RUNS=5
export MAX_TASKS_PER_DOMAIN=50

echo ">> New campaign RUN_ID..."
# shellcheck disable=SC1091
source scripts/init_run.sh

cat <<'EOF'

Paper campaign ready (profile: paper, max_tasks_per_domain=50, num_runs=5).

Suggested run order (your plan: hindsight first, then baseline + others):
  make build
  make stateless
  make hindsight
  make acontext
  make mem0
  make zep
  make aggregate

Memory utility vs stateless fills in after stateless completes; re-run
  make aggregate
after stateless if you aggregate before it finishes.

Monitor: make ui  →  http://localhost:8501

EOF
