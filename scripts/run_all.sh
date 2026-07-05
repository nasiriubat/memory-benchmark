#!/usr/bin/env bash
# Run the FULL benchmark: every memory system, one after another.
#
# For each memory system it:
#   1. starts fresh (in docker mode it tears down any running system first),
#   2. runs that system across the configured MODELS,
#   3. saves results into results/by_memory_system/{system}/{RUN_ID}/,
#   4. stops, then moves on to the next system.
# Finally it aggregates everything into results/aggregated/.
#
# All systems share ONE RUN_ID so gain-vs-stateless and cross-system
# aggregation line up. Stateless is run FIRST because memory-utility gain is
# computed relative to it.
#
# Usage:
#   scripts/run_all.sh                      # all systems, default 4 models, local
#   MODE=docker scripts/run_all.sh          # via Docker (self-hosted, stop-between)
#   scripts/run_all.sh stateless hindsight  # only these systems, in this order
#
# Tunables (env vars):
#   MODE=local|docker     execution mode (default: local)
#   MODELS=...            comma-separated model slots
#                         (default: top OpenAI, Meta, Claude, Gemini)
#   DEFAULT_NUM_RUNS=N    repetitions per task (default: from configs/benchmark.yaml = 5)
#   RUN_ID=...            explicit run id (default: auto-generated, shared by all)
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/_docker.sh
source "$(dirname "$0")/_docker.sh"

MODE="${MODE:-local}"
# All enabled frontier slots; ModelResolver resolves ~latest to concrete OpenRouter ids.
MODELS="${MODELS:-openai_frontier,anthropic_frontier,google_frontier,meta_frontier,deepseek_frontier}"

# Memory systems to run (in order). Stateless first (baseline).
if [[ "$#" -gt 0 ]]; then
  SYSTEMS=("$@")
else
  SYSTEMS=(stateless hindsight acontext mem0 zep)
fi

# One shared run id across every system.
export RUN_ID="${RUN_ID:-benchmark_$(date +%Y%m%dT%H%M%SZ)}"

echo "=================================================================="
echo " Full benchmark run"
echo "   RUN_ID : $RUN_ID"
echo "   MODE   : $MODE"
echo "   MODELS : $MODELS"
echo "   SYSTEMS: ${SYSTEMS[*]}"
echo "=================================================================="

for sys in "${SYSTEMS[@]}"; do
  echo ""
  echo ">>>>>> [$sys] starting ($(date +%H:%M:%S)) <<<<<<"
  if [[ "$MODE" == "docker" ]]; then
    # Stop only this Compose project (never other stacks on the machine).
    dc down --remove-orphans || true
    dc --profile "$sys" run --rm "$sys" \
      python scripts/run_memory_system.py --memory-system "$sys" --models "$MODELS" --no-aggregate
  else
    python scripts/run_memory_system.py --memory-system "$sys" --models "$MODELS" --no-aggregate
  fi
  echo ">>>>>> [$sys] finished ($(date +%H:%M:%S)) <<<<<<"
done

echo ""
echo ">>>>>> aggregating all results <<<<<<"
if [[ "$MODE" == "docker" ]]; then
  dc down --remove-orphans || true
  dc run --rm aggregator
else
  python scripts/aggregate_results.py
fi

echo ""
echo "=================================================================="
echo " Done. RUN_ID=$RUN_ID"
echo " Per-system results : results/by_memory_system/<system>/$RUN_ID/"
echo " Aggregated results : results/aggregated/"
echo " Open the dashboard : (docker) docker compose up ui  |  (local) streamlit run ui/app.py"
echo "=================================================================="
