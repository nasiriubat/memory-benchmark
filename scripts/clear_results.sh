#!/usr/bin/env bash
# Remove all benchmark outputs (keeps results/.gitkeep).
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> Stopping benchmark containers..."
COMPOSE_PROJECT_NAME=memory-agent-benchmark docker compose -p memory-agent-benchmark down --remove-orphans 2>/dev/null || true
pkill -f "scripts/run_all.sh" 2>/dev/null || true

echo ">> Clearing results/"
find results -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
touch results/.gitkeep
echo ">> Done. Results cleared."
