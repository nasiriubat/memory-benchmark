.PHONY: build stateless hindsight acontext mem0 zep supermemory aggregate ui down test snapshot run-all run-all-docker smoke init-run paper-init clear

# Clear all results and stop containers.
clear:
	bash scripts/clear_results.sh

# Clear one memory system only (keeps stateless / other systems). Usage: make clear-hindsight
clear-%:
	bash scripts/clear_system.sh $*

# Start a new RUN_ID campaign (prints export RUN_ID=...).
init-run:
	bash scripts/init_run.sh

# Clear results, set BENCHMARK_PROFILE=paper in .env, start new RUN_ID.
paper-init:
	bash scripts/init_paper_run.sh

# Build the shared benchmark image.
build:
	COMPOSE_PROJECT_NAME=memory-agent-benchmark docker compose -p memory-agent-benchmark build stateless ui aggregator

# Run EVERY memory system one after another, then aggregate.
# Default models: top OpenAI, Meta, Claude, Gemini. Override with MODELS=...
run-all:
	bash scripts/run_all.sh

# Same, but via Docker (self-hosted backends, stop-between guarantee).
run-all-docker:
	MODE=docker bash scripts/run_all.sh

# Quick, cheap smoke across all systems with strong cost guardrails:
# 1 model, 1 run, STATE-Bench only, 1 task per domain (a few dollars at most).
smoke:
	BENCHMARK_PROFILE=smoke MODELS=openai_frontier bash scripts/run_all.sh

# Run a single memory system (stops any other running system first).
stateless:
	bash scripts/run_one.sh stateless

hindsight:
	bash scripts/run_one.sh hindsight

acontext:
	bash scripts/run_one.sh acontext

mem0:
	bash scripts/run_one.sh mem0

zep:
	bash scripts/run_one.sh zep

supermemory:
	bash scripts/run_one.sh supermemory

# Aggregate all results into results/aggregated/.
aggregate:
	COMPOSE_PROJECT_NAME=memory-agent-benchmark docker compose -p memory-agent-benchmark run --rm aggregator

# Launch the dashboard on http://localhost:8501 (detached, isolated project).
ui:
	COMPOSE_PROJECT_NAME=memory-agent-benchmark docker compose -p memory-agent-benchmark up --build -d ui

# Full Docker benchmark (one system at a time) + dashboard.
bench: ui
	MODE=docker bash scripts/run_all.sh

# Snapshot the OpenRouter model catalog and resolved models.
snapshot:
	python scripts/snapshot_openrouter_models.py

# Stop everything.
down:
	docker compose down --remove-orphans

# Run the test suite locally.
test:
	pytest -q
