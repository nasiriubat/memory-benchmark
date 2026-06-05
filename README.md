# memory-agent-benchmark

A reproducible, Docker-managed benchmark harness that compares **LLM agent
memory systems** across **multiple models** on **Microsoft STATE-Bench (Agent
Learning Track)**, **MemoryArena**, and an optional local **DFx** benchmark.

It operationalizes the research questions and metrics from *Memory Matters:
Evaluating Learning Behavior in LLM-Based Agents* (ICSE TR). See
[docs/research_questions.md](docs/research_questions.md) for the RQ-to-metric map.

- Memory systems: `stateless`, `hindsight`, `acontext`, `mem0`, `zep`,
  `supermemory` (+ optional `automemory`).
- LLM provider: **OpenRouter only** (one reproducible provider surface).
- Strict isolation: every memory read/write is namespaced by
  `run_id / benchmark / domain / memory_system / model / task_id / seed`.

## Why run one memory system at a time

Each memory system is **self-hosted** (its own Docker service[s] on an internal
network). Running them one-at-a-time keeps the machine from being overloaded by
multiple backends (vector stores, graph DBs) at once, and keeps each run clean.
The `run_one.sh` wrapper stops any running memory system before starting the
next, so two never run concurrently.

## Why memory namespace isolation matters

If one model could read memory written by another model (or another
task/seed/benchmark/memory system), the benchmark would be **invalid** — you
could not attribute a result to the memory mechanism. Every adapter operation is
hard-scoped to a `MemoryNamespace`, and repeated runs use distinct seeds so
iterations never contaminate each other.

## Prerequisites

- **Docker** + **Docker Compose v2** (recommended path; nothing else needed).
- For the local (non-Docker) path: **Python 3.12+**. If your system Python is
  older, install an isolated 3.12 with [`uv`](https://docs.astral.sh/uv/):

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv
  uv python install 3.12
  uv venv --python 3.12 .venv
  source .venv/bin/activate
  ```
- An **OpenRouter API key** for real model calls (optional — the system runs
  offline with a deterministic mock when the key is absent).

## Installation

### Option A - Docker (recommended)

```bash
git clone <this-repo> memory-agent-benchmark && cd memory-agent-benchmark
cp .env.example .env        # then edit .env and add OPENROUTER_API_KEY
docker compose build        # builds the shared image used by all services
```

Verify the Compose file is valid:

```bash
docker compose config -q && echo "compose OK"
```

### Option B - Local Python (no Docker)

```bash
cp .env.example .env        # then edit .env and add OPENROUTER_API_KEY
# with the 3.12 venv from Prerequisites active:
uv pip install -e ".[ui,datasets,dev]"   # or: pip install -e ".[ui,datasets,dev]"
pytest -q                                 # optional: 31 tests should pass
```

The system is runnable **offline** (no API keys): the `stateless` system and
sample data work, and LLM calls fall back to a deterministic mock that is
clearly flagged. To force fully offline benchmark data (no Hugging Face network
calls), export `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

## Run everything with one command

Run **every** memory system one after another (stateless first), each across the
chosen models, saving per-system data, then aggregate — all in a single command:

```bash
# Local (no Docker). Default models: top OpenAI, Meta, Claude, Gemini.
make run-all
# or directly:
bash scripts/run_all.sh
```

All systems share one `RUN_ID`, so gain-vs-stateless and the cross-system
leaderboard line up. Customize without editing code:

```bash
# pick models (slots from configs/models.yaml)
MODELS=openai_frontier,meta_frontier,anthropic_frontier,google_frontier bash scripts/run_all.sh

# cheaper/faster repetitions (default num_runs is 5 from benchmark.yaml)
DEFAULT_NUM_RUNS=2 bash scripts/run_all.sh

# only some systems, in this order
bash scripts/run_all.sh stateless hindsight mem0

# via Docker (self-hosted backends; tears down the previous system before the next)
MODE=docker bash scripts/run_all.sh        # or: make run-all-docker

# quick, cheap smoke across all systems (1 model, 1 run each)
make smoke
```

## Paper-scale campaign (recommended before full sweep)

Mid-scale preset for ICSE-style analysis: **5 runs** (pass@5), **50 MemoryArena
tasks per config** (~6k rows per memory system with 5 models vs ~17k uncapped).

```bash
make paper-init          # clears results, sets BENCHMARK_PROFILE=paper in .env, new RUN_ID
make build
make hindsight           # your order: hindsight first
make stateless           # baseline for memory_utility_gain_vs_stateless
make acontext
make mem0
make zep
make supermemory
make aggregate
make ui                  # http://localhost:8501
```

Profiles live in `configs/benchmark.yaml` (`profiles.paper|full|smoke`). Env vars
`BENCHMARK_PROFILE`, `MAX_TASKS_PER_DOMAIN`, and `DEFAULT_NUM_RUNS` override the
YAML. After **stateless** finishes, run `make aggregate` again if you aggregated
earlier — gain-vs-stateless needs both systems under the same `RUN_ID`.

> Cost note: a full run is `models x systems x benchmarks x tasks x num_runs`
> real OpenRouter calls. Start with `make smoke` (or `DEFAULT_NUM_RUNS=1` and one
> model) to estimate cost before the full sweep. Per-call cost is logged and
> summed from live catalog pricing.

When it finishes, open the dashboard: `streamlit run ui/app.py` (local) or
`docker compose up ui` (Docker).

## Run (Docker, one system at a time)

Always run **stateless first** — memory-utility gain is computed relative to it.

```bash
docker compose run --rm stateless
docker compose run --rm hindsight
docker compose run --rm acontext
docker compose run --rm mem0
docker compose run --rm zep
docker compose run --rm supermemory
docker compose run --rm aggregator
docker compose up --build ui            # dashboard at http://localhost:8501
```

The helper enforces the stop-others-then-run-one guarantee:

```bash
bash scripts/run_one.sh stateless
make stateless          # equivalent via Makefile
```

## Run (local, without Docker)

```bash
pip install -e ".[ui,datasets,dev]"
python scripts/snapshot_openrouter_models.py
python scripts/run_memory_system.py --memory-system stateless
python scripts/run_memory_system.py --memory-system hindsight --models openai_frontier,google_frontier
python scripts/aggregate_results.py
streamlit run ui/app.py
```

## Using OpenRouter

1. Put your key in `.env` as `OPENROUTER_API_KEY`. **Do not commit `.env`.**
2. OpenRouter is the **only** LLM API provider used for model calls (via the
   OpenAI SDK pointed at `https://openrouter.ai/api/v1`).
3. `~latest` aliases are **resolved once per run** by `ModelResolver`, which
   saves the catalog snapshot and the resolved mapping under
   `results/model_snapshots/{run_id}/`.
4. **Concrete model ids are used after resolution**, for the whole run.
5. Both the requested alias and the resolved concrete model are logged on every
   `llm_response` event and stored on every result — this keeps runs
   **reproducible**.
6. For smoke tests you may relax routing (allow fallbacks, `require_parameters:
   false`) via `configs/openrouter.yaml`.
7. For **paper results, keep `strict_benchmark_mode: true`** (resolve once, use
   concrete ids, disable fallback routing, require parameters, deny data
   collection).

Snapshot models any time:

```bash
python scripts/snapshot_openrouter_models.py
```

Cost is computed from the **live OpenRouter catalog pricing** captured in
`resolved_models.yaml` (never hardcoded).

## Self-hosting memory systems

Each memory system has a Docker Compose **profile** containing the runner plus
its self-hosted backend(s), exposed on an internal network only (no host ports
except the UI on 8501). Adapters connect via `*_BASE_URL` env vars (e.g.
`MEM0_BASE_URL=http://mem0-qdrant:6333`, `ZEP_BASE_URL=http://zep:8000`). If a
backend is unreachable or its SDK is missing, the adapter **logs that it is
unavailable** and falls back to a local namespace-isolated store so the run
still completes. The `stateless` system needs no backend and always works.

## Results layout

```
results/
  raw/{run_id}/{memory_system}/{model}/{benchmark}/{domain}/events.jsonl
  by_memory_system/{memory_system}/{run_id}/
      results.jsonl  results.csv  results.parquet  run_metadata.json
  model_snapshots/{run_id}/openrouter_models.json  resolved_models.yaml
  aggregated/
      all_results.parquet
      summary_by_memory_model.csv  summary_by_benchmark.csv
      leaderboard.csv  failure_categories.csv
      views/*.csv  stats_anova.csv  stats_posthoc.csv
      effect_sizes.csv  pairwise_vs_stateless.csv
```

Results are also **downloadable from the UI** as JSON and CSV (filtered view and
per-memory-system bundles).

## How metrics are calculated

See [docs/research_questions.md](docs/research_questions.md). Summary:

- **Task success rate** = successful_tasks / total_tasks.
- **Rule compliance** = rules_satisfied / total_rules (official score first,
  then deterministic checks, then LLM judge).
- **Progress score** = correct_subtasks / total_subtasks.
- **Improvement over time** = OLS slope of score vs iteration + relative
  improvement (late_mean - early_mean) / max(|early_mean|, 1e-9).
- **Consistency** = 1 - contradictions against a per-task fact ledger
  (constraints, accepted decisions, feedback, prior recommendations, corrected
  mistakes, expected facts); deterministic first, LLM judge fallback.
- **Efficiency** = latency, tokens (input/output/cached/reasoning), cost, turns,
  tool calls, retrieved-memory count, store size.
- **Memory utility** = retrieval relevance + retrieval correctness + memory
  usage + `memory_utility_gain_vs_stateless` (= memory score - matched stateless
  score).
- **Reliability** = pass@1, pass@5, mean, std, coefficient of variation.
- **Failure categories** = the 11 categories in `FailureCategory`.
- **Statistical analysis** (paper III-J) = descriptive stats, two-way ANOVA
  (memory_system x model, main + interaction), Tukey HSD, effect sizes
  (eta-squared, Cohen's d), pairwise vs stateless.

Official metrics are namespaced `official_STATE_Bench_*` / `official_memoryarena_*`
and never silently mixed with the custom metrics.

## Extending

### Add a new memory system

1. Create `src/benchmark/memory/<system>.py` subclassing `StubMemoryAdapter`
   (or `MemoryAdapter` directly) and set `name`, `sdk_module`, `base_url_env`.
2. Register it in `configs/memory_systems.yaml` (`adapter: module:Class`).
3. Add a Compose profile + backend service in `docker-compose.yml` if needed.

### Add a new model

Add a slot under `models:` in `configs/models.yaml` with `provider_family`,
`requested_model` (optionally a `~latest` alias), `concrete_fallback_model`,
`benchmark_tier`, `temperature`, `max_output_tokens`, and `reasoning`. Nothing in
the runner hardcodes model names.

### Add DFx scenarios

Drop scenario JSON files into `data/dfx/*.json` (see
`data/dfx/sample_scenario.json`). Earlier subtasks introduce goals, decisions,
feedback, and constraints; later subtasks must reuse them. Set
`benchmarks.dfx.enabled: true` in `configs/benchmark.yaml`. Never include
confidential data.

## Testing

```bash
pytest -q
```
