# Installation & running (branch: `openai-memoryareena-nasir`)

This branch is a **narrow Docker campaign** — not the full multi-model OpenRouter sweep in `README.md`.

| What runs | What is disabled |
|-----------|------------------|
| **Docker only** (no local Python runner needed) | OpenRouter, multi-provider models |
| **MemoryArena only** | STATE-Bench, DFx |
| **100 random tasks** (same 100 for every memory system) | Full 701-task pool |
| **1 model:** `openai_frontier` → OpenAI `gpt-5.5` | Claude, Gemini, Meta, etc. |
| **5 runs per task** (pass@5 / reliability) | Paper-scale 50/domain preset |
| **One memory system at a time** | Parallel backends |

Each memory system produces **500 result rows** (100 tasks × 5 runs). The UI “tasks completed” column counts rows, not unique tasks — **500 when done is correct**.

---

## Prerequisites

- **Docker Desktop** (or Docker Engine) + **Docker Compose v2**
- **OpenAI API key** with access to `gpt-5.5`
- Network for first-run Hugging Face dataset download (~12 MB, then cached)

Check Docker works:

```bash
docker --version
docker compose version
```

---

## Step-by-step (copy in order)

All commands run from the **repo root** (the folder that contains `Makefile` and `docker-compose.yml`).

### Step 1 — Get this branch

```bash
git clone <repo-url> memory-agent-benchmark
cd memory-agent-benchmark
git checkout openai-memoryareena-nasir
```

### Step 2 — Create `.env`

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
# LLM — direct OpenAI (not OpenRouter)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...          # paste your key — never commit .env
OPENAI_MODEL=gpt-5.5

# Benchmark subset — MemoryArena, 100 random tasks
BENCHMARK_PROFILE=memoryarena_100
BENCHMARKS=memoryarena
MAX_TASKS_TOTAL=100
TASK_SAMPLE_STRATEGY=random
BENCHMARK_TASK_SEED=42
DEFAULT_NUM_RUNS=5

# Single model slot
MODELS=openai_frontier
```

For **Hindsight** (memory extraction inside its container), also set:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_API_KEY=sk-...   # same OpenAI key is fine
HINDSIGHT_API_LLM_MODEL=gpt-5.5
```

> `.env.example` on this branch already has these defaults. You mainly need to paste `OPENAI_API_KEY` (and the Hindsight copy if you run `make hindsight`).

### Step 3 — Verify Compose file

```bash
docker compose config -q && echo "compose OK"
```

### Step 4 — Build the Docker image

Required on first run and after any code change:

```bash
make build
```

Builds `memory-agent-benchmark:latest` (runners, aggregator, UI).

### Step 5 — Start a fresh campaign

Wipes old results and creates a shared `RUN_ID` for all systems:

```bash
make clear
source scripts/init_run.sh
```

You should see something like:

```
RUN_ID=benchmark_20260605T120220Z
Progress file: results/run_status.json
```

`RUN_ID` is saved in `results/.current_run_id`. **Every memory system in one campaign must use the same `RUN_ID`.**

### Step 6 — (Optional) Start the dashboard

In a **second terminal**, same repo folder:

```bash
make ui
```

Open **http://localhost:8501** to watch progress.

### Step 7 — Sanity-check config inside Docker

Run once before the long `make stateless` job (~thousands of real API calls):

```bash
make build   # skip if you just built in Step 4

COMPOSE_PROJECT_NAME=memory-agent-benchmark docker compose -p memory-agent-benchmark \
  --profile stateless run --rm stateless python -c "
from benchmark.config import load_config
from benchmark.llms.factory import llm_provider
import os
cfg = load_config()
print('LLM_PROVIDER:', llm_provider())
print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))
print('profile:', cfg.run.profile)
print('max_tasks_total:', cfg.run.max_tasks_total)
print('task_sample_strategy:', cfg.run.task_sample_strategy)
print('benchmark_task_seed:', cfg.run.benchmark_task_seed)
print('num_runs:', cfg.run.num_runs)
print('benchmarks:', [b for b, v in cfg.benchmarks.items() if getattr(v, 'enabled', False)])
"
```

**Expected output:**

```
LLM_PROVIDER: openai
OPENAI_API_KEY set: True
profile: memoryarena_100
max_tasks_total: 100
task_sample_strategy: random
benchmark_task_seed: 42
num_runs: 5
benchmarks: ['memoryarena']
```

If you see `profile: paper`, `max_tasks_total: 0`, or `LLM_PROVIDER: openrouter`, fix `.env` and run `make build` again.

### Step 8 — Run memory systems (one at a time)

Wait for each command to finish before starting the next. Each `make <system>` stops other memory containers, runs only that profile, and writes to `results/by_memory_system/<system>/<RUN_ID>/`.

```bash
# 1 — baseline (required first; needed for gain-vs-stateless)
make stateless

# 2–6 — memory systems (run any subset; order below is recommended)
make hindsight
make acontext
make mem0
make zep
make supermemory
```

You do **not** have to run all six. Run at least **stateless + one other** if you want comparisons.

### Step 9 — Aggregate results

After at least `stateless` and one memory system:

```bash
make aggregate
```

Outputs land in `results/aggregated/`.

---

## What to expect per system

| Metric | Value |
|--------|------:|
| Unique MemoryArena tasks | **100** (same IDs every system, seed 42) |
| Runs per task | **5** |
| Result rows per system | **500** |
| Models | **1** (`gpt-5.5`) |
| UI “tasks completed” when done | **500** |

### Good log lines

```
LLM provider: openai
Benchmark 'memoryarena': max_tasks_total=100 strategy=random seed=42 -> {...}
Summary [stateless]: tasks=500 ...
```

### Verify row vs unique task counts

```bash
python3 -c "
import json, pathlib
run_id = pathlib.Path('results/.current_run_id').read_text().strip()
p = pathlib.Path(f'results/by_memory_system/stateless/{run_id}/results.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print('result rows:', len(rows))
print('unique task_ids:', len({r['task_id'] for r in rows}))
"
```

Expect: `result rows: 500`, `unique task_ids: 100`.

---

## Config reference

| Variable | Value | Effect |
|----------|-------|--------|
| `LLM_PROVIDER` | `openai` | Direct OpenAI API (not OpenRouter) |
| `OPENAI_MODEL` | `gpt-5.5` | Model for `openai_frontier` slot |
| `BENCHMARK_PROFILE` | `memoryarena_100` | STATE-Bench off; MemoryArena on |
| `BENCHMARKS` | `memoryarena` | Extra filter |
| `MAX_TASKS_TOTAL` | `100` | 100 tasks sampled from 701-task pool |
| `TASK_SAMPLE_STRATEGY` | `random` | Random sample (not sequential) |
| `BENCHMARK_TASK_SEED` | `42` | Same 100 tasks for every memory system |
| `DEFAULT_NUM_RUNS` | `5` | 5 attempts per task |
| `MODELS` | `openai_frontier` | Single model only |

---

## Results layout

```
results/
  .current_run_id
  run_status.json
  by_memory_system/
    stateless/<RUN_ID>/results.jsonl
    hindsight/<RUN_ID>/results.jsonl
    ...
  model_snapshots/<RUN_ID>/resolved_models.yaml
  aggregated/                    # after make aggregate
```

---

## Useful commands

| Command | Purpose |
|---------|---------|
| `make build` | Rebuild image after code changes |
| `make clear` | Wipe all results, stop containers |
| `make clear-stateless` | Clear one system only (e.g. rerun after a fix) |
| `source scripts/init_run.sh` | New `RUN_ID` for a fresh campaign |
| `make ui` | Dashboard at http://localhost:8501 |
| `make down` | Stop all compose services |

### Cheaper / faster variants

```bash
# 1 run per task → 100 rows per system
DEFAULT_NUM_RUNS=1 make stateless

# Different random 100-task sample
BENCHMARK_TASK_SEED=99 make stateless

# Run only two systems
make stateless && make hindsight && make aggregate
```

---

## Troubleshooting

### Wrong config (5325+ rows, fast run, OpenRouter mock)

**Symptoms:** run finishes quickly, logs mention `OPENROUTER_API_KEY missing`, `profile: paper`, multiple models.

**Fix:**

```bash
make build
make clear
source scripts/init_run.sh
make stateless
```

### OpenAI 400: `max_tokens` not supported

`gpt-5.5` needs `max_completion_tokens`. Fixed in this branch — rebuild and rerun:

```bash
make build
make clear-stateless
make stateless
```

### OpenAI 400: `temperature` does not support 0

`gpt-5.5` only accepts the API default temperature. Rebuild as above.

### UI shows a number between 1 and 499

Run still in progress, or some tasks failed. **500** only when the `make` command exits successfully.

### Hindsight slow or failing

Hindsight starts its own container and needs `HINDSIGHT_API_LLM_*` pointing at OpenAI in `.env`. First start can take several minutes (healthcheck).

---

## Memory systems & OpenAI usage

| System | Answers benchmark tasks | Memory backend |
|--------|-------------------------|----------------|
| stateless | OpenAI `gpt-5.5` | None |
| hindsight | OpenAI `gpt-5.5` | Hindsight server (+ OpenAI for extraction) |
| acontext | OpenAI `gpt-5.5` | Optional backend; may fall back locally |
| mem0 | OpenAI `gpt-5.5` | Qdrant vector store |
| zep | OpenAI `gpt-5.5` | Zep + Postgres |
| supermemory | OpenAI `gpt-5.5` | Optional backend; may fall back locally |

The **benchmark agent** always uses `LLM_PROVIDER=openai` and `gpt-5.5`. Only Hindsight also calls OpenAI inside its container for memory extraction.

---

## Quick copy-paste — full workflow

```bash
git checkout openai-memoryareena-nasir
cp .env.example .env                    # edit OPENAI_API_KEY
docker compose config -q && echo "compose OK"
make build
make clear
source scripts/init_run.sh
make ui                                 # optional, http://localhost:8501

make stateless
make hindsight
make acontext
make mem0
make zep
make supermemory
make aggregate
```

**Cost warning:** each system is ~500 task iterations with many subtasks per task (real OpenAI calls). A full 6-system sweep is expensive — monitor logs and your OpenAI usage dashboard. Start with `make stateless` alone to confirm config before running the rest.

---

## Optional — local Python (contributors only)

Not needed to run benchmarks on this branch. Only for editing code or running tests:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[ui,datasets,dev]"
pytest -q
```
