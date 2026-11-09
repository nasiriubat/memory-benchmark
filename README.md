# memory-agent-benchmark

A reproducible, Docker-managed harness that compares **LLM-agent memory systems** on
**MemoryArena**, isolating the memory architecture so results are attributable to the memory
mechanism — not to the model, the tasks, or an internal helper LLM.

- **Memory systems (all real, self-hosted):** `stateless` (baseline), `hindsight`, `mem0`, `zep`,
  `acontext`. (Supermemory from the paper's Table I is **not evaluated** — its backend is
  closed-source and self-hosting is enterprise-only, so it is omitted rather than stubbed.)
- **LLM provider (this branch):** **direct OpenAI**. Agent = `gpt-5.5` at `reasoning.effort: medium`;
  LLM judge = cheap `gpt-4o-mini`; memory-system internal extraction = `gpt-4o-mini` (consistent &
  cheap). An OpenRouter multi-provider path also exists — see [Multi-provider](#multi-provider-openrouter).
- **Strict isolation:** every memory read/write is namespaced by
  `run_id / benchmark / domain / memory_system / model / task_id / seed`. Repeated runs use distinct
  seeds, so iterations never contaminate each other.

## Prerequisites

- **Docker** + **Docker Compose v2**. That's it — everything runs in containers.
- An **OpenAI API key** in `.env` (`OPENAI_API_KEY`). Without it, LLM calls fall back to a
  deterministic mock and results are invalid.

## Installation

```bash
git clone <this-repo> memory-agent-benchmark && cd memory-agent-benchmark
cp .env.example .env          # then set OPENAI_API_KEY (and keep the scope block below)
make build                    # builds the shared image used by every service
```

The `.env` already ships with a cost-efficient, correct default config (see [Configuration](#configuration)).

**AContext only:** it runs as its own 6-service stack and needs a secrets file
`acontext_server/.env` (gitignored) with `OPENAI_API_KEY`, `ACONTEXT_ROOT_TOKEN`,
`ROOT_SECRET_PEPPER`. `make acontext` starts the stack automatically; if that file is missing it
errors with instructions. The default-project SDK
key is `sk-ac-<ACONTEXT_ROOT_TOKEN>`, already wired as `ACONTEXT_API_KEY` in this project's `.env`.

---

## Run a campaign

Run **`stateless` first** (the memory-utility gain of every other system is computed relative to it),
then the memory systems one at a time. `make` stops the previous system before starting the next, so
two backends never run concurrently.

```bash
make clear                 # wipe old results (only when starting a brand-new campaign)
make init-run              # pin a fresh RUN_ID -> results/.current_run_id
make build                 # (re)build the image after any code change

make stateless             # baseline FIRST
make hindsight             # Hindsight server auto-starts (compose, healthcheck)
make mem0                  # Qdrant auto-starts
make zep                   # Zep CE auto-starts
make acontext              # AContext 6-service stack AUTO-STARTS (no manual step)

make aggregate             # combine into results/aggregated/
make ui                    # dashboard at http://localhost:8501
```

All systems share one `RUN_ID`, so gain-vs-stateless and the cross-system leaderboard line up.

**Verification gate (run after each system):** every row in
`results/by_memory_system/<system>/<RUN_ID>/results.jsonl` must have
`"memory_uses_local_fallback": false` — that confirms the row measured the *real* backend and not the
in-process fallback.

### Backends (what each `make` needs)

| System | Backend | Startup |
|---|---|---|
| `stateless` | none | — |
| `hindsight` | `ghcr.io/vectorize-io/hindsight` | auto (compose profile + healthcheck) |
| `mem0` | Qdrant (`qdrant/qdrant`) | auto (compose profile + healthcheck) |
| `zep` | Zep CE (`ghcr.io/getzep/zep` + Postgres) | auto (compose profile) |
| `acontext` | AContext 6-service stack (`acontext_server/`) | auto — `make acontext` runs `scripts/setup_acontext_server.sh` |

---

## Configuration

The active scope lives in `.env`; presets live in `configs/benchmark.yaml` (`BENCHMARK_PROFILE`
selects one; individual `.env` vars override it).

**Default (cost-efficient):** profile `memoryarena_shopping` — MemoryArena `bundled_shopping` only,
**40 of 150 tasks** (random, `seed=42`), **3 runs**. Shopping is a multi-turn product/ASIN recall
task where memory genuinely matters and token-F1 scoring is meaningful. Estimated ~$90/system,
~$360 for the four memory systems.

Knobs in `.env`:

```ini
BENCHMARK_PROFILE=memoryarena_shopping
MEMORYARENA_CONFIGS=bundled_shopping      # or: group_travel_planner, progressive_search, ...
MAX_TASKS_TOTAL=40                        # sample size (0 = all)
TASK_SAMPLE_STRATEGY=random               # deterministic with the seed below
BENCHMARK_TASK_SEED=42
DEFAULT_NUM_RUNS=3                         # repetitions (consistency / pass@k)
TASK_PARALLELISM=8                        # concurrent (task,iteration) jobs — main speed lever
```

Agent reasoning effort is in `configs/models.yaml` (`openai_frontier.reasoning.effort`, default
`medium`) and is **actually applied** to the OpenAI request — it drives the (expensive) reasoning
tokens, so it is the main quality/cost dial. The LLM judge is a separate cheap slot
(`configs/benchmark.yaml` → `evaluators.llm_judge_model: judge` → `gpt-4o-mini`) and is excluded
from the system-under-test set.

Cost dry-run (no LLM calls), using real pricing from `configs/pricing.yaml`:

```bash
docker compose -p memory-agent-benchmark run --rm \
  --entrypoint python stateless -m benchmark.cli estimate --memory-system mem0
```

---

## Metrics

Per result row (`results/by_memory_system/<system>/<run_id>/results.jsonl`):

- **Agent usage/cost** — `input_tokens`, `output_tokens`, `cached_input_tokens`, `reasoning_tokens`,
  `estimated_cost_usd`. Pricing is real (`configs/pricing.yaml`: gpt-5.5 $5 / $0.50 cached / $30 per
  1M); reasoning is billed inside completion tokens (no double-count).
- **Judge usage/cost** — tracked separately in `judge_input_tokens`, `judge_output_tokens`,
  `judge_reasoning_tokens`, `judge_cost_usd`. **`total_cost_usd = agent + judge`** is the true spend.
- **Quality** — `success` (all subtasks correct — strict), `progress_score` (fraction correct),
  `score` (mean token-F1). For graded signal read `progress_score` / `score`, not just binary
  `success`.
- **Memory utility** — `memory_relevance_score`, `memory_correctness_score`, `memory_usage_score`,
  `consistency_score`, and `memory_utility_gain_vs_stateless` (this row's `score` minus the matched
  stateless row's, joined on `model/benchmark/domain/task_id/seed`). **The gain is `null` in the
  per-system `results.jsonl` by design** — it's derived only at `make aggregate` (which needs the
  stateless baseline on the same config/tasks) and appears in `results/aggregated/` + the dashboard,
  not the per-system file.
- **Behavior** — `retrieved_memory_count`, `written_memory_count`, latencies,
  `memory_backend_available`, `memory_uses_local_fallback`.

Aggregates (leaderboard, ANOVA, pairwise-vs-stateless, failure categories) land in
`results/aggregated/` and the Streamlit dashboard.

## Results layout

```
results/
  .current_run_id
  raw/{run_id}/{memory_system}/{model}/{benchmark}/{domain}/events.jsonl
  by_memory_system/{memory_system}/{run_id}/results.jsonl  results.csv  results.parquet
  model_snapshots/{run_id}/resolved_models.yaml
  aggregated/all_results.parquet  leaderboard.csv  pairwise_vs_stateless.csv  ...
```

---

## Multi-provider (OpenRouter)

The harness is OpenRouter-capable (multi-model: Claude, Gemini, Grok, Llama, DeepSeek). To switch:
set `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`, set `MODELS=openai_frontier,anthropic_frontier,
google_frontier,...` (slots in `configs/models.yaml`), and add an `xai_frontier` (Grok) slot if
wanted. Pricing then comes automatically from the live OpenRouter catalog (the manual
`pricing.yaml` is only the OpenAI-direct fallback). `~latest` aliases resolve **once per `RUN_ID`**
and are snapshotted for reproducibility.

## Extending

**Add a memory system:** create `src/benchmark/memory/<system>.py` subclassing `MemoryAdapter`
(mirror `hindsight.py` / `mem0.py` — thread-local client, local-fallback only on real failure,
namespace isolation); register it in `configs/memory_systems.yaml`; add its backend to
`docker-compose.yml` (or its own stack like `acontext_server/`).

**Add a model:** add a slot under `models:` in `configs/models.yaml` (`provider_family`,
`openai_model` and/or `requested_model`, `benchmark_tier`, `reasoning`). Mark a judge-only slot with
`benchmark_tier: judge` so it is used for scoring but never benchmarked.

## Testing

```bash
make test        # pytest -q  (local Python env)
```
