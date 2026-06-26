# Benchmark Runbook (Docker) — current campaign

Run the remaining memory systems **one at a time via Docker**, each stopping the
previous, then aggregate against the **already-finished `stateless` baseline**.

## Campaign facts (cost-efficient config)

- **RUN_ID:** taken from `results/.current_run_id` (`.env` leaves `RUN_ID` empty). Start a fresh
  campaign with `make clear && make init-run` (the old formal-reasoning results were cleared
  because the config changed).
- **Config:** profile `memoryarena_shopping` — **MemoryArena `bundled_shopping` only**,
  `task_sample_strategy=random`, `max_tasks_total=40` (of 150), `seed=42`, `num_runs=3`. Shopping is
  a multi-turn product/ASIN recall task where memory matters and token-F1 scoring is meaningful.
- **Model:** only `openai_frontier` → `gpt-5.5`, now at **`reasoning.effort: medium`** (actually sent
  to the API — it was a silent no-op before). The cheap **`gpt-4o-mini` judge** scores memory
  signals and is excluded from the run set.
- Random sampling with `seed=42` is deterministic → **every system (incl. stateless) runs the same
  40 tasks**, so `memory_utility_gain_vs_stateless` (joined on `model/benchmark/domain/task_id/seed`)
  is valid. **Run `stateless` FIRST**, then the memory systems.

> **Metrics:** agent token/cost are in `input_tokens`/`output_tokens`/`estimated_cost_usd`; the
> judge's usage is now tracked separately in `judge_input_tokens`/`judge_output_tokens`/
> `judge_cost_usd`, and **`total_cost_usd = agent + judge`** is the true per-row spend. `success` is
> all-or-nothing across a task's subtasks — read `progress_score`/`score` for graded signal.
> **`memory_utility_gain_vs_stateless` is `null` in the per-system `results.jsonl` by design** — it is
> a derived metric, computed only at `make aggregate` (needs the `stateless` baseline on the same
> RUN_ID/tasks) and written to `results/aggregated/` + the dashboard, never the per-system file. Run
> order doesn't matter; you just need both `stateless` and the memory system present, then aggregate.

## `.env` scope block (already set)

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=...                 # MUST be set — empty = mock LLM = invalid results
OPENAI_MODEL=gpt-5.5
BENCHMARK_PROFILE=memoryarena_shopping
BENCHMARKS=memoryarena
MEMORYARENA_CONFIGS=bundled_shopping
MAX_TASKS_TOTAL=40
MAX_TASKS_PER_DOMAIN=0
TASK_SAMPLE_STRATEGY=random
BENCHMARK_TASK_SEED=42
DEFAULT_NUM_RUNS=3
MODELS=openai_frontier
MODEL_PARALLELISM=1
TASK_PARALLELISM=8                 # main speed lever; lower if you hit rate-limit backoff.
RUN_ID=                            # empty -> uses results/.current_run_id
```

---

## Step 0 — Pre-flight checks

```bash
# API key present (else runs use the mock and are invalid):
grep -qE '^OPENAI_API_KEY=.+' .env && echo "key OK" || echo "ADD OPENAI_API_KEY FIRST"
# Campaign id pinned to the stateless run:
cat results/.current_run_id        # expect benchmark_20260611T170455Z
```

## Step 1 — Rebuild the image (after code changes)

```bash
make build      # fast now: dependency layer is cached, only source is recopied
```

## Step 2 — (optional) Cost/size dry-run, no API calls

```bash
docker compose -p memory-agent-benchmark run --rm \
  --entrypoint python stateless -m benchmark.cli estimate --memory-system hindsight
```

## Step 3 — Run the memory systems one-by-one

`run_one.sh` (via `make`) stops the previous system + backend before starting the next,
and uses the pinned RUN_ID automatically.

```bash
export RUN_ID=$(cat results/.current_run_id)  # belt-and-braces; also auto-picked up

make stateless                                # baseline FIRST (memory_utility_gain needs it)
make hindsight
#  >>> run the VERIFICATION GATE below before continuing <<<
make mem0                                     # Qdrant auto-starts (verified)
make zep                                      # Zep CE auto-starts (ghcr.io/getzep/zep)
make acontext                                 # acontext server now AUTO-STARTS (no manual step)

make aggregate
make ui                                       # http://localhost:8501
```

> **supermemory was dropped** (its backend is closed-source; real self-hosting is
> enterprise-only). It is disabled in `configs/memory_systems.yaml`. To include it,
> point at the Supermemory cloud API (`SUPERMEMORY_API_KEY` + base URL) and re-enable it.

### Memory-system backends (what each `make` needs)

| System | Backend | Setup |
|---|---|---|
| `hindsight` | `ghcr.io/vectorize-io/hindsight` | auto (compose, healthcheck) |
| `mem0` | Qdrant (`qdrant/qdrant`) | auto (compose, healthcheck) — **verified** |
| `zep` | Zep CE (`ghcr.io/getzep/zep` + Postgres) | auto (compose) |
| `acontext` | AContext 6-service stack (`acontext_server/`) | **auto** — `make acontext` starts it |

**AContext is now fully automatic.** `make acontext` runs `scripts/setup_acontext_server.sh` first,
which idempotently brings up the 6-service stack in [`acontext_server/`](acontext_server/) (its own
compose project, `restart: unless-stopped`), creates the S3 bucket if missing, waits for the API to
be healthy, then runs the benchmark. No manual steps.

The only requirement: `acontext_server/.env` must exist (gitignored; holds `OPENAI_API_KEY`,
`ACONTEXT_ROOT_TOKEN`, `ROOT_SECRET_PEPPER`). Its default-project SDK key is
`sk-ac-<ACONTEXT_ROOT_TOKEN>`, already set as `ACONTEXT_API_KEY` in this project's `.env`. If that
file is ever lost, the script errors clearly; re-create it and re-run. **Verified:** the adapter
writes/retrieves real skills with `memory_uses_local_fallback=false`.

### Verification gate — confirm the task set matches (after `make hindsight`)

Sequential sampling is deterministic, so this should be a clean 100% overlap:

```bash
RID=benchmark_20260611T170455Z python3 - <<'PY'
import json, glob, os
RID=os.environ["RID"]
def keys(sys):
    s=set()
    for f in glob.glob(f"results/by_memory_system/{sys}/{RID}/results.jsonl"):
        for l in open(f):
            d=json.loads(l); s.add((d["model"],d["benchmark"],d["domain"],d["task_id"],d["seed"]))
    return s
a=keys("stateless"); b=keys("hindsight")
print("stateless keys:",len(a)," hindsight keys:",len(b)," overlap:",len(a&b))
print("MATCH OK" if b and b==a else "MISMATCH — investigate before continuing")
PY
```

### After `make hindsight` — confirm it used the REAL backend

```bash
RID=benchmark_20260611T170455Z python3 -c "
import json,collections,os
p=f'results/by_memory_system/hindsight/{os.environ[\"RID\"]}/results.jsonl'
print('hindsight uses_local_fallback:', dict(collections.Counter(json.loads(l).get('memory_uses_local_fallback') for l in open(p))))
"
```

Want `{False: N}`. `True` ⇒ the `hindsight-server` backend was unreachable and it measured
the local store. (mem0/zep/acontext/supermemory are local-store **stubs** and will report
`true` by design — only hindsight's flag is meaningful.)

---

## Step 4 — Aggregate + dashboard

```bash
make aggregate          # writes results/aggregated/ incl. pairwise_vs_stateless.csv
make ui                 # http://localhost:8501
```

Each system should have **300 rows** (same as stateless) in
`results/by_memory_system/<system>/$RUN_ID/results.jsonl`, and the leaderboard should show
every system on `gpt-5.5` with a gain-vs-stateless value.

---

## Gotchas

- **`OPENAI_API_KEY` must be set** — empty ⇒ deterministic mock ⇒ invalid results.
- **Don't edit the `.env` benchmark-scope block mid-campaign** — it splits the task set.
- **Rebuild after code edits** (`make build`); `src/` is baked into the image.
- **Rerunning a system** appends to its existing file — `make clear-<system>` first.
- **Stop everything**: `make down`. **Wipe all results**: `make clear` (also clears
  `results/.current_run_id`, so re-pin it afterward).
