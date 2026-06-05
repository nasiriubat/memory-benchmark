# Research Questions to Metrics Mapping

This harness operationalizes the research questions from *Memory Matters:
Evaluating Learning Behavior in LLM-Based Agents* (ICSE TR). Each RQ maps to
concrete metrics, aggregation views, and statistical analyses.

## RQ1 - How do memory mechanisms influence agent performance and learning behavior?

| Signal | Where |
| --- | --- |
| Task success rate, mean score | `metrics.task_success_rate`, `TaskResult.success/score` |
| Progress score | `metrics.progress_score` |
| Memory-system comparison | views `memory_system_x_model`, `memory_system_x_benchmark`, `memory_system_x_domain` |
| Main effect of memory system | `analysis.two_way_anova` (factor `memory_system`) |
| Post-hoc differences | `analysis.tukey_hsd` over `memory_system` |

## RQ2 - Do memory-augmented agents improve over repeated interactions and transfer knowledge vs stateless?

| Signal | Where |
| --- | --- |
| Improvement over time (OLS slope + relative) | `metrics.improvement_over_time` |
| Iteration trend | view `iteration_trend` |
| Gain vs stateless | `metrics.memory_utility_gain`, `analysis.compute_memory_utility_gain` |
| Pairwise vs stateless (mean diff, t-test, Cohen's d) | `analysis.pairwise_vs_stateless` |
| Cross-task transfer | `run.cross_task_memory = true` (shared memory namespace across task IDs) |

## RQ3 - How does the LLM backbone affect memory-based learning?

| Signal | Where |
| --- | --- |
| Per-model breakdown | views `memory_system_x_model`, `model_x_benchmark` |
| Interaction effect (memory_system x model) | `analysis.two_way_anova` interaction term |
| Provenance | `TaskResult.requested_model / resolved_model / provider_family / benchmark_tier` |

## RQ4 - How effectively do agents retrieve and use stored memory?

| Signal | Where |
| --- | --- |
| Retrieval relevance | `evaluators.memory_utility.retrieval_relevance_score` |
| Retrieval correctness | `evaluators.memory_utility.retrieval_correctness_score` |
| Memory usage | `evaluators.memory_utility.memory_usage_score` (prefers `used_memory_ids`) |
| Memory utility gain | `memory_utility_gain_vs_stateless` |
| Failure categories | `runner._classify_failure`, view/CSV `failure_categories` |

## Consistency & efficiency (cross-cutting)

- Consistency: `evaluators.consistency` with a per-task fact ledger
  (constraints, accepted decisions, user feedback, prior recommendations,
  corrected mistakes, expected facts).
- Efficiency: latency, tokens (input/output/cached/reasoning), cost, turn count,
  tool-call count, retrieved-memory count, memory store size.

## Official vs custom metrics

Official benchmark metrics are kept clearly namespaced (`official_STATE_Bench_*`,
`official_memoryarena_*`) and never silently mixed with the custom metrics above.

## Coverage notes (read before interpreting results)

| Area | Status |
| --- | --- |
| RQ1–RQ4 core metrics | Implemented (success, progress, consistency, memory utility, ANOVA/Tukey/Cohen's d) |
| Concrete model ids | `resolved_model` + `model_display` in results/UI (aliases resolved once per run) |
| STATE-Bench official runner | Used only if `state_bench` / `statebench` package is installed; otherwise offline samples |
| MemoryArena official runner | Used only if installed; otherwise Hugging Face / bundled fallback |
| RQ2 cross-task transfer | Requires `run.cross_task_memory: true` in `configs/benchmark.yaml` (default: false) |
| DFx expert scores | Only when `benchmarks.dfx.enabled` and expert scoring files exist |
| pass@5 / reliability | Computed at aggregation time across seeds per task (`pass_at_5` in summaries) |
