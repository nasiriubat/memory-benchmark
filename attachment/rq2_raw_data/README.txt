RQ2 raw data package - memory-agent benchmark (MemoryArena bundled_shopping)
Generated 2026-07-26.

CONTENTS
  raw_events/<run_id>/<memory_system>/<model_slot>/.../events.jsonl
      Canonical per-event logs (subtask-level source). 12 cells x 720 subtask
      observations = 8,640 scored subtasks, complete in every cell.
  task_level/<memory_system>/<run_id>/results.jsonl
      Canonical task-level rows: 1,440 = 4 systems x 3 backbones x 40 tasks x 3 seeds.
  benchmark_subtask_trace.xlsx   - subtask-level trace sheets (240 rows per cell)
  RQ2_subtask_analysis.xlsx      - RQ2-ready analysis workbook (+ embedded figures)
  subtask_trend.csv              - learning-curve data (mean score by position)

PROVENANCE NOTES
  * Backbone run ids: benchmark_20260626T142817Z = gpt-5.5 (OpenAI direct);
    benchmark_gemini35flash = Gemini 3.5 Flash (OpenRouter);
    benchmark_claudeopus48 = Claude Opus 4.8 (OpenRouter, Anthropic route).
  * stateless x Claude Opus 4.8 events: the original campaign's event log was truncated
    (15/40 tasks) by a file-locking issue; the cell was fully re-run on 2026-07-23
    (run id benchmark_claudeopus48_stateless_rerun; identical 40 tasks, seeds 1000-1002,
    prompts, judge, scripts). Its complete event log is the canonical one included here.
    Task-level rows remain the original campaign's (validated: mean score 0.312 vs 0.306).
  * mem0 x Claude Opus 4.8 events: 21/720 events lost to the same issue (tasks 61, 75, 130);
    those three tasks were re-run on 2026-07-26 (run id benchmark_claudeopus48_mem0_patch,
    all 3 seeds, real mem0/Qdrant backend) and spliced into the canonical log included here.
  * Claude Opus 4.8 re-runs omit the temperature parameter (Anthropic deprecated it for
    this model in July 2026); the original campaigns sent temperature=0.
  * Event records do not persist the seed (logged as 0), so the 3 seeds of a
    (task, position) appear as 3 unlabeled observations; seed-level matching is available
    at task level via task_level/*/results.jsonl.
