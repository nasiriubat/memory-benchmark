# RQ2 / RQ4 — What the data does and does not support

*An honest audit of our extracted data against the questions raised, with pointers to where each
signal lives (or a clear statement that it does not).*

## Short answer

We **did** measure learning behavior — specifically **within-task (within-episode) learning**: memory
agents accumulate information across a task's interdependent steps and reuse it to answer later steps,
clearly outperforming a stateless (no-memory) agent. What we did **not** measure is **cross-task /
long-term transfer** (task A → task B) or **memory lineage** (which specific stored item was reused).
The cross-task gap is a **property of the MemoryArena benchmark's dataset design** — its tasks are
independent by construction — not a misconfiguration on our side. We recommend keeping RQ2/RQ4 focused on
within-task learning + retrieval/use quality (which the data supports), and listing cross-task transfer
and lineage as future work.

## What learning did we measure? (the affirmative result)

**Within-task learning is present and measurable.** Per-subtask mean score, memory systems vs stateless
(averaged over the 3 models):

| subtask position | stateless | mem0 | hindsight | mem0 − stateless |
|---|---|---|---|---|
| 0 (first) | 0.406 | 0.432 | 0.440 | +0.03 |
| 1 | 0.105 | 0.255 | 0.291 | **+0.15** |
| 2 | 0.253 | 0.304 | 0.282 | +0.05 |
| 3 | 0.129 | 0.282 | 0.327 | **+0.15** |
| 4 | 0.197 | 0.311 | 0.270 | +0.11 |
| 5 | 0.226 | 0.328 | 0.254 | +0.10 |

At the **first** subtask, memory ≈ stateless — nothing has been stored yet (retrieval count = 0). From
subtask 1 onward, **stateless collapses** (0.41 → 0.11: it cannot recall what happened at step 0) while
**memory systems hold up** (~0.25–0.33). The **memory-vs-stateless gap opens from ~0 to +0.10–0.20**.
That widening gap **is** the learning signal: the agent stores information during the task and reuses it
on later, dependent steps; the memoryless agent cannot.

> Note: the *absolute* score curve is roughly flat across positions because **later subtasks are harder**
> (they depend on more prior context). The learning is therefore visible in the **gap vs stateless**, not
> in a rising absolute curve. Evidence: `benchmark_subtask_trace.xlsx` (sheet *Subtask progression*) and
> `docs/figures/fig3_learning_curve.png`; the aggregate version is the **gain-vs-stateless** column in
> `benchmark_by_system.xlsx`.

## Point-by-point

| Question raised | In our data? | Where / why not |
|---|---|---|
| What knowledge was learned in an earlier task | **No (cross-task)** | Tasks are independent by the MemoryArena dataset's design; memory is isolated per task. Knowledge learned *within* a task is used later in the same task, but nothing carries between tasks. |
| Where it was stored in memory | **No (not persisted)** | Memory content/IDs live only in the backend during a run; not written to results or event logs. |
| Whether it was retrieved in a later *related task* | **No cross-task; yes within-task** | No related later task exists in the dataset. *Within* a task, retrieval counts rise across subtasks (0 → ~3) — memory is actively retrieved on later steps (see trace). |
| Whether it was actually used in the final answer | **Partial** | `memory_usage_score` estimates this per task, but as a token-overlap / declared-id heuristic — correlation, not a traced "this memory → this answer". |
| Whether performance improved because of this reuse | **Yes, within a task** | Memory systems diverge **positively** from stateless as steps progress: gap ≈0 at step 0 → +0.10–0.20 by step 1+ (stateless collapses, memory holds). Not measured across tasks. |

## Evidence (files)

- **Within-task learning:** per-subtask memory-vs-stateless divergence in
  `results/aggregated/subtask_trend.csv` → `benchmark_subtask_trace.xlsx` and
  `docs/figures/fig3_learning_curve.png`; aggregate gain in `benchmark_by_system.xlsx`.
- **Task independence is by dataset design; memory isolated per task:** `MemoryNamespace` keys on
  `task_id` ([src/benchmark/schemas.py](../src/benchmark/schemas.py)); memory reset per task in
  [src/benchmark/runner.py](../src/benchmark/runner.py). Memory persists **only across subtasks within one
  task**.
- **No lineage in logs:** raw events record only `memory_retrieved {count}` and
  `memory_written {written: 0|1}` — **no memory ids or text**
  ([src/benchmark/benchmarks/base.py](../src/benchmark/benchmarks/base.py)). `used_memory_ids` computed
  for scoring but not persisted.
- **RQ4 metrics are heuristics:** relevance/correctness/usage are token-overlap (+ LLM fallback)
  ([src/benchmark/evaluators/memory_utility.py](../src/benchmark/evaluators/memory_utility.py)).
- **No session / transfer / lineage fields** exist anywhere.

## Is the cross-task gap our setup, or the benchmark?

It is the **benchmark / dataset design**. In the MemoryArena dataset, each "task" is a self-contained unit
of ordered, interdependent sub-questions (q0→qN); different tasks are unrelated by construction, so there
is no "task A teaches what task B needs" relationship to measure. Our pipeline exposes a
`cross_task_memory` switch (default off), but it is not the real lever — enabling it would only pool
unrelated tasks' memories (interference/noise, not transfer). Genuine related-task transfer needs a
**different dataset built with linked task sequences** — a separate study, not a config change on our
part.

## Recommended framing

- **RQ2:** *"Do memory-augmented agents outperform a stateless baseline and learn within a task —
  accumulating and reusing information across its interdependent steps?"* — **supported** (per-subtask
  divergence + gain-vs-stateless). Cross-task / lifelong transfer = future work.
- **RQ4:** *"How relevant, correct, and used is retrieved memory during task execution?"* — supported by
  aggregate relevance/correctness/usage + retrieval/write counts (proxy measures, not causal lineage).

## Threats to validity / future work

- **Cross-task transfer & sessions — a benchmark-design limitation:** not evaluable with the MemoryArena
  dataset (independent tasks). The `cross_task_memory` switch only pools unrelated memory. True A→B
  transfer needs a dataset with linked task sequences.
- **Memory lineage / causal influence (RQ4):** would require logging memory ids/text + `used_memory_ids`
  per subtask event and a re-run, to trace a specific memory from write → retrieve → use.

## Bottom line

We **do** have a learning result: **within-task** learning, shown by memory systems diverging from
stateless as a task's steps progress (gap ≈0 at step 0 → +0.10–0.20 later). We do **not** have cross-task
/ lifelong learning or memory lineage — the former because of the MemoryArena dataset's independent-task
design, the latter because ids/text aren't logged. Keep RQ2/RQ4 on the within-task claims the data
supports; list transfer and lineage as future work.
