"""Generate docs/paper_sections_input.docx — factual input for the memory paper.

Answers the supervisor's questions for six sections (pipeline, memory
configurations, LLM backbones, execution & logging, benchmark & task setting,
experimental matrix). Narrative facts are verified against the code
(file references inline); tables that carry numbers are loaded live from
results/aggregated/*.csv and the exported workbooks so the document can be
regenerated after any re-export.

Run:  .venv/bin/python3 scripts/make_paper_sections_docx.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "aggregated"
OUT = ROOT / "docs" / "paper_sections_input.docx"

MODEL_LABEL = {
    "gpt-5.5": "gpt-5.5",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "openai_frontier": "gpt-5.5",
    "anthropic_opus": "Claude Opus 4.8",
    "google_fast": "Gemini 3.5 Flash",
}


# ---------------------------------------------------------------- docx helpers
def _style_base(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)


def head(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def para(doc: Document, text: str, bold_label: str | None = None) -> None:
    p = doc.add_paragraph()
    if bold_label:
        r = p.add_run(bold_label + " ")
        r.bold = True
    p.add_run(text)


def bullets(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def numbered(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Number")


def table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9.5)
    doc.add_paragraph()


def fmt(x, nd=3) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------- the doc
def main() -> None:
    lb = pd.read_csv(AGG / "leaderboard.csv")
    sm = pd.read_csv(AGG / "summary_by_memory_model.csv")
    trend = pd.read_csv(AGG / "subtask_trend.csv")
    det = pd.read_excel(ROOT / "benchmark_subtask_trace.xlsx", sheet_name="By task (detail)")
    det_ct = pd.crosstab(det.memory_system, det.model)

    doc = Document()
    _style_base(doc)

    doc.add_heading("Technical and Experimental Details — Memory Paper (Sections Input)", level=0)
    para(doc, "Prepared 23 July 2026 from the memory-agent benchmark repository (code, configs, and the "
              "final result files). All statements below describe what was actually implemented and logged; "
              "no claims go beyond the recorded data. Source artefacts: benchmark_results.xlsx, "
              "benchmark_by_system.xlsx, benchmark_subtask_trace.xlsx, results/aggregated/*.csv, and the "
              "raw per-event logs (results/raw/**/events.jsonl).")

    # ================================================================ 1. Pipeline
    head(doc, "1  Common Agent-Memory Evaluation Pipeline (\\section, Overview)")
    para(doc, "The experiment uses one shared evaluation pipeline; the only components exchanged between "
              "conditions are (a) the memory adapter and (b) the LLM backbone. The pipeline is presented as "
              "a common evaluation harness for comparing existing memory mechanisms, not as a new memory "
              "approach.")

    para(doc, "Each benchmark task is delivered as an ordered list of interdependent sub-questions "
              "(subtasks) plus a small set of static background statements (scene-setting facts from the "
              "MemoryArena dataset). There is no single monolithic task prompt: the agent never sees the "
              "whole task at once. At task start the harness creates a fresh, isolated memory namespace for "
              "that (task, seed) combination and resets the memory backend, so every task starts with an "
              "empty store.", bold_label="Input at the beginning of a task.")

    para(doc, "Subtasks are executed strictly in order. For each subtask the harness performs four steps: "
              "(1) memory retrieval, (2) response generation, (3) scoring, (4) memory write. The loop is "
              "implemented once in SequentialBenchmarkAdapter (src/benchmark/benchmarks/base.py) and is "
              "byte-identical for every memory system and every model.", bold_label="Per-subtask steps.")

    para(doc, "Retrieval happens before the LLM call of every subtask. The query is the raw subtask "
              "question; the harness requests the top k = 3 records (config run.retrieve_top_k = 3). A "
              "retrieval failure yields an empty list and is logged; it never aborts the run. For the "
              "stateless baseline retrieval is a no-op that always returns an empty list. Retrieval latency "
              "and the number of returned records are logged per subtask.", bold_label="When and how memory is retrieved.")

    para(doc, "A single LLM call per subtask (no tool use, no multi-turn loop). The system prompt is fixed "
              "and instructs the agent to use any provided MEMORY to stay consistent with earlier "
              "decisions, and to answer with one JSON object {\"final_answer\", \"used_memory_ids\", "
              "\"reasoning_summary\"}. The user prompt concatenates three blocks: BACKGROUND (the task's "
              "background facts, re-supplied at every subtask), MEMORY (the retrieved records, each shown "
              "with its id and text; omitted when nothing is retrieved), and TASK (the current subtask "
              "question). Implementation: src/benchmark/agents/simple_tool_agent.py.",
         bold_label="How the LLM response is generated.")

    para(doc, "Each subtask answer is graded by an LLM judge (gpt-4o-mini, temperature 0) against the "
              "benchmark's expected answer using a fixed rubric; the judge returns a continuous score in "
              "[0, 1]. A subtask counts as correct when score ≥ 0.6. The judge, rubric, and threshold are "
              "identical for all conditions. (An exact/token-F1 scorer exists in the code but the LLM judge "
              "was used for these runs because MemoryArena reference answers contain internal product IDs.)",
         bold_label="How the response is evaluated.")

    para(doc, "None is returned to the agent. The judge score is logged and used only for analysis; it is "
              "never fed back into the prompt or into the memory store, and there is no retry or "
              "self-correction loop. A post-hoc failure-category label is assigned per task for analysis "
              "only.", bold_label="Feedback after evaluation.")

    para(doc, "After scoring, the harness writes exactly one memory record per subtask (for systems that "
              "support writing): the record text is the literal pair \"Q: <subtask question> / A: <agent's "
              "final answer>\", with the subtask id and the agent's one-sentence reasoning summary as "
              "metadata. What the backend then does with that record (verbatim storage, fact extraction, "
              "skill learning) is the memory system's own behaviour and is part of what is being compared. "
              "Write latency and success are logged. The stateless baseline's write is a no-op.",
         bold_label="When and how memory is written.")

    para(doc, "Nothing except the memory store. No conversation history, no previous answers, and no "
              "scores are carried forward in the prompt; the background block is static task material. "
              "Memory is therefore the only channel by which information can cross from one subtask to the "
              "next — this is the design point that makes the comparison meaningful.",
         bold_label="What is passed from one subtask to the next.")

    para(doc, "Every step emits a structured event to results/raw/<run>/<system>/<model>/.../events.jsonl: "
              "run_started, task_started, subtask_started, memory_retrieved, prompt_built, llm_response, "
              "evaluator_score, memory_written, subtask_finished, task_finished, run_finished. At task end "
              "one summary row (TaskResult, ~40 fields) is written to results/by_memory_system/.",
         bold_label="What is logged at each step.")

    para(doc, "The agent and both prompts, the subtask loop and its order, top-k, the judge model, rubric "
              "and correctness threshold, the consistency/memory-utility evaluators, the namespace "
              "isolation scheme, and the result schema are identical everywhere. Only the memory adapter "
              "(retrieve/write behaviour) and the backbone LLM client differ between conditions.",
         bold_label="Identical across all memory systems and models.")

    para(doc, "Step-by-step workflow (for the figure):", bold_label="")
    numbered(doc, [
        "Task input: ordered subtasks q0…qN + static backgrounds; fresh empty memory namespace.",
        "Memory retrieval: query = current subtask question; top-3 records (stateless: none).",
        "Prompt assembly: BACKGROUND + MEMORY (id + text) + TASK.",
        "Response generation: one LLM call; JSON answer with final_answer and used_memory_ids.",
        "Evaluation: LLM judge scores answer 0–1 vs. expected answer; correct if ≥ 0.6 (not shown to agent).",
        "Memory write: one record \"Q: … A: …\" appended to the store (stateless: no-op).",
        "Next subtask (repeat 2–6 until the task's subtasks are exhausted).",
        "Task-level scoring and logging: progress score, mean score, success, memory-utility metrics, "
        "token/cost/latency accounting.",
    ])

    # ============================================================ 2. Memory configs
    head(doc, "2  Memory Configurations (\\subsection)")
    para(doc, "Four configurations were evaluated: stateless, mem0, hindsight, and acontext. All run behind "
              "the same adapter interface (setup / reset / retrieve / write); each (task, seed) gets a "
              "dedicated namespace and the store is reset before every task, so memory never leaks across "
              "tasks, seeds, models, or systems.")

    table(doc,
          ["Memory configuration", "Memory strategy", "What is stored", "What is retrieved",
           "Update timing", "Role in experiment"],
          [
              ["stateless", "No memory (no-op reads and writes)", "Nothing", "Nothing (always empty list)",
               "Never", "Baseline; reference point for gain-vs-stateless"],
              ["mem0 (self-hosted OSS + Qdrant)", "LLM fact extraction on write; embedding-based semantic "
               "search on read (OpenAI text-embedding-3-small)", "Facts extracted by mem0's own LLM "
               "(gpt-4o-mini) from the Q/A record", "Top-3 most similar stored memories (vector similarity, "
               "per-namespace user id)", "After every subtask", "Extraction + vector-retrieval memory"],
              ["hindsight (self-hosted server)", "Server-side 'retain/recall' memory with internal "
               "reflection", "The Q/A record, retained and reorganised server-side", "First 3 results of "
               "recall(query); no per-item similarity scores exposed", "After every subtask",
               "Reflection-oriented memory"],
              ["acontext (self-hosted server)", "Skill learning: server distils stored messages into named "
               "skills", "The Q/A record as a session message; server learns skills from it "
               "(blocking wait, ≤ 60 s timeout)", "The list of learned skills (name + description), capped "
               "at 3; no semantic ranking", "After every subtask", "Skill-based memory"],
          ])

    para(doc, "The stateless agent receives only the current subtask question plus the task's static "
              "background statements — the same background block every memory-enabled agent also receives. "
              "It gets no conversation history, no previous sub-answers, and no retrieved memory (its "
              "MEMORY block is always absent). Its writes are no-ops. Every other pipeline component "
              "(prompt, judge, threshold, loop) is identical, so any difference from a memory-enabled "
              "condition is attributable to memory alone.", bold_label="How the stateless baseline works.")

    para(doc, "Important configuration settings.", bold_label="")
    bullets(doc, [
        "retrieve_top_k = 3 for all systems (configs/benchmark.yaml).",
        "mem0: self-hosted mem0 OSS over a Qdrant vector store; embedder OpenAI text-embedding-3-small; "
        "internal extraction LLM gpt-4o-mini (held constant across systems).",
        "hindsight: official client against the vectorize-io server container; recall queries are limited "
        "to 500 tokens server-side, so the harness truncates the query to 1,600 characters and "
        "halves-and-retries up to 3× if the server still rejects it.",
        "acontext: official SDK against a self-hosted server; each namespace maps to its own learning "
        "space + session; each write triggers skill learning and waits for completion (60 s timeout).",
        "All non-stateless adapters fall back to a local token-overlap store if their backend errors, and "
        "flag the row (memory_uses_local_fallback). In the final 1,440 task rows this flag is false "
        "everywhere — no degraded rows are included.",
    ])

    para(doc, "Observed behaviour / limitations during the experiment.", bold_label="")
    bullets(doc, [
        "acontext retrieves skills but the agent rarely uses them: its memory-usage score is much lower "
        "than mem0/hindsight (see Section 6 data), consistent with skill descriptions being less directly "
        "quotable than stored facts.",
        "hindsight exposes no per-item similarity scores; retrieval order is trusted as returned.",
        "acontext's blocking learn step makes it the slowest condition per subtask.",
        "A fifth system, zep, was attempted but excluded: its legacy backend never started, every zep run "
        "silently used the local fallback store, so all zep results were removed from the final data "
        "(preserved separately under results/EXCLUDED_zep_fallback/). supermemory (closed source) was "
        "listed in the original plan but not evaluated.",
    ])

    # ============================================================ 3. LLM backbones
    head(doc, "3  LLM Backbones (\\subsection)")
    table(doc,
          ["Model label in paper", "Full model name", "Provider", "Temperature", "Max tokens",
           "Other settings", "Notes"],
          [
              ["gpt-5.5", "gpt-5.5", "OpenAI (direct API)", "not sent (temperature-locked model; config "
               "value 0)", "8,192 (max_completion_tokens)", "reasoning_effort = medium; no top_p",
               "Run id benchmark_20260626T142817Z"],
              ["Claude Opus 4.8", "anthropic/claude-opus-4.8", "Anthropic via OpenRouter "
               "(Anthropic-direct route; Amazon Bedrock excluded)", "0", "4,096",
               "reasoning disabled; no top_p", "Run id benchmark_claudeopus48. The 23 July 2026 "
               "stateless replication run (Section 6) omits temperature — Anthropic deprecated the "
               "parameter for this model after the original campaign."],
              ["Gemini 3.5 Flash", "google/gemini-3.5-flash", "Google via OpenRouter", "0", "4,096",
               "reasoning disabled; no top_p", "Run id benchmark_gemini35flash"],
          ])
    bullets(doc, [
        "LLM judge for all runs and all conditions: gpt-4o-mini, temperature 0, max 2,048 output tokens.",
        "Memory-system internal LLM (mem0 extraction, hindsight server): gpt-4o-mini, held constant. Its "
        "API cost is not included in the logged agent/judge cost columns.",
        "top_p is never set anywhere in the codebase; defaults apply.",
        "Same prompt and pipeline for all models. Model-specific handling is limited to client plumbing: "
        "the OpenAI-direct client (retries = 6) omits temperature and uses max_completion_tokens for "
        "gpt-5.x models; the OpenRouter client (retries = 4) strips parameters a model does not support "
        "and applies the provider-routing policy in configs/openrouter.yaml (allow_fallbacks, "
        "require_parameters, data_collection = deny).",
        "Coverage: all 3 models were evaluated on all 4 memory systems and all 40 tasks × 3 seeds. The "
        "final task-level data is complete — 120 rows in every (model × memory system) cell, no failed or "
        "missing combinations. (An earlier Claude Sonnet 5 run exists only in the excluded zep tree and is "
        "not part of the final data.)",
    ])
    para(doc, "The three backbones were selected to span providers and capability tiers — a frontier "
              "OpenAI model (gpt-5.5), a frontier Anthropic model (Claude Opus 4.8), and a fast mid-tier "
              "Google model (Gemini 3.5 Flash) — so that memory behaviour can be compared across different "
              "LLM backbones under one identical memory pipeline.", bold_label="Why these models.")

    # ======================================================= 4. Execution & logging
    head(doc, "4  Execution and Logging (\\subsection)")
    para(doc, "One experimental run = one task × one memory system × one model × one seed, producing one "
              "TaskResult row. Counts: 40 tasks × 4 memory systems × 3 models × 3 seeds (1000, 1001, 1002) "
              "= 1,440 runs. Every task has 6 subtasks in this sample, so 8,640 subtask executions were "
              "performed in total. Repeats are identified by the seed column (there is no separate per-"
              "repeat run_id; run_id identifies the campaign — one campaign per backbone). subtask_index "
              "is the 0-based position of a sub-question within its task (0 = first, executed in order); "
              "observed range 0–5.", bold_label="Definition of a run.")

    para(doc, "Files and sheets.", bold_label="")
    table(doc, ["File", "Sheets / content", "Level"],
          [
              ["benchmark_results.xlsx (also in docs/ and attachment/)",
               "All results (raw) — 1,440 rows × 41 columns; Leaderboard; Per-system summary; Per-model "
               "summary; Learning curve; Failure taxonomy; Descriptive stats; Stats – ANOVA (2-way); "
               "Stats – vs stateless; Stats – post-hoc Tukey; Overview; Definitions", "task/run + summary"],
              ["benchmark_by_system.xlsx", "One sheet per memory system (3 rows = one per backbone, 26 "
               "metric columns) + Definitions", "summary"],
              ["benchmark_subtask_trace.xlsx", "Subtask progression (per system × model × subtask "
               "position); By task (detail) (per system × model × task × subtask position); About & "
               "definitions", "subtask"],
              ["results/aggregated/*.csv (+ all_results.parquet)", "leaderboard, summary_by_memory_model, "
               "subtask_trend, failure_categories, pairwise_vs_stateless, stats_anova, stats_posthoc, "
               "effect_sizes, descriptive_stats, paper_table", "summary"],
              ["results/raw/<run>/<system>/<model>/…/events.jsonl", "Per-event trace (11 event types "
               "listed in Section 1)", "event"],
          ])

    para(doc, "Logged variables. The table below lists every variable in the exported data, its level, "
              "and the research question it serves.", bold_label="")
    table(doc, ["Logged item", "Description", "Level", "Used for which RQ"],
          [
              ["run_id", "Campaign identifier (one per backbone: benchmark_20260626T142817Z = gpt-5.5, "
               "benchmark_claudeopus48, benchmark_gemini35flash)", "run-level", "provenance"],
              ["benchmark / domain", "memoryarena / bundled_shopping (constant)", "task-level", "provenance"],
              ["memory_system", "stateless | mem0 | hindsight | acontext", "task-level", "RQ1–RQ4"],
              ["model / requested_model / resolved_model / provider_family / benchmark_tier",
               "Model slot, requested id, exact resolved id, provider, tier", "task-level", "RQ3"],
              ["task_id", "MemoryArena task identifier (40 distinct)", "task-level", "RQ1–RQ4"],
              ["seed", "Repeat identifier: 1000 / 1001 / 1002", "run-level", "RQ1 (variance), pass@3"],
              ["success", "True iff every subtask in the task is correct (all-or-nothing)", "task-level", "RQ1"],
              ["score", "Mean LLM-judge score (0–1) over the task's subtasks", "task-level", "RQ1, RQ3"],
              ["progress_score", "Fraction of subtasks with judge score ≥ 0.6 (primary metric)",
               "task-level", "RQ1, RQ3"],
              ["consistency_score", "Agreement of answers with the task's accumulated fact ledger",
               "task-level", "RQ1, RQ4"],
              ["pass_at_3", "1 if any of the 3 seeds of a task succeeded (computed at aggregation)",
               "summary-level", "RQ1"],
              ["subtask_index", "0-based position of a sub-question within its task", "subtask-level", "RQ2"],
              ["mean score / correct (per subtask position)", "Judge score and correctness at each "
               "subtask position (Learning curve / subtask trace)", "subtask-level", "RQ2"],
              ["retrieved_memory_count", "Memory records retrieved over the task (and per subtask in the "
               "trace)", "task- & subtask-level", "RQ2, RQ4"],
              ["written_memory_count", "Memory records written over the task (and write rate per subtask)",
               "task- & subtask-level", "RQ4"],
              ["memory_retrieval_latency_ms / memory_write_latency_ms", "Cumulative memory-operation "
               "latencies", "task-level", "overhead analysis"],
              ["memory_relevance_score", "Overlap of retrieved records with gold memory facts / query "
               "(heuristic, LLM fallback)", "memory-level", "RQ4"],
              ["memory_correctness_score", "1 − fraction of retrieved records conflicting with the fact "
               "ledger", "memory-level", "RQ4"],
              ["memory_usage_score", "Fraction of retrieved records actually used in the answer (prefers "
               "the agent's used_memory_ids)", "memory-level", "RQ4"],
              ["memory_utility_gain_vs_stateless", "Task score minus the matched stateless task score "
               "(joined on model, task_id, seed; computed at aggregation)", "task-level", "RQ2, RQ3"],
              ["memory_backend_available / memory_uses_local_fallback", "Backend health; fallback flag "
               "(false in all 1,440 final rows)", "task-level", "validity check"],
              ["failure_category", "Post-hoc heuristic label, 11-value taxonomy (missing_memory_write, "
               "irrelevant_memory_write, failed_retrieval, incorrect_or_outdated_retrieval, "
               "incorrect_use_of_retrieved_memory, over_reliance_on_prior_feedback, "
               "inconsistency_across_iterations, model_reasoning_error, tool_use_error, "
               "benchmark_environment_error, unknown)", "task-level", "RQ4"],
              ["input/output/total_tokens, reasoning_tokens, judge_* tokens", "Token accounting, agent "
               "and judge separated", "task-level", "cost analysis"],
              ["estimated_cost_usd / judge_cost_usd / total_cost_usd", "Cost accounting (agent, judge, "
               "combined)", "task-level", "cost analysis"],
              ["latency_ms, turn_count, tool_call_count, iteration_count", "Wall-clock latency; number of "
               "subtasks (6); tool calls (always 0 — single-pass agent)", "task-level", "descriptive"],
              ["official_metrics", "MemoryArena-style extras (subtask/task success, final score, "
               "improvement slope), computed by the harness (marked non-official)", "task-level", "RQ2"],
          ])

    # ===================================================== 5. Benchmark & task setting
    head(doc, "5  Benchmark and Task Setting (\\subsection)")
    table(doc, ["Benchmark/task property", "Description"],
          [
              ["Benchmark name", "MemoryArena (HuggingFace dataset ZexueHe/memoryarena). The official "
               "MemoryArena runner was not installed; tasks were loaded from the dataset and scored by the "
               "harness's own LLM judge, so MemoryArena-style metrics are reported as non-official."],
              ["Task domain", "bundled_shopping — multi-step shopping scenarios (budgets, preferences, "
               "product/ASIN selection). One domain only; the adapter supports four further MemoryArena "
               "configs that were not run."],
              ["Number of tasks", "40, sampled deterministically (random sample of the ~150 available "
               "tasks, sampling seed 42) — identical 40 tasks for every system and model."],
              ["Subtasks per task", "6 in all sampled tasks (the count comes from the dataset per task and "
               "is variable in general; in this sample it is uniformly 6). Executed strictly in order."],
              ["Memory requirement", "Later sub-questions depend on decisions and constraints established "
               "in earlier ones (e.g. budget, preferred attributes, previously chosen products). Because "
               "the pipeline passes no conversation history between subtasks, memory is the only carrier "
               "of that information."],
              ["Scoring method", "Per subtask: LLM judge (gpt-4o-mini) scores the answer 0–1 against the "
               "expected answer with a fixed rubric; correct = score ≥ 0.6. Per task: score = mean judge "
               "score; progress_score = fraction of correct subtasks (primary metric); success = all "
               "subtasks correct (strict; observed rate ≈ 0.6%); pass@3 over the three seeds."],
              ["Task independence", "Tasks are mutually independent; memory is reset before every task "
               "and namespaced per (task, seed, system, model). Repeated seeds use isolated memory, so "
               "they measure variance, not cross-run learning."],
              ["Main limitation", "Within-task memory reuse only (see below); single domain; harness "
               "judge instead of official MemoryArena scoring; all-or-nothing success is very strict."],
          ])
    para(doc, "The benchmark, as configured in these runs (cross_task_memory = false, memory reset per "
              "task and per seed), does NOT support claims of cross-task transfer or long-term learning "
              "across independent sessions. It measures within-task memory reuse across dependent "
              "sub-questions — carrying information from earlier to later steps of the same task. Task A → "
              "Task B transfer was not measured and should not be claimed. (The harness has an optional "
              "cross-task mode, but it was off for all reported runs, and MemoryArena tasks are "
              "independent by construction, so enabling it would share unrelated memories rather than "
              "test genuine transfer.)", bold_label="Claim scoping (important).")

    # =========================================================== 6. Experimental matrix
    head(doc, "6  Experimental Matrix (\\subsection)")
    n_detail = len(det)
    trend_n = int(trend["n"].sum())
    table(doc, ["Factor", "Levels", "Count"],
          [
              ["Memory system", "stateless, mem0, hindsight, acontext", "4"],
              ["LLM backbone", "gpt-5.5, Claude Opus 4.8, Gemini 3.5 Flash", "3"],
              ["Tasks", "40 MemoryArena bundled_shopping tasks (fixed sample, seed 42)", "40"],
              ["Repeated runs/seeds", "seeds 1000, 1001, 1002 (isolated memory per seed)", "3"],
              ["Total runs (task rows)", "4 × 3 × 40 × 3", "1,440"],
              ["Subtasks executed", "1,440 runs × 6 subtasks", "8,640"],
              [f"Subtask-level observations in the event trace", "evaluator-score events aggregated in "
               f"the Learning curve / subtask trace (see completeness note)", f"{trend_n:,} events; "
               f"{n_detail:,} (task × subtask) trace rows"],
          ])

    para(doc, "The design is fully balanced at task level: every memory system × model × task × seed "
              "combination is present, 120 rows per (system × model) cell, and no cell relies on the "
              "local-fallback store.", bold_label="Balance.")

    para(doc, "Now fixed. In the original Claude Opus 4.8 campaign the stateless run's raw event log "
              "stopped being appended part-way (a file-locking issue on the OneDrive-synced disk), leaving "
              "event-level records for only 15 of 40 tasks, and an export bug additionally dropped even "
              "those rows from earlier versions of benchmark_subtask_trace.xlsx. Both problems were "
              "resolved on 23 July 2026: (1) the exporter bug was fixed, and (2) the stateless × Claude "
              "Opus 4.8 cell was re-executed as a dedicated replication run (run id "
              "benchmark_claudeopus48_stateless_rerun) with the identical configuration — the same 40 "
              "tasks, seeds 1000–1002, prompt, judge, and pipeline — producing a complete event log "
              "(720/720 subtask events over all 40 tasks). The subtask trace and Learning curve sheets now "
              "use this complete replication log for the stateless × Claude cell (240/240 trace rows; "
              "n = 120 per subtask position, like every other cell). One configuration difference was "
              "unavoidable: by July 2026 the Anthropic API had deprecated the temperature parameter for "
              "Opus 4.8, so the replication omits it, whereas the original campaign sent temperature = 0. "
              "The canonical task-level rows remain those of the original campaign (they were never "
              "affected; all 120 rows complete); the replication's task-level rows are kept separately "
              "(results/RERUN_stateless_claudeopus48_taskrows/) as a reproducibility check — cell means "
              "reproduce closely (mean score 0.312 vs 0.306; mean progress 0.318 vs 0.306; per-task "
              "Pearson r = 0.45 over the 120 matched task × seed pairs, reflecting normal LLM/judge "
              "run-to-run variance). The original partial event log is archived under "
              "results/EXCLUDED_partial_events_stateless_claudeopus48/. The smaller mem0 × Claude gap from "
              "the same logging issue (21 of 720 subtask events, tasks 61/75/130) was likewise fixed on "
              "26 July 2026 by re-running only those three tasks (all 3 seeds, real mem0 backend, "
              "identical configuration; run id benchmark_claudeopus48_mem0_patch) and splicing the "
              "recovered events into the canonical log — the subtask-level trace is now complete "
              "(720/720) in every cell.", bold_label="Stateless × Claude subtask rows — missing or fixed?")

    subtable = []
    for sys_name in ["stateless", "hindsight", "mem0", "acontext"]:
        row = [sys_name]
        for m in ["gpt-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8"]:
            row.append(str(int(det_ct.loc[sys_name, m])) if m in det_ct.columns else "0")
        subtable.append(row)
    para(doc, "Subtask-trace completeness after the fix (rows per cell; 240 = complete):", bold_label="")
    table(doc, ["Memory system", "gpt-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8"], subtable)

    table(doc, ["RQ", "Unit of analysis", "Required data", "Available?", "Notes"],
          [
              ["RQ1 (effect of memory mechanism)", "task/run level",
               "score, progress_score, success, memory_system; ANOVA + Tukey sheets", "Yes",
               "Fully balanced 1,440 rows"],
              ["RQ2 (improvement over repeated interactions)", "subtask level",
               "subtask_index, per-position score, retrieval count; gain vs stateless", "Yes",
               "Within-task curves only; stateless × Claude curve from the 23 July 2026 replication run "
               "(see above); cross-task transfer not measured"],
              ["RQ3 (effect of LLM backbone)", "task/run level",
               "model, memory_system, gain vs stateless, interaction ANOVA", "Yes",
               "All 12 cells complete, 120 rows each"],
              ["RQ4 (retrieval/usage effectiveness)", "memory-use level",
               "memory_relevance/correctness/usage scores, retrieved/written counts, failure taxonomy",
               "Yes, with caveats",
               "Scores are heuristics (token overlap + LLM fallback); individual memory ids/texts were "
               "not logged, so write→retrieve→use lineage cannot be traced"],
          ])

    para(doc, "Limitations of the matrix.", bold_label="")
    bullets(doc, [
        "RQ2's subtask-level analysis uses the event-derived trace, now complete (720/720) in all 12 "
        "cells; the stateless × Claude cell's trace comes from the 23 July 2026 replication run and "
        "the mem0 × Claude cell includes 3 tasks re-run on 26 July 2026 (see above). Task-level RQ2 "
        "quantities (gain vs stateless) use the complete, original 1,440-row data.",
        "RQ4 is correlational: relevance/correctness/usage are computed by heuristics, and memory content "
        "is not persisted, so no per-memory causal tracing is possible.",
        "Single benchmark domain (bundled_shopping); results may not generalise to other MemoryArena "
        "configs or other benchmarks.",
        "Cross-task transfer and long-term (cross-session) learning are outside the measured design.",
    ])

    # ------------------------------------------------------------ appendix numbers
    head(doc, "Appendix  Key aggregate numbers (for reference)")
    para(doc, "Loaded from results/aggregated/leaderboard.csv and summary_by_memory_model.csv (n = task "
              "rows per group).")
    lb_rows = [[r.memory_system, str(int(r.n)), fmt(r.progress_score), fmt(r.mean_score),
                fmt(r.task_success_rate), fmt(r.pass_at_3), fmt(r.memory_utility_gain_vs_stateless)]
               for r in lb.itertuples()]
    table(doc, ["Memory system", "n", "Progress score", "Mean score", "Task success rate", "pass@3",
                "Gain vs stateless"], lb_rows)
    sm_rows = [[r.memory_system, MODEL_LABEL.get(r.model, r.model), str(int(r.n)), fmt(r.progress_score),
                fmt(r.memory_utility_gain_vs_stateless), fmt(r.memory_relevance_score),
                fmt(r.memory_correctness_score), fmt(r.memory_usage_score)]
               for r in sm.sort_values(["memory_system", "model"]).itertuples()]
    table(doc, ["Memory system", "Model", "n", "Progress score", "Gain vs stateless", "Relevance",
                "Correctness", "Usage"], sm_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
