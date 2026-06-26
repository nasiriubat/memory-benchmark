"""Command-line interface and high-level run orchestration.

``run_memory_system`` performs the full per-run sequence:
  1. create run_id if missing
  2. load model + benchmark configs
  3. call ModelResolver (snapshot catalog + resolved models) ONCE
  4. run the benchmark using resolved concrete model ids
  5. save task results into the per-memory-system folder
  6. aggregate partial results + print a summary

Scripts in ``scripts/`` are thin wrappers around these functions.
"""

from __future__ import annotations

import os

import typer

from benchmark.config import AppConfig, load_config
from benchmark.evaluators.cost import estimate_cost_usd
from benchmark.llms.factory import build_model_resolver, llm_provider
from benchmark.logging import get_logger
from benchmark.progress import set_done, set_failed, set_running
from benchmark.runner import Runner, build_memory_adapter
from benchmark.schemas import RunMetadata
from benchmark.utils import resolve_run_id, results_root

_log = get_logger("benchmark.cli")

app = typer.Typer(add_completion=False, help="Memory Agent Benchmark CLI.")


def _resolve_models_once(cfg: AppConfig, run_id: str, root) -> dict:
    """Resolve models, reusing an existing snapshot for this RUN_ID if present.

    Every system in a campaign shares one RUN_ID and MUST use the same resolved
    concrete model ids (otherwise gain-vs-stateless compares different models).
    Re-resolving would also overwrite the snapshot that recorded what earlier
    systems used. So if ``model_snapshots/{run_id}/resolved_models.yaml`` already
    covers all enabled slots, reuse it. Set ``FORCE_MODEL_RESOLUTION=1`` to
    re-resolve and overwrite (e.g. starting a fresh campaign on an existing id).
    """
    from pathlib import Path

    import yaml

    from benchmark.llms.model_resolver import ResolvedModel

    enabled = {
        slot
        for slot, spec in cfg.models.get("models", {}).items()
        if spec.get("enabled", False)
    }
    snapshot = Path(root) / "model_snapshots" / run_id / "resolved_models.yaml"
    force = os.environ.get("FORCE_MODEL_RESOLUTION", "").strip() not in ("", "0", "false")

    if snapshot.exists() and not force:
        data = yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}
        saved = data.get("models", {}) or {}
        if enabled and enabled <= set(saved):
            resolved = {
                slot: ResolvedModel.from_dict(saved[slot]) for slot in enabled
            }
            _log.info(
                "Reusing resolved models from %s (RUN_ID=%s) — %d slot(s); "
                "set FORCE_MODEL_RESOLUTION=1 to re-resolve.",
                snapshot, run_id, len(resolved),
            )
            return {
                "resolved": resolved,
                "model_catalog_snapshot_path": data.get("model_catalog_snapshot_path"),
                "resolved_models_path": str(snapshot),
                "model_resolution_time": data.get("model_resolution_time"),
                "openrouter_base_url": data.get("openrouter_base_url"),
                "reused_snapshot": True,
            }
        _log.warning(
            "Existing snapshot %s does not cover all enabled slots (%s); "
            "re-resolving and OVERWRITING it.",
            snapshot, sorted(enabled - set(saved)),
        )

    resolver = build_model_resolver(results_root=Path(root), config=cfg)
    return resolver.resolve_all(run_id, cfg.models)


def run_memory_system(
    memory_system: str,
    *,
    models: list[str] | None = None,
    run_id: str | None = None,
    config: AppConfig | None = None,
) -> str:
    """Execute a full benchmark run for a single memory system."""
    cfg = config or load_config()
    run_id = resolve_run_id(run_id)
    root = results_root()
    set_running(run_id, memory_system, root=root)
    _log.info("=== Memory system '%s' | run_id=%s ===", memory_system, run_id)

    # Optionally restrict to a subset of model slots. Judge-tier slots
    # (benchmark_tier == "judge") are always kept regardless of the requested
    # MODELS set: they are needed to build the judge client but are never run as
    # systems-under-test (the Runner excludes them from the run loop).
    if models:
        wanted = set(models)
        cfg.models = {
            "models": {
                slot: spec
                for slot, spec in cfg.models.get("models", {}).items()
                if slot in wanted or spec.get("benchmark_tier") == "judge"
            }
        }

    # 1. Resolve models ONCE (reusing the campaign snapshot if present so every
    #    system uses identical concrete model ids).
    _log.info("LLM provider: %s", llm_provider())
    resolution = _resolve_models_once(cfg, run_id, root)
    resolved_models = resolution["resolved"]

    # 2. Build the memory adapter for this system.
    memory = build_memory_adapter(memory_system, cfg)
    if not memory.available:
        _log.warning(
            "Memory system '%s' is unavailable (%s); running with local fallback so "
            "the benchmark still produces results.",
            memory_system, memory.unavailable_reason,
        )

    # 3. Run.
    runner = Runner(
        config=cfg,
        run_id=run_id,
        memory_system=memory_system,
        memory=memory,
        resolved_models=resolved_models,
        results_root=root,
        run_metadata=resolution,
    )
    metadata = RunMetadata(
        run_id=run_id,
        memory_system=memory_system,
        models=list(resolved_models.keys()),
        benchmarks=list(cfg.enabled_benchmarks().keys()),
        model_catalog_snapshot_path=resolution.get("model_catalog_snapshot_path"),
        resolved_models_path=resolution.get("resolved_models_path"),
        model_resolution_time=resolution.get("model_resolution_time"),
        openrouter_base_url=resolution.get("openrouter_base_url"),
        memory_backend_available=memory.available,
        memory_uses_local_fallback=memory.uses_local_fallback,
        memory_unavailable_reason=memory.unavailable_reason,
        config={"run": cfg.run.__dict__},
    )
    runner.store.write_metadata(metadata)
    try:
        runner.run()
    except Exception as exc:
        set_failed(run_id, memory_system, str(exc), root=root)
        raise
    finally:
        memory.close()

    set_done(run_id, memory_system, root=root)
    _log.info("=== Done: '%s' run_id=%s ===", memory_system, run_id)
    return run_id


def estimate_run(
    memory_system: str,
    *,
    models: list[str] | None = None,
    config: AppConfig | None = None,
) -> dict:
    """Estimate the cost of a run without making any LLM calls (dry run).

    Enumerates the full job grid (models x benchmarks x domains x tasks x
    num_runs) and multiplies the per-call token assumptions by live OpenRouter
    catalog pricing. Token assumptions are rough; override via env vars
    ``EST_INPUT_TOKENS`` / ``EST_OUTPUT_TOKENS`` (agent) and
    ``EST_JUDGE_INPUT_TOKENS`` / ``EST_JUDGE_OUTPUT_TOKENS`` (judge).
    """
    cfg = config or load_config()
    run_id = resolve_run_id(None)
    root = results_root()
    if models:
        wanted = set(models)
        cfg.models = {
            "models": {
                slot: spec
                for slot, spec in cfg.models.get("models", {}).items()
                if slot in wanted or spec.get("benchmark_tier") == "judge"
            }
        }
    resolved = _resolve_models_once(cfg, run_id, root)["resolved"]
    available = {s: rm for s, rm in resolved.items() if rm.available}
    # Judge-tier slots are scored against, not benchmarked: keep them out of the
    # system-under-test (agent) cost and use them only for the judge estimate.
    run_models = {s: rm for s, rm in available.items() if rm.benchmark_tier != "judge"}

    memory = build_memory_adapter(memory_system, cfg)
    runner = Runner(
        config=cfg,
        run_id=run_id,
        memory_system=memory_system,
        memory=memory,
        resolved_models=resolved,
        results_root=root,
    )
    plan = runner.plan()
    memory.close()

    in_tok = int(os.environ.get("EST_INPUT_TOKENS", 1500))
    out_tok = int(os.environ.get("EST_OUTPUT_TOKENS", 400))
    j_in = int(os.environ.get("EST_JUDGE_INPUT_TOKENS", 600))
    j_out = int(os.environ.get("EST_JUDGE_OUTPUT_TOKENS", 40))
    agent_calls = plan["agent_calls"]
    # The judge runs only when memory is retrieved (~2 calls/subtask: relevance
    # + usage). Stateless retrieves nothing, so it incurs ~no judge cost.
    judge_calls_per_model = 0 if memory_system == "stateless" else agent_calls * 2

    judge_slot = cfg.benchmark.get("evaluators", {}).get("llm_judge_model")
    judge_rm = available.get(judge_slot) or next(iter(available.values()), None)

    per_model = {}
    agent_total = 0.0
    for slot, rm in run_models.items():
        unit = estimate_cost_usd(
            input_tokens=in_tok, output_tokens=out_tok, pricing=rm.pricing
        )
        cost = unit * agent_calls
        per_model[slot] = {"resolved_model": rm.resolved_model, "calls": agent_calls, "cost": cost}
        agent_total += cost

    judge_total = 0.0
    if judge_rm is not None and judge_calls_per_model:
        judge_unit = estimate_cost_usd(
            input_tokens=j_in, output_tokens=j_out, pricing=judge_rm.pricing
        )
        judge_total = judge_unit * judge_calls_per_model * len(run_models)

    result = {
        "plan": plan,
        "assumptions": {"input_tokens": in_tok, "output_tokens": out_tok,
                        "judge_input_tokens": j_in, "judge_output_tokens": j_out},
        "per_model": per_model,
        "agent_cost_usd": agent_total,
        "judge_cost_usd": judge_total,
        "judge_model": getattr(judge_rm, "slot", None),
        "total_cost_usd": agent_total + judge_total,
        "offline_no_pricing": not any(rm.pricing for rm in available.values()),
    }
    _print_estimate(result)
    return result


def _print_estimate(r: dict) -> None:
    plan = r["plan"]
    a = r["assumptions"]
    print("\n=== Cost estimate (DRY RUN — no LLM calls made) ===")
    print(f"Memory system: {plan['memory_system']} | profile: {plan['profile'] or '(none)'}")
    print(f"num_runs={plan['num_runs']}  tasks={plan['total_tasks']}  "
          f"subtasks={plan['total_subtasks']}  agent_calls/model={plan['agent_calls']}")
    for b, c in plan["benchmarks"].items():
        print(f"  - {b}: {c['domains']} domain(s), {c['tasks']} tasks, {c['subtasks']} subtasks")
    print(f"Token assumptions: agent in={a['input_tokens']} out={a['output_tokens']}, "
          f"judge in={a['judge_input_tokens']} out={a['judge_output_tokens']}")
    if r["offline_no_pricing"]:
        print("WARNING: no catalog pricing available (offline / no API key); costs are $0.00.")
    print("Per-model agent cost:")
    for slot, m in r["per_model"].items():
        print(f"  {slot} ({m['resolved_model']}): ${m['cost']:.4f}")
    print(f"Judge cost ({r['judge_model']}): ${r['judge_cost_usd']:.4f}")
    print(f"TOTAL ESTIMATE for '{plan['memory_system']}': ${r['total_cost_usd']:.4f}")
    print("Note: rough upper-bound; real cost depends on actual token usage and "
          "prompt caching.\n")


@app.command("estimate")
def cli_estimate(
    memory_system: str = typer.Option(..., "--memory-system", help="Memory system id."),
    models: str = typer.Option("", "--models", help="Comma-separated model slots."),
) -> None:
    """Estimate run cost without making any LLM calls."""
    model_list = [m.strip() for m in models.split(",") if m.strip()] or None
    estimate_run(memory_system, models=model_list)


@app.command("run")
def cli_run(
    memory_system: str = typer.Option(..., "--memory-system", help="Memory system id."),
    models: str = typer.Option("", "--models", help="Comma-separated model slots."),
    run_id: str = typer.Option("", "--run-id", help="Explicit run id (optional)."),
) -> None:
    """Run one memory system across the configured models."""
    model_list = [m.strip() for m in models.split(",") if m.strip()] or None
    run_memory_system(memory_system, models=model_list, run_id=run_id or None)


@app.command("aggregate")
def cli_aggregate() -> None:
    """Aggregate all results into results/aggregated/."""
    from benchmark.aggregate import aggregate

    aggregate(results_root())


@app.command("snapshot")
def cli_snapshot() -> None:
    """Snapshot the OpenRouter model catalog and print resolved models."""
    from benchmark.snapshot import snapshot

    snapshot()


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
