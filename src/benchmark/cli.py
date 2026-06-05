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

import typer

from benchmark.config import AppConfig, load_config
from benchmark.llms.model_resolver import ModelResolver
from benchmark.logging import get_logger
from benchmark.progress import set_done, set_failed, set_running
from benchmark.runner import Runner, build_memory_adapter
from benchmark.schemas import RunMetadata
from benchmark.utils import resolve_run_id, results_root

_log = get_logger("benchmark.cli")

app = typer.Typer(add_completion=False, help="Memory Agent Benchmark CLI.")


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

    # Optionally restrict to a subset of model slots.
    if models:
        wanted = set(models)
        cfg.models = {
            "models": {
                slot: spec
                for slot, spec in cfg.models.get("models", {}).items()
                if slot in wanted
            }
        }

    # 1. Resolve models ONCE (snapshot catalog + resolved mapping).
    policy = cfg.openrouter_policy
    resolver = ModelResolver(
        results_root=root,
        base_url=policy.get("base_url"),
        provider_policy=cfg.provider_policy,
        fail_if_model_unavailable=bool(policy.get("fail_if_model_unavailable", False)),
    )
    resolution = resolver.resolve_all(run_id, cfg.models)
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
