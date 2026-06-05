"""OpenRouter model snapshot helper.

Prints, for each enabled model slot: requested alias, resolved concrete model,
context length, pricing, supported parameters, and availability status. Also
saves the catalog + resolved mapping under results/model_snapshots/{run_id}/.
"""

from __future__ import annotations

from benchmark.config import load_config
from benchmark.llms.model_resolver import ModelResolver
from benchmark.logging import get_logger
from benchmark.utils import new_run_id, results_root

_log = get_logger("benchmark.snapshot")


def snapshot(run_id: str | None = None) -> dict:
    cfg = load_config()
    run_id = run_id or new_run_id("snapshot")
    policy = cfg.openrouter_policy
    resolver = ModelResolver(
        results_root=results_root(),
        base_url=policy.get("base_url"),
        provider_policy=cfg.provider_policy,
    )
    catalog = resolver.fetch_catalog()
    num_models = len(catalog.get("data", []) or [])
    resolution = resolver.resolve_all(run_id, cfg.models)
    resolved = resolution["resolved"]

    print(f"OpenRouter models found in catalog: {num_models}")
    print(f"Enabled model slots: {len(resolved)}")
    print("-" * 80)
    for slot, rm in resolved.items():
        print(f"slot:               {slot}")
        print(f"  requested alias:  {rm.requested_model}")
        print(f"  resolved model:   {rm.resolved_model}")
        print(f"  context length:   {rm.context_length}")
        print(f"  pricing:          {rm.pricing.get('raw', rm.pricing)}")
        print(f"  supported params: {rm.supported_parameters}")
        print(f"  availability:     {'available' if rm.available else rm.unavailable_reason}")
        print("-" * 80)
    print(f"Catalog snapshot:  {resolution['model_catalog_snapshot_path']}")
    print(f"Resolved models:   {resolution['resolved_models_path']}")
    return resolution
