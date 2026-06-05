"""Cost calculation from OpenRouter model-catalog pricing.

Pricing is NOT hardcoded. We use the per-token pricing fields captured from the
OpenRouter model catalog (saved in resolved_models.yaml). OpenRouter exposes
pricing as USD-per-token strings, e.g. ``{"prompt": "0.000003", "completion":
"0.000015", "input_cache_read": "0.0000003"}``.
"""

from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_pricing(catalog_pricing: dict[str, Any] | None) -> dict[str, float]:
    """Normalize OpenRouter catalog pricing into per-token floats.

    Returns a dict with keys: prompt, completion, cached_input, reasoning.
    """
    pricing = catalog_pricing or {}
    return {
        "prompt": _to_float(pricing.get("prompt")),
        "completion": _to_float(pricing.get("completion")),
        # OpenRouter uses input_cache_read for cached prompt tokens.
        "cached_input": _to_float(
            pricing.get("input_cache_read") or pricing.get("cached_input")
        ),
        # Reasoning tokens are typically billed at the completion rate unless a
        # dedicated field is present.
        "reasoning": _to_float(pricing.get("internal_reasoning"))
        or _to_float(pricing.get("completion")),
    }


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    pricing: dict[str, Any] | None = None,
) -> float:
    """Estimate request cost in USD from per-token pricing.

    ``pricing`` may be raw OpenRouter catalog pricing or already-normalized
    per-token floats. Cached input tokens are billed at the cached rate and are
    treated as a subset of input tokens (so non-cached input is billed at the
    prompt rate).
    """
    if pricing is None:
        return 0.0
    # Detect already-normalized pricing.
    if {"prompt", "completion"} <= set(pricing.keys()) and "input_cache_read" not in pricing:
        rates = {
            "prompt": _to_float(pricing.get("prompt")),
            "completion": _to_float(pricing.get("completion")),
            "cached_input": _to_float(pricing.get("cached_input")),
            "reasoning": _to_float(pricing.get("reasoning")) or _to_float(pricing.get("completion")),
        }
    else:
        rates = normalize_pricing(pricing)

    non_cached_input = max(input_tokens - cached_input_tokens, 0)
    cost = (
        non_cached_input * rates["prompt"]
        + cached_input_tokens * rates["cached_input"]
        + output_tokens * rates["completion"]
        + reasoning_tokens * rates["reasoning"]
    )
    return round(cost, 8)
