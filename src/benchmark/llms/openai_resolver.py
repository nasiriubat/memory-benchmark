"""Resolve model slots for direct OpenAI API usage.

When ``LLM_PROVIDER=openai``, model ids come from each slot's ``openai_model``
field, the ``OPENAI_MODEL`` env default, or ``concrete_fallback_model`` with any
``openai/`` prefix stripped. No OpenRouter catalog fetch is performed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from benchmark.evaluators.cost import normalize_pricing
from benchmark.llms.model_resolver import ResolvedModel
from benchmark.logging import get_logger
from benchmark.utils import ensure_dir, utcnow_iso

_log = get_logger("benchmark.llm.openai_resolver")


def _fallback_pricing(pricing_config: dict[str, Any], provider_family: str) -> dict[str, Any]:
    table = (pricing_config or {}).get("pricing", {}) or {}
    family = table.get(provider_family, {}) or table.get("default", {}) or {}
    per_1m = float(family.get("input_per_1m", 0.0) or 0.0)
    out_1m = float(family.get("output_per_1m", 0.0) or 0.0)
    cached_1m = float(family.get("cached_input_per_1m", 0.0) or 0.0)
    return normalize_pricing(
        {
            "prompt": str(per_1m / 1_000_000),
            "completion": str(out_1m / 1_000_000),
            "input_cache_read": str(cached_1m / 1_000_000),
        }
    )


def _resolve_openai_model_id(slot: str, spec: dict[str, Any]) -> str:
    if spec.get("openai_model"):
        return str(spec["openai_model"])
    env_default = os.environ.get("OPENAI_MODEL", "").strip()
    if env_default:
        return env_default
    fallback = str(spec.get("concrete_fallback_model", "")).strip()
    if fallback.startswith("openai/"):
        return fallback[len("openai/") :]
    if fallback:
        return fallback
    _log.warning("[%s] no openai_model configured; defaulting to gpt-4o", slot)
    return "gpt-4o"


class OpenAIModelResolver:
    """Resolve configured model slots for the direct OpenAI API."""

    def __init__(
        self,
        *,
        results_root: Path,
        base_url: str | None = None,
        api_key: str | None = None,
        pricing_config: dict[str, Any] | None = None,
        fail_if_model_unavailable: bool = False,
    ) -> None:
        self.results_root = results_root
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.pricing_config = pricing_config or {}
        self.fail_if_model_unavailable = fail_if_model_unavailable
        self._offline = not bool(self._api_key)

    def resolve_all(self, run_id: str, models_config: dict[str, Any]) -> dict[str, Any]:
        snapshot_dir = ensure_dir(self.results_root / "model_snapshots" / run_id)
        slots = (models_config or {}).get("models", {})
        resolved: dict[str, ResolvedModel] = {}

        for slot, spec in slots.items():
            if not spec.get("enabled", False):
                continue
            provider_family = str(spec.get("provider_family", "openai"))
            if provider_family != "openai":
                resolved[slot] = ResolvedModel(
                    slot=slot,
                    provider_family=provider_family,
                    benchmark_tier=spec.get("benchmark_tier"),
                    requested_model=str(spec.get("requested_model", "")),
                    resolved_model="",
                    available=False,
                    unavailable_reason=(
                        f"LLM_PROVIDER=openai only supports provider_family=openai "
                        f"(got {provider_family})"
                    ),
                    temperature=float(spec.get("temperature", 0.0)),
                    max_output_tokens=int(spec.get("max_output_tokens", 4096)),
                    reasoning=spec.get("reasoning", {}) or {},
                )
                if self.fail_if_model_unavailable:
                    raise RuntimeError(f"[{slot}] {resolved[slot].unavailable_reason}")
                _log.warning("[%s] skipped: %s", slot, resolved[slot].unavailable_reason)
                continue

            model_id = _resolve_openai_model_id(slot, spec)
            rm = ResolvedModel(
                slot=slot,
                provider_family=provider_family,
                benchmark_tier=spec.get("benchmark_tier"),
                requested_model=str(spec.get("requested_model", model_id)),
                resolved_model=model_id,
                available=not self._offline,
                unavailable_reason=None if not self._offline else "OPENAI_API_KEY missing",
                pricing={"raw": {}, **_fallback_pricing(self.pricing_config, provider_family)},
                temperature=float(spec.get("temperature", 0.0)),
                max_output_tokens=int(spec.get("max_output_tokens", 4096)),
                reasoning=spec.get("reasoning", {}) or {},
            )
            if self._offline and self.fail_if_model_unavailable:
                raise RuntimeError(f"[{slot}] OPENAI_API_KEY missing")
            resolved[slot] = rm
            _log.info(
                "[%s] requested=%s resolved=%s available=%s",
                slot,
                rm.requested_model,
                rm.resolved_model,
                rm.available,
            )

        resolved_path = snapshot_dir / "resolved_models.yaml"
        payload = {
            "run_id": run_id,
            "llm_provider": "openai",
            "openai_base_url": self.base_url,
            "model_resolution_time": utcnow_iso(),
            "models": {slot: rm.to_dict() for slot, rm in resolved.items()},
        }
        resolved_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        _log.info("Saved resolved models to %s", resolved_path)

        return {
            "resolved": resolved,
            "model_catalog_snapshot_path": None,
            "resolved_models_path": str(resolved_path),
            "model_resolution_time": payload["model_resolution_time"],
            "openrouter_base_url": None,
            "llm_provider": "openai",
            "openai_base_url": self.base_url,
        }
