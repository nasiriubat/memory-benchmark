"""ModelResolver: resolve model aliases once per run and snapshot the catalog.

Reproducibility is the core concern. A live "~latest" alias may change between
runs, so we:

1. Fetch the OpenRouter model catalog (GET /models).
2. Save the full catalog snapshot.
3. Resolve each configured model slot's alias to a concrete model id ONCE.
4. Save the resolved mapping (with pricing + supported parameters) so the whole
   run uses the same concrete ids, and both requested + resolved ids are logged.

When ``OPENROUTER_API_KEY`` is absent, resolution runs offline: aliases fall back
to ``concrete_fallback_model`` and the catalog snapshot is empty. This keeps the
harness runnable without keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmark.evaluators.cost import normalize_pricing
from benchmark.logging import get_logger
from benchmark.utils import ensure_dir, utcnow_iso

_log = get_logger("benchmark.llm.resolver")

_MODELS_ENDPOINT = "/models"


@dataclass
class ResolvedModel:
    """A configured model slot resolved to a concrete OpenRouter model id."""

    slot: str
    provider_family: str
    benchmark_tier: str | None
    requested_model: str
    resolved_model: str
    available: bool
    unavailable_reason: str | None = None
    context_length: int | None = None
    supported_parameters: list[str] = field(default_factory=list)
    pricing: dict[str, Any] = field(default_factory=dict)
    catalog_created_at: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 4096
    reasoning: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "provider_family": self.provider_family,
            "benchmark_tier": self.benchmark_tier,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "context_length": self.context_length,
            "supported_parameters": self.supported_parameters,
            "pricing": self.pricing,
            "catalog_created_at": self.catalog_created_at,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.reasoning,
        }


class ModelResolver:
    """Resolve configured model slots against the OpenRouter catalog."""

    def __init__(
        self,
        *,
        results_root: Path,
        base_url: str | None = None,
        api_key: str | None = None,
        provider_policy: dict[str, Any] | None = None,
        fail_if_model_unavailable: bool = False,
    ) -> None:
        self.results_root = results_root
        self.base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.provider_policy = provider_policy or {}
        self.fail_if_model_unavailable = fail_if_model_unavailable
        self._offline = not bool(self._api_key)

    # --------------------------------------------------------------- catalog
    def fetch_catalog(self) -> dict[str, Any]:
        """Fetch the full OpenRouter model catalog."""
        if self._offline:
            _log.warning("No OPENROUTER_API_KEY - skipping live catalog fetch (offline).")
            return {"data": [], "_offline": True}
        import httpx

        url = self.base_url.rstrip("/") + _MODELS_ENDPOINT
        headers = {"Authorization": f"Bearer {self._api_key}"}
        _log.info("Fetching OpenRouter model catalog from %s", url)
        resp = httpx.get(url, headers=headers, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------- resolve
    def resolve_all(self, run_id: str, models_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve every enabled model slot. Returns a run-metadata dict."""
        snapshot_dir = ensure_dir(self.results_root / "model_snapshots" / run_id)
        catalog = self.fetch_catalog()
        catalog_path = snapshot_dir / "openrouter_models.json"
        catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        _log.info("Saved model catalog snapshot to %s", catalog_path)

        catalog_index = self._index_catalog(catalog)
        catalog_created_at = str(catalog.get("created_at") or "")

        slots = (models_config or {}).get("models", {})
        resolved: dict[str, ResolvedModel] = {}
        for slot, spec in slots.items():
            if not spec.get("enabled", False):
                continue
            resolved[slot] = self._resolve_slot(slot, spec, catalog_index, catalog_created_at)

        resolved_path = snapshot_dir / "resolved_models.yaml"
        payload = {
            "run_id": run_id,
            "openrouter_base_url": self.base_url,
            "model_resolution_time": utcnow_iso(),
            "models": {slot: rm.to_dict() for slot, rm in resolved.items()},
        }
        resolved_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        _log.info("Saved resolved models to %s", resolved_path)

        return {
            "resolved": resolved,
            "model_catalog_snapshot_path": str(catalog_path),
            "resolved_models_path": str(resolved_path),
            "model_resolution_time": payload["model_resolution_time"],
            "openrouter_base_url": self.base_url,
        }

    def _resolve_slot(
        self,
        slot: str,
        spec: dict[str, Any],
        catalog_index: dict[str, dict[str, Any]],
        catalog_created_at: str,
    ) -> ResolvedModel:
        requested = str(spec.get("requested_model", ""))
        fallback = str(spec.get("concrete_fallback_model", ""))
        provider_family = str(spec.get("provider_family", "unknown"))
        use_alias = bool(spec.get("use_latest_alias", False))

        rm = ResolvedModel(
            slot=slot,
            provider_family=provider_family,
            benchmark_tier=spec.get("benchmark_tier"),
            requested_model=requested,
            resolved_model=fallback,
            available=False,
            catalog_created_at=catalog_created_at or None,
            temperature=float(spec.get("temperature", 0.0)),
            max_output_tokens=int(spec.get("max_output_tokens", 4096)),
            reasoning=spec.get("reasoning", {}) or {},
        )

        # Strip a leading "~" alias marker for lookup.
        alias_target = requested[1:] if requested.startswith("~") else requested

        concrete: str | None = None
        if use_alias and alias_target in catalog_index:
            concrete = alias_target
        elif use_alias and self._offline:
            _log.info("[%s] offline - using concrete fallback %s", slot, fallback)
        elif use_alias:
            concrete = self._probe_resolve(alias_target)

        if concrete is None and fallback in catalog_index:
            concrete = fallback
        elif concrete is None and not self._offline and fallback:
            # Fallback may still be valid even if not in our index parsing.
            concrete = fallback if self._probe_resolve(fallback) else None

        if concrete is None:
            # Offline or unresolved: trust the configured fallback id.
            concrete = fallback or alias_target

        entry = catalog_index.get(concrete, {})
        if entry:
            rm.available = True
            rm.resolved_model = entry.get("id", concrete)
            rm.context_length = entry.get("context_length")
            rm.supported_parameters = list(entry.get("supported_parameters", []) or [])
            rm.pricing = self._extract_pricing(entry)
        elif self._offline:
            # Offline: accept the fallback id without catalog validation.
            rm.available = True
            rm.resolved_model = concrete
            rm.unavailable_reason = None
        else:
            rm.available = False
            rm.unavailable_reason = f"model '{concrete}' not found in OpenRouter catalog"
            if self.fail_if_model_unavailable:
                raise RuntimeError(f"[{slot}] {rm.unavailable_reason}")
            _log.warning("[%s] unavailable: %s - will be skipped", slot, rm.unavailable_reason)

        _log.info(
            "[%s] requested=%s resolved=%s available=%s",
            slot,
            requested,
            rm.resolved_model,
            rm.available,
        )
        return rm

    def _probe_resolve(self, model_id: str) -> str | None:
        """Tiny probe request to resolve a model id; returns response.model.

        Only used when the catalog does not already expose the concrete target.
        """
        if self._offline or not model_id:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self._api_key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return resp.model or model_id
        except Exception as exc:  # noqa: BLE001 - log and report unavailability
            _log.warning("Probe resolution failed for %s: %s", model_id, exc)
            return None

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _index_catalog(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for entry in catalog.get("data", []) or []:
            model_id = entry.get("id")
            if model_id:
                index[model_id] = entry
        return index

    @staticmethod
    def _extract_pricing(entry: dict[str, Any]) -> dict[str, Any]:
        pricing = entry.get("pricing", {}) or {}
        normalized = normalize_pricing(pricing)
        # Keep both raw catalog pricing and normalized per-token rates.
        return {"raw": pricing, **normalized}
