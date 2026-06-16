"""Direct OpenAI API client.

Used when ``LLM_PROVIDER=openai``. Reads ``OPENAI_API_KEY`` from the environment
(never logged). Falls back to the same deterministic offline mock as
:class:`OpenRouterClient` when the key is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from benchmark.evaluators.cost import estimate_cost_usd
from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.schemas import LLMResponse

_log = get_logger("benchmark.llm.openai")

# Models that only accept the API default temperature (omit the parameter).
_TEMPERATURE_LOCKED_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_custom_temperature(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return not any(mid.startswith(p) for p in _TEMPERATURE_LOCKED_PREFIXES)


class OpenAIClient(LLMClient):
    """LLM client backed by the official OpenAI API via the OpenAI SDK."""

    def __init__(
        self,
        *,
        requested_model: str,
        resolved_model: str,
        provider_family: str = "openai",
        benchmark_tier: str | None = None,
        pricing: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 4,
    ) -> None:
        self.requested_model = requested_model
        self.resolved_model = resolved_model
        self.provider_family = provider_family
        self.benchmark_tier = benchmark_tier
        self.pricing = pricing or {}
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self._client = None
        self._offline = not bool(self._api_key)
        if self._offline:
            _log.warning(
                "OPENAI_API_KEY missing - OpenAIClient running in offline "
                "deterministic mock mode (model=%s).",
                resolved_model,
            )
        else:
            self._client = self._build_sdk_client()

    def _build_sdk_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAIClient") from exc

        return OpenAI(base_url=self.base_url, api_key=self._api_key)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        metadata = metadata or {}
        want_json = bool(metadata.get("json_output"))

        if self._offline:
            return self._mock_generate(messages, want_json=want_json)

        # gpt-5.x requires max_completion_tokens and rejects custom temperature.
        params: dict[str, Any] = {
            "model": self.resolved_model,
            "messages": messages,
            "max_completion_tokens": self.max_output_tokens,
        }
        if _supports_custom_temperature(self.resolved_model):
            params["temperature"] = self.temperature
        else:
            _log.debug(
                "Omitting temperature for model %s (API default only).",
                self.resolved_model,
            )
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if want_json:
            params["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        completion = self._call_with_retry(params)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._parse_completion(completion, latency_ms)

    def _call_with_retry(self, params: dict[str, Any]) -> Any:
        from openai import APIError, APITimeoutError, RateLimitError

        attempt = 0
        delay = 1.0
        while True:
            attempt += 1
            try:
                return self._client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError) as exc:
                if attempt > self.max_retries:
                    _log.error("OpenAI transient failure after %d attempts: %s", attempt, exc)
                    raise
                _log.warning(
                    "Transient OpenAI error (attempt %d/%d), backing off %.1fs: %s",
                    attempt,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    _log.error(
                        "OpenAI invalid request (status=%s), not retrying: %s",
                        status,
                        exc,
                    )
                    raise
                if attempt > self.max_retries:
                    raise
                _log.warning("OpenAI server error (attempt %d), backing off: %s", attempt, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _parse_completion(self, completion: Any, latency_ms: float) -> LLMResponse:
        raw = completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        usage = raw.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details", {}) or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        completion_details = usage.get("completion_tokens_details", {}) or {}
        reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)

        cost = estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=reasoning_tokens,
            pricing=self.pricing,
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            raw_response=raw,
            requested_model=self.requested_model,
            resolved_model=raw.get("model") or self.resolved_model,
            provider_family=self.provider_family,
            estimated_cost_usd=cost,
        )

    def _mock_generate(self, messages: list[dict[str, Any]], *, want_json: bool) -> LLMResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        digest = hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:8]
        if want_json:
            text = json.dumps(
                {
                    "final_answer": f"[offline-mock:{digest}]",
                    "used_memory_ids": [],
                    "reasoning_summary": "Offline deterministic mock response.",
                }
            )
        else:
            text = f"[offline-mock:{digest}] {last_user[:120]}"
        approx_in = max(len(last_user) // 4, 1)
        approx_out = max(len(text) // 4, 1)
        return LLMResponse(
            text=text,
            input_tokens=approx_in,
            output_tokens=approx_out,
            latency_ms=0.0,
            requested_model=self.requested_model,
            resolved_model=self.resolved_model,
            provider_family=self.provider_family,
            estimated_cost_usd=0.0,
            model_unavailable_reason="offline_mock_no_api_key",
            raw_response={"offline": True},
        )
