"""OpenRouter LLM client (the only live LLM client).

Uses the OpenAI Python SDK pointed at the OpenRouter base URL. API keys are read
from the environment and never logged. Unsupported parameters (per the model
snapshot's ``supported_parameters``) are stripped before the request and logged
as warnings. Transient failures are retried with exponential backoff; invalid
request errors (e.g. unsupported parameters) are NOT retried.

When ``OPENROUTER_API_KEY`` is absent the client runs in an offline deterministic
mock mode so the harness remains runnable (stateless + sample data) without keys.
The mock is clearly flagged via ``model_unavailable_reason``.
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

_log = get_logger("benchmark.llm.openrouter")


class OpenRouterClient(LLMClient):
    """LLM client backed by OpenRouter via the OpenAI SDK."""

    def __init__(
        self,
        *,
        requested_model: str,
        resolved_model: str,
        provider_family: str = "unknown",
        benchmark_tier: str | None = None,
        supported_parameters: list[str] | None = None,
        pricing: dict[str, Any] | None = None,
        provider_policy: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        reasoning: dict[str, Any] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        app_title: str | None = None,
        http_referer: str | None = None,
        max_retries: int = 4,
    ) -> None:
        self.requested_model = requested_model
        self.resolved_model = resolved_model
        self.provider_family = provider_family
        self.benchmark_tier = benchmark_tier
        self.supported_parameters = supported_parameters or []
        self.pricing = pricing or {}
        self.provider_policy = provider_policy or {}
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning or {}
        self.max_retries = max_retries

        self.base_url = base_url or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.app_title = app_title or os.environ.get("OPENROUTER_APP_TITLE", "memory-agent-benchmark")
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER", "")

        self._client = None
        self._offline = not bool(self._api_key)
        if self._offline:
            _log.warning(
                "OPENROUTER_API_KEY missing - OpenRouterClient running in offline "
                "deterministic mock mode (model=%s).",
                resolved_model,
            )
        else:
            self._client = self._build_client()

    # ------------------------------------------------------------------ build
    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency present in image
            raise RuntimeError("openai package is required for OpenRouterClient") from exc

        default_headers: dict[str, str] = {}
        if self.http_referer:
            default_headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            default_headers["X-Title"] = self.app_title
        # NOTE: api_key is passed to the SDK only; it is never logged.
        return OpenAI(
            base_url=self.base_url,
            api_key=self._api_key,
            default_headers=default_headers or None,
        )

    # --------------------------------------------------------------- generate
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

        params, stripped = self._build_params(messages, tools, want_json)
        if stripped:
            _log.warning(
                "Stripped unsupported parameters for model %s: %s",
                self.resolved_model,
                ", ".join(sorted(stripped)),
            )

        start = time.perf_counter()
        completion = self._call_with_retry(params)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._parse_completion(completion, latency_ms, stripped)

    # --------------------------------------------------------- param building
    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        want_json: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build request params.

        Standard OpenAI-compatible fields go at the top level; OpenRouter
        extensions (reasoning, provider routing, usage accounting) go in
        ``extra_body`` (the OpenAI SDK rejects unknown top-level kwargs).
        Parameters the model does not support (per the catalog's
        ``supported_parameters``) are stripped and recorded.
        """
        supported = set(self.supported_parameters)

        def supports(name: str) -> bool:
            # When we don't know the supported set, optimistically include.
            return (not supported) or (name in supported)

        params: dict[str, Any] = {"model": self.resolved_model, "messages": messages}
        stripped: list[str] = []

        # temperature (None = configured "do not send": some endpoints, e.g.
        # Anthropic Claude Opus 4.8, reject the parameter as deprecated even
        # while the catalog still lists it as supported)
        if self.temperature is not None and supports("temperature"):
            params["temperature"] = self.temperature
        else:
            stripped.append("temperature")

        # max output tokens (use whichever variant the model supports)
        if not supported or "max_tokens" in supported:
            params["max_tokens"] = self.max_output_tokens
        elif "max_completion_tokens" in supported:
            params["max_completion_tokens"] = self.max_output_tokens
        else:
            stripped.append("max_tokens")

        # tools (only when the model supports them)
        if tools:
            if supports("tools"):
                params["tools"] = tools
                params["tool_choice"] = "auto"
            else:
                stripped.append("tools")

        # structured JSON output (only when supported)
        if want_json:
            if supports("response_format") or supports("structured_outputs"):
                params["response_format"] = {"type": "json_object"}
            else:
                stripped.append("response_format")

        # ---- OpenRouter extensions live in extra_body ----
        extra_body: dict[str, Any] = {}

        # reasoning (OpenRouter-specific; NOT a top-level OpenAI kwarg)
        if self.reasoning.get("enabled"):
            if supports("reasoning"):
                extra_body["reasoning"] = {"effort": self.reasoning.get("effort", "medium")}
            else:
                stripped.append("reasoning")

        provider_pref: dict[str, Any] = {}
        if "allow_fallbacks" in self.provider_policy:
            provider_pref["allow_fallbacks"] = self.provider_policy["allow_fallbacks"]
        if self.provider_policy.get("require_parameters") is not None:
            provider_pref["require_parameters"] = self.provider_policy["require_parameters"]
        if self.provider_policy.get("data_collection"):
            provider_pref["data_collection"] = self.provider_policy["data_collection"]
        if self.provider_policy.get("sort"):
            provider_pref["sort"] = self.provider_policy["sort"]
        # Exclude / prefer specific upstream providers (e.g. skip Amazon Bedrock
        # when it is disabled for the account and returns "organization disabled").
        if self.provider_policy.get("ignore"):
            provider_pref["ignore"] = self.provider_policy["ignore"]
        if self.provider_policy.get("order"):
            provider_pref["order"] = self.provider_policy["order"]
        if provider_pref:
            extra_body["provider"] = provider_pref

        # Ask OpenRouter to include usage/cost accounting in the response.
        extra_body["usage"] = {"include": True}

        params["extra_body"] = extra_body
        return params, stripped

    # ----------------------------------------------------------------- retry
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
                    _log.error("OpenRouter transient failure after %d attempts: %s", attempt, exc)
                    raise
                _log.warning(
                    "Transient OpenRouter error (attempt %d/%d), backing off %.1fs: %s",
                    attempt,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                # 4xx (except 429) are invalid-request errors: do NOT retry.
                if status is not None and 400 <= status < 500 and status != 429:
                    _log.error(
                        "OpenRouter invalid request (status=%s), not retrying: %s",
                        status,
                        exc,
                    )
                    raise
                if attempt > self.max_retries:
                    raise
                _log.warning("OpenRouter server error (attempt %d), backing off: %s", attempt, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    # --------------------------------------------------------------- parsing
    def _parse_completion(
        self, completion: Any, latency_ms: float, stripped: list[str]
    ) -> LLMResponse:
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
            supported_parameters=self.supported_parameters,
            stripped_parameters=stripped,
            estimated_cost_usd=cost,
            openrouter_generation_id=raw.get("id"),
            openrouter_request_id=raw.get("id"),
        )

    # ------------------------------------------------------------------ mock
    def _mock_generate(self, messages: list[dict[str, Any]], *, want_json: bool) -> LLMResponse:
        """Deterministic offline response so the harness runs without keys."""
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
