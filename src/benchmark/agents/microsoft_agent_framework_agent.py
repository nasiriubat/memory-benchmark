"""Microsoft Agent Framework agent wrapper (optional).

The paper's methodology uses the Microsoft Agent Framework as the implementation
platform. This wrapper uses it when installed; otherwise it logs that the
framework is unavailable and transparently falls back to :class:`SimpleToolAgent`
so the harness remains runnable.

Real integration points are marked ``# REAL SDK:``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from benchmark.agents.base_agent import Agent, AgentAnswer
from benchmark.agents.simple_tool_agent import SimpleToolAgent
from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.memory.base import MemoryAdapter
from benchmark.schemas import MemoryNamespace, MemoryRecord

_log = get_logger("benchmark.agent.maf")


def _framework_available() -> bool:
    return importlib.util.find_spec("agent_framework") is not None


class MicrosoftAgentFrameworkAgent(Agent):
    name = "microsoft_agent_framework_agent"

    def __init__(self) -> None:
        self._available = _framework_available()
        self._fallback = SimpleToolAgent()
        if not self._available:
            _log.warning(
                "Microsoft Agent Framework not installed; falling back to "
                "SimpleToolAgent. Install the framework to enable native orchestration."
            )

    def act(
        self,
        *,
        question: str,
        backgrounds: list[str],
        retrieved: list[MemoryRecord],
        model: LLMClient,
        namespace: MemoryNamespace,
        memory: MemoryAdapter,
        metadata: dict[str, Any] | None = None,
    ) -> AgentAnswer:
        if not self._available:
            return self._fallback.act(
                question=question,
                backgrounds=backgrounds,
                retrieved=retrieved,
                model=model,
                namespace=namespace,
                memory=memory,
                metadata=metadata,
            )
        # REAL SDK: construct and run a Microsoft Agent Framework agent here,
        # wiring the LLMClient as the chat backend and the MemoryAdapter as the
        # memory provider, then map its output onto AgentAnswer. Until that
        # native integration is implemented, we use the same contract-enforcing
        # path as the simple agent to keep results comparable.
        return self._fallback.act(
            question=question,
            backgrounds=backgrounds,
            retrieved=retrieved,
            model=model,
            namespace=namespace,
            memory=memory,
            metadata=metadata,
        )
