"""Base agent interface and shared answer contract.

Agents orchestrate one subtask: build a prompt (optionally with retrieved
memory), call the LLM, and parse a machine-readable answer. The agent must
return a structured answer:

    {
      "final_answer": "...",
      "used_memory_ids": ["..."],
      "reasoning_summary": "short public summary only"
    }

Only the short public reasoning summary is stored. Private chain-of-thought is
never stored.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from benchmark.llms.base import LLMClient
from benchmark.memory.base import MemoryAdapter
from benchmark.schemas import LLMResponse, MemoryNamespace, MemoryRecord


@dataclass
class AgentAnswer:
    """Structured agent answer for a single subtask."""

    final_answer: str
    used_memory_ids: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    llm: LLMResponse | None = None
    tool_call_count: int = 0
    retrieved: list[MemoryRecord] = field(default_factory=list)


class Agent(abc.ABC):
    """Abstract agent."""

    name: str = "base"

    @abc.abstractmethod
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
        """Produce an answer for a single subtask."""
        raise NotImplementedError
