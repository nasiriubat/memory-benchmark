"""SimpleToolAgent: default single-pass agent enforcing the answer contract.

Builds a prompt that includes retrieved memory, instructs the model to return a
strict JSON object ``{final_answer, used_memory_ids, reasoning_summary}``, and
parses it. Falls back to treating raw text as the final answer if JSON parsing
fails (logged, never silently hidden).
"""

from __future__ import annotations

from typing import Any

from benchmark.agents.base_agent import Agent, AgentAnswer
from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.memory.base import MemoryAdapter
from benchmark.schemas import MemoryNamespace, MemoryRecord
from benchmark.utils import extract_json_object

_log = get_logger("benchmark.agent.simple")

_SYSTEM_PROMPT = (
    "You are a careful task-solving agent operating in an iterative, "
    "interdependent workflow. Use any provided MEMORY to stay consistent with "
    "earlier decisions, feedback, and constraints. "
    "Respond ONLY with a single JSON object of the form: "
    '{"final_answer": "<your answer>", "used_memory_ids": ["<ids you actually '
    'used>"], "reasoning_summary": "<one short public sentence; no private '
    'chain-of-thought>"}.'
)


class SimpleToolAgent(Agent):
    name = "simple_tool_agent"

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
        metadata = metadata or {}
        prompt = self._build_prompt(question, backgrounds, retrieved)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = model.generate(messages, tools=None, metadata={"json_output": True})

        parsed = extract_json_object(response.text)
        if parsed is None:
            _log.warning(
                "Agent answer was not valid JSON; using raw text as final_answer."
            )
            return AgentAnswer(
                final_answer=response.text.strip(),
                used_memory_ids=[],
                reasoning_summary="",
                llm=response,
                tool_call_count=len(response.tool_calls),
                retrieved=retrieved,
            )

        used = parsed.get("used_memory_ids") or []
        if not isinstance(used, list):
            used = []
        return AgentAnswer(
            final_answer=str(parsed.get("final_answer", "")).strip(),
            used_memory_ids=[str(x) for x in used],
            # Only store the short public reasoning summary.
            reasoning_summary=str(parsed.get("reasoning_summary", ""))[:500],
            llm=response,
            tool_call_count=len(response.tool_calls),
            retrieved=retrieved,
        )

    @staticmethod
    def _build_prompt(
        question: str, backgrounds: list[str], retrieved: list[MemoryRecord]
    ) -> str:
        parts: list[str] = []
        if backgrounds:
            parts.append("BACKGROUND:\n" + "\n".join(f"- {b}" for b in backgrounds))
        if retrieved:
            mem_lines = [
                f"- (id={r.id}) {r.text}" for r in retrieved if r.text
            ]
            if mem_lines:
                parts.append("MEMORY (may be relevant to earlier steps):\n" + "\n".join(mem_lines))
        parts.append("TASK:\n" + question)
        return "\n\n".join(parts)
