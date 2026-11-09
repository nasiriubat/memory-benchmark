"""Logging utilities: rich console logging + structured JSONL event logging.

Two distinct concerns live here:

* :func:`get_logger` returns a configured ``logging.Logger`` using ``rich`` for
  human-readable console output.
* :class:`EventLogger` writes machine-readable JSONL event streams, one event
  per line, to the per-namespace path required by the spec.

Exceptions are never silently swallowed; callers log them with context via the
``error`` event type and the rich logger.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from benchmark.schemas import MemoryNamespace
from benchmark.utils import ensure_dir, utcnow_iso

_CONFIGURED = False


def get_logger(name: str = "benchmark") -> logging.Logger:
    """Return a rich-configured logger (idempotent)."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=False, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)


# Canonical event types (spec). Used for validation/documentation.
EVENT_TYPES = frozenset(
    {
        "run_started",
        "run_finished",
        "task_started",
        "task_finished",
        "subtask_started",
        "subtask_finished",
        "prompt_built",
        "memory_retrieved",
        "memory_written",
        "memory_updated",
        "llm_response",
        "evaluator_score",
        "error",
    }
)


class EventLogger:
    """Append-only JSONL event logger scoped to a namespace path.

    Every event line carries the full envelope: timestamp, run_id, benchmark,
    domain, memory_system, model, task_id, subtask_id, iteration, seed, payload.
    """

    def __init__(self, path: Path, namespace: MemoryNamespace) -> None:
        self._path = path
        self._ns = namespace
        self._lock = threading.Lock()
        ensure_dir(path.parent)
        self._log = get_logger("benchmark.events")

    @property
    def path(self) -> Path:
        return self._path

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        iteration: int = 0,
    ) -> None:
        """Write a single event line."""
        if event_type not in EVENT_TYPES:
            # Do not hide this: warn loudly but still record the event.
            self._log.warning("Unknown event type emitted: %s", event_type)
        event = {
            "timestamp": utcnow_iso(),
            "event_type": event_type,
            "run_id": self._ns.run_id,
            "benchmark": self._ns.benchmark,
            "domain": self._ns.domain,
            "memory_system": self._ns.memory_system,
            "model": self._ns.model,
            "task_id": task_id if task_id is not None else self._ns.task_id,
            "subtask_id": subtask_id,
            "iteration": iteration,
            "seed": self._ns.seed,
            "payload": payload or {},
        }
        line = json.dumps(event, default=str)
        # Cloud-synced filesystems (e.g. OneDrive) intermittently reject the
        # append with EAGAIN/EDEADLK ("Resource deadlock avoided") while their own
        # sync process holds the file. Retry briefly, then drop this one diagnostic
        # line rather than crash the run. IMPORTANT: only the file write is under
        # the lock; the backoff sleep is OUTSIDE it so a stall on one thread does
        # not block the other worker threads (that throttled the whole run).
        for attempt in range(4):
            try:
                with self._lock:
                    with self._path.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                return
            except OSError as exc:
                if attempt == 3:
                    self._log.warning("event log line dropped (%s): %s", self._path, exc)
                    return
                time.sleep(0.02 * (attempt + 1))

    def error(self, message: str, *, context: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Convenience: log an error to both console and the JSONL stream."""
        payload = {"message": message, "context": context or {}}
        self._log.error("%s | context=%s", message, context or {})
        self.emit("error", payload, **kwargs)
