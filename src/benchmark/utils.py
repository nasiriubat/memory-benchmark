"""Shared utility helpers."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    """Generate a fresh, time-sortable, reproducible-friendly run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def resolve_run_id(explicit: str | None = None) -> str:
    """Resolve the run id from an explicit value, env var, or a fresh id."""
    if explicit:
        return explicit
    env = os.environ.get("RUN_ID")
    if env:
        return env
    return new_run_id()


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def project_root() -> Path:
    """Return the project root (two levels above this file's package)."""
    # src/benchmark/utils.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def results_root() -> Path:
    """Return the results directory, honoring an optional override env var."""
    override = os.environ.get("RESULTS_DIR")
    if override:
        return Path(override)
    return project_root() / "results"


def configs_root() -> Path:
    override = os.environ.get("CONFIGS_DIR")
    if override:
        return Path(override)
    return project_root() / "configs"


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if needed; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model text.

    Handles fenced code blocks and stray prose around the JSON. Returns None if
    no valid object can be parsed (callers must handle the fallback explicitly;
    we never silently swallow errors elsewhere).
    """
    if not text:
        return None
    # Try direct parse first.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Strip code fences.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Grab the first balanced-looking object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: str) -> set[str]:
    """Return the set of normalized tokens in a string."""
    return set(normalize_text(text).split())
