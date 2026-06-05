"""Make ``src`` importable when scripts are run directly (python scripts/foo.py).

Also loads ``.env`` from the project root so local runs pick up secrets (e.g.
OPENROUTER_API_KEY) the same way Docker Compose does. Secrets stay in .env only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load .env if present (no-op if python-dotenv is missing or no .env exists).
try:
    from dotenv import load_dotenv

    _env = _ROOT / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass
