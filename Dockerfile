# Python 3.12 base image for the memory-agent-benchmark harness.
# Secrets are never baked into the image. All API keys are provided at runtime
# through environment variables (see .env.example) and never copied in.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install uv for fast, reproducible dependency installation (falls back to pip).
RUN pip install --no-cache-dir uv

# --- Dependency layer (cached across source edits) ----------------------------
# Copy only dependency metadata plus an empty package skeleton, then install all
# third-party dependencies + extras. Because this layer does NOT depend on the
# real source, editing code in src/ does not invalidate it: rebuilds reuse the
# cached dependency install instead of re-downloading the whole scientific stack.
COPY pyproject.toml README.md ./
RUN mkdir -p src/benchmark && touch src/benchmark/__init__.py
RUN uv pip install --system --no-cache ".[ui,datasets,memory-hindsight,memory]"

# --- Source layer (cheap; re-runs on every code edit) -------------------------
# The real source overrides the empty skeleton at import time via PYTHONPATH
# (/app/src is on sys.path), so no project reinstall is needed after edits.
COPY src ./src

# Copy the remainder of the project (configs/data/results are also mounted at runtime).
COPY . .

# Default command is overridden per-service in docker-compose.yml.
CMD ["python", "scripts/run_memory_system.py", "--memory-system", "stateless"]
