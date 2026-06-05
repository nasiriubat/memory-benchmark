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

# Copy only dependency metadata first to leverage Docker layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the project with UI + datasets extras so the same image serves both
# the benchmark runner services and the Streamlit UI service.
RUN uv pip install --system --no-cache ".[ui,datasets,memory-hindsight]"

# Copy the remainder of the project (configs/data/results are also mounted at runtime).
COPY . .

# Default command is overridden per-service in docker-compose.yml.
CMD ["python", "scripts/run_memory_system.py", "--memory-system", "stateless"]
