# Data Quality Auto-Fixing & Validation Agent
# Build:  docker compose build     Run:  docker compose up
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    APP_HOST=0.0.0.0 \
    APP_PORT=8100

WORKDIR /app

# Install dependencies first so Docker caches this layer between code changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[all-providers]"

# The sample CSVs ship inside the package (dq_agent/resources).

# Run as a non-root user; /app/data is the only writable location (SQLite + settings.json).
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/api/health').status==200 else 1)"

CMD ["uvicorn", "dq_agent.api.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8100"]
