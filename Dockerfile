# Anukriti ADK — container for the testing surface + ADK agent.
# Multi-stage, non-root, healthchecked. Runs fully offline-deterministic.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code.
COPY pyproject.toml README.md ./
COPY anukriti_adk ./anukriti_adk
COPY eval ./eval
COPY harness ./harness

# Non-root runtime user.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin anukriti
USER 10001

EXPOSE 8080

# Cloud Run sets $PORT; default to 8080 locally.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/healthz')" || exit 1

# Serve the authenticated testing surface. Set ANUKRITI_API_KEY at deploy time.
CMD ["sh", "-c", "uvicorn anukriti_adk.serve:app --host 0.0.0.0 --port ${PORT:-8080}"]
