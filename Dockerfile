# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────
# EMA — single-image production build.
#
# One image hosts both the FastAPI backend and the pre-built React SPA:
# backend/main.py mounts frontend/dist and falls back to index.html for
# any non-API route, so the backend container *is* the web server.
#
# The BGE-M3 embedding model is NOT baked into the image.  It is mounted
# from the host at runtime: docker-compose mounts ./docker/models to
# /home/ema/.cache/huggingface (the runtime HOME), where
# backend/service/embedding_service.py's local_files_only=True load finds
# it.  The image stays model-free (~3GB); the model lives on the host and
# can be swapped without rebuilding.
#   To switch the model, change EMBEDDING_MODEL in .env and prepare the
#   matching HF cache under docker/models (see docs/deployment.md).
# ─────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=22

# ── Stage 1: build the React SPA ─────────────────────────────────────
FROM node:${NODE_VERSION}-alpine AS frontend-build
WORKDIR /app/frontend

# VITE_* vars are consumed by Vite at build time.  The frontend build must
# receive VITE_EMA_API_KEY here as a build-arg (docker-compose passes it via
# build.args) — the root .env is deliberately .dockerignored and is only
# injected into the *runtime* container, which Vite never sees.
ARG VITE_EMA_API_KEY=""

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ── Stage 2: backend runtime (serves API + built SPA) ────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_ENDPOINT=https://hf-mirror.com

COPY requirements.txt ./
# torch==<ver>+cpu is published only on PyTorch's own CPU index — the
# +cpu local-version suffix does not exist on pypi.org, so the two
# installs are split: torch comes from PyTorch's index (--no-deps, so
# pip does not resolve its dependency tree there), everything else from
# PyPI.  A single --extra-index-url would pollute the resolution of
# shared packages (e.g. charset-normalizer) with the PyTorch index's old
# snapshots, so the split must stay separate.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --no-deps "torch==2.13.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# requirements 走 pip 镜像源（默认清华镜像，与本机 pip 全局配置一致；
# 直连 files.pythonhosted.org 在此网络下不稳定，曾导致构建读超时）。
# 需要时可用 --build-arg PIP_INDEX_URL=https://pypi.org/simple 覆盖。
ARG PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    --index-url ${PIP_INDEX_URL} \
    -r requirements.txt

# psycopg's pure-Python implementation loads libpq at runtime; the
# python:*-slim image has no system libpq, so AsyncPostgresSaver would
# silently fall back to InMemorySaver (checkpoints lost on restart) unless
# the runtime library is installed.  libpq5 alone suffices — no headers.
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# ── Embedding model: mounted at runtime, not baked ───────────────────
# The model lives on the host under docker/models and is mounted to
# /home/ema/.cache/huggingface by docker-compose (runtime HOME is
# /home/ema, uid 10001).  See the header comment and docs/deployment.md.

COPY alembic.ini ./
COPY migrations/ migrations/
COPY backend/ backend/
# frontend/public assets (favicon) are copied verbatim into dist by Vite.
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Run as non-root.
RUN useradd --create-home --uid 10001 ema
USER ema

EXPOSE 8000

# Single process: EMA is single-instance by design (see docs/deployment.md)
# — in-memory circuit breakers / usage buffer / auto-memory throttle are
# process-local state, so --workers 1 is correct, not a fallback.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
