# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────
# EMA — single-image production build.
#
# One image hosts both the FastAPI backend and the pre-built React SPA:
# backend/main.py mounts frontend/dist and falls back to index.html for
# any non-API route, so the backend container *is* the web server.
#
# The local BGE-M3 embedding model is baked into the image at build time
# (backend/service/embedding_service.py loads with local_files_only=True,
# so it must already be on disk).  Baking it here makes the runtime fully
# offline — no HF download at container start.
#   To switch the model, pass --build-arg EMBEDDING_MODEL=<model-id>.
#   To skip baking (e.g. mounting the model cache as a volume instead),
#   comment out the "Bake BGE-M3" step.
# ─────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=22
ARG EMBEDDING_MODEL=BAAI/bge-m3

# ── Stage 1: build the React SPA ─────────────────────────────────────
FROM node:${NODE_VERSION}-alpine AS frontend-build
WORKDIR /app/frontend

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

# psycopg / pygit2 / onnxruntime publish manylinux wheels; only git is
# needed by pygit2 at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

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
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 120 --retries 5 \
    -r requirements.txt

# ── Bake the embedding model into the image (offline runtime) ────────
# SentenceTransformer caches under $HOME/.cache/huggingface; HOME=/root,
# so the runtime's local_files_only=True load hits this cache directly.
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('${EMBEDDING_MODEL}')"

COPY agent/ agent/
COPY backend/ backend/
COPY --from=frontend-build /app/frontend/dist frontend/dist
COPY frontend/static frontend/static

# Run as non-root.
RUN useradd --create-home --uid 10001 ema
USER ema

EXPOSE 8000

# Single process: EMA is single-instance by design (see docs/deployment.md)
# — in-memory circuit breakers / usage buffer / auto-memory throttle are
# process-local state, so --workers 1 is correct, not a fallback.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
