"""EMA — Engineering Memory Agent — FastAPI Application."""

from __future__ import annotations

import os

# ── Offline mode & BLAS env vars — MUST be set BEFORE any import that
# may transitively pull in transformers / huggingface_hub / numpy / torch.
# langchain-core → transformers → huggingface_hub reads HF_HUB_OFFLINE
# at import time; setting it later has no effect.
for _k, _v in {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ[_k] = _v

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

# ── Windows: psycopg 3 (used by AsyncPostgresSaver) requires
#    SelectorEventLoop — ProActorEventLoop is incompatible.
#    uvicorn creates its own event loop after importing this module,
#    so set_event_loop here has no lasting effect.  On Windows the
#    checkpointer falls back to InMemorySaver — acceptable for dev.
#    Production should run in a Linux container where ProactorEventLoop
#    doesn't exist and psycopg async works natively.

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sqlalchemy import text

from backend.api.router import api_router
from backend.api.ratelimit import RateLimitMiddleware
from backend.db import close_db, get_session_factory
from backend.db.schema import init_db
from backend.shared.config import config, validate_config
from backend.shared.runtime_metrics import MetricsMiddleware, render_metrics

# Paths relative to backend/main.py
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure logging, validate config, init DB, start patrol."""

    # ── Logging from LOG_LEVEL + LOG_FORMAT (json enables structured output
    #    for log collection; text keeps the default human-readable lines).
    from backend.shared.logging_config import setup_logging

    setup_logging()

    # ── Fail fast on invalid configuration instead of mid-request errors or
    #    silently invalid schedules (e.g. PATROL_DAILY_HOUR=25).
    _problems = validate_config()
    if _problems:
        raise RuntimeError("Invalid configuration:\n- " + "\n- ".join(_problems))

    await init_db()

    # ── LLM usage tracing flusher ───────────────────────────────────
    # Drains the in-memory usage buffer into ``llm_usage`` every
    # USAGE_FLUSH_INTERVAL_SECONDS; flushed once more on shutdown so no
    # buffered rows are lost on a clean exit.
    _usage_task: asyncio.Task[None] | None = None
    if config.usage_enabled:
        from backend.service.usage import usage_flusher_loop

        _usage_task = asyncio.create_task(usage_flusher_loop())

    # Register connectors at startup.  Connectors missing required
    # configuration (API keys, etc.) are still registered but flagged
    # as "pending" so the frontend can show their status.  Gated on
    # ``connectors_active`` (default off, ADR-011) — with the flag unset the
    # connector/webhook routes are not mounted either, so this registration
    # would be dead work.
    if config.connectors_active:
        try:
            from backend.connectors.registry import register_connector
            from backend.connectors.ci import CIConnector
            from backend.connectors.feishu import FeishuConnector
            from backend.connectors.pingcode import PingCodeConnector

            _pingcode_secret = os.getenv("WEBHOOK_PINGCODE_SECRET", "")
            register_connector(
                "pingcode",
                PingCodeConnector(),
                status="active" if _pingcode_secret else "pending",
            )
            _ci_secret = os.getenv("WEBHOOK_CI_SECRET", "")
            register_connector(
                "ci",
                CIConnector(),
                status="active" if _ci_secret else "pending",
            )
            _feishu_secret = os.getenv("WEBHOOK_FEISHU_SECRET", "")
            register_connector(
                "feishu",
                FeishuConnector(),
                status="active" if _feishu_secret else "pending",
            )
        except Exception:

            logging.getLogger(__name__).warning(
                "Failed to register connectors — webhook endpoints will be unavailable."
            )

    # Create checkpoint table for conversation persistence
    try:
        from backend.service.agent_service import _setup_checkpointer

        await _setup_checkpointer()
    except Exception as _exc:

        logging.getLogger(__name__).warning(
            "Failed to setup checkpointer — conversations will be ephemeral. "
            "Error: %s",
            _exc,
        )

    # Warm the embedding model so the first request doesn't block for 30+
    # seconds loading BGE-M3.  Awaited before serving (not fire-and-forget):
    # the async loader builds the provider in a background thread while the
    # event loop yields, so startup waits for the model but a request arriving
    # mid-load (or a health probe) can never block the event loop on the
    # singleton's threading.Lock — the failure mode that used to freeze the
    # whole loop for 30s+.
    try:
        from backend.service.embedding_service import get_embedding_provider_async

        async def _warm_embedding() -> None:
            try:
                await get_embedding_provider_async()
            except Exception:

                logging.getLogger(__name__).warning(
                    "Embedding warmup failed — model will load on first "
                    "use (may delay the first request)."
                )

        await _warm_embedding()
    except Exception:
        pass

    # ── Phase 3: start patrol scheduler ─────────────────────────────
    _scheduler = None
    # Catch-up patrol tasks (fired when a slot was missed while the server
    # was down) — tracked so shutdown can cancel them instead of leaving
    # detached tasks that warn on destruction.
    _catchup_tasks: list[asyncio.Task[None]] = []
    if config.patrol_active:
        try:
            from backend.service.patrol import (
                get_patrol_prompt,
                mark_stale_patrols_failed,
                run_patrol,
            )
            from backend.service.scheduler import (
                PatrolScheduler,
                previous_daily_slot,
                previous_weekly_slot,
                should_catch_up,
            )

            # A previous process may have left patrols mid-run; re-mark them
            # failed before scheduling so the logs don't lie about in-flight.
            await mark_stale_patrols_failed()

            _scheduler = PatrolScheduler()

            async def _run_daily_patrol(trigger: str = "cron") -> None:
                prompt = get_patrol_prompt("daily")
                await run_patrol(
                    patrol_type="daily",
                    trigger=trigger,
                    system_prompt=prompt,
                )

            # A daily slot that elapsed while the server was down (restart /
            # deploy / crash) is caught up once.  The catch-up runs as a
            # background task so startup isn't gated on a full patrol; its
            # trigger records the reason and the overlap guard in run_patrol
            # keeps it from colliding with a concurrent run.
            daily_slot = previous_daily_slot(config.patrol_daily_hour)
            if await should_catch_up("daily", daily_slot):
                logging.getLogger(__name__).info(
                    "Daily patrol missed its %s slot — running catch-up",
                    daily_slot.isoformat(),
                )
                _catchup_tasks.append(
                    asyncio.create_task(_run_daily_patrol("cron_catchup"))
                )

            _scheduler.schedule_daily(
                hour=config.patrol_daily_hour,
                callback=_run_daily_patrol,
            )

            if config.patrol_weekly_enabled:
                async def _run_weekly_patrol(trigger: str = "cron") -> None:
                    prompt = get_patrol_prompt("weekly")
                    await run_patrol(
                        patrol_type="weekly",
                        trigger=trigger,
                        system_prompt=prompt,
                    )

                weekly_slot = previous_weekly_slot(
                    config.patrol_weekly_day, config.patrol_weekly_hour
                )
                if await should_catch_up("weekly", weekly_slot):
                    logging.getLogger(__name__).info(
                        "Weekly patrol missed its %s slot — running catch-up",
                        weekly_slot.isoformat(),
                    )
                    _catchup_tasks.append(
                        asyncio.create_task(_run_weekly_patrol("cron_catchup"))
                    )

                _scheduler.schedule_weekly(
                    day=config.patrol_weekly_day,
                    hour=config.patrol_weekly_hour,
                    callback=_run_weekly_patrol,
                )

                # ── Phase 4: tech debt radar weekly scan ────────────
                async def _run_tech_debt_scan() -> None:
                    """Run tech debt scenario via compose function.

                    The scenario's agent may call notify_feishu_tool to push
                    findings to the team channel.  Persistence is handled by
                    the agent's memory tools — no separate DB write needed.
                    """

                    _log = logging.getLogger(__name__)
                    try:
                        from backend.service.scenarios.tech_debt import (
                            compose_tech_debt_report,
                        )

                        report = await compose_tech_debt_report()
                        _log.info(
                            "Tech debt scan completed (%d chars)", len(report)
                        )
                    except Exception:
                        _log.exception("Tech debt scan failed")

                _scheduler.schedule_weekly(
                    day=config.patrol_weekly_day,
                    hour=config.patrol_weekly_hour + 2,
                    callback=_run_tech_debt_scan,
                )

            await _scheduler.start()
            logging.getLogger(__name__).info("Patrol scheduler started")
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to start patrol scheduler — proactive features disabled."
            )

    yield

    # ── Shutdown: stop patrol scheduler + catch-up tasks ───────────
    if _scheduler is not None:
        try:
            await _scheduler.stop()
        except Exception:
            logging.getLogger(__name__).warning("Failed to stop patrol scheduler")

    # Cancel any catch-up patrol still in flight (same trade-off as the
    # scheduler loops: a mid-run patrol is re-marked failed on next startup).
    pending_catchup = [t for t in _catchup_tasks if not t.done()]
    for t in pending_catchup:
        t.cancel()
    if pending_catchup:
        await asyncio.gather(*pending_catchup, return_exceptions=True)

    # Stop the usage flusher and drain the buffer one last time.
    if _usage_task is not None:
        _usage_task.cancel()
        await asyncio.gather(_usage_task, return_exceptions=True)
        try:
            from backend.service.usage import flush_usage_buffer

            await flush_usage_buffer()
        except Exception:

            logging.getLogger(__name__).warning(
                "Failed to flush usage buffer on shutdown", exc_info=True
            )

    await close_db()

    # Close checkpointer pool on shutdown
    try:
        from backend.service.agent_service import _close_checkpointer

        await _close_checkpointer()
    except Exception:

        logging.getLogger(__name__).warning("Failed to close checkpointer pool")


app = FastAPI(
    title="EMA — Engineering Memory Agent",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)

# Runtime health metrics: per-route request count / latency / status for
# every HTTP request (including /health and /metrics themselves).
#
# Middleware order matters: RateLimitMiddleware is registered first, so
# MetricsMiddleware sits *outside* it and records the 429s the limiter
# produces — rate-limit rejections stay visible in the HTTP status
# distribution instead of disappearing before metrics sees them.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(MetricsMiddleware)

app.include_router(api_router, prefix="/api")


@app.get("/health", include_in_schema=False)
async def health_check() -> JSONResponse:
    """Liveness probe: DB reachability + AI provider configuration/health.

    Returns 200 ``{"status": "ok", ...}`` when the API is up and PostgreSQL
    is reachable; 503 ``{"status": "degraded", ...}`` otherwise (e.g. DB
    down, network partition).  Provider liveness is reported cheaply — config
    presence and circuit-breaker state — without issuing a real LLM/embedding
    call on every probe (that would burn tokens and latency on a frequently
    polled endpoint).
    Registered before the SPA catch-all so the route is actually reachable.
    """
    db_ok = True
    try:
        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
        logging.getLogger(__name__).warning("Health check: database unreachable")

    from backend.shared.resilience import get_circuit_breaker

    # The provider classes key their circuit breaker per endpoint|model (see
    # ``primary_breaker_name``), so the health probe reads the *same* breaker
    # the primary provider guards with.
    from backend.service.llm_service import primary_breaker_name
    from backend.service.embedding_service import embedding_breaker_name

    llm_breaker = get_circuit_breaker(primary_breaker_name())
    # Local BGE embeddings make no remote calls and have no breaker; only the
    # remote OpenAI-compatible embedder participates in resilience.  The name
    # follows the provider's per-endpoint scheme (``base_url|model``) so this
    # reports the same breaker the primary provider guards with.
    if config.embedding.provider == "openai":
        embed_breaker = get_circuit_breaker(embedding_breaker_name())
        embed_circuit = "open" if embed_breaker.is_open else "closed"
    else:
        embed_circuit = "n/a"

    payload = {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "llm": {
            "provider": config.llm.provider,
            "configured": bool(config.llm.api_key),
            "circuit": "open" if llm_breaker.is_open else "closed",
        },
        "embedding": {
            "provider": config.embedding.provider,
            "configured": config.embedding.provider == "local" or bool(config.embedding.api_key),
            "circuit": embed_circuit,
        },
    }
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content=payload,
    )

@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    """Prometheus scrape endpoint — runtime health metrics in text format.

    Serves the in-memory time series (HTTP request latency / status, LLM
    call count / latency / tokens, circuit-breaker state, agent concurrency,
    ReAct step distribution) when ``METRICS_ENABLED=true``.  Registered
    before the SPA catch-all so scraping actually reaches it.
    """
    if not config.metrics_enabled:
        return PlainTextResponse(
            "Metrics disabled (METRICS_ENABLED=false)",
            status_code=404,
        )
    return PlainTextResponse(render_metrics())


# ── SPA static files & fallback ──
# Mount the built JS/CSS so the browser can load them.  The favicon (and any
# other static assets) live under frontend/public — Vite's convention — and
# are copied verbatim into frontend/dist at build time; the dev server serves
# them at / directly, so no mount is needed for them here.
if _FRONTEND_DIST.joinpath("assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")


@app.get("/brain_favicon.png", include_in_schema=False)
async def favicon():
    """Serve the favicon (Vite copies frontend/public/ into dist at build)."""
    path = _FRONTEND_DIST / "brain_favicon.png"
    if path.is_file():
        return FileResponse(str(path))
    return JSONResponse({"detail": "favicon not built"}, status_code=404)


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """Catch-all: serve index.html for non-API paths (SPA client-side routing).

    ``api/...`` paths are excluded: an unmounted or misspelled API route must
    return 404 rather than the SPA index.  The breadth layers (connectors,
    scenarios, patrol) are conditionally mounted under ``/api`` (ADR-011) and
    the frontend probes them — a 200 HTML here would mask a disabled feature
    as available.
    """
    if full_path == "api" or full_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    index_path = _FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return {"detail": "Frontend not built — run: cd frontend && npm run build"}
