"""Health check endpoint.

Returns the overall service status together with per-dependency probes for the
database, Redis, and loaded ML models. Used by load balancers and uptime monitors.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.common import HealthResponse
from app.core.config import Settings, get_settings
from app.db.session import get_db_session

router = APIRouter(tags=["health"])

# Capture the process start time at import so uptime can be computed on every request.
_PROCESS_START: float = time.monotonic()


async def _probe_database(db: AsyncSession) -> str:
    """Execute a lightweight SQL probe; return 'healthy' or 'unhealthy'."""
    try:
        await db.execute(text("SELECT 1"))
        return "healthy"
    except Exception:
        return "unhealthy"


async def _probe_redis(settings: Settings) -> str:
    """Attempt a Redis PING; return 'healthy', 'unhealthy', or 'unavailable'."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import-untyped]

        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await client.ping()
        await client.aclose()
        return "healthy"
    except ImportError:
        return "unavailable"
    except Exception:
        return "unhealthy"


async def _probe_models() -> dict[str, str]:
    """Check which ML models are currently loaded via the model registry.

    Returns a dict with per-model status (``'loaded'`` / ``'not_loaded'``)
    plus an overall ``'_status'`` key that is ``'loaded'`` when at least
    one model is ready, ``'not_loaded'`` when none are.
    """
    from app.services.model_registry import health_check as registry_health

    statuses = registry_health()
    per_model = {k: ("loaded" if v else "not_loaded") for k, v in statuses.items()}
    any_loaded = any(statuses.values())
    per_model["_status"] = "loaded" if any_loaded else "not_loaded"
    return per_model


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the overall service health status together with per-dependency "
        "connectivity probes for the database, Redis cache, and ML models. "
        "An HTTP 200 with status='healthy' or status='degraded' means the service "
        "is operational. status='unhealthy' is returned with HTTP 503 when critical "
        "dependencies are unavailable."
    ),
    responses={
        200: {"description": "Service is healthy or degraded"},
        503: {"description": "Service is unhealthy — critical dependencies unavailable"},
    },
)
async def health_check(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return the aggregated health status of all service dependencies."""
    db_status = await _probe_database(db)
    redis_status = await _probe_redis(settings)
    model_info = await _probe_models()
    model_status = model_info.pop("_status", "not_loaded")

    dependencies: dict[str, str] = {
        "database": db_status,
        "redis": redis_status,
        "models": model_status,
        **{f"model_{k}": v for k, v in model_info.items()},
    }

    # Determine overall status based on critical dependency health.
    if db_status == "unhealthy":
        overall = "unhealthy"
    elif redis_status == "unhealthy":
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        dependencies=dependencies,
        uptime_seconds=time.monotonic() - _PROCESS_START,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description=(
        "Kubernetes liveness probe. Returns HTTP 200 as long as the process is running. "
        "Does not check dependencies."
    ),
)
async def liveness() -> dict[str, str]:
    """Kubernetes liveness probe — returns alive immediately."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Kubernetes readiness probe. Returns HTTP 200 once the service can accept traffic "
        "(database reachable, models loaded). Returns HTTP 503 otherwise."
    ),
)
async def readiness(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, str]:
    """Kubernetes readiness probe — checks database connectivity."""
    db_status = await _probe_database(db)
    if db_status == "unhealthy":
        return {"status": "not_ready", "reason": "Database unreachable"}
    return {"status": "ready"}
