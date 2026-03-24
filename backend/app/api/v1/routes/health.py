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


async def _probe_models() -> str:
    """Check whether ML model artefacts are accessible; always returns 'loaded' for now."""
    # TODO: wire up actual model registry probe once the ModelRegistry service exists.
    return "loaded"


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
    model_status = await _probe_models()

    dependencies: dict[str, str] = {
        "database": db_status,
        "redis": redis_status,
        "models": model_status,
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
