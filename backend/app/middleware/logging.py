"""Structured logging middleware."""

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with structured JSON logging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Attach a unique request ID, log start/finish, and measure latency.

        Adds ``X-Request-ID`` and ``X-Process-Time`` response headers for
        downstream tracing and performance monitoring.
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Bind request context to logger
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        logger.info(
            "request_started",
            query_params=str(request.query_params),
            user_agent=request.headers.get("user-agent", ""),
        )

        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000

            logger.info(
                "request_completed",
                status_code=response.status_code,
                process_time_ms=round(process_time, 2),
            )

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

            return response

        except Exception as exc:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                "request_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                process_time_ms=round(process_time, 2),
            )
            raise


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structlog for the application."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
