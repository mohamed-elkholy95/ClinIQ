"""Main FastAPI application."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.middleware.rate_limit import RateLimitMiddleware

from app.api.v1.routes import api_router
from app.core.config import get_settings
from app.core.exceptions import ClinIQError
from app.db.session import close_db, init_db
from app.middleware.logging import configure_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting ClinIQ API...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    # Configure structured logging
    configure_logging(settings.log_level, settings.log_format)

    logger.info("ClinIQ API started successfully")

    yield

    # Cleanup
    logger.info("Shutting down ClinIQ API...")
    await close_db()
    logger.info("ClinIQ API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ClinIQ API",
    description="""
Clinical Intelligence & Query Platform API

An AI-powered clinical NLP platform that processes unstructured medical text
and transforms it into structured, actionable clinical intelligence.

## Features

- **Medical Entity Recognition (NER)**: Extract diseases, symptoms, medications, procedures
- **ICD-10 Code Prediction**: Predict diagnosis codes from clinical text
- **Clinical Summarization**: Generate concise summaries of clinical notes
- **Risk Scoring**: Calculate patient risk scores based on clinical context

## Authentication

The API supports two authentication methods:
- **JWT Bearer Token**: For user authentication via `/auth/token`
- **API Key**: For programmatic access via `X-API-Key` header

## Rate Limiting

Free tier: 100 requests/day
Pro tier: 10,000 requests/day
    """,
    version=settings.app_version,
    docs_url=settings.docs_url,
    openapi_url=settings.openapi_url,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    return response


# Exception handlers
@app.exception_handler(ClinIQError)
async def cliniq_exception_handler(request: Request, exc: ClinIQError) -> JSONResponse:
    """Handle ClinIQ custom exceptions."""
    logger.error(f"ClinIQ error: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=_error_code_to_http_status(exc.error_code),
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {} if not settings.debug else {"exception": str(exc)},
        },
    )


def _error_code_to_http_status(error_code: str) -> int:
    """Map error codes to HTTP status codes."""
    mapping = {
        "VALIDATION_ERROR": 400,
        "AUTHENTICATION_ERROR": 401,
        "AUTHORIZATION_ERROR": 403,
        "NOT_FOUND": 404,
        "RATE_LIMIT_EXCEEDED": 429,
        "MODEL_NOT_FOUND": 404,
        "MODEL_LOAD_ERROR": 503,
        "INFERENCE_ERROR": 500,
        "DOCUMENT_ERROR": 400,
        "BATCH_ERROR": 400,
        "CONFIGURATION_ERROR": 500,
        "DATABASE_ERROR": 500,
    }
    return mapping.get(error_code, 500)


# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, redis_url=settings.redis_url)

# Include routers
app.include_router(api_router, prefix=settings.api_v1_prefix)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Root endpoint redirect to docs."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": settings.docs_url or "/docs",
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
    )
