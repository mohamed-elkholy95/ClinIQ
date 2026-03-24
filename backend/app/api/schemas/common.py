"""Common response models shared across all API endpoints."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ErrorDetail(BaseModel):
    """Single error detail."""

    field: str | None = Field(
        default=None,
        description="Field that caused the error, if applicable",
    )
    message: str = Field(description="Human-readable error message")
    code: str | None = Field(
        default=None,
        description="Machine-readable error code (e.g. 'validation_error', 'not_found')",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "field": "text",
                "message": "text must not exceed 100000 characters",
                "code": "validation_error",
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error envelope returned on all 4xx / 5xx responses."""

    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(description="Short error summary")
    details: list[ErrorDetail] | None = Field(
        default=None,
        description="Per-field validation errors or additional context",
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for tracing",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the error",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": "Validation failed",
                "details": [
                    {
                        "field": "text",
                        "message": "text must not exceed 100000 characters",
                        "code": "validation_error",
                    }
                ],
                "request_id": "req_01hx8y3k4q",
                "timestamp": "2026-03-24T10:00:00Z",
            }
        }
    }


class SuccessResponse(BaseModel):
    """Lightweight success acknowledgement with no payload."""

    success: bool = Field(default=True, description="Always true for success responses")
    message: str = Field(description="Human-readable success message")
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for tracing",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Resource deleted successfully",
                "request_id": "req_01hx8y3k4q",
            }
        }
    }


class PaginationMeta(BaseModel):
    """Pagination metadata embedded in list responses."""

    total: int = Field(ge=0, description="Total number of items matching the query")
    page: int = Field(ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(ge=1, le=500, description="Number of items per page")
    total_pages: int = Field(ge=0, description="Total number of pages")
    has_next: bool = Field(description="Whether a next page exists")
    has_previous: bool = Field(description="Whether a previous page exists")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 250,
                "page": 2,
                "page_size": 25,
                "total_pages": 10,
                "has_next": True,
                "has_previous": True,
            }
        }
    }


class PaginatedResponse(BaseModel, Generic[DataT]):
    """Generic paginated list wrapper."""

    success: bool = Field(default=True)
    data: list[DataT] = Field(description="Page of results")
    pagination: PaginationMeta = Field(description="Pagination metadata")

    model_config = {"json_schema_extra": {"example": {"success": True, "data": [], "pagination": {}}}}


class ModelInfo(BaseModel):
    """Metadata about the ML model that produced a result."""

    name: str = Field(description="Model identifier (e.g. 'en_ner_bc5cdr_md')")
    version: str = Field(description="Semantic version of the deployed model")
    processing_time_ms: float = Field(
        ge=0,
        description="Wall-clock inference time in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "rule-based-ner",
                "version": "1.0.0",
                "processing_time_ms": 42.7,
            }
        }
    }


class HealthResponse(BaseModel):
    """API and dependency health check response."""

    status: str = Field(description="Overall health status: 'healthy' | 'degraded' | 'unhealthy'")
    version: str = Field(description="Application version string")
    environment: str = Field(description="Deployment environment: development | staging | production")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the health check",
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="Per-dependency status map (e.g. {'database': 'healthy', 'redis': 'healthy'})",
    )
    uptime_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Process uptime in seconds",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "environment": "production",
                "timestamp": "2026-03-24T10:00:00Z",
                "dependencies": {
                    "database": "healthy",
                    "redis": "healthy",
                    "minio": "healthy",
                },
                "uptime_seconds": 3600.5,
            }
        }
    }
