"""Custom exceptions for the ClinIQ application."""

from typing import Any


class ClinIQError(Exception):
    """Base exception for ClinIQ application."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ClinIQError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class AuthenticationError(ClinIQError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR")


class AuthorizationError(ClinIQError):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, "AUTHORIZATION_ERROR")


class NotFoundError(ClinIQError):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, identifier: str | int | None = None):
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"
        super().__init__(message, "NOT_FOUND")


class RateLimitError(ClinIQError):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int | None = None):
        message = "Rate limit exceeded"
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class ModelNotFoundError(ClinIQError):
    """Raised when a requested model is not found."""

    def __init__(self, model_name: str, model_version: str | None = None):
        identifier = f"{model_name}"
        if model_version:
            identifier = f"{model_name}:{model_version}"
        super().__init__(
            f"Model '{identifier}' not found",
            "MODEL_NOT_FOUND",
            {"model_name": model_name, "model_version": model_version},
        )


class ModelLoadError(ClinIQError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: str | None = None):
        message = f"Failed to load model '{model_name}'"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR", {"model_name": model_name})


class InferenceError(ClinIQError):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, reason: str | None = None):
        message = f"Inference failed for model '{model_name}'"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, "INFERENCE_ERROR", {"model_name": model_name})


class DocumentProcessingError(ClinIQError):
    """Raised when document processing fails."""

    def __init__(self, reason: str, document_id: str | None = None):
        details = {}
        if document_id:
            details["document_id"] = document_id
        super().__init__(f"Document processing failed: {reason}", "DOCUMENT_ERROR", details)


class BatchProcessingError(ClinIQError):
    """Raised when batch processing fails."""

    def __init__(self, job_id: str, reason: str):
        super().__init__(
            f"Batch job '{job_id}' failed: {reason}",
            "BATCH_ERROR",
            {"job_id": job_id},
        )


class ConfigurationError(ClinIQError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")


class DatabaseError(ClinIQError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, reason: str | None = None):
        message = f"Database operation '{operation}' failed"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, "DATABASE_ERROR")
