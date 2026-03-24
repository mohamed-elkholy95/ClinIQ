"""Core module initialization."""

from app.core.config import Settings, SettingsDependency, get_settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BatchProcessingError,
    ClinIQError,
    ConfigurationError,
    DatabaseError,
    DocumentProcessingError,
    InferenceError,
    ModelLoadError,
    ModelNotFoundError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_api_key,
    get_password_hash,
    hash_api_key,
    verify_api_key,
    verify_password,
)

__all__ = [
    # Config
    "Settings",
    "SettingsDependency",
    "get_settings",
    # Exceptions
    "ClinIQError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "DocumentProcessingError",
    "BatchProcessingError",
    "ConfigurationError",
    "DatabaseError",
    # Security
    "create_access_token",
    "decode_access_token",
    "generate_api_key",
    "get_password_hash",
    "hash_api_key",
    "verify_api_key",
    "verify_password",
]
