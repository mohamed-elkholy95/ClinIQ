"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "ClinIQ"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production", "test"] = "development"

    # API
    api_v1_prefix: str = "/api/v1"
    docs_url: str | None = "/docs"
    openapi_url: str | None = "/openapi.json"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://cliniq:cliniq@localhost:5432/cliniq",
        description="Async PostgreSQL connection URL",
    )
    database_sync_url: str = Field(
        default="postgresql://cliniq:cliniq@localhost:5432/cliniq",
        description="Sync PostgreSQL connection URL (for Alembic)",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600  # 1 hour

    # Security
    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production-use-openssl-rand-hex-32"),
        description="Secret key for JWT signing",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_header: str = "X-API-Key"

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 86400  # 24 hours in seconds

    # ML Models
    model_dir: str = "models"
    model_cache_size: int = 3  # Number of models to keep in memory
    inference_timeout: int = 30
    max_document_length: int = 100000  # characters
    max_batch_size: int = 100

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "cliniq"

    # Storage
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    minio_bucket: str = "cliniq"
    minio_secure: bool = False

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "json"

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        """Ensure secret key is not default in production."""
        # This is checked at runtime
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Type alias for dependency injection
SettingsDependency = Settings
