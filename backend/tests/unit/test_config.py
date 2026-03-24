"""Unit tests for application configuration (Settings)."""

import pytest

from app.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Tests for default Settings values."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Provide a default Settings instance."""
        return Settings()

    def test_default_app_name(self, settings: Settings):
        """Test default application name."""
        assert settings.app_name == "ClinIQ"

    def test_default_app_version(self, settings: Settings):
        """Test default application version is set."""
        assert isinstance(settings.app_version, str)
        assert len(settings.app_version) > 0

    def test_default_debug_is_false(self, settings: Settings):
        """Test that debug mode is off by default."""
        assert settings.debug is False

    def test_default_environment(self, settings: Settings):
        """Test that default environment is 'development'."""
        assert settings.environment == "development"

    def test_default_api_prefix(self, settings: Settings):
        """Test default API v1 prefix."""
        assert settings.api_v1_prefix == "/api/v1"

    def test_default_algorithm(self, settings: Settings):
        """Test default JWT algorithm."""
        assert settings.algorithm == "HS256"

    def test_default_access_token_expire_minutes(self, settings: Settings):
        """Test default token expiry in minutes."""
        assert settings.access_token_expire_minutes == 30

    def test_default_rate_limit_requests(self, settings: Settings):
        """Test default rate limit request count."""
        assert settings.rate_limit_requests == 100

    def test_default_max_document_length(self, settings: Settings):
        """Test default maximum document length."""
        assert settings.max_document_length == 100000

    def test_default_max_batch_size(self, settings: Settings):
        """Test default maximum batch size."""
        assert settings.max_batch_size == 100

    def test_default_db_pool_size(self, settings: Settings):
        """Test default database pool size."""
        assert settings.db_pool_size == 10

    def test_default_db_max_overflow(self, settings: Settings):
        """Test default database max overflow."""
        assert settings.db_max_overflow == 20

    def test_default_model_cache_size(self, settings: Settings):
        """Test default model cache size."""
        assert settings.model_cache_size == 3

    def test_default_log_level(self, settings: Settings):
        """Test default log level."""
        assert settings.log_level == "INFO"

    def test_default_log_format(self, settings: Settings):
        """Test default log format."""
        assert settings.log_format == "json"

    def test_default_cors_origins_is_list(self, settings: Settings):
        """Test that cors_origins is a list."""
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0

    def test_secret_key_is_set(self, settings: Settings):
        """Test that secret_key has a value."""
        value = settings.secret_key.get_secret_value()
        assert isinstance(value, str)
        assert len(value) > 0

    def test_database_url_is_set(self, settings: Settings):
        """Test that database_url is configured."""
        assert isinstance(settings.database_url, str)
        assert len(settings.database_url) > 0

    def test_redis_url_is_set(self, settings: Settings):
        """Test that redis_url is configured."""
        assert isinstance(settings.redis_url, str)
        assert settings.redis_url.startswith("redis://")

    def test_minio_bucket(self, settings: Settings):
        """Test default MinIO bucket name."""
        assert settings.minio_bucket == "cliniq"


class TestSettingsProperties:
    """Tests for computed Settings properties."""

    def test_is_development_true_for_development_environment(self):
        """Test is_development returns True when environment='development'."""
        settings = Settings(environment="development")
        assert settings.is_development is True
        assert settings.is_production is False

    def test_is_production_true_for_production_environment(self):
        """Test is_production returns True when environment='production'."""
        settings = Settings(environment="production")
        assert settings.is_production is True
        assert settings.is_development is False

    def test_is_production_false_for_staging(self):
        """Test is_production returns False for staging environment."""
        settings = Settings(environment="staging")
        assert settings.is_production is False
        assert settings.is_development is False

    def test_is_development_false_for_production(self):
        """Test is_development returns False for production environment."""
        settings = Settings(environment="production")
        assert settings.is_development is False

    def test_is_development_false_for_staging(self):
        """Test is_development returns False for staging environment."""
        settings = Settings(environment="staging")
        assert settings.is_development is False


class TestSettingsCustomisation:
    """Tests for customising Settings values."""

    def test_custom_app_name(self):
        """Test overriding app_name."""
        settings = Settings(app_name="CustomApp")
        assert settings.app_name == "CustomApp"

    def test_custom_debug(self):
        """Test enabling debug mode."""
        settings = Settings(debug=True)
        assert settings.debug is True

    def test_custom_cors_origins(self):
        """Test providing custom CORS origins."""
        origins = ["http://localhost:3000", "https://myapp.example.com"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_custom_rate_limit(self):
        """Test overriding rate limit settings."""
        settings = Settings(rate_limit_requests=500, rate_limit_period=3600)
        assert settings.rate_limit_requests == 500
        assert settings.rate_limit_period == 3600

    def test_custom_token_expiry(self):
        """Test overriding access token expiry."""
        settings = Settings(access_token_expire_minutes=60)
        assert settings.access_token_expire_minutes == 60

    def test_custom_max_document_length(self):
        """Test overriding max document length."""
        settings = Settings(max_document_length=50000)
        assert settings.max_document_length == 50000

    def test_environment_literal_values(self):
        """Test that environment only accepts valid literal values."""
        for env in ("development", "staging", "production"):
            settings = Settings(environment=env)
            assert settings.environment == env


class TestGetSettings:
    """Tests for the get_settings() cached factory function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings() returns a Settings object."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test that get_settings() returns the same instance on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
