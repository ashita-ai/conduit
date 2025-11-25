"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError

from conduit.core.config import (
    APIConfig,
    ArbiterConfig,
    DatabaseConfig,
    MLConfig,
    ProviderConfig,
    RedisConfig,
    Settings,
    TelemetryConfig,
)


class TestSettings:
    """Tests for Settings configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        # Minimal config with required fields only
        settings = Settings(database_url="postgresql://localhost/test")

        assert settings.database_pool_size == 20
        # Redis URL may come from environment (.env file) or default
        assert settings.redis_url.startswith("redis://")  # Accept any valid redis URL
        assert settings.redis_cache_ttl == 86400  # 24 hours default
        # embedding_model defaults to empty string (uses provider default)
        assert settings.embedding_model == ""
        assert settings.exploration_rate == 0.1
        assert settings.api_rate_limit == 100
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_key == ""  # Empty by default
        assert settings.api_require_auth == False  # Disabled by default
        assert settings.log_level == "INFO"
        assert settings.environment == "development"

    def test_reward_weights_default(self):
        """Test default reward weights sum to 1.0."""
        settings = Settings(database_url="postgresql://localhost/test")

        assert settings.reward_weight_quality == 0.5
        assert settings.reward_weight_cost == 0.3
        assert settings.reward_weight_latency == 0.2
        assert sum([
            settings.reward_weight_quality,
            settings.reward_weight_cost,
            settings.reward_weight_latency
        ]) == 1.0

    def test_reward_weights_validation_success(self):
        """Test reward weights validation passes when sum is 1.0."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            reward_weight_quality=0.4,
            reward_weight_cost=0.4,
            reward_weight_latency=0.2
        )
        assert settings.reward_weight_quality == 0.4
        assert settings.reward_weight_cost == 0.4
        assert settings.reward_weight_latency == 0.2

    def test_reward_weights_validation_failure(self):
        """Test reward weights validation fails when sum is not 1.0."""
        with pytest.raises(ValidationError, match="Reward weights must sum to 1.0"):
            Settings(
                database_url="postgresql://localhost/test",
                reward_weight_quality=0.6,
                reward_weight_cost=0.3,
                reward_weight_latency=0.3  # Sum = 1.2
            )

    def test_reward_weights_property(self):
        """Test reward_weights property returns dict."""
        settings = Settings(database_url="postgresql://localhost/test")

        weights = settings.reward_weights
        assert isinstance(weights, dict)
        assert weights["quality"] == 0.5
        assert weights["cost"] == 0.3
        assert weights["latency"] == 0.2

    def test_is_production_property(self):
        """Test is_production property."""
        dev_settings = Settings(
            database_url="postgresql://localhost/test",
            environment="development"
        )
        assert dev_settings.is_production is False

        prod_settings = Settings(
            database_url="postgresql://localhost/test",
            environment="production"
        )
        assert prod_settings.is_production is True

    def test_default_models_list(self):
        """Test default models list."""
        settings = Settings(database_url="postgresql://localhost/test")

        assert "gpt-4o-mini" in settings.default_models
        assert "gpt-4o" in settings.default_models
        assert "claude-3-5-sonnet-20241022" in settings.default_models
        assert "claude-3-opus-20240229" in settings.default_models

    def test_api_port_validation(self):
        """Test API port must be valid range."""
        with pytest.raises(ValidationError):
            Settings(
                database_url="postgresql://localhost/test",
                api_port=70000  # Out of range
            )

        with pytest.raises(ValidationError):
            Settings(
                database_url="postgresql://localhost/test",
                api_port=0  # Too low
            )


class TestNestedConfigClasses:
    """Tests for nested configuration classes."""

    def test_database_config_defaults(self):
        """Test DatabaseConfig default values."""
        config = DatabaseConfig()
        assert config.url == ""
        assert config.pool_size == 20

    def test_database_config_validation(self):
        """Test DatabaseConfig validation."""
        config = DatabaseConfig(url="postgresql://localhost/test", pool_size=50)
        assert config.url == "postgresql://localhost/test"
        assert config.pool_size == 50

        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=0)  # Below minimum

        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=200)  # Above maximum

    def test_redis_config_defaults(self):
        """Test RedisConfig default values."""
        config = RedisConfig()
        assert config.url == "redis://localhost:6379"
        assert config.cache_enabled is True
        assert config.cache_ttl == 86400
        assert config.max_retries == 3
        assert config.timeout == 5
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 300

    def test_redis_config_validation(self):
        """Test RedisConfig validation."""
        config = RedisConfig(cache_ttl=3600, max_retries=5)
        assert config.cache_ttl == 3600
        assert config.max_retries == 5

        with pytest.raises(ValidationError):
            RedisConfig(cache_ttl=30)  # Below minimum (60)

    def test_provider_config_defaults(self):
        """Test ProviderConfig default values."""
        config = ProviderConfig()
        assert config.openai_api_key == ""
        assert config.anthropic_api_key == ""
        assert config.google_api_key == ""
        assert config.aws_region == "us-east-1"

    def test_ml_config_defaults(self):
        """Test MLConfig default values."""
        config = MLConfig()
        assert config.embedding_provider == "huggingface"
        assert config.embedding_model == ""
        assert config.use_pca is False
        assert config.pca_dimensions == 64
        assert config.exploration_rate == 0.1
        assert config.reward_weight_quality == 0.5
        assert config.reward_weight_cost == 0.3
        assert config.reward_weight_latency == 0.2

    def test_ml_config_reward_weights_validation(self):
        """Test MLConfig reward weights must sum to 1.0."""
        with pytest.raises(ValidationError, match="Reward weights must sum to 1.0"):
            MLConfig(
                reward_weight_quality=0.5,
                reward_weight_cost=0.5,
                reward_weight_latency=0.5,  # Sum = 1.5
            )

    def test_ml_config_reward_weights_property(self):
        """Test MLConfig reward_weights property."""
        config = MLConfig()
        weights = config.reward_weights
        assert weights == {"quality": 0.5, "cost": 0.3, "latency": 0.2}

    def test_api_config_defaults(self):
        """Test APIConfig default values."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.rate_limit == 100
        assert config.key == ""
        assert config.require_auth is False
        assert config.max_request_size == 10_000

    def test_api_config_validation(self):
        """Test APIConfig validation."""
        with pytest.raises(ValidationError):
            APIConfig(port=70000)  # Out of range

    def test_arbiter_config_defaults(self):
        """Test ArbiterConfig default values."""
        config = ArbiterConfig()
        assert config.enabled is False
        assert config.sample_rate == 0.1
        assert config.daily_budget == 10.0
        assert config.model == "gpt-4o-mini"

    def test_arbiter_config_validation(self):
        """Test ArbiterConfig validation."""
        with pytest.raises(ValidationError):
            ArbiterConfig(sample_rate=1.5)  # Above maximum (1.0)

    def test_telemetry_config_defaults(self):
        """Test TelemetryConfig default values."""
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.service_name == "conduit-router"
        assert config.exporter_otlp_endpoint == "http://localhost:4318"
        assert config.traces_enabled is True
        assert config.metrics_enabled is True


class TestNestedConfigProperties:
    """Tests for Settings nested config property accessors."""

    def test_database_property(self):
        """Test Settings.database property returns DatabaseConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            database_pool_size=30,
        )
        db = settings.database
        assert isinstance(db, DatabaseConfig)
        assert db.url == "postgresql://localhost/test"
        assert db.pool_size == 30

    def test_redis_property(self):
        """Test Settings.redis property returns RedisConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            redis_url="redis://custom:6379",
            redis_cache_enabled=False,
            redis_cache_ttl=7200,
        )
        redis = settings.redis
        assert isinstance(redis, RedisConfig)
        assert redis.url == "redis://custom:6379"
        assert redis.cache_enabled is False
        assert redis.cache_ttl == 7200

    def test_providers_property(self):
        """Test Settings.providers property returns ProviderConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            openai_api_key="sk-test123",
            anthropic_api_key="sk-ant-test",
        )
        providers = settings.providers
        assert isinstance(providers, ProviderConfig)
        assert providers.openai_api_key == "sk-test123"
        assert providers.anthropic_api_key == "sk-ant-test"

    def test_ml_property(self):
        """Test Settings.ml property returns MLConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            embedding_provider="openai",
            use_pca=True,
            pca_dimensions=128,
        )
        ml = settings.ml
        assert isinstance(ml, MLConfig)
        assert ml.embedding_provider == "openai"
        assert ml.use_pca is True
        assert ml.pca_dimensions == 128

    def test_api_property(self):
        """Test Settings.api property returns APIConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            api_host="127.0.0.1",
            api_port=9000,
            api_require_auth=True,
        )
        api = settings.api
        assert isinstance(api, APIConfig)
        assert api.host == "127.0.0.1"
        assert api.port == 9000
        assert api.require_auth is True

    def test_arbiter_property(self):
        """Test Settings.arbiter property returns ArbiterConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            arbiter_enabled=True,
            arbiter_sample_rate=0.25,
            arbiter_daily_budget=50.0,
        )
        arbiter = settings.arbiter
        assert isinstance(arbiter, ArbiterConfig)
        assert arbiter.enabled is True
        assert arbiter.sample_rate == 0.25
        assert arbiter.daily_budget == 50.0

    def test_telemetry_property(self):
        """Test Settings.telemetry property returns TelemetryConfig."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            otel_enabled=True,
            otel_service_name="my-service",
        )
        telemetry = settings.telemetry
        assert isinstance(telemetry, TelemetryConfig)
        assert telemetry.enabled is True
        assert telemetry.service_name == "my-service"

    def test_backwards_compatibility(self):
        """Test flat attribute access still works (backwards compatibility)."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            redis_cache_ttl=7200,
            api_port=9000,
        )
        # Both flat and nested access should work
        assert settings.redis_cache_ttl == 7200
        assert settings.redis.cache_ttl == 7200
        assert settings.api_port == 9000
        assert settings.api.port == 9000
