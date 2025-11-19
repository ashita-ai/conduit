"""Unit tests for configuration management."""

import pytest
from pydantic import ValidationError

from conduit.core.config import Settings


class TestSettings:
    """Tests for Settings configuration model."""

    def test_default_values(self):
        """Test default configuration values."""
        # Minimal config with required fields only
        settings = Settings(database_url="postgresql://localhost/test")

        assert settings.database_pool_size == 20
        # Redis URL may come from environment (.env file) or default
        assert settings.redis_url.startswith("redis://")  # Accept any valid redis URL
        assert settings.redis_ttl == 3600
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.exploration_rate == 0.1
        assert settings.api_rate_limit == 100
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
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
        assert "claude-3.5-sonnet" in settings.default_models
        assert "claude-opus-4" in settings.default_models

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
