"""Unit tests for configuration management."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from conduit.core.config import (
    Settings,
    detect_available_models,
    get_models_with_fallback,
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
        # embedding_model may be set by conduit.yaml or None (provider default)
        assert settings.embedding_model is None or isinstance(
            settings.embedding_model, str
        )
        assert settings.exploration_rate == 0.1
        assert settings.api_rate_limit == 100
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        # api_key may come from environment (.env file) or default to empty string
        assert isinstance(settings.api_key, str)
        assert settings.api_require_auth == False  # Disabled by default
        assert settings.log_level == "INFO"
        assert settings.environment == "development"

    def test_reward_weights_default(self):
        """Test default reward weights sum to 1.0."""
        settings = Settings(database_url="postgresql://localhost/test")

        assert settings.reward_weight_quality == 0.5
        assert settings.reward_weight_cost == 0.3
        assert settings.reward_weight_latency == 0.2
        assert (
            sum(
                [
                    settings.reward_weight_quality,
                    settings.reward_weight_cost,
                    settings.reward_weight_latency,
                ]
            )
            == 1.0
        )

    def test_reward_weights_validation_success(self):
        """Test reward weights validation passes when sum is 1.0."""
        settings = Settings(
            database_url="postgresql://localhost/test",
            reward_weight_quality=0.4,
            reward_weight_cost=0.4,
            reward_weight_latency=0.2,
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
                reward_weight_latency=0.3,  # Sum = 1.2
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
            database_url="postgresql://localhost/test", environment="development"
        )
        assert dev_settings.is_production is False

        prod_settings = Settings(
            database_url="postgresql://localhost/test", environment="production"
        )
        assert prod_settings.is_production is True

    def test_default_models_list(self):
        """Test default models list is populated."""
        settings = Settings(database_url="postgresql://localhost/test")

        # Verify default_models is a non-empty list of model IDs
        assert isinstance(settings.default_models, list)
        assert len(settings.default_models) > 0
        # All entries should be strings (model IDs)
        assert all(isinstance(m, str) for m in settings.default_models)

    def test_api_port_validation(self):
        """Test API port must be valid range."""
        with pytest.raises(ValidationError):
            Settings(
                database_url="postgresql://localhost/test",
                api_port=70000,  # Out of range
            )

        with pytest.raises(ValidationError):
            Settings(database_url="postgresql://localhost/test", api_port=0)  # Too low


class TestModelAutoDetection:
    """Tests for auto-detection of available models from API keys."""

    def test_detect_openai_models(self):
        """Test detection of OpenAI models when API key is set."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.anthropic_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = detect_available_models()

            assert "gpt-4o-mini" in models
            assert "gpt-4o" in models
            assert len(models) == 2

    def test_detect_anthropic_models(self):
        """Test detection of Anthropic models when API key is set."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = "sk-ant-test-key"
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = detect_available_models()

            assert "claude-3-5-sonnet-20241022" in models
            assert "claude-3-haiku-20240307" in models
            assert len(models) == 2

    def test_detect_multiple_providers(self):
        """Test detection when multiple API keys are set."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.anthropic_api_key = "sk-ant-test-key"
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = detect_available_models()

            assert "gpt-4o-mini" in models
            assert "claude-3-5-sonnet-20241022" in models
            assert len(models) == 4

    def test_detect_no_api_keys(self):
        """Test detection returns empty list when no API keys set."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = detect_available_models()

            assert models == []

    def test_get_models_with_fallback_uses_configured(self):
        """Test fallback uses configured models when available."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.default_models = ["custom-model-1", "custom-model-2"]

            models = get_models_with_fallback()

            assert models == ["custom-model-1", "custom-model-2"]

    def test_get_models_with_fallback_uses_detection(self):
        """Test fallback uses auto-detection when no configured models."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.default_models = []
            mock_settings.openai_api_key = "sk-test-key"
            mock_settings.anthropic_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = get_models_with_fallback()

            assert "gpt-4o-mini" in models
            assert "gpt-4o" in models

    def test_get_models_with_fallback_ultimate_fallback(self):
        """Test fallback returns hardcoded default when nothing available."""
        with patch("conduit.core.config.settings.settings") as mock_settings:
            mock_settings.default_models = []
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            mock_settings.google_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.mistral_api_key = None

            models = get_models_with_fallback()

            assert models == ["gpt-4o-mini"]
