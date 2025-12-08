"""Tests for conduit_litellm/config.py configuration module."""

import pytest

from conduit_litellm.config import ConduitLiteLLMConfig, DEFAULT_CONFIG


class TestConduitLiteLLMConfig:
    """Tests for ConduitLiteLLMConfig dataclass."""

    def test_default_values(self) -> None:
        """Config has expected default values."""
        config = ConduitLiteLLMConfig()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.cache_enabled is False
        assert config.redis_url is None
        assert config.cache_ttl == 3600

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = ConduitLiteLLMConfig(
            embedding_model="paraphrase-MiniLM-L6-v2",
            cache_enabled=True,
            redis_url="redis://localhost:6379/0",
            cache_ttl=7200,
        )

        assert config.embedding_model == "paraphrase-MiniLM-L6-v2"
        assert config.cache_enabled is True
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.cache_ttl == 7200

    def test_partial_custom_values(self) -> None:
        """Config allows partial customization with defaults for others."""
        config = ConduitLiteLLMConfig(cache_enabled=True)

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.cache_enabled is True
        assert config.redis_url is None
        assert config.cache_ttl == 3600


class TestToConduitConfig:
    """Tests for to_conduit_config method."""

    def test_basic_config_without_redis(self) -> None:
        """Converts config without Redis settings."""
        config = ConduitLiteLLMConfig()
        result = config.to_conduit_config()

        assert result == {
            "embedding_model": "all-MiniLM-L6-v2",
            "cache_enabled": False,
        }
        assert "redis_url" not in result
        assert "cache_ttl" not in result

    def test_config_with_redis(self) -> None:
        """Converts config with Redis settings."""
        config = ConduitLiteLLMConfig(
            embedding_model="custom-model",
            cache_enabled=True,
            redis_url="redis://localhost:6379",
            cache_ttl=1800,
        )
        result = config.to_conduit_config()

        assert result == {
            "embedding_model": "custom-model",
            "cache_enabled": True,
            "redis_url": "redis://localhost:6379",
            "cache_ttl": 1800,
        }

    def test_cache_enabled_without_redis_url(self) -> None:
        """Config with cache_enabled but no redis_url omits Redis settings."""
        config = ConduitLiteLLMConfig(cache_enabled=True)
        result = config.to_conduit_config()

        assert result == {
            "embedding_model": "all-MiniLM-L6-v2",
            "cache_enabled": True,
        }
        assert "redis_url" not in result

    def test_redis_url_includes_cache_ttl(self) -> None:
        """When redis_url is set, cache_ttl is included."""
        config = ConduitLiteLLMConfig(
            redis_url="redis://host:6379",
            cache_ttl=600,
        )
        result = config.to_conduit_config()

        assert result["redis_url"] == "redis://host:6379"
        assert result["cache_ttl"] == 600


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""

    def test_default_config_exists(self) -> None:
        """DEFAULT_CONFIG is defined and is a ConduitLiteLLMConfig instance."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, ConduitLiteLLMConfig)

    def test_default_config_has_default_values(self) -> None:
        """DEFAULT_CONFIG has the expected default values."""
        assert DEFAULT_CONFIG.embedding_model == "all-MiniLM-L6-v2"
        assert DEFAULT_CONFIG.cache_enabled is False
        assert DEFAULT_CONFIG.redis_url is None
        assert DEFAULT_CONFIG.cache_ttl == 3600

    def test_default_config_matches_new_instance(self) -> None:
        """DEFAULT_CONFIG values match a new default instance."""
        new_config = ConduitLiteLLMConfig()

        assert DEFAULT_CONFIG.embedding_model == new_config.embedding_model
        assert DEFAULT_CONFIG.cache_enabled == new_config.cache_enabled
        assert DEFAULT_CONFIG.redis_url == new_config.redis_url
        assert DEFAULT_CONFIG.cache_ttl == new_config.cache_ttl
