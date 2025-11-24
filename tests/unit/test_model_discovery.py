"""Tests for model discovery functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from conduit.core.model_discovery import (
    ModelDiscovery,
    ProviderModels,
    PROVIDER_DEFAULTS,
)


@pytest.fixture
def mock_settings():
    """Mock settings object with API keys."""
    settings = MagicMock()
    # Default: no API keys configured
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.google_api_key = ""
    settings.groq_api_key = ""
    settings.mistral_api_key = ""
    settings.cohere_api_key = ""
    settings.huggingface_api_key = ""
    settings.default_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    return settings


def test_no_api_keys_uses_default_models(mock_settings):
    """When no API keys configured, should return settings.default_models."""
    discovery = ModelDiscovery(mock_settings)
    models = discovery.get_models()

    assert models == mock_settings.default_models


def test_single_provider_discovery(mock_settings):
    """Should discover models for single configured provider."""
    mock_settings.openai_api_key = "sk-test-key"

    discovery = ModelDiscovery(mock_settings)
    models = discovery.get_models()

    # Should get OpenAI models
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert "gpt-4-turbo" in models
    assert len(models) == 3  # Only OpenAI models


def test_multiple_providers_discovery(mock_settings):
    """Should discover models for all configured providers."""
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = "sk-ant-test"

    discovery = ModelDiscovery(mock_settings)
    models = discovery.get_models()

    # Should get OpenAI models
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models

    # Should get Anthropic models
    assert "claude-3-5-sonnet-20241022" in models
    assert "claude-3-5-haiku-20241022" in models
    assert "claude-3-opus-20240229" in models

    # Total: 3 OpenAI + 3 Anthropic
    assert len(models) == 6


def test_all_providers_discovery(mock_settings):
    """Should discover models for all providers when all API keys set."""
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = "sk-ant-test"
    mock_settings.google_api_key = "test-key"
    mock_settings.groq_api_key = "gsk-test"
    mock_settings.mistral_api_key = "test-key"
    mock_settings.cohere_api_key = "test-key"

    discovery = ModelDiscovery(mock_settings)
    models = discovery.get_models()

    # Should get models from all providers
    # OpenAI (3) + Anthropic (3) + Google (3) + Groq (3) + Mistral (3) + Cohere (3) = 18
    assert (
        len(models) >= 15
    )  # At least 15 models (allowing for provider config changes)

    # Check representative models
    assert "gpt-4o" in models  # OpenAI
    assert "claude-3-5-sonnet-20241022" in models  # Anthropic
    assert "gemini-1.5-pro" in models  # Google
    assert "llama-3.1-70b-versatile" in models  # Groq
    assert "mistral-large-latest" in models  # Mistral
    assert "command-r-plus" in models  # Cohere


def test_yaml_config_override(mock_settings):
    """YAML config should override auto-discovery."""
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = "sk-ant-test"

    # Create temporary YAML config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "models": [
                    "custom-model-1",
                    "custom-model-2",
                ]
            },
            f,
        )
        config_path = Path(f.name)

    try:
        discovery = ModelDiscovery(mock_settings, config_path=config_path)
        models = discovery.get_models()

        # Should use YAML config, not auto-discovery
        assert models == ["custom-model-1", "custom-model-2"]
        assert "gpt-4o" not in models  # Auto-discovery ignored

    finally:
        config_path.unlink()


def test_yaml_config_missing_models_key(mock_settings):
    """YAML config with missing 'models' key should fallback to auto-discovery."""
    mock_settings.openai_api_key = "sk-test"

    # Create YAML without 'models' key
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"other_key": "value"}, f)
        config_path = Path(f.name)

    try:
        discovery = ModelDiscovery(mock_settings, config_path=config_path)
        models = discovery.get_models()

        # Should fallback to auto-discovery
        assert "gpt-4o" in models

    finally:
        config_path.unlink()


def test_yaml_config_invalid_format(mock_settings):
    """Invalid YAML should fallback to auto-discovery."""
    mock_settings.openai_api_key = "sk-test"

    # Create invalid YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("models: not a list")
        config_path = Path(f.name)

    try:
        discovery = ModelDiscovery(mock_settings, config_path=config_path)
        models = discovery.get_models()

        # Should fallback to auto-discovery
        assert "gpt-4o" in models

    finally:
        config_path.unlink()


def test_yaml_config_nonexistent_file(mock_settings):
    """Non-existent YAML file should fallback to auto-discovery."""
    mock_settings.openai_api_key = "sk-test"

    discovery = ModelDiscovery(
        mock_settings, config_path=Path("/nonexistent/file.yaml")
    )
    models = discovery.get_models()

    # Should fallback to auto-discovery
    assert "gpt-4o" in models


def test_get_providers(mock_settings):
    """Should return list of configured providers."""
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = "sk-ant-test"

    discovery = ModelDiscovery(mock_settings)
    providers = discovery.get_providers()

    assert "openai" in providers
    assert "anthropic" in providers
    assert len(providers) == 2


def test_get_models_by_provider(mock_settings):
    """Should return models grouped by provider."""
    mock_settings.openai_api_key = "sk-test"
    mock_settings.anthropic_api_key = "sk-ant-test"

    discovery = ModelDiscovery(mock_settings)
    models_by_provider = discovery.get_models_by_provider()

    assert "openai" in models_by_provider
    assert "anthropic" in models_by_provider
    assert len(models_by_provider) == 2

    # Check OpenAI models
    assert "gpt-4o" in models_by_provider["openai"]
    assert "gpt-4o-mini" in models_by_provider["openai"]

    # Check Anthropic models
    assert "claude-3-5-sonnet-20241022" in models_by_provider["anthropic"]


def test_custom_provider_defaults(mock_settings):
    """Should support custom provider configurations."""
    custom_providers = [
        ProviderModels(
            provider="test_provider",
            models=["test-model-1", "test-model-2"],
            requires_key="test_api_key",
        )
    ]

    # Add test API key to settings
    mock_settings.test_api_key = "test-key"

    discovery = ModelDiscovery(mock_settings, provider_defaults=custom_providers)
    models = discovery.get_models()

    assert "test-model-1" in models
    assert "test-model-2" in models


def test_empty_api_key_ignored(mock_settings):
    """Empty or whitespace-only API keys should be ignored."""
    mock_settings.openai_api_key = "   "  # Whitespace only
    mock_settings.anthropic_api_key = ""  # Empty string

    discovery = ModelDiscovery(mock_settings)
    models = discovery.get_models()

    # Should use default_models since no valid API keys
    assert models == mock_settings.default_models


def test_provider_defaults_structure():
    """Verify PROVIDER_DEFAULTS has expected structure."""
    assert (
        len(PROVIDER_DEFAULTS) >= 5
    )  # At least OpenAI, Anthropic, Google, Groq, Mistral

    # Check each provider has required fields
    for provider in PROVIDER_DEFAULTS:
        assert provider.provider
        assert len(provider.models) >= 2  # At least 2 models per provider
        assert provider.requires_key

        # Check all models are strings
        assert all(isinstance(m, str) for m in provider.models)


def test_openai_models_in_defaults():
    """Verify OpenAI models are in PROVIDER_DEFAULTS."""
    openai_config = next((p for p in PROVIDER_DEFAULTS if p.provider == "openai"), None)

    assert openai_config is not None
    assert openai_config.requires_key == "openai_api_key"
    assert "gpt-4o" in openai_config.models or "gpt-4o-mini" in openai_config.models


def test_anthropic_models_in_defaults():
    """Verify Anthropic models are in PROVIDER_DEFAULTS."""
    anthropic_config = next(
        (p for p in PROVIDER_DEFAULTS if p.provider == "anthropic"), None
    )

    assert anthropic_config is not None
    assert anthropic_config.requires_key == "anthropic_api_key"
    assert "claude-3-5-sonnet-20241022" in anthropic_config.models


def test_google_models_in_defaults():
    """Verify Google models are in PROVIDER_DEFAULTS."""
    google_config = next((p for p in PROVIDER_DEFAULTS if p.provider == "google"), None)

    assert google_config is not None
    assert google_config.requires_key == "google_api_key"
    assert any("gemini" in m.lower() for m in google_config.models)
