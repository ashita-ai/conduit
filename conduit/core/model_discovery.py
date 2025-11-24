"""Programmatic model discovery based on available API keys.

Automatically detects which LLM providers are configured and returns
a sensible default set of models (3 representative models per provider).

Principles:
- Simple: No phased rollout complexity
- Smart defaults: 3 recent, popular models per provider
- Override friendly: Explicit lists or YAML configs take precedence
- Provider detection: Based on which API keys are set
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProviderModels(BaseModel):
    """Representative models for a provider."""

    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    models: list[str] = Field(..., description="Model IDs for this provider")
    requires_key: str = Field(
        ..., description="API key name required (e.g., 'openai_api_key')"
    )


# Representative models per provider (3 each: flagship, balanced, cheap)
# These are current as of 2025-11-24
PROVIDER_DEFAULTS = [
    ProviderModels(
        provider="openai",
        models=[
            "gpt-4o",  # Flagship - best overall
            "gpt-4o-mini",  # Balanced - good quality, fast, cheap
            "gpt-4-turbo",  # Premium - high quality
        ],
        requires_key="openai_api_key",
    ),
    ProviderModels(
        provider="anthropic",
        models=[
            "claude-3-5-sonnet-20241022",  # Flagship - current best
            "claude-3-5-haiku-20241022",  # Fast, cheap, recent
            "claude-3-opus-20240229",  # Premium quality
        ],
        requires_key="anthropic_api_key",
    ),
    ProviderModels(
        provider="google",
        models=[
            "gemini-1.5-pro",  # Flagship - best quality
            "gemini-1.5-flash",  # Fast and cheap
            "gemini-2.0-flash-exp",  # Latest experimental
        ],
        requires_key="google_api_key",
    ),
    ProviderModels(
        provider="groq",
        models=[
            "llama-3.1-70b-versatile",  # Best quality
            "llama-3.1-8b-instant",  # Fastest
            "mixtral-8x7b-32768",  # Good middle ground
        ],
        requires_key="groq_api_key",
    ),
    ProviderModels(
        provider="mistral",
        models=[
            "mistral-large-latest",  # Flagship
            "mistral-medium-latest",  # Balanced
            "mistral-small-latest",  # Fast and cheap
        ],
        requires_key="mistral_api_key",
    ),
    ProviderModels(
        provider="cohere",
        models=[
            "command-r-plus",  # Best quality
            "command-r",  # Balanced
            "command",  # Fast baseline
        ],
        requires_key="cohere_api_key",
    ),
]


class ModelDiscovery:
    """Discover available models based on configured API keys.

    Example:
        >>> from conduit.core.config import settings
        >>> discovery = ModelDiscovery(settings)
        >>> models = discovery.get_models()
        >>> print(f"Auto-detected {len(models)} models from configured providers")

    YAML Override:
        Create `models.yaml`:
        ```yaml
        models:
          - gpt-4o-mini
          - claude-3-5-sonnet-20241022
          - gemini-1.5-flash
        ```

        >>> discovery = ModelDiscovery(settings, config_path="models.yaml")
        >>> models = discovery.get_models()  # Uses YAML instead of auto-detection
    """

    def __init__(
        self,
        settings: Any,
        config_path: str | Path | None = None,
        provider_defaults: list[ProviderModels] | None = None,
    ):
        """Initialize model discovery.

        Args:
            settings: Settings object with API keys
            config_path: Optional YAML config file path
            provider_defaults: Override default provider models (for testing)
        """
        self.settings = settings
        self.config_path = Path(config_path) if config_path else None
        self.provider_defaults = provider_defaults or PROVIDER_DEFAULTS

    def get_models(self) -> list[str]:
        """Get list of available models.

        Priority:
        1. YAML config file (if provided and exists)
        2. Auto-detection from API keys
        3. Fallback to settings.default_models

        Returns:
            List of model IDs to use for routing
        """
        # 1. Try YAML config first
        if self.config_path and self.config_path.exists():
            models = self._load_from_yaml()
            if models:
                logger.info(f"Loaded {len(models)} models from {self.config_path}")
                return models

        # 2. Auto-detect from API keys
        models = self._discover_from_api_keys()
        if models:
            logger.info(f"Auto-detected {len(models)} models from configured providers")
            return models

        # 3. Fallback to settings
        logger.warning(
            "No API keys configured and no YAML config found, "
            f"using settings.default_models ({len(self.settings.default_models)} models)"
        )
        return self.settings.default_models

    def _load_from_yaml(self) -> list[str] | None:
        """Load models from YAML config file.

        Expected format:
            models:
              - gpt-4o-mini
              - claude-3-5-sonnet-20241022

        Returns:
            List of model IDs or None if load fails
        """
        if not self.config_path or not self.config_path.exists():
            return None

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            models = config.get("models", [])
            if not models:
                logger.warning(f"YAML config {self.config_path} has no 'models' key")
                return None

            if not isinstance(models, list):
                logger.error(f"YAML config 'models' must be a list, got {type(models)}")
                return None

            # Validate all are strings
            if not all(isinstance(m, str) for m in models):
                logger.error("All models in YAML config must be strings")
                return None

            logger.info(f"Loaded {len(models)} models from YAML: {models}")
            return models

        except Exception as e:
            logger.error(f"Failed to load YAML config from {self.config_path}: {e}")
            return None

    def _discover_from_api_keys(self) -> list[str]:
        """Auto-detect available models based on configured API keys.

        For each provider with a configured API key, adds 3 representative models.

        Returns:
            List of model IDs from all configured providers
        """
        available_models: list[str] = []

        for provider_config in self.provider_defaults:
            # Check if API key is set
            key_name = provider_config.requires_key
            api_key = getattr(self.settings, key_name, "")

            if api_key and api_key.strip():
                # API key is configured, add this provider's models
                available_models.extend(provider_config.models)
                logger.info(
                    f"Provider '{provider_config.provider}' configured, "
                    f"adding {len(provider_config.models)} models: {provider_config.models}"
                )
            else:
                logger.debug(
                    f"Provider '{provider_config.provider}' not configured (no {key_name}), skipping"
                )

        return available_models

    def get_providers(self) -> list[str]:
        """Get list of configured providers.

        Returns:
            List of provider names (e.g., ['openai', 'anthropic'])
        """
        providers = []
        for provider_config in self.provider_defaults:
            key_name = provider_config.requires_key
            api_key = getattr(self.settings, key_name, "")
            if api_key and api_key.strip():
                providers.append(provider_config.provider)
        return providers

    def get_models_by_provider(self) -> dict[str, list[str]]:
        """Get models grouped by provider.

        Returns:
            Dictionary mapping provider name to list of model IDs
        """
        result: dict[str, list[str]] = {}
        for provider_config in self.provider_defaults:
            key_name = provider_config.requires_key
            api_key = getattr(self.settings, key_name, "")
            if api_key and api_key.strip():
                result[provider_config.provider] = provider_config.models
        return result
