"""Model registry with dynamic pricing from llm-prices.com.

Pricing data automatically fetched from https://www.llm-prices.com/current-v1.json
- Cached for 24 hours in-memory
- Falls back to static pricing if API unavailable
- Supports all models listed on llm-prices.com (90+ models)

Source: https://github.com/simonw/llm-prices (community-maintained)
"""

import logging
from typing import Any

from conduit.engines.bandits.base import ModelArm
from conduit.models.pricing_fetcher import (
    estimate_quality,
    fetch_pricing_sync,
    get_fallback_pricing,
)

logger = logging.getLogger(__name__)

# Providers supported by PydanticAI (filter llm-prices.com to these only)
PYDANTIC_AI_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "amazon",  # via bedrock
    "mistral",
    "cohere",
    "groq",
    "huggingface",
}


# Legacy PRICING dict - DEPRECATED
# Use create_model_registry() which fetches from llm-prices.com dynamically
# This is kept for backwards compatibility only
PRICING: dict[str, dict[str, dict[str, float]]] = {}


def create_model_registry(use_cache: bool = True) -> list[ModelArm]:
    """Create comprehensive model registry from llm-prices.com.

    Fetches current pricing dynamically from https://www.llm-prices.com/current-v1.json
    with 24-hour caching. Falls back to static pricing if API unavailable.

    Args:
        use_cache: Use cached pricing if available (default: True)

    Returns:
        List of ModelArm instances for all supported models (90+ models)

    Example:
        >>> registry = create_model_registry()
        >>> len(registry)  # 90+ models
        >>> registry[0].model_id
        "openai:gpt-4o"
    """
    # Fetch pricing data (with caching)
    try:
        pricing_data = fetch_pricing_sync() if use_cache else get_fallback_pricing()
    except Exception as e:
        logger.warning(f"Failed to fetch pricing from llm-prices.com: {e}")
        logger.info("Using fallback pricing")
        pricing_data = get_fallback_pricing()

    # Convert to ModelArm instances (filter to PydanticAI-supported providers only)
    models = []
    skipped = 0
    for model_data in pricing_data["prices"]:
        model_id = model_data["id"]
        provider = model_data["vendor"]
        model_name = model_data["name"]

        # Skip providers not supported by PydanticAI
        if provider not in PYDANTIC_AI_PROVIDERS:
            skipped += 1
            continue

        # Convert from per-million to per-1K tokens
        input_cost = float(model_data["input"]) / 1000
        output_cost = float(model_data["output"]) / 1000

        # Estimate quality (llm-prices doesn't provide this)
        quality = estimate_quality(model_id, model_name)

        arm = ModelArm(
            model_id=f"{provider}:{model_id}",
            provider=provider,
            model_name=model_id,  # Use ID not name for consistency
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            expected_quality=quality,
            metadata={
                "pricing_source": "llm-prices.com",
                "pricing_updated": pricing_data["updated_at"],
                "quality_estimate_source": "conduit_heuristics",
                "display_name": model_name,
            },
        )
        models.append(arm)

    if skipped > 0:
        logger.debug(
            f"Skipped {skipped} models from unsupported providers "
            f"(not in PydanticAI)"
        )

    logger.info(
        f"Created registry with {len(models)} models "
        f"(source: {pricing_data['updated_at']})"
    )
    return models


def get_model_by_id(model_id: str, registry: list[ModelArm]) -> ModelArm | None:
    """Get model from registry by ID.

    Args:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")
        registry: Model registry

    Returns:
        ModelArm if found, None otherwise

    Example:
        >>> registry = create_model_registry()
        >>> model = get_model_by_id("openai:gpt-4o-mini", registry)
        >>> model.cost_per_input_token
        0.00015
    """
    for model in registry:
        if model.model_id == model_id:
            return model
    return None


def get_models_by_provider(provider: str, registry: list[ModelArm]) -> list[ModelArm]:
    """Get all models from specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        registry: Model registry

    Returns:
        List of ModelArm instances for that provider

    Example:
        >>> registry = create_model_registry()
        >>> openai_models = get_models_by_provider("openai", registry)
        >>> len(openai_models)
        4
    """
    return [model for model in registry if model.provider == provider]


def filter_models(
    registry: list[ModelArm],
    min_quality: float | None = None,
    max_cost: float | None = None,
    providers: list[str] | None = None,
) -> list[ModelArm]:
    """Filter models by criteria.

    Args:
        registry: Model registry
        min_quality: Minimum expected quality (0-1 scale)
        max_cost: Maximum average cost per token
        providers: List of allowed providers

    Returns:
        Filtered list of ModelArm instances

    Example:
        >>> registry = create_model_registry()
        >>> # Get high-quality, low-cost models
        >>> filtered = filter_models(
        ...     registry,
        ...     min_quality=0.85,
        ...     max_cost=0.001,
        ...     providers=["openai", "anthropic"]
        ... )
    """
    filtered = registry.copy()

    if min_quality is not None:
        filtered = [m for m in filtered if m.expected_quality >= min_quality]

    if max_cost is not None:
        filtered = [
            m
            for m in filtered
            if (m.cost_per_input_token + m.cost_per_output_token) / 2 <= max_cost
        ]

    if providers is not None:
        filtered = [m for m in filtered if m.provider in providers]

    return filtered


def get_registry_stats(registry: list[ModelArm]) -> dict[str, Any]:
    """Get statistics about model registry.

    Args:
        registry: Model registry

    Returns:
        Dictionary with registry statistics

    Example:
        >>> registry = create_model_registry()
        >>> stats = get_registry_stats(registry)
        >>> print(stats["total_models"])
        17
        >>> print(stats["providers"])
        ["openai", "anthropic", "google", "groq", "mistral", "cohere"]
    """
    providers = sorted(set(m.provider for m in registry))
    models_by_provider = {p: len(get_models_by_provider(p, registry)) for p in providers}

    costs = [(m.cost_per_input_token + m.cost_per_output_token) / 2 for m in registry]
    qualities = [m.expected_quality for m in registry]

    return {
        "total_models": len(registry),
        "providers": providers,
        "models_by_provider": models_by_provider,
        "cost_range": {
            "min": min(costs),
            "max": max(costs),
            "median": sorted(costs)[len(costs) // 2],
        },
        "quality_range": {
            "min": min(qualities),
            "max": max(qualities),
            "median": sorted(qualities)[len(qualities) // 2],
        },
    }


# Provider API key environment variable mapping
# Maps provider names to their environment variable names
# Only includes providers with pricing data from llm-prices.com
PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",  # Also works with GEMINI_API_KEY
    "mistral": "MISTRAL_API_KEY",
    "amazon": "AWS_ACCESS_KEY_ID",  # AWS Bedrock
    # Commented out until llm-prices.com adds them:
    # "groq": "GROQ_API_KEY",
    # "cohere": "COHERE_API_KEY",
    # "huggingface": "HUGGINGFACE_API_KEY",
}


def supported_models(
    providers: list[str] | None = None,
    min_quality: float | None = None,
    max_cost: float | None = None,
) -> list[ModelArm]:
    """Get all models supported by Conduit.

    Returns all models from the registry, optionally filtered by criteria.
    This shows what Conduit CAN use, not what YOU can use (see available_models).

    Args:
        providers: Filter to specific providers (e.g., ["openai", "anthropic"])
        min_quality: Minimum expected quality (0-1 scale)
        max_cost: Maximum average cost per token

    Returns:
        List of ModelArm instances matching criteria

    Example:
        >>> # All models
        >>> all_models = supported_models()
        >>> len(all_models)
        17

        >>> # High-quality budget models
        >>> good_cheap = supported_models(min_quality=0.85, max_cost=0.001)

        >>> # OpenAI only
        >>> openai = supported_models(providers=["openai"])
    """
    registry = create_model_registry()
    return filter_models(registry, min_quality, max_cost, providers)


def available_models(
    dotenv_path: str = ".env",
    providers: list[str] | None = None,
    min_quality: float | None = None,
    max_cost: float | None = None,
) -> list[ModelArm]:
    """Get models YOU can actually use based on API keys in environment.

    Auto-detects API keys from .env file and returns only models you have
    credentials for. This is what you should use to see your routing options.

    Args:
        dotenv_path: Path to .env file (default: ".env")
        providers: Further filter to specific providers
        min_quality: Minimum expected quality (0-1 scale)
        max_cost: Maximum average cost per token

    Returns:
        List of ModelArm instances you can use

    Example:
        >>> # What can I use?
        >>> my_models = available_models()

        >>> # What high-quality models can I use?
        >>> my_good = available_models(min_quality=0.90)
    """
    import os
    from pathlib import Path

    # Load .env file if it exists
    env_path = Path(dotenv_path)
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    # Detect which providers have API keys set
    available_providers = []
    for provider, env_var in PROVIDER_ENV_VARS.items():
        if os.getenv(env_var):
            available_providers.append(provider)

    # If user specified providers, intersect with available
    if providers is not None:
        available_providers = [p for p in providers if p in available_providers]

    # Return models for available providers
    return supported_models(
        providers=available_providers if available_providers else None,
        min_quality=min_quality,
        max_cost=max_cost,
    )


def get_available_providers(dotenv_path: str = ".env") -> list[str]:
    """Get list of providers you have API keys for.

    Args:
        dotenv_path: Path to .env file (default: ".env")

    Returns:
        List of provider names with configured API keys

    Example:
        >>> providers = get_available_providers()
        >>> print(providers)
        ['openai', 'anthropic', 'google']
    """
    import os
    from pathlib import Path

    env_path = Path(dotenv_path)
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    return [
        provider
        for provider, env_var in PROVIDER_ENV_VARS.items()
        if os.getenv(env_var)
    ]


# Create default registry
DEFAULT_REGISTRY = create_model_registry()
