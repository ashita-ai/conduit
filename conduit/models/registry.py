"""Model registry with comprehensive PydanticAI model catalog.

Pricing data from:
- OpenAI: https://openai.com/api/pricing/
- Anthropic: https://www.anthropic.com/pricing
- Google: https://ai.google.dev/pricing
- Groq: https://groq.com/pricing/
- Mistral: https://mistral.ai/technology/#pricing
- Cohere: https://cohere.com/pricing

Last updated: 2025-01-19
Prices in USD per 1M tokens (converted to per-1K for storage).
"""

from typing import Any

from conduit.engines.bandits.base import ModelArm


# Pricing constants (USD per 1M tokens → divide by 1000 for per-1K)
PRICING = {
    # OpenAI models
    "openai": {
        "gpt-4o": {"input": 2.50 / 1000, "output": 10.00 / 1000, "quality": 0.95},
        "gpt-4o-mini": {"input": 0.15 / 1000, "output": 0.60 / 1000, "quality": 0.85},
        "gpt-4-turbo": {"input": 10.00 / 1000, "output": 30.00 / 1000, "quality": 0.93},
        "gpt-3.5-turbo": {"input": 0.50 / 1000, "output": 1.50 / 1000, "quality": 0.75},
    },
    # Anthropic models
    "anthropic": {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00 / 1000,
            "output": 15.00 / 1000,
            "quality": 0.96,
        },
        "claude-3-opus-20240229": {
            "input": 15.00 / 1000,
            "output": 75.00 / 1000,
            "quality": 0.97,
        },
        "claude-3-haiku-20240307": {
            "input": 0.25 / 1000,
            "output": 1.25 / 1000,
            "quality": 0.80,
        },
    },
    # Google models
    "google": {
        "gemini-1.5-pro": {"input": 1.25 / 1000, "output": 5.00 / 1000, "quality": 0.92},
        "gemini-1.5-flash": {"input": 0.075 / 1000, "output": 0.30 / 1000, "quality": 0.82},
        "gemini-1.0-pro": {"input": 0.50 / 1000, "output": 1.50 / 1000, "quality": 0.78},
    },
    # Groq models (ultra-fast inference, competitive pricing)
    "groq": {
        "llama-3.1-70b-versatile": {
            "input": 0.59 / 1000,
            "output": 0.79 / 1000,
            "quality": 0.88,
        },
        "llama-3.1-8b-instant": {
            "input": 0.05 / 1000,
            "output": 0.08 / 1000,
            "quality": 0.72,
        },
        "mixtral-8x7b-32768": {
            "input": 0.24 / 1000,
            "output": 0.24 / 1000,
            "quality": 0.85,
        },
    },
    # Mistral models
    "mistral": {
        "mistral-large-latest": {
            "input": 2.00 / 1000,
            "output": 6.00 / 1000,
            "quality": 0.91,
        },
        "mistral-medium-latest": {
            "input": 0.70 / 1000,
            "output": 2.10 / 1000,
            "quality": 0.86,
        },
        "mistral-small-latest": {
            "input": 0.20 / 1000,
            "output": 0.60 / 1000,
            "quality": 0.79,
        },
    },
    # Cohere models
    "cohere": {
        "command-r-plus": {"input": 3.00 / 1000, "output": 15.00 / 1000, "quality": 0.90},
        "command-r": {"input": 0.50 / 1000, "output": 1.50 / 1000, "quality": 0.83},
    },
}


def create_model_registry() -> list[ModelArm]:
    """Create comprehensive model registry from pricing data.

    Returns:
        List of ModelArm instances for all supported models

    Example:
        >>> registry = create_model_registry()
        >>> len(registry)
        17
        >>> registry[0].model_id
        "openai:gpt-4o"
    """
    models = []

    for provider, provider_models in PRICING.items():
        for model_name, pricing in provider_models.items():
            model_id = f"{provider}:{model_name}"

            arm = ModelArm(
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                cost_per_input_token=pricing["input"],
                cost_per_output_token=pricing["output"],
                expected_quality=pricing["quality"],
                metadata={
                    "pricing_last_updated": "2025-01-19",
                    "quality_estimate_source": "vendor_benchmarks_and_community",
                },
            )
            models.append(arm)

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


# Create default registry
DEFAULT_REGISTRY = create_model_registry()
