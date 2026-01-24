"""Model detection and utility functions for Conduit configuration.

These functions handle model auto-detection, fallback selection,
and model-related configuration helpers.
"""


def get_arbiter_model() -> str:
    """Get the model ID to use for arbiter evaluation.

    Returns the arbiter_model from settings, falling back to 'o4-mini'.

    Returns:
        Model ID string for evaluation.
    """
    from conduit.core.config.settings import settings

    return settings.arbiter_model or "o4-mini"


def get_fallback_model() -> str:
    """Get the fallback model ID for routing when no specific model is chosen.

    Returns the first model from default_models, or 'o4-mini' as ultimate fallback.

    Returns:
        Model ID string.
    """
    from conduit.core.config.settings import settings

    if settings.default_models:
        return settings.default_models[0]
    return "o4-mini"


def detect_available_models() -> list[str]:
    """Auto-detect available models based on configured API keys.

    Checks for provider API keys and returns default models for each available
    provider. This enables zero-config usage where models are automatically
    detected from environment variables.

    Detection order (by provider):
        1. OpenAI (OPENAI_API_KEY)
        2. Anthropic (ANTHROPIC_API_KEY)
        3. Google (GOOGLE_API_KEY)
        4. Groq (GROQ_API_KEY)
        5. Mistral (MISTRAL_API_KEY)

    Returns:
        List of model IDs available based on detected API keys.
        Empty list if no API keys found.

    Example:
        >>> # With OPENAI_API_KEY and ANTHROPIC_API_KEY set:
        >>> models = detect_available_models()
        >>> print(models)
        ['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307']

        >>> # With no API keys set:
        >>> models = detect_available_models()
        >>> print(models)
        []
    """
    from conduit.core.config.settings import settings

    available: list[str] = []

    # OpenAI models
    if settings.openai_api_key:
        available.extend(["gpt-4o-mini", "gpt-4o"])

    # Anthropic models
    if settings.anthropic_api_key:
        available.extend(["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"])

    # Google/Gemini models
    if settings.google_api_key:
        available.extend(["gemini-1.5-flash", "gemini-1.5-pro"])

    # Groq models (fast inference)
    if settings.groq_api_key:
        available.extend(["groq/llama-3.1-70b-versatile", "groq/llama-3.1-8b-instant"])

    # Mistral models
    if settings.mistral_api_key:
        available.extend(
            ["mistral/mistral-large-latest", "mistral/mistral-small-latest"]
        )

    return available


def get_models_with_fallback() -> list[str]:
    """Get models to use for routing with smart fallback.

    Priority order:
        1. Models from conduit.yaml priors (if configured)
        2. Auto-detected models from API keys
        3. Hardcoded defaults (if nothing else available)

    This enables three usage patterns:
        - Explicit: Configure models in conduit.yaml
        - Auto-detect: Set API keys, models auto-discovered
        - Fallback: No config needed for quick testing

    Returns:
        List of model IDs to use for routing.

    Raises:
        ConfigurationError: If no models available and no API keys configured.

    Example:
        >>> # Auto-detect from API keys
        >>> models = get_models_with_fallback()
        >>> print(models)
        ['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet-20241022', ...]
    """
    from conduit.core.config.settings import settings

    # Try configured models first (from YAML)
    if settings.default_models:
        return settings.default_models

    # Try auto-detection from API keys
    detected = detect_available_models()
    if detected:
        return detected

    # Ultimate fallback for testing
    return ["gpt-4o-mini"]
