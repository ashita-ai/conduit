"""Dynamic pricing fetcher from llm-prices.com with caching and fallback.

This module fetches current LLM pricing from llm-prices.com and caches it
in-memory with TTL. Falls back to static pricing if fetch fails.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# llm-prices.com API endpoint
LLM_PRICES_API = "https://www.llm-prices.com/current-v1.json"

# Cache configuration
CACHE_TTL_HOURS = 24
_pricing_cache: dict[str, Any] | None = None
_cache_expires_at: datetime | None = None


# Quality estimates by model family (used when llm-prices doesn't provide quality)
# These are rough estimates based on community benchmarks and vendor claims
QUALITY_ESTIMATES = {
    # OpenAI
    "gpt-4o": 0.95,
    "gpt-4o-mini": 0.85,
    "gpt-4-turbo": 0.93,
    "gpt-4": 0.92,
    "gpt-3.5-turbo": 0.75,
    # Anthropic
    "claude-3.7": 0.97,
    "claude-3.5": 0.96,
    "claude-3-opus": 0.97,
    "claude-3-sonnet": 0.94,
    "claude-3-haiku": 0.80,
    # Google
    "gemini-2.0": 0.93,
    "gemini-1.5-pro": 0.92,
    "gemini-1.5-flash": 0.82,
    "gemini-1.0-pro": 0.78,
    # Groq (Llama models)
    "llama-3.3": 0.90,
    "llama-3.1-70b": 0.88,
    "llama-3.1-8b": 0.72,
    "llama-3-70b": 0.86,
    "llama-3-8b": 0.70,
    "mixtral-8x7b": 0.85,
    # Mistral
    "mistral-large": 0.91,
    "mistral-medium": 0.86,
    "mistral-small": 0.79,
    # Cohere
    "command-r-plus": 0.90,
    "command-r": 0.83,
    # Default fallback
    "default": 0.75,
}


def estimate_quality(model_id: str, model_name: str) -> float:
    """Estimate model quality based on model family.

    Args:
        model_id: Model identifier from llm-prices.com
        model_name: Human-readable model name

    Returns:
        Quality estimate (0-1 scale)
    """
    # Check for exact match first
    for pattern, quality in QUALITY_ESTIMATES.items():
        if pattern in model_id.lower() or pattern in model_name.lower():
            return quality

    # Fallback to default
    return QUALITY_ESTIMATES["default"]


async def fetch_pricing_from_api() -> dict[str, Any]:
    """Fetch current pricing from llm-prices.com API.

    Returns:
        Dictionary with 'updated_at' and 'prices' keys

    Raises:
        httpx.HTTPError: If API request fails
    """
    logger.info(f"Fetching pricing from {LLM_PRICES_API}")

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(LLM_PRICES_API)
        response.raise_for_status()
        data = response.json()

    logger.info(
        f"Fetched {len(data['prices'])} models (updated: {data['updated_at']})"
    )
    return data


def get_cached_pricing() -> dict[str, Any] | None:
    """Get pricing from cache if valid.

    Returns:
        Cached pricing data or None if cache invalid/expired
    """
    global _pricing_cache, _cache_expires_at

    if _pricing_cache is None or _cache_expires_at is None:
        return None

    if datetime.now() >= _cache_expires_at:
        logger.info("Pricing cache expired")
        return None

    logger.debug("Using cached pricing data")
    return _pricing_cache


def set_pricing_cache(data: dict[str, Any]) -> None:
    """Set pricing cache with TTL.

    Args:
        data: Pricing data to cache
    """
    global _pricing_cache, _cache_expires_at

    _pricing_cache = data
    _cache_expires_at = datetime.now() + timedelta(hours=CACHE_TTL_HOURS)
    logger.info(f"Cached pricing data (expires: {_cache_expires_at})")


async def fetch_pricing_with_cache() -> dict[str, Any]:
    """Fetch pricing with caching.

    Returns:
        Pricing data (from cache or API)

    Raises:
        httpx.HTTPError: If API request fails and no cache available
    """
    # Try cache first
    cached = get_cached_pricing()
    if cached is not None:
        return cached

    # Fetch from API
    data = await fetch_pricing_from_api()
    set_pricing_cache(data)
    return data


def fetch_pricing_sync() -> dict[str, Any]:
    """Synchronous wrapper for fetching pricing (for non-async contexts).

    Returns:
        Pricing data (from cache or API)

    Note:
        This uses asyncio.run() internally. Use fetch_pricing_with_cache()
        in async contexts for better performance.
    """
    import asyncio

    return asyncio.run(fetch_pricing_with_cache())


# Static fallback pricing (minimal set, only what we know works)
# This is ONLY used if llm-prices.com is unreachable
# Also includes providers missing from llm-prices.com (cohere, groq, huggingface)
FALLBACK_PRICING = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00, "quality": 0.95},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "quality": 0.85},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "quality": 0.96},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00, "quality": 0.97},
    },
    "cohere": {
        "command-r-plus": {"input": 3.00, "output": 15.00, "quality": 0.90},
        "command-r": {"input": 0.50, "output": 1.50, "quality": 0.83},
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79, "quality": 0.88},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08, "quality": 0.72},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24, "quality": 0.85},
    },
}


def get_fallback_pricing() -> dict[str, Any]:
    """Get fallback pricing (used when API is unavailable).

    Returns:
        Fallback pricing data in llm-prices.com format
    """
    prices = []
    for provider, models in FALLBACK_PRICING.items():
        for model_name, pricing in models.items():
            prices.append(
                {
                    "id": model_name,
                    "vendor": provider,
                    "name": model_name,
                    "input": pricing["input"],
                    "output": pricing["output"],
                    "input_cached": None,
                }
            )

    return {"updated_at": "fallback", "prices": prices}
