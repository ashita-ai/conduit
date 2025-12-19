"""Pricing models and helpers for LLM cost calculation.

This module provides pricing lookup using LiteLLM's bundled model_cost database,
which contains up-to-date pricing for all major LLM providers.

Benefits of using LiteLLM pricing:
- No external API calls (pricing bundled with package)
- Exact model ID matching (no normalization needed)
- Comprehensive coverage (all providers, regions, aliases)
- Includes cache pricing (creation and read costs)
- Tiered pricing support (e.g., >200k token rates)
- Updates with `uv update litellm`
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import litellm
from pydantic import BaseModel, Field, computed_field

logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    """Per-model pricing information.

    All prices are expressed in cost per one million tokens to match common
    provider conventions. Helper properties expose per-token prices for use
    in cost calculations.
    """

    model_id: str = Field(..., description="Provider model identifier")
    input_cost_per_million: float = Field(
        ...,
        ge=0.0,
        description="Cost per 1M input tokens in dollars",
    )
    output_cost_per_million: float = Field(
        ...,
        ge=0.0,
        description="Cost per 1M output tokens in dollars",
    )
    cached_input_cost_per_million: float | None = Field(
        default=None,
        ge=0.0,
        description="Cost per 1M cached input tokens in dollars (cache read)",
    )
    cache_creation_cost_per_million: float | None = Field(
        default=None,
        ge=0.0,
        description="Cost per 1M tokens for cache creation",
    )
    source: str | None = Field(
        default=None,
        description="Pricing source identifier",
    )
    snapshot_at: datetime | None = Field(
        default=None,
        description="Timestamp when this pricing snapshot was recorded",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token in dollars."""
        return self.input_cost_per_million / 1_000_000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token in dollars."""
        return self.output_cost_per_million / 1_000_000.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cached_input_cost_per_token(self) -> float | None:
        """Cost per single cached input token in dollars (cache read)."""
        if self.cached_input_cost_per_million is None:
            return None
        return self.cached_input_cost_per_million / 1_000_000.0


def get_model_pricing(model_id: str) -> ModelPricing | None:
    """Get pricing for a model from LiteLLM's bundled database.

    Args:
        model_id: Model identifier (e.g., "claude-sonnet-4-5-20250929", "gpt-4o-mini")

    Returns:
        ModelPricing if found, None if model not in LiteLLM's database

    Example:
        >>> pricing = get_model_pricing("claude-sonnet-4-5-20250929")
        >>> if pricing:
        ...     print(f"Input: ${pricing.input_cost_per_million}/1M tokens")
        Input: $3.0/1M tokens
    """
    model_info = litellm.model_cost.get(model_id)

    if model_info is None:
        logger.debug(f"No pricing found for model: {model_id}")
        return None

    # LiteLLM stores per-token costs, convert to per-million
    input_cost = model_info.get("input_cost_per_token", 0.0)
    output_cost = model_info.get("output_cost_per_token", 0.0)
    cache_read_cost = model_info.get("cache_read_input_token_cost")
    cache_creation_cost = model_info.get("cache_creation_input_token_cost")

    return ModelPricing(
        model_id=model_id,
        input_cost_per_million=input_cost * 1_000_000,
        output_cost_per_million=output_cost * 1_000_000,
        cached_input_cost_per_million=(
            cache_read_cost * 1_000_000 if cache_read_cost else None
        ),
        cache_creation_cost_per_million=(
            cache_creation_cost * 1_000_000 if cache_creation_cost else None
        ),
        source="litellm",
        snapshot_at=datetime.now(timezone.utc),
    )


def get_all_model_pricing() -> dict[str, ModelPricing]:
    """Get pricing for all models in LiteLLM's database.

    Filters to chat models only (excludes embeddings, image generation, etc.)

    Returns:
        Dict mapping model_id to ModelPricing

    Example:
        >>> all_pricing = get_all_model_pricing()
        >>> len(all_pricing)
        500  # approximate
    """
    pricing: dict[str, ModelPricing] = {}

    for model_id, model_info in litellm.model_cost.items():
        # Skip non-chat models and the sample spec
        if model_id == "sample_spec":
            continue
        mode = model_info.get("mode", "")
        if mode and mode != "chat":
            continue

        # Skip models without input pricing (indicates incomplete data)
        if "input_cost_per_token" not in model_info:
            continue

        model_pricing = get_model_pricing(model_id)
        if model_pricing:
            pricing[model_id] = model_pricing

    logger.info(f"Loaded pricing for {len(pricing)} models from LiteLLM")
    return pricing


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Compute total cost for a model call.

    Args:
        input_tokens: Number of input tokens (excluding cached)
        output_tokens: Number of output tokens
        model_id: Model identifier for pricing lookup
        cache_read_tokens: Tokens read from cache (cheaper than input)
        cache_creation_tokens: Tokens written to cache (more expensive)

    Returns:
        Total cost in dollars

    Example:
        >>> cost = compute_cost(1000, 500, "claude-sonnet-4-5-20250929")
        >>> print(f"${cost:.6f}")
        $0.010500
    """
    pricing = get_model_pricing(model_id)

    if pricing is None:
        # Conservative fallback: use GPT-4 tier pricing to never underreport
        # $10/1M input, $30/1M output (upper bound for most models)
        FALLBACK_INPUT_PER_TOKEN = 10.0 / 1_000_000
        FALLBACK_OUTPUT_PER_TOKEN = 30.0 / 1_000_000
        fallback_cost = (
            input_tokens * FALLBACK_INPUT_PER_TOKEN
            + output_tokens * FALLBACK_OUTPUT_PER_TOKEN
        )
        logger.warning(
            f"No pricing for {model_id}, using conservative fallback: "
            f"${fallback_cost:.6f} ({input_tokens} in, {output_tokens} out)"
        )
        return fallback_cost

    # Base cost
    cost = (
        input_tokens * pricing.input_cost_per_token
        + output_tokens * pricing.output_cost_per_token
    )

    # Cache read cost (if applicable)
    if cache_read_tokens > 0 and pricing.cached_input_cost_per_token is not None:
        cost += cache_read_tokens * pricing.cached_input_cost_per_token

    # Cache creation cost (if applicable)
    if (
        cache_creation_tokens > 0
        and pricing.cache_creation_cost_per_million is not None
    ):
        cache_creation_per_token = pricing.cache_creation_cost_per_million / 1_000_000
        cost += cache_creation_tokens * cache_creation_per_token

    return cost
