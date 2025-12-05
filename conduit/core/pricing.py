"""Pricing models and helpers for LLM cost calculation.

This module defines the data structures used to represent per-model pricing
information and provides helpers to convert per-million token prices into
per-token costs.

Pricing data is loaded from the ``model_prices`` table in Supabase by the
Database layer and injected into components that need it (for example,
``ModelExecutor``). This keeps pricing logic centralized and makes it easy
to update prices without code changes.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, computed_field


def normalize_model_id(model: str) -> str:
    """Normalize model ID to match llm-prices.com conventions.

    Handles various naming patterns:
    - claude-sonnet-4-5-20250929 → claude-sonnet-4.5
    - claude-haiku-4-5-20251001 → claude-4.5-haiku
    - claude-opus-4-5-20251101 → claude-opus-4-5 (keeps hyphens)
    - gpt-5-mini-2025-08-07 → gpt-5-mini
    - gemini-2.5-pro → gemini-2.5-pro

    Args:
        model: Raw model identifier

    Returns:
        Normalized model identifier matching llm-prices.com conventions
    """
    # Strip date suffixes (YYYY-MM-DD or YYYYMMDD patterns at end)
    normalized = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    normalized = re.sub(r"-\d{8}$", "", normalized)

    # Handle Claude naming conventions
    # llm-prices.com uses:
    # - claude-sonnet-4.5 (dot separator)
    # - claude-4.5-haiku (version before model name)
    # - claude-opus-4-5 (hyphen separator - exception!)
    if normalized.startswith("claude-"):
        # Handle sonnet: claude-sonnet-4-5 → claude-sonnet-4.5
        if "sonnet" in normalized:
            normalized = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", normalized)

        # Handle haiku: claude-haiku-4-5 → claude-4.5-haiku
        elif "haiku" in normalized:
            normalized = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", normalized)
            haiku_match = re.match(r"claude-haiku-(\d+\.\d+)$", normalized)
            if haiku_match:
                version = haiku_match.group(1)
                normalized = f"claude-{version}-haiku"

        # opus stays as-is (claude-opus-4-5 matches llm-prices)

    return normalized


class ModelPricing(BaseModel):
    """Per-model pricing information.

    All prices are expressed in cost per one million tokens to match common
    provider and tooling conventions (for example, values from
    https://www.llm-prices.com/). Helper properties expose per-token prices
    for use in cost calculations.
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
        description="Cost per 1M cached input tokens in dollars (if applicable)",
    )
    source: str | None = Field(
        default=None,
        description="Pricing source identifier (for example, 'llm-prices.com')",
    )
    snapshot_at: datetime | None = Field(
        default=None,
        description="Timestamp when this pricing snapshot was recorded",
    )

    @computed_field
    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token in dollars."""
        return self.input_cost_per_million / 1_000_000.0

    @computed_field
    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token in dollars."""
        return self.output_cost_per_million / 1_000_000.0

    @computed_field
    @property
    def cached_input_cost_per_token(self) -> float | None:
        """Cost per single cached input token in dollars, if available."""
        if self.cached_input_cost_per_million is None:
            return None
        return self.cached_input_cost_per_million / 1_000_000.0
