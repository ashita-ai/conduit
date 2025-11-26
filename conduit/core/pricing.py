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

from datetime import datetime

from pydantic import BaseModel, Field


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

    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token in dollars."""
        return self.input_cost_per_million / 1_000_000.0

    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token in dollars."""
        return self.output_cost_per_million / 1_000_000.0

    @property
    def cached_input_cost_per_token(self) -> float | None:
        """Cost per single cached input token in dollars, if available."""
        if self.cached_input_cost_per_million is None:
            return None
        return self.cached_input_cost_per_million / 1_000_000.0
