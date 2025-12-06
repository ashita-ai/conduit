"""Cost-based model filtering for budget enforcement.

This module provides pre-routing cost filtering to enforce max_cost constraints.
Models that would exceed the budget are excluded before bandit selection.

Uses tiktoken for accurate token estimation (100% accurate for OpenAI models,
reasonable approximation for others).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import tiktoken

from conduit.core.pricing import get_model_pricing
from conduit.engines.bandits.base import ModelArm

logger = logging.getLogger(__name__)

# Default tiktoken encoding (cl100k_base works well across providers)
DEFAULT_ENCODING = "cl100k_base"


@dataclass
class CostEstimate:
    """Cost estimation result for a model."""

    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class FilterResult:
    """Result of cost-based filtering."""

    models: list[ModelArm]
    was_relaxed: bool
    original_count: int
    filtered_count: int
    budget: float
    cheapest_model: str | None


class CostFilter:
    """Filter models by cost budget before bandit selection.

    This class estimates query costs using tiktoken for token counting
    and LiteLLM pricing data, then filters to models within budget.

    Attributes:
        output_ratio: Multiplier for output tokens (output = input * ratio).
            Default 1.0 assumes output length equals input length.
        fallback_on_empty: If True, use cheapest model when none fit budget.
            If False, raise ValueError.

    Example:
        >>> filter = CostFilter(output_ratio=1.0, fallback_on_empty=True)
        >>> result = filter.filter_by_budget(models, max_cost=0.01, query_text="Hello")
        >>> print(f"Filtered to {len(result.models)} models")
    """

    def __init__(
        self,
        output_ratio: float = 1.0,
        fallback_on_empty: bool = True,
    ) -> None:
        """Initialize cost filter.

        Args:
            output_ratio: Multiplier for estimating output tokens from input tokens.
                Examples: 1.0 (equal), 2.0 (output twice as long), 0.5 (output half).
            fallback_on_empty: If True and no models fit budget, return cheapest model.
                If False, raise ValueError when no models fit budget.
        """
        self.output_ratio = output_ratio
        self.fallback_on_empty = fallback_on_empty
        self._encoding: tiktoken.Encoding | None = None

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Lazily initialize tiktoken encoding."""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
        return self._encoding

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using tiktoken.

        Uses cl100k_base encoding which is accurate for OpenAI models
        and provides reasonable estimates for other providers (typically
        within 10-20% for Claude, Gemini).

        Args:
            text: Input text to tokenize

        Returns:
            Estimated token count

        Example:
            >>> filter = CostFilter()
            >>> tokens = filter.estimate_tokens("Hello, world!")
            >>> print(tokens)
            4
        """
        return len(self.encoding.encode(text))

    def estimate_cost(
        self,
        model: ModelArm,
        input_tokens: int,
    ) -> CostEstimate:
        """Estimate total cost for a model given input tokens.

        Uses model's pricing from LiteLLM if available, falls back to
        ModelArm's cost_per_input_token and cost_per_output_token.

        Args:
            model: Model arm with pricing information
            input_tokens: Number of input tokens

        Returns:
            CostEstimate with breakdown of input/output costs
        """
        output_tokens = int(input_tokens * self.output_ratio)

        # Try LiteLLM pricing first (most accurate)
        pricing = get_model_pricing(model.model_id)
        if pricing:
            input_cost = input_tokens * pricing.input_cost_per_token
            output_cost = output_tokens * pricing.output_cost_per_token
        else:
            # Fall back to ModelArm pricing (per 1K tokens)
            input_cost = (input_tokens / 1000) * model.cost_per_input_token
            output_cost = (output_tokens / 1000) * model.cost_per_output_token

        return CostEstimate(
            model_id=model.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

    def filter_by_budget(
        self,
        models: list[ModelArm],
        max_cost: float,
        query_text: str,
    ) -> FilterResult:
        """Filter models to those within cost budget.

        Args:
            models: List of available model arms
            max_cost: Maximum allowed cost in dollars
            query_text: Query text for token estimation

        Returns:
            FilterResult with filtered models and metadata

        Raises:
            ValueError: If no models fit budget and fallback_on_empty=False

        Example:
            >>> result = filter.filter_by_budget(models, 0.01, "Summarize this")
            >>> if result.was_relaxed:
            ...     print(f"Budget exceeded, using {result.cheapest_model}")
        """
        if not models:
            return FilterResult(
                models=[],
                was_relaxed=False,
                original_count=0,
                filtered_count=0,
                budget=max_cost,
                cheapest_model=None,
            )

        input_tokens = self.estimate_tokens(query_text)

        # Calculate costs and filter
        within_budget: list[ModelArm] = []
        cost_estimates: dict[str, CostEstimate] = {}

        for model in models:
            estimate = self.estimate_cost(model, input_tokens)
            cost_estimates[model.model_id] = estimate

            if estimate.total_cost <= max_cost:
                within_budget.append(model)

        # Find cheapest model for potential fallback
        cheapest_model = min(
            models,
            key=lambda m: cost_estimates[m.model_id].total_cost,
        )
        cheapest_estimate = cost_estimates[cheapest_model.model_id]

        # Handle empty result
        if not within_budget:
            if self.fallback_on_empty:
                logger.warning(
                    f"No models under ${max_cost:.4f} budget "
                    f"(cheapest: {cheapest_model.model_id} at ${cheapest_estimate.total_cost:.4f}). "
                    f"Falling back to cheapest model."
                )
                return FilterResult(
                    models=[cheapest_model],
                    was_relaxed=True,
                    original_count=len(models),
                    filtered_count=1,
                    budget=max_cost,
                    cheapest_model=cheapest_model.model_id,
                )
            else:
                raise ValueError(
                    f"No models fit budget of ${max_cost:.4f}. "
                    f"Cheapest model ({cheapest_model.model_id}) costs "
                    f"${cheapest_estimate.total_cost:.4f}"
                )

        logger.debug(
            f"Cost filter: {len(within_budget)}/{len(models)} models "
            f"within ${max_cost:.4f} budget"
        )

        return FilterResult(
            models=within_budget,
            was_relaxed=False,
            original_count=len(models),
            filtered_count=len(within_budget),
            budget=max_cost,
            cheapest_model=cheapest_model.model_id,
        )

    def get_cost_breakdown(
        self,
        models: list[ModelArm],
        query_text: str,
    ) -> list[CostEstimate]:
        """Get cost estimates for all models (useful for debugging/logging).

        Args:
            models: List of model arms
            query_text: Query text for token estimation

        Returns:
            List of CostEstimate objects sorted by total cost (ascending)
        """
        input_tokens = self.estimate_tokens(query_text)
        estimates = [self.estimate_cost(model, input_tokens) for model in models]
        return sorted(estimates, key=lambda e: e.total_cost)
