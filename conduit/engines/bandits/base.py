"""Base classes and interfaces for bandit algorithms.

All bandit algorithms in Conduit use QueryFeatures from conduit.core.models
for context, ensuring consistency across the routing system.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from conduit.core.models import QueryFeatures


class ModelArm(BaseModel):
    """Represents a model (arm) in the multi-armed bandit.

    Attributes:
        model_id: Unique identifier (e.g., "openai:gpt-4o-mini")
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model_name: Model name within provider
        cost_per_input_token: Cost in USD per 1K input tokens
        cost_per_output_token: Cost in USD per 1K output tokens
        expected_quality: Prior estimate of quality (0-1 scale)
        metadata: Additional model characteristics
    """

    model_id: str
    provider: str
    model_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    expected_quality: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get full model name for PydanticAI."""
        return f"{self.provider}:{self.model_name}"


class BanditFeedback(BaseModel):
    """Feedback from executing a model selection.

    Attributes:
        model_id: Which model was selected
        cost: Actual cost incurred (USD)
        quality_score: Quality score from evaluation (0-1 scale)
        latency: Response latency in seconds
        success: Whether execution succeeded
        metadata: Additional feedback data (token counts, etc.)

    Multi-Objective Reward Function (Phase 3):
        Composite reward combines quality, cost, and latency using configurable weights.
        Default weights: 70% quality, 20% cost, 10% latency
        All metrics normalized to [0, 1] range where higher is better.
    """

    model_id: str
    cost: float = Field(..., ge=0.0)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    latency: float = Field(..., ge=0.0)
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def calculate_reward(
        self,
        quality_weight: float = 0.70,
        cost_weight: float = 0.20,
        latency_weight: float = 0.10,
    ) -> float:
        """Calculate composite reward from quality, cost, and latency.

        Uses asymptotic normalization for cost and latency (no population stats needed):
        - Quality: Already in [0, 1], higher is better (use directly)
        - Cost: Normalized as 1 / (1 + cost), inverted so lower cost = higher reward
        - Latency: Normalized as 1 / (1 + latency), inverted so lower latency = higher reward

        Args:
            quality_weight: Weight for quality component (default: 0.70)
            cost_weight: Weight for cost component (default: 0.20)
            latency_weight: Weight for latency component (default: 0.10)

        Returns:
            Composite reward in [0, 1] range

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> reward = feedback.calculate_reward()
            >>> print(f"{reward:.3f}")  # ~0.90 (high quality dominates)
            0.903
        """
        # Validate weights sum to 1.0
        total_weight = quality_weight + cost_weight + latency_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total_weight:.3f}"
            )

        # Normalize components to [0, 1] where higher is better
        quality_norm = self.quality_score  # Already [0, 1], higher is better

        # Cost: Lower is better, so invert. Asymptotic normalization.
        # cost=0 → 1.0, cost=1 → 0.5, cost=10 → 0.09
        cost_norm = 1.0 / (1.0 + self.cost)

        # Latency: Lower is better, so invert. Asymptotic normalization.
        # latency=0 → 1.0, latency=1 → 0.5, latency=10 → 0.09
        latency_norm = 1.0 / (1.0 + self.latency)

        # Weighted combination
        reward = (
            quality_weight * quality_norm
            + cost_weight * cost_norm
            + latency_weight * latency_norm
        )

        return reward


class BanditAlgorithm(ABC):
    """Abstract base class for multi-armed bandit algorithms.

    All bandit algorithms must implement:
    1. select_arm: Choose which model to use for a query
    2. update: Update internal state with feedback
    3. reset: Reset algorithm state

    Uses Conduit's QueryFeatures for context instead of custom data structures.
    """

    def __init__(self, name: str, arms: list[ModelArm]) -> None:
        """Initialize bandit algorithm.

        Args:
            name: Algorithm name for identification
            arms: List of available model arms
        """
        self.name = name
        self.arms = {arm.model_id: arm for arm in arms}
        self.arm_list = arms
        self.n_arms = len(arms)
        self.total_queries = 0

    @abstractmethod
    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select which model arm to pull for this query.

        Args:
            features: Query features from QueryAnalyzer

        Returns:
            Selected model arm

        Example:
            >>> features = QueryFeatures(
            ...     embedding=[0.1] * 384,
            ...     token_count=10,
            ...     complexity_score=0.5,
            ...     domain="general",
            ...     domain_confidence=0.8
            ... )
            >>> arm = await algorithm.select_arm(features)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        pass

    @abstractmethod
    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update algorithm state with feedback from arm pull.

        Args:
            feedback: Feedback from model execution
            features: Original query features

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await algorithm.update(feedback, features)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters and history.
        Useful for running multiple independent experiments.

        Example:
            >>> algorithm.reset()
            >>> algorithm.total_queries
            0
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get algorithm statistics and state.

        Returns:
            Dictionary with algorithm-specific statistics

        Example:
            >>> stats = algorithm.get_stats()
            >>> print(stats["total_queries"])
            1000
        """
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "n_arms": self.n_arms,
        }
