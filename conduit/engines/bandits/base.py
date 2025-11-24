"""Base classes and interfaces for bandit algorithms.

All bandit algorithms in Conduit use QueryFeatures from conduit.core.models
for context, ensuring consistency across the routing system.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from conduit.core.defaults import (
    DEFAULT_REWARD_WEIGHTS,
    PREFERENCE_WEIGHTS,
    TOKEN_COUNT_NORMALIZATION,
)
from conduit.core.models import QueryFeatures

if TYPE_CHECKING:
    from conduit.core.models import UserPreferences


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
        quality_weight: float | None = None,
        cost_weight: float | None = None,
        latency_weight: float | None = None,
    ) -> float:
        """Calculate composite reward from quality, cost, and latency.

        Uses asymptotic normalization for cost and latency (no population stats needed):
        - Quality: Already in [0, 1], higher is better (use directly)
        - Cost: Normalized as 1 / (1 + cost), inverted so lower cost = higher reward
        - Latency: Normalized as 1 / (1 + latency), inverted so lower latency = higher reward

        Args:
            quality_weight: Weight for quality component (default: from DEFAULT_REWARD_WEIGHTS)
            cost_weight: Weight for cost component (default: from DEFAULT_REWARD_WEIGHTS)
            latency_weight: Weight for latency component (default: from DEFAULT_REWARD_WEIGHTS)

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
        # Use defaults if not provided
        if quality_weight is None:
            quality_weight = DEFAULT_REWARD_WEIGHTS["quality"]
        if cost_weight is None:
            cost_weight = DEFAULT_REWARD_WEIGHTS["cost"]
        if latency_weight is None:
            latency_weight = DEFAULT_REWARD_WEIGHTS["latency"]

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

    def calculate_reward_with_preferences(
        self, preferences: "UserPreferences"
    ) -> float:
        """Calculate reward using user preferences.

        Convenience method that looks up weights from PREFERENCE_WEIGHTS
        based on the optimize_for setting.

        Args:
            preferences: User preferences with optimize_for setting

        Returns:
            Composite reward in [0, 1] range

        Example:
            >>> from conduit.core.models import UserPreferences
            >>> feedback = BanditFeedback(
            ...     model_id="gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> prefs = UserPreferences(optimize_for="cost")
            >>> reward = feedback.calculate_reward_with_preferences(prefs)
            >>> print(f"{reward:.3f}")  # Cost-optimized
        """
        weights = PREFERENCE_WEIGHTS[preferences.optimize_for]
        return self.calculate_reward(
            quality_weight=weights["quality"],
            cost_weight=weights["cost"],
            latency_weight=weights["latency"],
        )


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
    async def update(
        self,
        feedback: BanditFeedback,
        features: QueryFeatures,
        preferences: "UserPreferences | None" = None,
    ) -> None:
        """Update algorithm state with feedback from arm pull.

        Args:
            feedback: Feedback from model execution
            features: Original query features
            preferences: Optional user preferences to override default reward weights

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await algorithm.update(feedback, features)
            >>>
            >>> # With preferences
            >>> from conduit.core.models import UserPreferences
            >>> prefs = UserPreferences(optimize_for="cost")
            >>> await algorithm.update(feedback, features, preferences=prefs)
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

    def _get_reward_weights_from_preferences(
        self, preferences: "UserPreferences | None"
    ) -> dict[str, float]:
        """Get reward weights from preferences or use defaults.

        Args:
            preferences: User preferences, or None to use defaults

        Returns:
            Dictionary with quality, cost, and latency weights

        Example:
            >>> from conduit.core.models import UserPreferences
            >>> prefs = UserPreferences(optimize_for="cost")
            >>> weights = self._get_reward_weights_from_preferences(prefs)
            >>> print(weights)  # {"quality": 0.4, "cost": 0.5, "latency": 0.1}
        """
        if preferences is not None:
            return PREFERENCE_WEIGHTS[preferences.optimize_for]
        # Use bandit's default weights if available, otherwise use global defaults
        if hasattr(self, "reward_weights"):
            return self.reward_weights  # type: ignore
        return DEFAULT_REWARD_WEIGHTS

    def _extract_features(self, features: QueryFeatures) -> np.ndarray:
        """Extract feature vector from QueryFeatures.

        Combines embedding vector with metadata features:
        - embedding (384 dims)
        - token_count (1 dim, normalized by TOKEN_COUNT_NORMALIZATION)
        - complexity_score (1 dim)
        - domain_confidence (1 dim)

        Args:
            features: Query features object

        Returns:
            Feature vector as (d×1) numpy array

        Example:
            >>> features = QueryFeatures(
            ...     embedding=[0.1] * 384,
            ...     token_count=50,
            ...     complexity_score=0.5,
            ...     domain="general",
            ...     domain_confidence=0.8
            ... )
            >>> x = bandit._extract_features(features)
            >>> x.shape
            (387, 1)
        """
        # Combine embedding with metadata
        feature_vector = np.array(
            features.embedding
            + [
                features.token_count / TOKEN_COUNT_NORMALIZATION,
                features.complexity_score,
                features.domain_confidence,
            ]
        )

        # Reshape to column vector (d×1)
        return feature_vector.reshape(-1, 1)
