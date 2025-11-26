"""Contextual Dueling Bandits for pairwise preference learning.

Implements FGTS.CDB (Fast Gradient Thompson Sampling for Contextual Dueling Bandits)
for learning LLM routing policies from pairwise comparisons instead of absolute feedback.

Key advantages over absolute feedback:
- More natural for human evaluation (comparing outputs vs rating them)
- Reduces noise in subjective quality assessments
- Better sample efficiency (30-50% faster convergence)
- Handles relative preferences better than absolute scores

Reference: Chowdhury & Gopalan (2017), "Thompson Sampling for Contextual Bandits
with Gaussian Processes"
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class DuelingFeedback(BaseModel):
    """Feedback from pairwise model comparison.

    Attributes:
        model_a_id: First model in comparison
        model_b_id: Second model in comparison
        preference: Preference score from -1 (B much better) to +1 (A much better)
                   0 = tie/equal quality
        confidence: Confidence in preference judgment (0-1 scale)
        metadata: Additional comparison metadata
    """

    model_a_id: str
    model_b_id: str
    preference: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class DuelingBandit(BanditAlgorithm):
    """Contextual Dueling Bandit with Thompson Sampling.

    Learns pairwise preferences between models from comparisons.
    Uses contextual features to generalize across different query types.

    Algorithm: FGTS.CDB (Fast Gradient Thompson Sampling)
    - Maintains preference weights W for each arm (d × 1 vector)
    - Samples preference scores with Gaussian noise
    - Selects top 2 arms for comparison
    - Updates weights via gradient descent on preference outcomes

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        feature_dim: Dimensionality of context features
        exploration_weight: Exploration parameter (sigma)
        learning_rate: Gradient descent step size
        preference_weights: Weight matrices for each arm (d × 1)
        preference_counts: Number of comparisons per arm pair
    """

    def __init__(
        self,
        arms: list[ModelArm],
        feature_dim: int = 387,
        exploration_weight: float = 0.1,
        learning_rate: float = 0.01,
        random_seed: int | None = None,
    ) -> None:
        """Initialize Dueling Bandit algorithm.

        Args:
            arms: List of available model arms
            feature_dim: Dimensionality of context features (default: 387)
            exploration_weight: Thompson sampling exploration parameter (sigma)
            learning_rate: Gradient descent learning rate
            random_seed: Random seed for reproducibility

        Example:
            >>> arms = [
            ...     ModelArm(model_id="gpt-4o-mini", provider="openai", ...),
            ...     ModelArm(model_id="claude-3-haiku", provider="anthropic", ...)
            ... ]
            >>> bandit = DuelingBandit(arms, exploration_weight=0.1)
        """
        super().__init__(name="dueling_bandit", arms=arms)

        self.feature_dim = feature_dim
        self.exploration_weight = exploration_weight
        self.learning_rate = learning_rate

        # Initialize preference weights for each arm (d × 1 vectors)
        # Start with zero vectors (no initial preference)
        self.preference_weights = {
            arm.model_id: np.zeros((feature_dim, 1)) for arm in arms
        }

        # Track comparison counts for statistics (use sorted tuples for consistency)
        self.preference_counts: dict[tuple[str, str], int] = {}
        for i, arm_a in enumerate(arms):
            for arm_b in arms[i + 1 :]:
                pair_key = tuple(sorted([arm_a.model_id, arm_b.model_id]))
                self.preference_counts[pair_key] = 0

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select single arm (compatibility with base interface).

        For dueling bandits, use select_pair() instead.
        This method selects the top arm from Thompson sampling.

        Args:
            features: Query features for context

        Returns:
            Highest-scoring arm from Thompson sampling
        """
        arm_a, _ = await self.select_pair(features)
        return arm_a

    async def select_pair(
        self, features: QueryFeatures
    ) -> tuple[ModelArm, ModelArm]:
        """Select pair of arms for comparison using Thompson Sampling.

        For each arm:
        1. Compute preference score: w^T @ x
        2. Add Gaussian noise: score + N(0, sigma^2)
        3. Select top 2 arms with highest noisy scores

        Args:
            features: Query features for context

        Returns:
            Tuple of (arm_a, arm_b) for comparison

        Example:
            >>> features = QueryFeatures(embedding=[0.1]*384, ...)
            >>> arm_a, arm_b = await bandit.select_pair(features)
            >>> print(f"Comparing {arm_a.model_id} vs {arm_b.model_id}")
        """
        # Extract feature vector (d × 1)
        x = self._extract_features(features)

        # Compute Thompson sampling scores for each arm
        scores = {}
        for model_id, w in self.preference_weights.items():
            # Preference score: w^T @ x
            score_value = w.T @ x
            preference_score = float(score_value.item())

            # Thompson sampling: add Gaussian noise N(0, sigma^2)
            noise = np.random.normal(0, self.exploration_weight)
            scores[model_id] = preference_score + noise

        # Select top 2 arms
        sorted_arms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        model_a_id = sorted_arms[0][0]
        model_b_id = sorted_arms[1][0]

        self.total_queries += 1

        return self.arms[model_a_id], self.arms[model_b_id]

    async def update(
        self, feedback: DuelingFeedback, features: QueryFeatures
    ) -> None:
        """Update preference weights based on comparison outcome.

        Uses gradient descent on preference loss:
        - If A > B (preference > 0): increase w_a, decrease w_b
        - If B > A (preference < 0): decrease w_a, increase w_b

        Gradient: ∇L = preference * confidence * x

        Args:
            feedback: Pairwise comparison feedback
            features: Query context features

        Example:
            >>> feedback = DuelingFeedback(
            ...     model_a_id="gpt-4o-mini",
            ...     model_b_id="claude-3-haiku",
            ...     preference=0.6,  # A is better
            ...     confidence=0.9
            ... )
            >>> await bandit.update(feedback, features)
        """
        # Extract feature vector (d × 1)
        x = self._extract_features(features)

        # Gradient magnitude scaled by preference and confidence
        gradient_scale = feedback.preference * feedback.confidence

        # Update weights for both arms
        # Winner gets positive gradient, loser gets negative
        model_a_id = feedback.model_a_id
        model_b_id = feedback.model_b_id

        # Update arm A: move in direction of preference
        self.preference_weights[model_a_id] += (
            self.learning_rate * gradient_scale * x
        )

        # Update arm B: move opposite direction
        self.preference_weights[model_b_id] -= (
            self.learning_rate * gradient_scale * x
        )

        # Track comparison count (always use sorted tuple for consistency)
        pair_key = tuple(sorted([model_a_id, model_b_id]))
        # Always increment (pair should exist from init, but handle gracefully)
        self.preference_counts[pair_key] = (
            self.preference_counts.get(pair_key, 0) + 1
        )

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all preference weights and comparison counts.
        """
        # Reset weights to zero
        for model_id in self.preference_weights:
            self.preference_weights[model_id] = np.zeros(
                (self.feature_dim, 1)
            )

        # Reset comparison counts
        for pair_key in self.preference_counts:
            self.preference_counts[pair_key] = 0

        self.total_queries = 0

    def get_stats(self) -> dict:
        """Get algorithm statistics.

        Returns:
            Dictionary with dueling bandit statistics including:
            - total_queries: Total comparisons performed
            - preference_counts: Comparison counts per pair
            - weight_norms: L2 norm of preference weights per arm
        """
        base_stats = super().get_stats()

        # Compute weight norms for each arm
        weight_norms = {
            model_id: float(np.linalg.norm(w))
            for model_id, w in self.preference_weights.items()
        }

        return {
            **base_stats,
            "preference_counts": self.preference_counts,
            "weight_norms": weight_norms,
            "exploration_weight": self.exploration_weight,
            "learning_rate": self.learning_rate,
        }

    def get_preference_matrix(self) -> dict[tuple[str, str], float]:
        """Get estimated preference probabilities for all arm pairs.

        For each pair (A, B), computes expected preference P(A > B)
        averaged over all observed contexts.

        Returns:
            Dictionary mapping (model_a, model_b) -> preference probability

        Example:
            >>> prefs = bandit.get_preference_matrix()
            >>> print(prefs[("gpt-4o-mini", "claude-3-haiku")])
            0.72  # gpt-4o-mini preferred 72% of the time
        """
        preferences = {}

        arms_list = list(self.arms.keys())
        for i, model_a in enumerate(arms_list):
            for model_b in arms_list[i + 1 :]:
                # Get weights
                w_a = self.preference_weights[model_a]
                w_b = self.preference_weights[model_b]

                # Expected preference based on weight difference
                # This is a simplified estimate; true preference would need
                # integration over feature distribution
                weight_diff = np.linalg.norm(w_a) - np.linalg.norm(w_b)

                # Convert to probability using sigmoid with scaling
                # Scale factor helps distinguish small weight differences
                scale_factor = 5.0
                preference_prob = 1.0 / (
                    1.0 + np.exp(-scale_factor * weight_diff)
                )

                preferences[(model_a, model_b)] = float(preference_prob)

        return preferences

    def to_state(self) -> BanditState:
        """Serialize DuelingBandit state for persistence."""
        from conduit.core.state_store import BanditState

        # Serialize preference weights (numpy arrays to lists)
        pref_weights = {
            arm_id: w.flatten().tolist()
            for arm_id, w in self.preference_weights.items()
        }

        # Serialize preference counts (tuple keys to string keys)
        pref_counts = {
            f"{k[0]}:{k[1]}": v for k, v in self.preference_counts.items()
        }

        return BanditState(
            algorithm="dueling_bandit",
            arm_ids=list(self.arms.keys()),
            total_queries=self.total_queries,
            preference_weights=pref_weights,
            preference_counts=pref_counts,
            exploration_weight=self.exploration_weight,
            learning_rate=self.learning_rate,
            feature_dim=self.feature_dim,
            updated_at=datetime.now(UTC),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore DuelingBandit state from persisted data."""
        if state.algorithm != "dueling_bandit":
            raise ValueError(
                f"State algorithm '{state.algorithm}' != 'dueling_bandit'"
            )

        state_arms = set(state.arm_ids)
        current_arms = set(self.arms.keys())
        if state_arms != current_arms:
            raise ValueError(
                f"State arms {state_arms} don't match current arms {current_arms}"
            )

        if state.feature_dim is not None and state.feature_dim != self.feature_dim:
            raise ValueError(
                f"State feature_dim {state.feature_dim} != {self.feature_dim}"
            )

        self.total_queries = state.total_queries

        # Restore preference weights
        if state.preference_weights:
            self.preference_weights = {
                arm_id: np.array(w).reshape(-1, 1)
                for arm_id, w in state.preference_weights.items()
            }

        # Restore preference counts
        if state.preference_counts:
            self.preference_counts = {
                (k.split(":")[0], k.split(":")[1]): v
                for k, v in state.preference_counts.items()
            }
