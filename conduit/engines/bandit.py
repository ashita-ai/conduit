"""Thompson Sampling contextual bandit for model selection."""

import logging
from datetime import datetime, timezone

import numpy as np  # type: ignore[import-untyped,unused-ignore]

from conduit.core.models import ModelState, QueryFeatures

logger = logging.getLogger(__name__)


class ContextualBandit:
    """Thompson Sampling for multi-armed bandit model selection.

    Maintains Beta distribution for each model's success rate.
    Samples from distributions and selects model with highest sample.

    Algorithm:
        1. For each model, sample success probability from Beta(α, β)
        2. Weight sample by predicted reward for this query
        3. Select model with highest weighted sample
        4. Update Beta parameters based on observed reward
    """

    def __init__(self, models: list[str]):
        """Initialize bandit with uniform priors.

        Args:
            models: List of model IDs (e.g., ["gpt-4o-mini", "claude-sonnet-4"])
        """
        self.model_states: dict[str, ModelState] = {
            model: ModelState(model_id=model) for model in models
        }

    def select_model(
        self, features: QueryFeatures, models: list[str] | None = None
    ) -> str:
        """Select model using Thompson Sampling.

        Args:
            features: Extracted query features
            models: Optional subset of models to consider (for constraint filtering)

        Returns:
            Selected model ID

        Example:
            >>> bandit = ContextualBandit(["gpt-4o-mini", "gpt-4o"])
            >>> features = QueryFeatures(...)
            >>> model = bandit.select_model(features)
            >>> model in ["gpt-4o-mini", "gpt-4o"]
            True
        """
        if models is None:
            models = list(self.model_states.keys())

        if not models:
            raise ValueError("No models available for selection")

        # Sample from each model's Beta distribution
        samples: dict[str, float] = {}
        for model in models:
            state = self.model_states[model]

            # Sample success probability from Beta(α, β)
            theta = np.random.beta(state.alpha, state.beta)

            # Weight by predicted reward for this query
            predicted_reward = self._predict_reward(model, features)
            samples[model] = theta * predicted_reward

            logger.debug(
                f"Thompson sample: {model} θ={theta:.3f} "
                f"predicted_reward={predicted_reward:.3f} "
                f"weighted={samples[model]:.3f}"
            )

        # Select model with highest weighted sample
        selected_model = max(samples, key=samples.get)  # type: ignore

        logger.info(
            f"Selected {selected_model} with score {samples[selected_model]:.3f}"
        )

        return selected_model

    def update(
        self, model: str, reward: float, query_id: str, success_threshold: float = 0.7
    ) -> None:
        """Update model's Beta distribution using Thompson Sampling.

        Args:
            model: Model that was used
            reward: Observed reward (0.0-1.0)
            query_id: For audit trail
            success_threshold: Reward threshold to count as success (default 0.7)

        Note:
            Thompson Sampling requires binary success/failure updates.
            We convert continuous reward to binary outcome using threshold.

        Example:
            >>> bandit = ContextualBandit(["gpt-4o-mini"])
            >>> bandit.update("gpt-4o-mini", reward=0.85, query_id="q123")
            >>> bandit.model_states["gpt-4o-mini"].alpha > 1.0
            True
        """
        if model not in self.model_states:
            raise ValueError(f"Unknown model: {model}")

        state = self.model_states[model]

        # Bayesian update with binary success/failure
        if reward >= success_threshold:
            state.alpha += 1.0  # Count success
            success = True
        else:
            state.beta += 1.0  # Count failure
            success = False

        state.total_requests += 1
        state.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Updated {model}: α={state.alpha:.2f}, β={state.beta:.2f}, "
            f"reward={reward:.2f}, success={success}, query={query_id}"
        )

    def _predict_reward(self, model: str, features: QueryFeatures) -> float:
        """Predict expected reward for model on this query.

        Phase 1: Simple heuristic based on complexity.
        Phase 2+: Neural network or linear model.

        Args:
            model: Model identifier
            features: Query features

        Returns:
            Predicted reward (0.0-1.0)
        """
        # Model cost tiers (approximate)
        cost_tier = {
            "gpt-4o-mini": 0.1,
            "gpt-4o": 0.5,
            "claude-opus-4": 1.0,
            "claude-sonnet-4": 0.3,
        }.get(model, 0.5)

        complexity = features.complexity_score

        # Simple heuristic: match model tier to complexity
        if complexity < 0.3:
            # Simple queries: prefer cheap models
            reward = 1.0 if cost_tier < 0.3 else 0.5
        elif complexity < 0.7:
            # Medium complexity: prefer balanced models
            reward = 1.0 if 0.2 <= cost_tier <= 0.6 else 0.7
        else:
            # Complex queries: prefer premium models
            reward = 1.0 if cost_tier >= 0.5 else 0.3

        return reward

    def get_confidence(self, model: str) -> float:
        """Get confidence in model's performance.

        Args:
            model: Model identifier

        Returns:
            Confidence score based on Beta distribution variance

        Example:
            >>> bandit = ContextualBandit(["gpt-4o-mini"])
            >>> confidence = bandit.get_confidence("gpt-4o-mini")
            >>> 0.0 <= confidence <= 1.0
            True
        """
        if model not in self.model_states:
            return 0.0

        state = self.model_states[model]

        # Confidence inversely proportional to variance
        # High variance (uncertain) → low confidence
        # Low variance (certain) → high confidence
        variance = state.variance

        # Normalize to 0.0-1.0 range
        confidence = 1.0 / (1.0 + variance * 10)

        return min(1.0, max(0.0, confidence))

    def get_model_state(self, model: str) -> ModelState:
        """Get current state for a model.

        Args:
            model: Model identifier

        Returns:
            ModelState with Beta parameters

        Raises:
            ValueError: If model not found
        """
        if model not in self.model_states:
            raise ValueError(f"Unknown model: {model}")

        return self.model_states[model]

    def load_states(self, states: dict[str, ModelState]) -> None:
        """Load model states from database.

        Args:
            states: Dictionary mapping model_id to ModelState
        """
        self.model_states.update(states)
        logger.info(f"Loaded {len(states)} model states from database")
