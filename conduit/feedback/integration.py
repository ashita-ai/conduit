"""Feedback integration for Thompson Sampling bandit.

Converts explicit and implicit feedback signals into reward updates
for the Thompson Sampling contextual bandit. Implements weighted
combination of explicit user ratings and implicit behavioral signals.
"""

import logging
from typing import Any

from conduit.core.models import Feedback, ImplicitFeedback, QueryFeatures
from conduit.engines.bandit import ContextualBandit

logger = logging.getLogger(__name__)


class FeedbackIntegrator:
    """Integrates feedback signals with Thompson Sampling bandit.

    Converts behavioral signals (errors, latency, retries) and explicit
    user feedback (ratings, expectations) into reward updates for the
    contextual bandit's Beta distributions.

    Reward Mapping:
        - Error occurred: 0.0 (complete failure)
        - Retry detected: 0.3 (dissatisfaction, user tried again)
        - Low latency tolerance: 0.5 (user waited but slow)
        - Medium latency tolerance: 0.7 (acceptable speed)
        - High latency tolerance: 0.9 (fast response)
        - Explicit negative: 0.2 (user rated poorly)
        - Explicit positive: 1.0 (user rated highly)

    Weighting Strategy:
        - Explicit feedback: 0.7 weight (user explicitly rates quality)
        - Implicit feedback: 0.3 weight (behavioral signals less reliable)
        - Combined: weighted_reward = explicit * 0.7 + implicit * 0.3

    Example:
        >>> integrator = FeedbackIntegrator(bandit)
        >>> integrator.update_from_explicit(
        ...     model="gpt-4o-mini",
        ...     features=query_features,
        ...     feedback=explicit_feedback
        ... )
        >>> integrator.update_from_implicit(
        ...     model="gpt-4o-mini",
        ...     features=query_features,
        ...     feedback=implicit_feedback
        ... )
    """

    def __init__(
        self,
        bandit: ContextualBandit,
        explicit_weight: float = 0.7,
        implicit_weight: float = 0.3,
    ):
        """Initialize feedback integrator.

        Args:
            bandit: Contextual bandit to update
            explicit_weight: Weight for explicit user feedback (0.7 default)
            implicit_weight: Weight for implicit behavioral signals (0.3 default)

        Note:
            Weights should sum to 1.0 for proper reward scaling.
        """
        self.bandit = bandit
        self.explicit_weight = explicit_weight
        self.implicit_weight = implicit_weight

        if abs(explicit_weight + implicit_weight - 1.0) > 0.01:
            logger.warning(
                f"Feedback weights don't sum to 1.0: "
                f"explicit={explicit_weight}, implicit={implicit_weight}"
            )

    def update_from_explicit(
        self,
        model: str,
        features: QueryFeatures,
        feedback: Feedback,
    ) -> None:
        """Update bandit from explicit user feedback.

        Args:
            model: Model that generated response
            features: Query features (context for contextual bandit)
            feedback: Explicit feedback (quality score, expectations met)

        Algorithm:
            1. Convert explicit feedback to reward (0.0-1.0)
            2. Apply explicit weight (0.7)
            3. Update Thompson Sampling Beta distributions
        """
        # Convert explicit feedback to reward
        reward = self._explicit_to_reward(feedback)

        # Apply explicit weight
        weighted_reward = reward * self.explicit_weight

        # Update bandit
        self.bandit.update(
            model=model,
            reward=weighted_reward,
            features=features,
        )

        logger.debug(
            f"Explicit feedback update: model={model}, "
            f"raw_reward={reward:.2f}, weighted_reward={weighted_reward:.2f}"
        )

    def update_from_implicit(
        self,
        model: str,
        features: QueryFeatures,
        feedback: ImplicitFeedback,
    ) -> None:
        """Update bandit from implicit behavioral signals.

        Args:
            model: Model that generated response
            features: Query features (context for contextual bandit)
            feedback: Implicit feedback (errors, latency, retries)

        Algorithm:
            1. Convert implicit signals to reward (0.0-1.0)
            2. Apply implicit weight (0.3)
            3. Update Thompson Sampling Beta distributions

        Note:
            Multiple signals can be detected simultaneously.
            Priority: Error > Retry > Latency (most to least reliable).
        """
        # Convert implicit signals to reward
        reward = self._implicit_to_reward(feedback)

        # Apply implicit weight
        weighted_reward = reward * self.implicit_weight

        # Update bandit
        self.bandit.update(
            model=model,
            reward=weighted_reward,
            features=features,
        )

        logger.debug(
            f"Implicit feedback update: model={model}, "
            f"raw_reward={reward:.2f}, weighted_reward={weighted_reward:.2f}, "
            f"signals=[error={feedback.error_occurred}, "
            f"retry={feedback.retry_detected}, "
            f"latency={feedback.latency_tolerance}]"
        )

    def update_combined(
        self,
        model: str,
        features: QueryFeatures,
        explicit: Feedback | None = None,
        implicit: ImplicitFeedback | None = None,
    ) -> None:
        """Update bandit from combined explicit and implicit feedback.

        Args:
            model: Model that generated response
            features: Query features
            explicit: Optional explicit user feedback
            implicit: Optional implicit behavioral signals

        Note:
            If both provided, applies proper weighting (0.7 + 0.3).
            If only one provided, uses that signal with full weight.
        """
        if explicit is not None:
            self.update_from_explicit(model, features, explicit)

        if implicit is not None:
            self.update_from_implicit(model, features, implicit)

    def _explicit_to_reward(self, feedback: Feedback) -> float:
        """Convert explicit feedback to reward score.

        Args:
            feedback: Explicit user feedback

        Returns:
            Reward score (0.0 to 1.0)

        Algorithm:
            1. Start with quality_score (already 0-1 scale)
            2. Adjust based on met_expectations boolean
            3. Combine with 60% quality, 40% expectations weighting
        """
        # Quality score is already 0-1
        quality_reward = feedback.quality_score

        # Expectations met: 1.0 if true, 0.0 if false
        expectations_reward = 1.0 if feedback.met_expectations else 0.0

        # Weighted combination: 60% quality, 40% expectations
        reward = (quality_reward * 0.6) + (expectations_reward * 0.4)

        return float(reward)

    def _implicit_to_reward(self, feedback: ImplicitFeedback) -> float:
        """Convert implicit signals to reward score.

        Args:
            feedback: Implicit behavioral signals

        Returns:
            Reward score (0.0 to 1.0)

        Algorithm (Priority Order):
            1. Error occurred → 0.0 (complete failure)
            2. Retry detected → 0.3 (user dissatisfied, tried again)
            3. Latency tolerance:
               - high (fast): 0.9
               - medium (acceptable): 0.7
               - low (slow but patient): 0.5

        Note:
            Takes first applicable signal in priority order.
            Error is most reliable, latency least reliable.
        """
        # Priority 1: Error occurred (most reliable signal)
        if feedback.error_occurred:
            return 0.0

        # Priority 2: Retry detected (strong dissatisfaction signal)
        if feedback.retry_detected:
            return 0.3

        # Priority 3: Latency tolerance (weakest signal, but informative)
        latency_rewards = {
            "high": 0.9,  # Fast response, no patience needed
            "medium": 0.7,  # Acceptable speed
            "low": 0.5,  # Slow but user waited patiently
        }

        reward = latency_rewards.get(feedback.latency_tolerance, 0.7)
        return float(reward)

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback integration statistics.

        Returns:
            Dictionary with update counts and reward distributions

        Note:
            Phase 1 returns basic stats. Phase 2+ will add:
            - Reward distribution histograms
            - Signal frequency analysis
            - Model-specific feedback patterns
        """
        return {
            "explicit_weight": self.explicit_weight,
            "implicit_weight": self.implicit_weight,
            "bandit_states": {
                model: {
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "mean_success_rate": state.mean_success_rate,
                }
                for model, state in self.bandit.model_states.items()
            },
        }
