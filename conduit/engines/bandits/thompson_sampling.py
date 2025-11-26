"""Thompson Sampling bandit algorithm for LLM routing.

Thompson Sampling uses Bayesian inference to balance exploration and exploitation.
For each arm, we maintain a Beta distribution Beta(α, β) representing our belief
about the arm's quality. We sample from these distributions and select the arm
with the highest sample.

Supports sliding window for non-stationarity: maintains only recent N observations
to adapt to model quality/cost changes over time.

Reference: https://en.wikipedia.org/wiki/Thompson_sampling
"""

from __future__ import annotations

import random
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np  # type: ignore[import-untyped,unused-ignore]

from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class ThompsonSamplingBandit(BanditAlgorithm):
    """Thompson Sampling with Beta-Bernoulli conjugate prior.

    Maintains Beta(α, β) distribution for each arm where:
    - α (alpha): Number of successes + prior
    - β (beta): Number of failures + prior

    For quality scores in [0, 1], we treat score as success probability.

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        alpha: Success counts for each arm (Beta distribution)
        beta: Failure counts for each arm (Beta distribution)
        prior_alpha: Prior belief about success rate (default: 1.0)
        prior_beta: Prior belief about failure rate (default: 1.0)
    """

    def __init__(
        self,
        arms: list[ModelArm],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        random_seed: int | None = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
        success_threshold: float = 0.85,
    ) -> None:
        """Initialize Thompson Sampling algorithm.

        Args:
            arms: List of available model arms
            prior_alpha: Prior successes (higher = more optimistic)
            prior_beta: Prior failures (higher = more pessimistic)
            random_seed: Random seed for reproducibility
            reward_weights: Multi-objective reward weights. If None, uses defaults
                (quality: 0.70, cost: 0.20, latency: 0.10)
            window_size: Sliding window size for non-stationarity.
                0 = unlimited history (default), N = keep only last N rewards per arm
            success_threshold: Reward threshold for counting successes (default: 0.85)
                Only used for statistics, not algorithm decisions

        Example:
            >>> arms = [
            ...     ModelArm(model_id="gpt-4o", provider="openai", ...),
            ...     ModelArm(model_id="claude-3-5-sonnet", provider="anthropic", ...)
            ... ]
            >>> # Unlimited history (stationary environment)
            >>> bandit1 = ThompsonSamplingBandit(arms, prior_alpha=1.0, prior_beta=1.0)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit2 = ThompsonSamplingBandit(arms, window_size=1000)
        """
        super().__init__(name="thompson_sampling", arms=arms)

        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.window_size = window_size
        self.success_threshold = success_threshold

        # Multi-objective reward weights (Phase 3)
        if reward_weights is None:
            self.reward_weights = {"quality": 0.70, "cost": 0.20, "latency": 0.10}
        else:
            self.reward_weights = reward_weights

        # Sliding window: Store recent rewards per arm (Phase 3 - Non-stationarity)
        # If window_size > 0, use deque with maxlen. Otherwise, use list (unlimited).
        self.reward_history: dict[str, deque[float]]
        if window_size > 0:
            self.reward_history = {
                arm.model_id: deque(maxlen=window_size) for arm in arms
            }
        else:
            # Use list for unlimited history (no maxlen)
            self.reward_history = {arm.model_id: deque() for arm in arms}

        # Initialize Beta distributions for each arm
        self.alpha = {arm.model_id: prior_alpha for arm in arms}
        self.beta = {arm.model_id: prior_beta for arm in arms}

        # Track arm pulls for statistics
        self.arm_pulls = {arm.model_id: 0 for arm in arms}
        self.arm_successes = {arm.model_id: 0 for arm in arms}

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm using Thompson Sampling.

        For each arm, sample from Beta(α, β) and select arm with highest sample.

        Args:
            features: Query features (not used in basic Thompson Sampling)

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
            >>> arm = await bandit.select_arm(features)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        # Sample from Beta distribution for each arm
        samples = {}
        for model_id in self.arms:
            alpha = self.alpha[model_id]
            beta = self.beta[model_id]
            # Sample from Beta(α, β)
            samples[model_id] = np.random.beta(alpha, beta)

        # Select arm with highest sample
        selected_id = max(samples, key=samples.get)  # type: ignore
        selected_arm = self.arms[selected_id]

        # Track total queries only (arm_pulls incremented by update())
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update Beta distribution with feedback.

        Uses multi-objective reward (quality + cost + latency) as success probability.

        With sliding window (window_size > 0):
        - Stores reward in history deque (automatically drops oldest when full)
        - Recalculates α, β from all rewards in current window + prior

        Without window (window_size = 0):
        - Incremental update: α += r, β += (1 - r)

        Args:
            feedback: Feedback from model execution
            features: Original query features (not used)

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await bandit.update(feedback, features)
        """
        model_id = feedback.model_id

        # Validate model_id exists in available arms
        if model_id not in self.arms:
            raise ValueError(
                f"Model ID '{model_id}' not in arms. "
                f"Available: {list(self.arms.keys())}"
            )

        # Calculate composite reward from quality, cost, and latency (Phase 3)
        reward = feedback.calculate_reward(
            quality_weight=self.reward_weights["quality"],
            cost_weight=self.reward_weights["cost"],
            latency_weight=self.reward_weights["latency"],
        )

        # Add reward to history
        self.reward_history[model_id].append(reward)

        # Recalculate alpha/beta from windowed history
        # α = prior_alpha + sum(rewards in window)
        # β = prior_beta + sum(1 - reward for reward in window)
        window_rewards = list(self.reward_history[model_id])
        self.alpha[model_id] = self.prior_alpha + sum(window_rewards)
        self.beta[model_id] = self.prior_beta + sum(1.0 - r for r in window_rewards)

        # Track statistics
        self.arm_pulls[model_id] += 1  # Always increment for feedback count
        if reward >= self.success_threshold:
            self.arm_successes[model_id] += 1

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters, reward history, and reverts to prior.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
        """
        self.alpha = {arm.model_id: self.prior_alpha for arm in self.arm_list}
        self.beta = {arm.model_id: self.prior_beta for arm in self.arm_list}
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.arm_successes = {arm.model_id: 0 for arm in self.arm_list}

        # Clear reward history
        for model_id in self.reward_history:
            self.reward_history[model_id].clear()

        self.total_queries = 0

    def get_stats(self) -> dict[str, Any]:
        """Get algorithm statistics.

        Returns:
            Dictionary with statistics including:
            - total_queries: Total number of queries processed
            - arm_pulls: Number of times each arm was selected
            - arm_success_rates: Success rate for each arm
            - arm_distributions: Current Beta distribution parameters

        Example:
            >>> stats = bandit.get_stats()
            >>> print(stats["arm_pulls"])
            {"openai:gpt-4o-mini": 500, "claude-3-5-sonnet": 300, ...}
        """
        base_stats = super().get_stats()

        # Calculate success rates
        success_rates = {}
        for model_id in self.arms:
            pulls = self.arm_pulls[model_id]
            if pulls > 0:
                success_rates[model_id] = self.arm_successes[model_id] / pulls
            else:
                success_rates[model_id] = 0.0

        # Get current distribution parameters
        distributions = {}
        for model_id in self.arms:
            distributions[model_id] = {
                "alpha": self.alpha[model_id],
                "beta": self.beta[model_id],
                "mean": self.alpha[model_id]
                / (self.alpha[model_id] + self.beta[model_id]),
            }

        return {
            **base_stats,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "arm_pulls": self.arm_pulls,
            "arm_successes": self.arm_successes,
            "arm_success_rates": success_rates,
            "arm_distributions": distributions,
        }

    def to_state(self) -> BanditState:
        """Serialize Thompson Sampling state for persistence.

        Returns:
            BanditState object containing all Thompson Sampling state

        Example:
            >>> state = bandit.to_state()
            >>> state.algorithm
            "thompson_sampling"
        """
        from conduit.core.state_store import BanditState

        # Convert reward history deques to list of dicts for serialization
        reward_history_serialized = []
        for arm_id, rewards in self.reward_history.items():
            for reward in rewards:
                reward_history_serialized.append({"arm_id": arm_id, "reward": reward})

        return BanditState(
            algorithm="thompson_sampling",
            arm_ids=list(self.arms.keys()),
            arm_pulls=self.arm_pulls.copy(),
            arm_successes=self.arm_successes.copy(),
            total_queries=self.total_queries,
            alpha_params=self.alpha.copy(),
            beta_params=self.beta.copy(),
            reward_history=reward_history_serialized,
            window_size=self.window_size if self.window_size > 0 else None,
            updated_at=datetime.now(UTC),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore Thompson Sampling state from persisted data.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with current configuration

        Example:
            >>> state = await store.load_bandit_state("router-1", "thompson_sampling")
            >>> bandit.from_state(state)
        """
        if state.algorithm != "thompson_sampling":
            raise ValueError(
                f"State algorithm '{state.algorithm}' mismatch: expected 'thompson_sampling'"
            )

        # Verify arms match
        state_arms = set(state.arm_ids)
        current_arms = set(self.arms.keys())
        if state_arms != current_arms:
            raise ValueError(
                f"State arms {state_arms} don't match current arms {current_arms}"
            )

        # Restore counters
        self.total_queries = state.total_queries
        self.arm_pulls = state.arm_pulls.copy()
        self.arm_successes = state.arm_successes.copy()
        self.alpha = state.alpha_params.copy()
        self.beta = state.beta_params.copy()

        # Restore reward history
        for arm_id in self.arms:
            self.reward_history[arm_id].clear()

        for entry in state.reward_history:
            arm_id = entry["arm_id"]
            reward = entry["reward"]
            if arm_id in self.reward_history:
                self.reward_history[arm_id].append(reward)
