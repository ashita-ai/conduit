"""Upper Confidence Bound (UCB) bandit algorithms.

UCB algorithms select arms based on optimistic estimates, choosing the arm
with the highest upper confidence bound on its reward. This balances exploration
(uncertainty) and exploitation (expected reward).

Supports sliding window for non-stationarity: maintains only recent N observations
to adapt to model quality/cost changes over time.

Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Upper_Confidence_Bounds
"""

from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from conduit.core.config import load_algorithm_config
from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class UCB1Bandit(BanditAlgorithm):
    """UCB1 algorithm for multi-armed bandits.

    Selects arm with highest upper confidence bound:
        UCB(arm) = mean_reward(arm) + c * sqrt(ln(total_pulls) / pulls(arm))

    Where:
    - mean_reward: Average reward received from this arm
    - c: Exploration parameter (default: sqrt(2))
    - total_pulls: Total number of arm pulls across all arms
    - pulls(arm): Number of times this specific arm was pulled

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        c: Exploration parameter (higher = more exploration)
        mean_reward: Average reward for each arm
        sum_reward: Cumulative reward for each arm
        arm_pulls: Number of pulls for each arm
    """

    def __init__(
        self,
        arms: list[ModelArm],
        c: float | None = None,
        random_seed: int | None = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
        success_threshold: float | None = None,
    ) -> None:
        """Initialize UCB1 algorithm.

        Args:
            arms: List of available model arms
            c: Exploration parameter (default: loaded from config)
            random_seed: Random seed for tie-breaking
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
            >>> bandit1 = UCB1Bandit(arms, c=1.5)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit2 = UCB1Bandit(arms, c=1.5, window_size=1000)
        """
        super().__init__(name="ucb1", arms=arms)

        # Load config if parameters not provided
        if c is None or success_threshold is None:
            config = load_algorithm_config("ucb1")
            if c is None:
                c = config["c"]
            if success_threshold is None:
                success_threshold = config.get("success_threshold", 0.85)

        self.c = c
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
            # Use deque for unlimited history (no maxlen)
            self.reward_history = {arm.model_id: deque() for arm in arms}

        # Initialize statistics for each arm
        self.mean_reward = {arm.model_id: 0.0 for arm in arms}
        self.sum_reward = {arm.model_id: 0.0 for arm in arms}
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

        # Track successes for statistics
        self.arm_successes = {arm.model_id: 0 for arm in arms}

        # Track explored arms for exploration phase (separate from feedback count)
        self.explored_arms: set[str] = set()

        if random_seed is not None:
            np.random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm using UCB1 policy.

        Initially, pull each arm once (exploration phase).
        Then select arm with highest upper confidence bound.

        Args:
            context: Query context (not used in basic UCB1)

        Returns:
            Selected model arm

        Example:
            >>> features = QueryFeatures(embedding=[0.1]*384, token_count=10, complexity_score=0.5, domain="general", domain_confidence=0.8, query_text="What is 2+2?")
            >>> arm = await bandit.select_arm(context)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        # Exploration phase: Pull each arm at least once
        for model_id in self.arms:
            if model_id not in self.explored_arms:
                self.explored_arms.add(model_id)
                selected_arm = self.arms[model_id]
                self.total_queries += 1
                return selected_arm

        # Exploitation phase: Calculate UCB for each arm
        ucb_values = {}
        for model_id in self.arms:
            mean = self.mean_reward[model_id]
            pulls = self.arm_pulls[model_id]

            # If arm has no pulls yet (explored but no feedback), use infinite UCB
            if pulls == 0:
                ucb_values[model_id] = float("inf")
            else:
                # UCB = mean + c * sqrt(ln(total) / pulls)
                exploration_term = self.c * math.sqrt(
                    math.log(self.total_queries) / pulls
                )
                ucb_values[model_id] = mean + exploration_term

        # Select arm with highest UCB
        selected_id = max(ucb_values, key=ucb_values.get)  # type: ignore
        selected_arm = self.arms[selected_id]

        # Track total queries only (arm_pulls incremented by update())
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update arm statistics with feedback.

        Updates running mean reward for the selected arm using multi-objective
        reward function (quality + cost + latency).

        With sliding window (window_size > 0):
        - Stores reward in history deque (automatically drops oldest when full)
        - Recalculates mean and sum from all rewards in current window

        Without window (window_size = 0):
        - Incremental update: sum += reward, mean = sum / count

        Args:
            feedback: Feedback from model execution
            context: Original query context (not used)

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await bandit.update(feedback, context)
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

        # Recalculate mean and sum from windowed history
        window_rewards = list(self.reward_history[model_id])
        self.sum_reward[model_id] = sum(window_rewards)
        self.mean_reward[model_id] = (
            self.sum_reward[model_id] / len(window_rewards) if window_rewards else 0.0
        )

        # Track statistics (arm_pulls always = len(history) for consistency)
        self.arm_pulls[model_id] += 1  # Always increment for feedback count

        # Track successes (reward above threshold)
        if reward >= self.success_threshold:
            self.arm_successes[model_id] += 1

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters and reward history.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
        """
        self.mean_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.sum_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.arm_successes = {arm.model_id: 0 for arm in self.arm_list}
        self.explored_arms = set()

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
            - arm_mean_reward: Average reward for each arm
            - arm_ucb_values: Current UCB values for each arm

        Example:
            >>> stats = bandit.get_stats()
            >>> print(stats["arm_mean_reward"])
            {"openai:gpt-4o-mini": 0.92, "claude-3-5-sonnet": 0.89, ...}
        """
        base_stats = super().get_stats()

        # Calculate current UCB values
        ucb_values = {}
        for model_id in self.arms:
            pulls = self.arm_pulls[model_id]
            if pulls > 0 and self.total_queries > 0:
                mean = self.mean_reward[model_id]
                exploration = self.c * math.sqrt(math.log(self.total_queries) / pulls)
                ucb_values[model_id] = mean + exploration
            else:
                ucb_values[model_id] = float("inf")  # Not yet pulled

        # Calculate success rates
        success_rates = {}
        for model_id in self.arms:
            pulls = self.arm_pulls[model_id]
            if pulls > 0:
                success_rates[model_id] = self.arm_successes[model_id] / pulls
            else:
                success_rates[model_id] = 0.0

        return {
            **base_stats,
            "c": self.c,
            "arm_pulls": self.arm_pulls,
            "arm_mean_reward": self.mean_reward,
            "arm_sum_reward": self.sum_reward,
            "arm_ucb_values": ucb_values,
            "arm_successes": self.arm_successes,
            "arm_success_rates": success_rates,
        }

    def to_state(self) -> BanditState:
        """Serialize UCB1 state for persistence.

        Returns:
            BanditState object containing all UCB1 state

        Example:
            >>> state = bandit.to_state()
            >>> state.algorithm
            "ucb1"
        """
        from conduit.core.state_store import BanditState

        # Convert reward history deques to list of dicts for serialization
        reward_history_serialized = []
        for arm_id, rewards in self.reward_history.items():
            for reward in rewards:
                reward_history_serialized.append({"arm_id": arm_id, "reward": reward})

        return BanditState(
            algorithm="ucb1",
            arm_ids=list(self.arms.keys()),
            arm_pulls=self.arm_pulls.copy(),
            arm_successes=self.arm_successes.copy(),
            total_queries=self.total_queries,
            mean_reward=self.mean_reward.copy(),
            sum_reward=self.sum_reward.copy(),
            explored_arms=list(self.explored_arms),
            reward_history=reward_history_serialized,
            window_size=self.window_size if self.window_size > 0 else None,
            updated_at=datetime.now(UTC),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore UCB1 state from persisted data.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with current configuration

        Example:
            >>> state = await store.load_bandit_state("router-1", "ucb1")
            >>> bandit.from_state(state)
        """
        if state.algorithm != "ucb1":
            raise ValueError(f"State algorithm '{state.algorithm}' != 'ucb1'")

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
        self.mean_reward = state.mean_reward.copy()
        self.sum_reward = state.sum_reward.copy()
        self.explored_arms = set(state.explored_arms)

        # Restore reward history
        for arm_id in self.arms:
            self.reward_history[arm_id].clear()

        for entry in state.reward_history:
            arm_id = entry["arm_id"]
            reward = entry["reward"]
            if arm_id in self.reward_history:
                self.reward_history[arm_id].append(reward)
