"""Epsilon-Greedy bandit algorithm.

Epsilon-Greedy is one of the simplest exploration/exploitation strategies.
With probability ε (epsilon), explore by selecting a random arm.
With probability (1-ε), exploit by selecting the arm with highest mean reward.

Supports sliding window for non-stationarity: maintains only recent N observations
to adapt to model quality/cost changes over time.

Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions
"""

from __future__ import annotations

import random
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from conduit.core.config import load_algorithm_config
from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class EpsilonGreedyBandit(BanditAlgorithm):
    """Epsilon-Greedy algorithm with decaying exploration rate.

    Selects arm using epsilon-greedy policy:
    - With probability ε: Select random arm (exploration)
    - With probability (1-ε): Select arm with highest mean reward (exploitation)

    Supports epsilon decay to reduce exploration over time.

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        epsilon: Exploration probability (0-1)
        decay: Epsilon decay rate per query (default: no decay)
        min_epsilon: Minimum epsilon value (default: 0.01)
        mean_reward: Average reward for each arm
        sum_reward: Cumulative reward for each arm
        arm_pulls: Number of pulls for each arm
    """

    def __init__(
        self,
        arms: list[ModelArm],
        epsilon: float | None = None,
        decay: float | None = None,
        min_epsilon: float | None = None,
        random_seed: int | None = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
        success_threshold: float | None = None,
    ) -> None:
        """Initialize Epsilon-Greedy algorithm.

        Args:
            arms: List of available model arms
            epsilon: Initial exploration probability (default: EPSILON_GREEDY_DEFAULT)
            decay: Epsilon decay multiplier per query (default: EPSILON_DECAY_DEFAULT)
            min_epsilon: Minimum epsilon value after decay (default: EPSILON_MIN_DEFAULT)
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
            >>> # Static epsilon (10% exploration forever)
            >>> bandit1 = EpsilonGreedyBandit(arms, epsilon=0.1)
            >>>
            >>> # Decaying epsilon (start 20%, decay to 1% over time)
            >>> bandit2 = EpsilonGreedyBandit(arms, epsilon=0.2, decay=0.999, min_epsilon=0.01)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit3 = EpsilonGreedyBandit(arms, window_size=1000)
        """
        # Load config if parameters not provided
        if (
            epsilon is None
            or decay is None
            or min_epsilon is None
            or success_threshold is None
        ):
            config = load_algorithm_config("epsilon_greedy")
            if epsilon is None:
                epsilon = config["epsilon"]
            if decay is None:
                decay = config["decay"]
            if min_epsilon is None:
                min_epsilon = config["min_epsilon"]
            if success_threshold is None:
                success_threshold = config.get("success_threshold", 0.85)

        # Validate epsilon parameter
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"Epsilon must be between 0 and 1, got {epsilon}")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError(f"min_epsilon must be between 0 and 1, got {min_epsilon}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(
                f"decay must be between 0 and 1 (exclusive of 0), got {decay}"
            )

        super().__init__(name="epsilon_greedy", arms=arms)

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.window_size = window_size
        self.success_threshold = success_threshold

        # Multi-objective reward weights (Phase 3)
        if reward_weights is None:
            self.reward_weights = {"quality": 0.70, "cost": 0.20, "latency": 0.10}
        else:
            self.reward_weights = reward_weights

        # Sliding window: Store recent rewards per arm (Phase 3 - Non-stationarity)
        # If window_size > 0, use deque with maxlen. Otherwise, use deque (unlimited).
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

        # Track successes and exploration/exploitation counts
        self.arm_successes = {arm.model_id: 0 for arm in arms}
        self.exploration_count = 0
        self.exploitation_count = 0

        # Set random seed for reproducibility
        self.random_state = (
            random.Random(random_seed) if random_seed is not None else random
        )
        if random_seed is not None:
            np.random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm using epsilon-greedy policy.

        With probability ε: random arm (exploration)
        With probability (1-ε): best arm by mean reward (exploitation)

        Args:
            context: Query context (not used in basic epsilon-greedy)

        Returns:
            Selected model arm

        Example:
            >>> features = QueryFeatures(embedding=[0.1]*384, token_count=10, complexity_score=0.5, domain="general", domain_confidence=0.8, query_text="What is 2+2?")
            >>> arm = await bandit.select_arm(context)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        # Decide: explore or exploit?
        if self.random_state.random() < self.epsilon:
            # EXPLORE: Random arm
            selected_id = self.random_state.choice(list(self.arms.keys()))
            selected_arm = self.arms[selected_id]
            self.exploration_count += 1
        else:
            # EXPLOIT: Best arm by mean reward
            # For arms never pulled, use expected_quality from model metadata
            best_reward = -float("inf")
            best_id = None

            for model_id, arm in self.arms.items():
                # Use observed mean if available, else prior expected quality
                if self.arm_pulls[model_id] > 0:
                    reward = self.mean_reward[model_id]
                else:
                    reward = arm.expected_quality

                if reward > best_reward:
                    best_reward = reward
                    best_id = model_id

            selected_id = best_id or list(self.arms.keys())[0]  # Fallback to first arm
            selected_arm = self.arms[selected_id]
            self.exploitation_count += 1

        # Track total queries only (arm_pulls incremented by update())
        self.total_queries += 1

        # Decay epsilon
        if self.decay < 1.0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

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

        Clears all learned parameters, reward history, and restores initial epsilon.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
            >>> bandit.epsilon == bandit.initial_epsilon
            True
        """
        self.epsilon = self.initial_epsilon
        self.mean_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.sum_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.arm_successes = {arm.model_id: 0 for arm in self.arm_list}
        self.exploration_count = 0
        self.exploitation_count = 0

        # Clear reward history
        for model_id in self.reward_history:
            self.reward_history[model_id].clear()

        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
        """Get algorithm statistics.

        Returns:
            Dictionary with statistics including:
            - total_queries: Total number of queries processed
            - current_epsilon: Current exploration probability
            - exploration_count: Number of exploration actions
            - exploitation_count: Number of exploitation actions
            - arm_pulls: Number of times each arm was selected
            - arm_mean_reward: Average reward for each arm

        Example:
            >>> stats = bandit.get_stats()
            >>> print(f"Exploration: {stats['exploration_count']}")
            >>> print(f"Exploitation: {stats['exploitation_count']}")
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

        # Calculate exploration ratio
        total_actions = self.exploration_count + self.exploitation_count
        exploration_ratio = (
            self.exploration_count / total_actions if total_actions > 0 else 0.0
        )

        return {
            **base_stats,
            "initial_epsilon": self.initial_epsilon,
            "current_epsilon": self.epsilon,
            "decay": self.decay,
            "min_epsilon": self.min_epsilon,
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "exploration_ratio": exploration_ratio,
            "arm_pulls": self.arm_pulls,
            "arm_mean_reward": self.mean_reward,
            "arm_sum_reward": self.sum_reward,
            "arm_successes": self.arm_successes,
            "arm_success_rates": success_rates,
        }

    def to_state(self) -> BanditState:
        """Serialize Epsilon-Greedy state for persistence.

        Returns:
            BanditState object containing all Epsilon-Greedy state

        Example:
            >>> state = bandit.to_state()
            >>> state.algorithm
            "epsilon_greedy"
        """
        from conduit.core.state_store import BanditState

        # Convert reward history deques to list of dicts for serialization
        reward_history_serialized = []
        for arm_id, rewards in self.reward_history.items():
            for reward in rewards:
                reward_history_serialized.append({"arm_id": arm_id, "reward": reward})

        return BanditState(
            algorithm="epsilon_greedy",
            arm_ids=list(self.arms.keys()),
            arm_pulls=self.arm_pulls.copy(),
            arm_successes=self.arm_successes.copy(),
            total_queries=self.total_queries,
            mean_reward=self.mean_reward.copy(),
            sum_reward=self.sum_reward.copy(),
            reward_history=reward_history_serialized,
            epsilon=self.epsilon,
            window_size=self.window_size if self.window_size > 0 else None,
            updated_at=datetime.now(UTC),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore Epsilon-Greedy state from persisted data.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with current configuration

        Example:
            >>> state = await store.load_bandit_state("router-1", "epsilon_greedy")
            >>> bandit.from_state(state)
        """
        if state.algorithm != "epsilon_greedy":
            raise ValueError(f"State algorithm '{state.algorithm}' != 'epsilon_greedy'")

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

        # Restore epsilon if present
        if state.epsilon is not None:
            self.epsilon = state.epsilon

        # Restore reward history
        for arm_id in self.arms:
            self.reward_history[arm_id].clear()

        for entry in state.reward_history:
            arm_id = entry["arm_id"]
            reward = entry["reward"]
            if arm_id in self.reward_history:
                self.reward_history[arm_id].append(reward)
