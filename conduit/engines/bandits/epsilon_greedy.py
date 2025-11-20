"""Epsilon-Greedy bandit algorithm.

Epsilon-Greedy is one of the simplest exploration/exploitation strategies.
With probability ε (epsilon), explore by selecting a random arm.
With probability (1-ε), exploit by selecting the arm with highest mean reward.

Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions
"""

import random
from typing import Optional

import numpy as np

from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm


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
        epsilon: float = 0.1,
        decay: float = 1.0,
        min_epsilon: float = 0.01,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize Epsilon-Greedy algorithm.

        Args:
            arms: List of available model arms
            epsilon: Initial exploration probability (default: 0.1 = 10% exploration)
            decay: Epsilon decay multiplier per query (default: 1.0 = no decay)
            min_epsilon: Minimum epsilon value after decay (default: 0.01)
            random_seed: Random seed for reproducibility

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
        """
        super().__init__(name=f"epsilon_greedy_eps{epsilon}", arms=arms)

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

        # Initialize statistics for each arm
        self.mean_reward = {arm.model_id: 0.0 for arm in arms}
        self.sum_reward = {arm.model_id: 0.0 for arm in arms}
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

        # Track successes and exploration/exploitation counts
        self.arm_successes = {arm.model_id: 0 for arm in arms}
        self.exploration_count = 0
        self.exploitation_count = 0

        if random_seed is not None:
            random.seed(random_seed)
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
            >>> context = BanditContext(query_text="What is 2+2?")
            >>> arm = await bandit.select_arm(context)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        # Decide: explore or exploit?
        if random.random() < self.epsilon:
            # EXPLORE: Random arm
            selected_id = random.choice(list(self.arms.keys()))
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

        # Track selection
        self.arm_pulls[selected_id] += 1
        self.total_queries += 1

        # Decay epsilon
        if self.decay < 1.0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update arm statistics with feedback.

        Updates running mean reward for the selected arm.

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
        reward = feedback.quality_score

        # Update running statistics
        self.sum_reward[model_id] += reward
        pulls = self.arm_pulls[model_id]

        if pulls > 0:
            self.mean_reward[model_id] = self.sum_reward[model_id] / pulls

        # Track successes (quality above threshold)
        if reward >= 0.85:
            self.arm_successes[model_id] += 1

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters and restores initial epsilon.

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
