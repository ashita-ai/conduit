"""Upper Confidence Bound (UCB) bandit algorithms.

UCB algorithms select arms based on optimistic estimates, choosing the arm
with the highest upper confidence bound on its reward. This balances exploration
(uncertainty) and exploitation (expected reward).

Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Upper_Confidence_Bounds
"""

import math
from typing import Optional

import numpy as np

from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm


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
        c: float = np.sqrt(2),
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize UCB1 algorithm.

        Args:
            arms: List of available model arms
            c: Exploration parameter (default: sqrt(2) from UCB1 paper)
            random_seed: Random seed for tie-breaking

        Example:
            >>> arms = [
            ...     ModelArm(model_id="gpt-4o", provider="openai", ...),
            ...     ModelArm(model_id="claude-3-5-sonnet", provider="anthropic", ...)
            ... ]
            >>> bandit = UCB1Bandit(arms, c=1.5)
        """
        super().__init__(name=f"ucb1_c{c}", arms=arms)

        self.c = c

        # Initialize statistics for each arm
        self.mean_reward = {arm.model_id: 0.0 for arm in arms}
        self.sum_reward = {arm.model_id: 0.0 for arm in arms}
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

        # Track successes for statistics
        self.arm_successes = {arm.model_id: 0 for arm in arms}

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
            >>> context = BanditContext(query_text="What is 2+2?")
            >>> arm = await bandit.select_arm(context)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        # Exploration phase: Pull each arm at least once
        for model_id in self.arms:
            if self.arm_pulls[model_id] == 0:
                selected_arm = self.arms[model_id]
                self.arm_pulls[model_id] += 1
                self.total_queries += 1
                return selected_arm

        # Exploitation phase: Calculate UCB for each arm
        ucb_values = {}
        for model_id in self.arms:
            mean = self.mean_reward[model_id]
            pulls = self.arm_pulls[model_id]

            # UCB = mean + c * sqrt(ln(total) / pulls)
            exploration_term = self.c * math.sqrt(math.log(self.total_queries) / pulls)
            ucb_values[model_id] = mean + exploration_term

        # Select arm with highest UCB
        selected_id = max(ucb_values, key=ucb_values.get)  # type: ignore
        selected_arm = self.arms[selected_id]

        # Track selection
        self.arm_pulls[selected_id] += 1
        self.total_queries += 1

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

        Clears all learned parameters.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
        """
        self.mean_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.sum_reward = {arm.model_id: 0.0 for arm in self.arm_list}
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.arm_successes = {arm.model_id: 0 for arm in self.arm_list}
        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
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
