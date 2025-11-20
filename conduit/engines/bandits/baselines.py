"""Baseline algorithms for comparison.

These baselines provide reference points for evaluating bandit performance:
- Random: Uniform random selection (no learning)
- Oracle: Perfect knowledge of best arm (theoretical optimum)
- AlwaysBest: Always select highest quality model (ignores cost)
- AlwaysCheapest: Always select cheapest model (ignores quality)
"""

import random
from typing import Optional

from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm


class RandomBaseline(BanditAlgorithm):
    """Random arm selection baseline.

    Selects arms uniformly at random. No learning occurs.
    Provides lower bound on expected performance.

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        arm_pulls: Number of pulls for each arm (for statistics)
    """

    def __init__(self, arms: list[ModelArm], random_seed: Optional[int] = None) -> None:
        """Initialize random baseline.

        Args:
            arms: List of available model arms
            random_seed: Random seed for reproducibility

        Example:
            >>> arms = [ModelArm(...), ModelArm(...), ...]
            >>> baseline = RandomBaseline(arms, random_seed=42)
        """
        super().__init__(name="random", arms=arms)

        self.arm_pulls = {arm.model_id: 0 for arm in arms}

        if random_seed is not None:
            random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select random arm uniformly.

        Args:
            context: Query context (ignored)

        Returns:
            Randomly selected model arm
        """
        selected_id = random.choice(list(self.arms.keys()))
        selected_arm = self.arms[selected_id]

        self.arm_pulls[selected_id] += 1
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """No-op update (random baseline doesn't learn)."""
        pass

    def reset(self) -> None:
        """Reset statistics."""
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
        """Get statistics."""
        base_stats = super().get_stats()
        return {**base_stats, "arm_pulls": self.arm_pulls}


class OracleBaseline(BanditAlgorithm):
    """Oracle with perfect knowledge of arm rewards.

    Requires ground truth rewards to be provided via update().
    Always selects the arm that would give highest reward for given context.

    This represents the theoretical optimum (zero regret).

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        oracle_rewards: Ground truth rewards for each (context, arm) pair
        arm_pulls: Number of pulls for each arm
    """

    def __init__(self, arms: list[ModelArm]) -> None:
        """Initialize oracle baseline.

        Args:
            arms: List of available model arms

        Example:
            >>> arms = [ModelArm(...), ModelArm(...), ...]
            >>> oracle = OracleBaseline(arms)
        """
        super().__init__(name="oracle", arms=arms)

        # Store observed rewards for each query+arm combination
        # Key: (query_text_hash, model_id), Value: quality_score
        self.oracle_rewards: dict[tuple[int, str], float] = {}
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

        # Track which queries we've seen all arm rewards for
        self.complete_queries: set[int] = set()

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm with highest known reward for this context.

        If rewards are unknown, defaults to arm with highest expected_quality.

        Args:
            features: Query features from analyzer

        Returns:
            Optimal model arm for this query
        """
        query_hash = hash(tuple(features.embedding))

        # Check if we have oracle knowledge for this query
        best_reward = -float("inf")
        best_id = None

        for model_id, arm in self.arms.items():
            key = (query_hash, model_id)
            if key in self.oracle_rewards:
                # Use oracle knowledge
                reward = self.oracle_rewards[key]
            else:
                # Fall back to prior expected quality
                reward = arm.expected_quality

            if reward > best_reward:
                best_reward = reward
                best_id = model_id

        selected_id = best_id or list(self.arms.keys())[0]
        selected_arm = self.arms[selected_id]

        self.arm_pulls[selected_id] += 1
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Store oracle reward for this query+arm combination.

        Args:
            feedback: Feedback containing actual reward
            features: Query features from analyzer
        """
        query_hash = hash(tuple(features.embedding))
        key = (query_hash, feedback.model_id)
        self.oracle_rewards[key] = feedback.quality_score

    def reset(self) -> None:
        """Reset oracle knowledge."""
        self.oracle_rewards = {}
        self.complete_queries = set()
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
        """Get statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "arm_pulls": self.arm_pulls,
            "oracle_knowledge_size": len(self.oracle_rewards),
        }


class AlwaysBestBaseline(BanditAlgorithm):
    """Always select highest quality model (ignores cost).

    Provides upper bound on quality, lower bound on cost efficiency.

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        best_arm: Arm with highest expected quality
    """

    def __init__(self, arms: list[ModelArm]) -> None:
        """Initialize always-best baseline.

        Args:
            arms: List of available model arms

        Example:
            >>> arms = [ModelArm(...), ModelArm(...), ...]
            >>> baseline = AlwaysBestBaseline(arms)
        """
        super().__init__(name="always_best", arms=arms)

        # Find arm with highest expected quality
        self.best_arm = max(arms, key=lambda arm: arm.expected_quality)
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Always select highest quality arm.

        Args:
            context: Query context (ignored)

        Returns:
            Highest quality model arm
        """
        self.arm_pulls[self.best_arm.model_id] += 1
        self.total_queries += 1
        return self.best_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """No-op update (static policy)."""
        pass

    def reset(self) -> None:
        """Reset statistics."""
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
        """Get statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "best_arm": self.best_arm.model_id,
            "best_arm_quality": self.best_arm.expected_quality,
            "arm_pulls": self.arm_pulls,
        }


class AlwaysCheapestBaseline(BanditAlgorithm):
    """Always select cheapest model (ignores quality).

    Provides upper bound on cost efficiency, lower bound on quality.

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        cheapest_arm: Arm with lowest expected cost
    """

    def __init__(self, arms: list[ModelArm]) -> None:
        """Initialize always-cheapest baseline.

        Args:
            arms: List of available model arms

        Example:
            >>> arms = [ModelArm(...), ModelArm(...), ...]
            >>> baseline = AlwaysCheapestBaseline(arms)
        """
        super().__init__(name="always_cheapest", arms=arms)

        # Find arm with lowest expected cost (average of input/output)
        self.cheapest_arm = min(
            arms,
            key=lambda arm: (arm.cost_per_input_token + arm.cost_per_output_token) / 2,
        )
        self.arm_pulls = {arm.model_id: 0 for arm in arms}

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Always select cheapest arm.

        Args:
            context: Query context (ignored)

        Returns:
            Cheapest model arm
        """
        self.arm_pulls[self.cheapest_arm.model_id] += 1
        self.total_queries += 1
        return self.cheapest_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """No-op update (static policy)."""
        pass

    def reset(self) -> None:
        """Reset statistics."""
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.total_queries = 0

    def get_stats(self) -> dict[str, any]:  # type: ignore
        """Get statistics."""
        base_stats = super().get_stats()
        avg_cost = (
            self.cheapest_arm.cost_per_input_token + self.cheapest_arm.cost_per_output_token
        ) / 2
        return {
            **base_stats,
            "cheapest_arm": self.cheapest_arm.model_id,
            "cheapest_arm_avg_cost": avg_cost,
            "arm_pulls": self.arm_pulls,
        }
