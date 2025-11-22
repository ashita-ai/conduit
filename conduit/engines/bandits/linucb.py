"""LinUCB (Linear Upper Confidence Bound) contextual bandit algorithm.

LinUCB uses ridge regression to model the expected reward as a linear function
of context features. It maintains a separate linear model for each arm and
selects arms using an upper confidence bound that balances exploitation
(expected reward) and exploration (uncertainty).

Supports sliding window for non-stationarity: maintains only recent N observations
to adapt to model quality/cost changes over time. With sliding window, recomputes
A and b matrices from windowed history on each update.

Reference: https://arxiv.org/abs/1003.0146 (Li et al. 2010)
Tutorial: https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
"""

import numpy as np
from collections import deque
from typing import Any, Optional

from conduit.core.models import QueryFeatures
from .base import BanditAlgorithm, BanditFeedback, ModelArm


class LinUCBBandit(BanditAlgorithm):
    """LinUCB Disjoint algorithm for contextual bandits.

    Maintains a ridge regression model for each arm:
    - A: d×d design matrix (feature covariance)
    - b: d×1 response vector (feature-weighted rewards)
    - theta: d×1 coefficient vector (computed from A and b)

    Selection uses Upper Confidence Bound:
        UCB(arm) = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)

    Where:
    - theta: Estimated reward coefficients
    - x: Context feature vector
    - alpha: Exploration parameter (higher = more exploration)
    - A_inv: Inverse of design matrix

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        alpha: Exploration parameter (default: 1.0)
        feature_dim: Dimensionality of context features
        A: Design matrix for each arm (d×d)
        b: Response vector for each arm (d×1)
        arm_pulls: Number of times each arm was selected
    """

    def __init__(
        self,
        arms: list[ModelArm],
        alpha: float = 1.0,
        feature_dim: int = 387,  # 384 embedding + 3 metadata
        random_seed: Optional[int] = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
    ) -> None:
        """Initialize LinUCB algorithm.

        Args:
            arms: List of available model arms
            alpha: Exploration parameter (higher = more exploration, default: 1.0)
            feature_dim: Dimensionality of context features (default: 387)
            random_seed: Random seed for reproducibility (not used in LinUCB)
            reward_weights: Multi-objective reward weights. If None, uses defaults
                (quality: 0.70, cost: 0.20, latency: 0.10)
            window_size: Sliding window size for non-stationarity.
                0 = unlimited history (default), N = keep only last N observations per arm

        Example:
            >>> arms = [
            ...     ModelArm(model_id="gpt-4o", provider="openai", ...),
            ...     ModelArm(model_id="claude-3-5-sonnet", provider="anthropic", ...)
            ... ]
            >>> # Unlimited history (stationary environment)
            >>> bandit1 = LinUCBBandit(arms, alpha=1.5)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit2 = LinUCBBandit(arms, alpha=1.5, window_size=1000)
        """
        super().__init__(name="linucb", arms=arms)

        self.alpha = alpha
        self.feature_dim = feature_dim
        self.window_size = window_size

        # Multi-objective reward weights (Phase 3)
        if reward_weights is None:
            self.reward_weights = {"quality": 0.70, "cost": 0.20, "latency": 0.10}
        else:
            self.reward_weights = reward_weights

        # Sliding window: Store recent observations (x, r) per arm (Phase 3 - Non-stationarity)
        # Each observation is a tuple: (feature_vector, reward)
        if window_size > 0:
            self.observation_history: dict[str, deque[tuple[np.ndarray, float]]] = {
                arm.model_id: deque(maxlen=window_size) for arm in arms
            }
        else:
            # Use deque for unlimited history (no maxlen)
            self.observation_history: dict[str, deque[tuple[np.ndarray, float]]] = {
                arm.model_id: deque() for arm in arms
            }

        # Initialize A as identity matrix and b as zero vector for each arm
        self.A = {
            arm.model_id: np.identity(feature_dim) for arm in arms
        }
        self.b = {
            arm.model_id: np.zeros((feature_dim, 1)) for arm in arms
        }

        # Track arm pulls and successes
        self.arm_pulls = {arm.model_id: 0 for arm in arms}
        self.arm_successes = {arm.model_id: 0 for arm in arms}

        if random_seed is not None:
            np.random.seed(random_seed)

    def _extract_features(self, features: QueryFeatures) -> np.ndarray:
        """Extract feature vector from QueryFeatures.

        Combines embedding vector with metadata features:
        - embedding (384 dims)
        - token_count (1 dim, normalized)
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
                features.token_count / 1000.0,  # Normalize token count
                features.complexity_score,
                features.domain_confidence,
            ]
        )

        # Reshape to column vector (d×1)
        return feature_vector.reshape(-1, 1)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm using LinUCB policy.

        For each arm, compute:
            UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)

        Where theta = A_inv @ b (ridge regression coefficients)

        Args:
            features: Query features for context

        Returns:
            Selected model arm with highest UCB

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
            "openai:gpt-4o"
        """
        x = self._extract_features(features)

        ucb_values = {}
        for model_id in self.arms:
            # Compute theta = A_inv @ b (ridge regression coefficients)
            A_inv = np.linalg.inv(self.A[model_id])
            theta = A_inv @ self.b[model_id]

            # Compute UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)
            mean_reward = float((theta.T @ x)[0, 0])  # Expected reward
            uncertainty = float(np.sqrt((x.T @ A_inv @ x)[0, 0]))  # Confidence radius
            ucb = mean_reward + self.alpha * uncertainty

            ucb_values[model_id] = float(ucb)

        # Select arm with highest UCB
        selected_id = max(ucb_values, key=ucb_values.get)  # type: ignore
        selected_arm = self.arms[selected_id]

        # Track queries
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update ridge regression parameters with feedback.

        Uses multi-objective reward (quality + cost + latency) for regression updates.

        With sliding window (window_size > 0):
        - Stores observation (x, r) in history deque (automatically drops oldest when full)
        - Recalculates A and b from all observations in current window:
            A = I + sum(x_i @ x_i^T for all i in window)
            b = sum(r_i * x_i for all i in window)

        Without window (window_size = 0):
        - Incremental update: A += x @ x^T, b += r * x

        Args:
            feedback: Feedback from model execution
            features: Original query features

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o",
            ...     cost=0.001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await bandit.update(feedback, features)
        """
        model_id = feedback.model_id

        # Calculate composite reward from quality, cost, and latency (Phase 3)
        reward = feedback.calculate_reward(
            quality_weight=self.reward_weights["quality"],
            cost_weight=self.reward_weights["cost"],
            latency_weight=self.reward_weights["latency"],
        )

        x = self._extract_features(features)

        # Add observation to history
        self.observation_history[model_id].append((x, reward))

        # Recalculate A and b from windowed history
        # A = I + sum(x_i @ x_i^T for all i)
        # b = sum(r_i * x_i for all i)
        self.A[model_id] = np.identity(self.feature_dim)
        self.b[model_id] = np.zeros((self.feature_dim, 1))

        for obs_x, obs_r in self.observation_history[model_id]:
            self.A[model_id] += obs_x @ obs_x.T
            self.b[model_id] += obs_r * obs_x

        # Track statistics
        self.arm_pulls[model_id] += 1
        if reward >= 0.85:  # Success threshold
            self.arm_successes[model_id] += 1

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters, observation history, and reverts to prior.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
        """
        self.A = {
            arm.model_id: np.identity(self.feature_dim) for arm in self.arm_list
        }
        self.b = {
            arm.model_id: np.zeros((self.feature_dim, 1)) for arm in self.arm_list
        }
        self.arm_pulls = {arm.model_id: 0 for arm in self.arm_list}
        self.arm_successes = {arm.model_id: 0 for arm in self.arm_list}

        # Clear observation history
        for model_id in self.observation_history:
            self.observation_history[model_id].clear()

        self.total_queries = 0

    def get_stats(self) -> dict[str, Any]:
        """Get algorithm statistics.

        Returns:
            Dictionary with statistics including:
            - total_queries: Total number of queries processed
            - arm_pulls: Number of times each arm was selected
            - arm_success_rates: Success rate for each arm
            - arm_theta_norms: L2 norm of theta for each arm (model confidence)

        Example:
            >>> stats = bandit.get_stats()
            >>> print(stats["arm_pulls"])
            {"openai:gpt-4o": 150, "claude-3-5-sonnet": 100, ...}
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

        # Calculate theta norms (model confidence)
        theta_norms = {}
        for model_id in self.arms:
            A_inv = np.linalg.inv(self.A[model_id])
            theta = A_inv @ self.b[model_id]
            theta_norms[model_id] = float(np.linalg.norm(theta))

        return {
            **base_stats,
            "alpha": self.alpha,
            "feature_dim": self.feature_dim,
            "arm_pulls": self.arm_pulls,
            "arm_successes": self.arm_successes,
            "arm_success_rates": success_rates,
            "arm_theta_norms": theta_norms,
        }
