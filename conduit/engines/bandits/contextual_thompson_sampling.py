"""Contextual Thompson Sampling with Bayesian Linear Regression.

Contextual Thompson Sampling combines Thompson Sampling's Bayesian exploration
strategy with contextual features. It maintains a Bayesian linear regression
model for each arm and samples from the posterior distribution to select arms.

This combines the best of both worlds:
- Thompson Sampling: Natural exploration via sampling from posterior
- Contextual: Uses query features for smarter routing decisions

Reference: Agrawal & Goyal (2013) "Thompson Sampling for Contextual Bandits with Linear Payoffs"
Tutorial: http://proceedings.mlr.press/v28/agrawal13.pdf
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from conduit.core.config import load_algorithm_config, load_feature_dimensions
from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class ContextualThompsonSamplingBandit(BanditAlgorithm):
    """Contextual Thompson Sampling with Bayesian Linear Regression.

    Maintains a Bayesian linear regression model for each arm:
    - Prior: theta ~ N(mu_0, Sigma_0)
    - Posterior: theta | D ~ N(mu_n, Sigma_n)

    Where:
    - theta: d×1 coefficient vector (reward parameters)
    - mu: d×1 mean vector (posterior mean)
    - Sigma: d×d covariance matrix (posterior uncertainty)

    Selection strategy:
    1. Sample theta_hat ~ N(mu_n, Sigma_n) for each arm
    2. Compute expected reward: r_hat = theta_hat^T @ x
    3. Select arm with highest r_hat

    Posterior update (Bayesian linear regression):
    - Sigma_n = (Sigma_0^-1 + lambda * sum(x_i @ x_i^T))^-1
    - mu_n = Sigma_n @ (Sigma_0^-1 @ mu_0 + lambda * sum(r_i * x_i))

    Attributes:
        name: Algorithm identifier
        arms: Available model arms
        lambda_reg: Regularization parameter (precision of reward noise)
        feature_dim: Dimensionality of context features
        mu: Posterior mean for each arm (d×1)
        Sigma: Posterior covariance for each arm (d×d)
        arm_pulls: Number of times each arm was selected
    """

    def __init__(
        self,
        arms: list[ModelArm],
        lambda_reg: float | None = None,
        feature_dim: int | None = None,
        random_seed: int | None = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
        success_threshold: float | None = None,
    ) -> None:
        """Initialize Contextual Thompson Sampling algorithm.

        Args:
            arms: List of available model arms
            lambda_reg: Regularization parameter / noise precision (default: THOMPSON_LAMBDA_DEFAULT)
                Higher = less uncertainty (tighter posterior), more regularization
            feature_dim: Dimensionality of context features (default: 387)
            random_seed: Random seed for reproducibility
            reward_weights: Multi-objective reward weights. If None, uses defaults
                (quality: 0.70, cost: 0.20, latency: 0.10)
            window_size: Sliding window size for non-stationarity.
                0 = unlimited history (default), N = keep only last N observations per arm
            success_threshold: Reward threshold for counting successes (default: 0.85)
                Only used for statistics, not algorithm decisions

        Example:
            >>> arms = [
            ...     ModelArm(model_id="gpt-4o", provider="openai", ...),
            ...     ModelArm(model_id="claude-3-5-sonnet", provider="anthropic", ...)
            ... ]
            >>> # Unlimited history (stationary environment)
            >>> bandit1 = ContextualThompsonSamplingBandit(arms, lambda_reg=1.0)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit2 = ContextualThompsonSamplingBandit(arms, lambda_reg=1.0, window_size=1000)
        """
        super().__init__(name="contextual_thompson_sampling", arms=arms)

        # Load config if parameters not provided
        if lambda_reg is None or success_threshold is None:
            algo_config = load_algorithm_config("thompson_sampling")
            if lambda_reg is None:
                lambda_reg = algo_config["lambda"]
            if success_threshold is None:
                success_threshold = algo_config.get("success_threshold", 0.85)

        if feature_dim is None:
            feature_config = load_feature_dimensions()
            feature_dim = feature_config["full_dim"]

        self.lambda_reg = lambda_reg
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.success_threshold = success_threshold

        # Multi-objective reward weights (Phase 3)
        if reward_weights is None:
            self.reward_weights = {"quality": 0.70, "cost": 0.20, "latency": 0.10}
        else:
            self.reward_weights = reward_weights

        # Sliding window: Store recent observations (x, r) per arm (Phase 3 - Non-stationarity)
        # Each observation is a tuple: (feature_vector, reward)
        self.observation_history: dict[str, deque[tuple[np.ndarray, float]]]
        if window_size > 0:
            self.observation_history = {
                arm.model_id: deque(maxlen=window_size) for arm in arms
            }
        else:
            # Use deque for unlimited history (no maxlen)
            self.observation_history = {
                arm.model_id: deque() for arm in arms
            }

        # Initialize posterior parameters
        # Prior: theta ~ N(0, I) (uninformative prior)
        self.mu = {
            arm.model_id: np.zeros((feature_dim, 1)) for arm in arms
        }
        self.Sigma = {
            arm.model_id: np.identity(feature_dim) for arm in arms
        }

        # Track arm pulls and successes
        self.arm_pulls = {arm.model_id: 0 for arm in arms}
        self.arm_successes = {arm.model_id: 0 for arm in arms}

        if random_seed is not None:
            np.random.seed(random_seed)

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select arm using Thompson Sampling with Bayesian linear regression.

        For each arm:
        1. Sample theta_hat ~ N(mu, Sigma) from posterior
        2. Compute expected reward: r_hat = theta_hat^T @ x
        3. Select arm with highest r_hat

        Args:
            features: Query features for context

        Returns:
            Selected model arm with highest sampled reward

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

        sampled_rewards = {}
        for model_id in self.arms:
            # Sample theta from posterior: theta ~ N(mu, Sigma)
            mu = self.mu[model_id]
            sigma = self.Sigma[model_id]

            # Sample: theta_hat = mu + Sigma^(1/2) @ z, where z ~ N(0, I)
            # Using Cholesky decomposition: Sigma = L @ L^T
            try:
                L = np.linalg.cholesky(sigma)
                z = np.random.randn(self.feature_dim, 1)
                theta_hat = mu + L @ z
            except np.linalg.LinAlgError:
                # If Sigma is not positive definite, use mu directly (no sampling)
                theta_hat = mu

            # Compute expected reward: r = theta^T @ x
            expected_reward = float((theta_hat.T @ x)[0, 0])
            sampled_rewards[model_id] = expected_reward

        # Select arm with highest sampled reward
        selected_id = max(sampled_rewards, key=sampled_rewards.get)  # type: ignore
        selected_arm = self.arms[selected_id]

        # Track queries
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update posterior distribution with feedback.

        Uses Bayesian linear regression to update posterior:
        - Sigma_n = (Sigma_0^-1 + lambda * sum(x_i @ x_i^T))^-1
        - mu_n = Sigma_n @ (Sigma_0^-1 @ mu_0 + lambda * sum(r_i * x_i))

        With sliding window (window_size > 0):
        - Stores observation (x, r) in history deque (automatically drops oldest when full)
        - Recalculates mu and Sigma from all observations in current window

        Without window (window_size = 0):
        - Recalculates from all historical observations

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

        # Recalculate posterior from windowed history
        # Prior: theta ~ N(0, I) (uninformative)
        # Sigma_n = (I + lambda * sum(x_i @ x_i^T))^-1
        # mu_n = Sigma_n @ (lambda * sum(r_i * x_i))

        # Compute sufficient statistics from window
        Sigma_inv = np.identity(self.feature_dim)  # Prior precision (I)
        weighted_sum = np.zeros((self.feature_dim, 1))  # Weighted feature sum

        for obs_x, obs_r in self.observation_history[model_id]:
            Sigma_inv += self.lambda_reg * (obs_x @ obs_x.T)
            weighted_sum += self.lambda_reg * obs_r * obs_x

        # Update posterior
        self.Sigma[model_id] = np.linalg.inv(Sigma_inv)  # type: ignore[assignment]
        self.mu[model_id] = self.Sigma[model_id] @ weighted_sum

        # Track statistics
        self.arm_pulls[model_id] += 1
        if reward >= self.success_threshold:
            self.arm_successes[model_id] += 1

    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters, observation history, and reverts to prior.

        Example:
            >>> bandit.reset()
            >>> bandit.total_queries
            0
        """
        # Reset to prior: theta ~ N(0, I)
        self.mu = {
            arm.model_id: np.zeros((self.feature_dim, 1)) for arm in self.arm_list
        }
        self.Sigma = {
            arm.model_id: np.identity(self.feature_dim) for arm in self.arm_list
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
            - arm_mu_norms: L2 norm of mu for each arm (posterior mean confidence)
            - arm_sigma_traces: Trace of Sigma for each arm (total uncertainty)

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

        # Calculate posterior statistics
        mu_norms = {}
        sigma_traces = {}
        for model_id in self.arms:
            mu_norms[model_id] = float(np.linalg.norm(self.mu[model_id]))
            sigma_traces[model_id] = float(np.trace(self.Sigma[model_id]))

        return {
            **base_stats,
            "lambda_reg": self.lambda_reg,
            "feature_dim": self.feature_dim,
            "arm_pulls": self.arm_pulls,
            "arm_successes": self.arm_successes,
            "arm_success_rates": success_rates,
            "arm_mu_norms": mu_norms,
            "arm_sigma_traces": sigma_traces,
        }

    def to_state(self) -> BanditState:
        """Serialize Contextual Thompson Sampling state for persistence.

        Converts numpy arrays to nested lists for JSON serialization.

        Returns:
            BanditState object containing all CTS state

        Example:
            >>> state = bandit.to_state()
            >>> state.algorithm
            "contextual_thompson_sampling"
            >>> len(state.mu_vectors)
            5
        """
        from conduit.core.state_store import BanditState

        # Serialize mu vectors and Sigma matrices
        mu_vectors = {
            arm_id: vec.flatten().tolist() for arm_id, vec in self.mu.items()
        }
        sigma_matrices = {
            arm_id: mat.tolist() for arm_id, mat in self.Sigma.items()
        }

        # Serialize observation history (feature vectors and rewards)
        observation_history_serialized = []
        for arm_id, observations in self.observation_history.items():
            for feature_vec, reward in observations:
                observation_history_serialized.append({
                    "arm_id": arm_id,
                    "features": feature_vec.flatten().tolist(),
                    "reward": reward,
                })

        return BanditState(
            algorithm="contextual_thompson_sampling",
            arm_ids=list(self.arms.keys()),
            arm_pulls=self.arm_pulls.copy(),
            arm_successes=self.arm_successes.copy(),
            total_queries=self.total_queries,
            mu_vectors=mu_vectors,
            sigma_matrices=sigma_matrices,
            observation_history=observation_history_serialized,
            feature_dim=self.feature_dim,
            window_size=self.window_size if self.window_size > 0 else None,
            updated_at=datetime.now(UTC),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore Contextual Thompson Sampling state from persisted data.

        Deserializes numpy arrays from nested lists.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with current configuration

        Example:
            >>> state = await store.load_bandit_state("router-1", "contextual_thompson_sampling")
            >>> bandit.from_state(state)
        """
        if state.algorithm != "contextual_thompson_sampling":
            raise ValueError(
                f"State algorithm '{state.algorithm}' != 'contextual_thompson_sampling'"
            )

        # Verify arms match
        state_arms = set(state.arm_ids)
        current_arms = set(self.arms.keys())
        if state_arms != current_arms:
            raise ValueError(
                f"State arms {state_arms} don't match current arms {current_arms}"
            )

        # Verify feature dimension matches
        if state.feature_dim is not None and state.feature_dim != self.feature_dim:
            raise ValueError(
                f"State feature_dim {state.feature_dim} != {self.feature_dim}"
            )

        # Restore counters
        self.total_queries = state.total_queries
        self.arm_pulls = state.arm_pulls.copy()
        self.arm_successes = state.arm_successes.copy()

        # Restore mu vectors and Sigma matrices
        self.mu = {
            arm_id: np.array(vec).reshape(-1, 1)
            for arm_id, vec in state.mu_vectors.items()
        }
        self.Sigma = {
            arm_id: np.array(mat) for arm_id, mat in state.sigma_matrices.items()
        }

        # Restore observation history
        for arm_id in self.arms:
            self.observation_history[arm_id].clear()

        for entry in state.observation_history:
            arm_id = entry["arm_id"]
            features = np.array(entry["features"]).reshape(-1, 1)
            reward = entry["reward"]
            if arm_id in self.observation_history:
                self.observation_history[arm_id].append((features, reward))
