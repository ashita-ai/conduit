"""LinUCB (Linear Upper Confidence Bound) contextual bandit algorithm.

LinUCB uses ridge regression to model the expected reward as a linear function
of context features. It maintains a separate linear model for each arm and
selects arms using an upper confidence bound that balances exploitation
(expected reward) and exploration (uncertainty).

Supports sliding window for non-stationarity: maintains only recent N observations
to adapt to model quality/cost changes over time. With sliding window, uses Woodbury
identity for O(d²) incremental updates (downdate oldest + update newest) instead of
O(W·d² + d³) full recalculation.

Reference: https://arxiv.org/abs/1003.0146 (Li et al. 2010)
Tutorial: https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from conduit.core.config import load_algorithm_config, load_feature_dimensions
from conduit.core.models import QueryFeatures

from .base import BanditAlgorithm, BanditFeedback, ModelArm

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


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
        A_inv: Cached inverse of A for each arm (updated incrementally via Sherman-Morrison)
        arm_pulls: Number of times each arm was selected
    """

    def __init__(
        self,
        arms: list[ModelArm],
        alpha: float | None = None,
        feature_dim: int | None = None,
        random_seed: int | None = None,
        reward_weights: dict[str, float] | None = None,
        window_size: int = 0,
        success_threshold: float | None = None,
    ) -> None:
        """Initialize LinUCB algorithm.

        Args:
            arms: List of available model arms
            alpha: Exploration parameter (higher = more exploration, default: LINUCB_ALPHA_DEFAULT)
            feature_dim: Dimensionality of context features (default: 386)
            random_seed: Random seed for reproducibility (not used in LinUCB)
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
            >>> bandit1 = LinUCBBandit(arms, alpha=1.5)
            >>>
            >>> # Sliding window of 1000 (non-stationary environment)
            >>> bandit2 = LinUCBBandit(arms, alpha=1.5, window_size=1000)
        """
        super().__init__(name="linucb", arms=arms)

        # Load config if parameters not provided
        if alpha is None or success_threshold is None:
            algo_config = load_algorithm_config("linucb")
            if alpha is None:
                alpha = algo_config["alpha"]
            if success_threshold is None:
                success_threshold = algo_config.get("success_threshold", 0.85)

        if feature_dim is None:
            feature_config = load_feature_dimensions()
            feature_dim = int(feature_config["full_dim"])

        self.alpha = alpha
        self.feature_dim: int = feature_dim
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
            self.observation_history = {arm.model_id: deque() for arm in arms}

        # Initialize A as identity matrix and b as zero vector for each arm
        self.A = {arm.model_id: np.identity(feature_dim) for arm in arms}
        self.b = {arm.model_id: np.zeros((feature_dim, 1)) for arm in arms}
        # Store A_inv for efficient computation (Sherman-Morrison incremental update)
        self.A_inv = {arm.model_id: np.identity(feature_dim) for arm in arms}

        # Track arm pulls and successes
        self.arm_pulls = {arm.model_id: 0 for arm in arms}
        self.arm_successes = {arm.model_id: 0 for arm in arms}

        if random_seed is not None:
            np.random.seed(random_seed)

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
            ...     domain="general",            ... )
            >>> arm = await bandit.select_arm(features)
            >>> print(arm.model_id)
            "openai:gpt-4o"
        """
        x = self._extract_features(features)

        ucb_values = {}
        for model_id in self.arms:
            # Use cached A_inv (no inversion needed - Sherman-Morrison keeps it updated)
            theta = self.A_inv[model_id] @ self.b[model_id]

            # Compute UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)
            mean_reward = float((theta.T @ x)[0, 0])  # Expected reward
            uncertainty = float(
                np.sqrt((x.T @ self.A_inv[model_id] @ x)[0, 0])
            )  # Confidence radius
            ucb = mean_reward + self.alpha * uncertainty

            ucb_values[model_id] = float(ucb)

        # Select arm with highest UCB
        selected_id = max(ucb_values, key=ucb_values.get)  # type: ignore[arg-type]  # dict.get returns Optional but all keys exist
        selected_arm = self.arms[selected_id]

        # Track queries
        self.total_queries += 1

        return selected_arm

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update ridge regression parameters with feedback.

        Uses multi-objective reward (quality + cost + latency) for regression updates.

        With sliding window (window_size > 0):
        - Uses Woodbury identity for O(d²) incremental updates
        - Downdates A_inv when oldest observation drops (rank-1 downdate)
        - Updates A_inv with new observation (Sherman-Morrison rank-1 update)
        - Complexity: O(d²) per update (vs O(W·d² + d³) for full recalculation)

        Without window (window_size = 0):
        - Incremental update: A += x @ x^T, b += r * x
        - Uses Sherman-Morrison for O(d²) A_inv updates

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

        x = self._extract_features(features)

        # Two update strategies:
        # 1. Sliding window: Use Woodbury identity for incremental downdate + Sherman-Morrison update
        # 2. No window: Use Sherman-Morrison for incremental A_inv update

        if self.window_size > 0:
            # Sliding window: Optimized incremental update using Woodbury identity
            # Check if window is full (will drop oldest observation)
            history = self.observation_history[model_id]
            will_drop_oldest = len(history) == self.window_size

            if will_drop_oldest:
                # Get oldest observation that will be dropped
                oldest_x, oldest_r = history[0]

                # Downdate A_inv using Woodbury identity: (A - xx^T)^-1 = A^-1 + (A^-1 x)(x^T A^-1) / (1 - x^T A^-1 x)
                a_inv_oldest_x = self.A_inv[model_id] @ oldest_x  # d×1 vector
                downdate_denominator = 1.0 - float(
                    (oldest_x.T @ a_inv_oldest_x)[0, 0]
                )  # scalar

                # Numerical stability check for downdate
                if abs(downdate_denominator) > 1e-10:
                    # Downdate A_inv (remove oldest observation)
                    self.A_inv[model_id] += (
                        a_inv_oldest_x @ a_inv_oldest_x.T
                    ) / downdate_denominator
                    # Downdate A and b
                    self.A[model_id] -= oldest_x @ oldest_x.T
                    self.b[model_id] -= oldest_r * oldest_x
                else:
                    # Fallback: Rebuild from history if numerical issues
                    self.A[model_id] = np.identity(self.feature_dim)
                    self.b[model_id] = np.zeros((self.feature_dim, 1))
                    for obs_x, obs_r in history:
                        self.A[model_id] += obs_x @ obs_x.T
                        self.b[model_id] += obs_r * obs_x
                    self.A_inv[model_id] = np.linalg.inv(self.A[model_id])  # type: ignore[assignment]

            # Add new observation to history (deque automatically drops oldest if full)
            history.append((x, reward))

            # Update A and b with new observation
            self.A[model_id] += x @ x.T
            self.b[model_id] += reward * x

            # Update A_inv using Sherman-Morrison formula: (A + xx^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
            a_inv_x = self.A_inv[model_id] @ x  # d×1 vector
            update_denominator = 1.0 + float((x.T @ a_inv_x)[0, 0])  # scalar

            # Numerical stability check for update
            if update_denominator > 1e-10:
                # Update A_inv incrementally using Sherman-Morrison
                self.A_inv[model_id] -= (a_inv_x @ a_inv_x.T) / update_denominator
            else:
                # Fallback: Rebuild from history if numerical issues
                self.A[model_id] = np.identity(self.feature_dim)
                self.b[model_id] = np.zeros((self.feature_dim, 1))
                for obs_x, obs_r in history:
                    self.A[model_id] += obs_x @ obs_x.T
                    self.b[model_id] += obs_r * obs_x
                self.A_inv[model_id] = np.linalg.inv(self.A[model_id])  # type: ignore[assignment]
        else:
            # No sliding window: Use Sherman-Morrison incremental update
            # Update A and b incrementally
            self.A[model_id] += x @ x.T
            self.b[model_id] += reward * x

            # Sherman-Morrison formula: (A + xx^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
            a_inv_x = self.A_inv[model_id] @ x  # d×1 vector
            denominator = 1.0 + float((x.T @ a_inv_x)[0, 0])  # scalar

            # Numerical stability check: denominator should be positive
            if denominator > 1e-10:
                # Update A_inv incrementally using Sherman-Morrison
                self.A_inv[model_id] -= (a_inv_x @ a_inv_x.T) / denominator
            else:
                # Fallback to full inversion if numerical issues detected
                self.A_inv[model_id] = np.linalg.inv(self.A[model_id])  # type: ignore[assignment]  # np.linalg.inv returns compatible dtype

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
        self.A = {arm.model_id: np.identity(self.feature_dim) for arm in self.arm_list}
        self.b = {
            arm.model_id: np.zeros((self.feature_dim, 1)) for arm in self.arm_list
        }
        self.A_inv = {
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
            theta = self.A_inv[model_id] @ self.b[model_id]
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

    def compute_scores(self, features: QueryFeatures) -> dict[str, dict[str, float]]:
        """Compute UCB scores for all arms without selecting.

        Exposes the internal UCB computation for audit logging:
            UCB = theta^T @ x + alpha * sqrt(x^T @ A_inv @ x)

        Args:
            features: Query features for context

        Returns:
            Dictionary mapping arm_id to score components:
            {
                "arm_id": {
                    "mean": expected reward (theta^T @ x),
                    "uncertainty": exploration bonus (sqrt(x^T @ A_inv @ x)),
                    "total": UCB score (mean + alpha * uncertainty)
                }
            }

        Example:
            >>> features = QueryFeatures(embedding=[0.1] * 384, ...)
            >>> scores = bandit.compute_scores(features)
            >>> scores["gpt-4o-mini"]
            {"mean": 0.72, "uncertainty": 0.15, "total": 0.87}
        """
        x = self._extract_features(features)
        scores: dict[str, dict[str, float]] = {}

        for model_id in self.arms:
            theta = self.A_inv[model_id] @ self.b[model_id]
            mean_reward = float((theta.T @ x)[0, 0])
            uncertainty = float(np.sqrt((x.T @ self.A_inv[model_id] @ x)[0, 0]))
            ucb = mean_reward + self.alpha * uncertainty

            scores[model_id] = {
                "mean": mean_reward,
                "uncertainty": uncertainty,
                "total": ucb,
            }

        return scores

    def to_state(self) -> BanditState:
        """Serialize LinUCB state for persistence.

        Converts numpy arrays to nested lists for JSON serialization.
        A_inv is not persisted (recomputed from A on restore).

        Returns:
            BanditState object containing all LinUCB state

        Example:
            >>> state = bandit.to_state()
            >>> state.algorithm
            "linucb"
            >>> len(state.A_matrices)
            5
        """
        from conduit.core.state_store import BanditState, serialize_bandit_matrices

        # Serialize A matrices and b vectors
        A_matrices, b_vectors = serialize_bandit_matrices(self.A, self.b)

        # Serialize observation history (feature vectors and rewards)
        observation_history_serialized = []
        for arm_id, observations in self.observation_history.items():
            for feature_vec, reward in observations:
                observation_history_serialized.append(
                    {
                        "arm_id": arm_id,
                        "features": feature_vec.flatten().tolist(),
                        "reward": reward,
                    }
                )

        return BanditState(
            algorithm="linucb",
            arm_ids=list(self.arms.keys()),
            arm_pulls=self.arm_pulls.copy(),
            arm_successes=self.arm_successes.copy(),
            total_queries=self.total_queries,
            A_matrices=A_matrices,
            b_vectors=b_vectors,
            observation_history=observation_history_serialized,
            alpha=self.alpha,
            feature_dim=self.feature_dim,
            window_size=self.window_size if self.window_size > 0 else None,
            updated_at=datetime.now(timezone.utc),
        )

    def from_state(self, state: BanditState) -> None:
        """Restore LinUCB state from persisted data.

        Deserializes numpy arrays from nested lists and recomputes A_inv.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with current configuration

        Example:
            >>> state = await store.load_bandit_state("router-1", "linucb")
            >>> bandit.from_state(state)
        """
        from conduit.core.state_store import deserialize_bandit_matrices

        if state.algorithm != "linucb":
            raise ValueError(f"State algorithm '{state.algorithm}' != 'linucb'")

        # Verify arms match
        state_arms = set(state.arm_ids)
        current_arms = set(self.arms.keys())
        if state_arms != current_arms:
            raise ValueError(
                f"State arms {state_arms} don't match current arms {current_arms}"
            )

        # Verify feature dimension matches (Phase 1: dimension safety)
        if state.feature_dim is not None and state.feature_dim != self.feature_dim:
            # Build helpful error message with embedding provider context
            provider_info = ""
            if state.embedding_provider:
                provider_info = f"\n  Saved with: {state.embedding_provider}"
                if state.embedding_dimensions:
                    provider_info += f" ({state.embedding_dimensions} dims"
                    if state.pca_enabled:
                        provider_info += f" → PCA {state.pca_dimensions} dims"
                    provider_info += ")"

            raise ValueError(
                f"Cannot load state: dimension mismatch.\n"
                f"  Saved state: {state.feature_dim} total dims{provider_info}\n"
                f"  Current config: {self.feature_dim} total dims\n"
                f"\n"
                f"  This usually happens when switching embedding providers or PCA settings.\n"
                f"  Solutions:\n"
                f"  1. Revert to the original embedding configuration (preserve learning)\n"
                f"  2. Delete saved state to start fresh (lose learning but use new config)\n"
                f"  3. Use explicit feature_dim parameter matching saved state"
            )

        # Restore counters
        self.total_queries = state.total_queries
        self.arm_pulls = state.arm_pulls.copy()
        self.arm_successes = state.arm_successes.copy()

        # Restore A matrices and b vectors
        self.A, self.b = deserialize_bandit_matrices(state.A_matrices, state.b_vectors)

        # Recompute A_inv from A (not stored to save space)
        # numpy stubs dtype mismatch for np.linalg.inv - functionally correct
        self.A_inv = {
            arm_id: np.linalg.inv(A_mat)  # type: ignore[misc]
            for arm_id, A_mat in self.A.items()
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
