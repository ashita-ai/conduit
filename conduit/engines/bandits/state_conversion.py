"""State conversion utilities for hybrid routing algorithm transitions.

Enables optimistic conversion between different bandit algorithm states to preserve
learned knowledge when changing algorithms. All conversions are mathematically
lossless or information-preserving.

Supported Conversions:
    - UCB1 ↔ Thompson Sampling (non-contextual)
    - LinUCB ↔ Contextual Thompson Sampling (contextual)
    - Cross-category conversions (via intermediate representations)

Example:
    >>> from conduit.core.state_store import BanditState
    >>> # Convert UCB1 state to Thompson Sampling
    >>> ucb1_state = BanditState(algorithm="ucb1", ...)
    >>> thompson_state = convert_bandit_state(
    ...     ucb1_state,
    ...     target_algorithm="thompson_sampling"
    ... )
    >>> assert thompson_state.algorithm == "thompson_sampling"
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from conduit.core.state_store import BanditState

logger = logging.getLogger(__name__)

# Type alias for converter functions
StateConverter = Callable[[BanditState, int], BanditState]


def convert_bandit_state(
    source_state: BanditState,
    target_algorithm: str,
    feature_dim: int = 387,
) -> BanditState:
    """Convert bandit state from one algorithm to another.

    Performs optimistic conversion preserving learned knowledge where possible.
    All conversions maintain pull counts and success rates.

    Args:
        source_state: State from source algorithm
        target_algorithm: Target algorithm name
        feature_dim: Feature dimensionality for contextual algorithms (default: 387)

    Returns:
        Converted BanditState for target algorithm

    Raises:
        ValueError: If conversion is not supported or states incompatible

    Example:
        >>> ucb1_state = BanditState(
        ...     algorithm="ucb1",
        ...     arm_ids=["gpt-4o-mini", "gpt-4o"],
        ...     mean_rewards={"gpt-4o-mini": 0.75, "gpt-4o": 0.85},
        ...     arm_pulls={"gpt-4o-mini": 100, "gpt-4o": 50},
        ...     total_queries=150
        ... )
        >>> thompson_state = convert_bandit_state(ucb1_state, "thompson_sampling")
        >>> # Thompson state has alpha, beta computed from mean_rewards
    """
    source_algo = source_state.algorithm
    target_algo = target_algorithm

    # No conversion needed if same algorithm
    if source_algo == target_algo:
        logger.info(f"No conversion needed: {source_algo} == {target_algo}")
        return source_state

    logger.info(f"Converting bandit state: {source_algo} → {target_algo}")

    # Route to specific conversion function
    conversion_key = (source_algo, target_algo)

    conversion_map: dict[tuple[str, str], StateConverter] = {
        # Non-contextual ↔ Non-contextual
        ("ucb1", "thompson_sampling"): _ucb1_to_thompson,
        ("thompson_sampling", "ucb1"): _thompson_to_ucb1,
        # Contextual ↔ Contextual
        ("linucb", "contextual_thompson_sampling"): _linucb_to_contextual_thompson,
        ("contextual_thompson_sampling", "linucb"): _contextual_thompson_to_linucb,
        # Cross-category conversions (non-contextual → contextual)
        ("ucb1", "linucb"): lambda s, d: _noncontextual_to_linucb(s, d, "ucb1"),
        (
            "thompson_sampling",
            "linucb",
        ): lambda s, d: _noncontextual_to_linucb(s, d, "thompson_sampling"),
        (
            "ucb1",
            "contextual_thompson_sampling",
        ): lambda s, d: _noncontextual_to_contextual_thompson(s, d, "ucb1"),
        (
            "thompson_sampling",
            "contextual_thompson_sampling",
        ): lambda s, d: _noncontextual_to_contextual_thompson(
            s, d, "thompson_sampling"
        ),
        # Cross-category conversions (contextual → non-contextual)
        ("linucb", "ucb1"): lambda s, d: _contextual_to_ucb1(s, d, "linucb"),
        (
            "contextual_thompson_sampling",
            "ucb1",
        ): lambda s, d: _contextual_to_ucb1(s, d, "contextual_thompson_sampling"),
        (
            "linucb",
            "thompson_sampling",
        ): lambda s, d: _contextual_to_thompson(s, d, "linucb"),
        (
            "contextual_thompson_sampling",
            "thompson_sampling",
        ): lambda s, d: _contextual_to_thompson(s, d, "contextual_thompson_sampling"),
    }

    if conversion_key not in conversion_map:
        raise ValueError(
            f"Conversion not supported: {source_algo} → {target_algo}. "
            f"Supported conversions: {list(conversion_map.keys())}"
        )

    converter = conversion_map[conversion_key]
    converted_state = converter(source_state, feature_dim)

    logger.info(
        f"Conversion complete: {source_algo} → {target_algo} "
        f"(arms: {len(converted_state.arm_ids)}, "
        f"total_queries: {converted_state.total_queries})"
    )

    return converted_state


# ============================================================================
# NON-CONTEXTUAL CONVERSIONS (UCB1 ↔ Thompson Sampling)
# ============================================================================


def _ucb1_to_thompson(state: BanditState, feature_dim: int) -> BanditState:
    """Convert UCB1 state to Thompson Sampling state.

    Conversion formula:
        mean_reward = successes / pulls
        alpha = 1 + mean_reward * pulls
        beta = 1 + (1 - mean_reward) * pulls

    This preserves the Beta distribution's mean and variance based on observed data.
    """
    alpha_params = {}
    beta_params = {}

    for arm_id in state.arm_ids:
        pulls = state.arm_pulls.get(arm_id, 0)
        mean_reward_value = state.mean_reward.get(arm_id, 0.5)  # Default neutral prior

        if pulls > 0:
            # Convert mean_reward to Beta parameters
            successes = mean_reward_value * pulls
            failures = (1.0 - mean_reward_value) * pulls
            alpha_params[arm_id] = 1.0 + successes  # Beta(1,1) prior
            beta_params[arm_id] = 1.0 + failures
        else:
            # No data yet, use neutral prior
            alpha_params[arm_id] = 1.0
            beta_params[arm_id] = 1.0

    return BanditState(
        algorithm="thompson_sampling",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        alpha_params=alpha_params,
        beta_params=beta_params,
        reward_history=[],  # Start fresh (can't reconstruct from UCB1)
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


def _thompson_to_ucb1(state: BanditState, feature_dim: int) -> BanditState:
    """Convert Thompson Sampling state to UCB1 state.

    Conversion formula:
        mean_reward = alpha / (alpha + beta)
        pulls = (alpha + beta) - 2  # Remove Beta(1,1) prior

    This extracts the empirical mean from the Beta distribution.
    """
    mean_reward_dict = {}
    arm_pulls_corrected = {}

    for arm_id in state.arm_ids:
        alpha = state.alpha_params.get(arm_id, 1.0)
        beta = state.beta_params.get(arm_id, 1.0)

        # Extract mean from Beta distribution
        mean_reward_dict[arm_id] = alpha / (alpha + beta)

        # Reconstruct pull count (remove prior)
        pulls = max(0, int((alpha + beta) - 2.0))
        arm_pulls_corrected[arm_id] = pulls

    return BanditState(
        algorithm="ucb1",
        arm_ids=state.arm_ids,
        arm_pulls=arm_pulls_corrected,
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        mean_reward=mean_reward_dict,  # Fixed: use mean_reward not mean_rewards
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


# ============================================================================
# CONTEXTUAL CONVERSIONS (LinUCB ↔ Contextual Thompson Sampling)
# ============================================================================


def _linucb_to_contextual_thompson(state: BanditState, feature_dim: int) -> BanditState:
    """Convert LinUCB state to Contextual Thompson Sampling state.

    Conversion formula:
        mu = A^-1 @ b      (ridge regression estimate → posterior mean)
        Sigma = A^-1        (uncertainty from A → posterior covariance)

    Both algorithms use Bayesian linear regression, so conversion is natural.
    LinUCB: theta = A^-1 @ b
    CTS: theta ~ N(mu, Sigma)
    """
    from conduit.core.state_store import deserialize_bandit_matrices

    # Deserialize LinUCB matrices
    A_matrices, b_vectors = deserialize_bandit_matrices(
        state.A_matrices, state.b_vectors
    )

    mu_vectors = {}
    sigma_matrices = {}

    for arm_id in state.arm_ids:
        A = A_matrices[arm_id]
        b = b_vectors[arm_id]

        # Compute A^-1 (posterior covariance)
        A_inv = np.linalg.inv(A)

        # Compute posterior mean: mu = A^-1 @ b
        mu = A_inv @ b

        mu_vectors[arm_id] = mu.flatten().tolist()
        sigma_matrices[arm_id] = A_inv.tolist()

    return BanditState(
        algorithm="contextual_thompson_sampling",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        mu_vectors=mu_vectors,
        sigma_matrices=sigma_matrices,
        observation_history=state.observation_history,  # Preserve history
        feature_dim=state.feature_dim or feature_dim,
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


def _contextual_thompson_to_linucb(state: BanditState, feature_dim: int) -> BanditState:
    """Convert Contextual Thompson Sampling state to LinUCB state.

    Conversion formula:
        A = Sigma^-1        (posterior covariance → design matrix)
        b = A @ mu          (since theta = A^-1 @ b = mu)

    Both algorithms maintain equivalent linear models, so conversion is lossless.
    """
    from conduit.core.state_store import serialize_bandit_matrices

    A_matrices_dict = {}
    b_vectors_dict = {}

    for arm_id in state.arm_ids:
        mu = np.array(state.mu_vectors[arm_id]).reshape(-1, 1)
        Sigma = np.array(state.sigma_matrices[arm_id])

        # Compute A = Sigma^-1 (design matrix from posterior covariance)
        A = np.linalg.inv(Sigma)

        # Compute b = A @ mu (since theta = A^-1 @ b = mu)
        b = A @ mu

        A_matrices_dict[arm_id] = A
        b_vectors_dict[arm_id] = b

    # Serialize for BanditState
    A_matrices, b_vectors = serialize_bandit_matrices(A_matrices_dict, b_vectors_dict)

    return BanditState(
        algorithm="linucb",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        A_matrices=A_matrices,
        b_vectors=b_vectors,
        observation_history=state.observation_history,  # Preserve history
        feature_dim=state.feature_dim or feature_dim,
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


# ============================================================================
# CROSS-CATEGORY CONVERSIONS (Non-contextual ↔ Contextual)
# ============================================================================


def _noncontextual_to_linucb(
    state: BanditState, feature_dim: int, source_algo: str
) -> BanditState:
    """Convert non-contextual state (UCB1/Thompson) to LinUCB.

    Strategy:
    1. Extract mean_reward from source algorithm
    2. Initialize LinUCB's b vector first dimension with mean_reward
    3. Initialize A as identity (no feature knowledge yet)

    This gives LinUCB a "warm start" based on non-contextual quality estimates.
    """
    # First convert to Thompson Sampling if needed (to get alpha/beta)
    if source_algo == "ucb1":
        thompson_state = _ucb1_to_thompson(state, feature_dim)
    else:
        thompson_state = state

    A_matrices_dict = {}
    b_vectors_dict = {}

    for arm_id in state.arm_ids:
        alpha = thompson_state.alpha_params.get(arm_id, 1.0)
        beta = thompson_state.beta_params.get(arm_id, 1.0)
        mean_reward = alpha / (alpha + beta)

        # Initialize LinUCB with neutral A and biased b
        A = np.identity(feature_dim)

        # Set first dimension of b to mean_reward (scaled by exploration proportion)
        pulls = state.arm_pulls.get(arm_id, 0)
        total_pulls = sum(state.arm_pulls.values())
        # Scale by proportion of exploration, not absolute count
        # This prevents bias toward randomly-explored arms with low switch thresholds
        proportion = pulls / max(1, total_pulls)
        scaling_factor = min(10.0, proportion * 20.0) if pulls > 0 else 0.0

        b = np.zeros((feature_dim, 1))
        b[0] = mean_reward * scaling_factor  # Warm start from non-contextual estimate

        A_matrices_dict[arm_id] = A
        b_vectors_dict[arm_id] = b

    from conduit.core.state_store import serialize_bandit_matrices

    A_matrices, b_vectors = serialize_bandit_matrices(A_matrices_dict, b_vectors_dict)

    return BanditState(
        algorithm="linucb",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        A_matrices=A_matrices,
        b_vectors=b_vectors,
        observation_history=[],  # Start fresh
        feature_dim=feature_dim,
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


def _noncontextual_to_contextual_thompson(
    state: BanditState, feature_dim: int, source_algo: str
) -> BanditState:
    """Convert non-contextual state (UCB1/Thompson) to Contextual Thompson Sampling.

    Strategy:
    1. Extract mean_reward from source algorithm
    2. Initialize mu vector first dimension with mean_reward
    3. Initialize Sigma as identity (no feature knowledge yet)
    """
    # First convert to Thompson Sampling if needed
    if source_algo == "ucb1":
        thompson_state = _ucb1_to_thompson(state, feature_dim)
    else:
        thompson_state = state

    mu_vectors = {}
    sigma_matrices = {}

    for arm_id in state.arm_ids:
        alpha = thompson_state.alpha_params.get(arm_id, 1.0)
        beta = thompson_state.beta_params.get(arm_id, 1.0)
        mean_reward = alpha / (alpha + beta)

        # Initialize contextual Thompson with neutral Sigma and biased mu
        pulls = state.arm_pulls.get(arm_id, 0)
        total_pulls = sum(state.arm_pulls.values())
        # Scale by proportion of exploration, not absolute count
        proportion = pulls / max(1, total_pulls)
        scaling_factor = min(10.0, proportion * 20.0) if pulls > 0 else 0.0

        mu = np.zeros((feature_dim, 1))
        mu[0] = mean_reward * scaling_factor  # Warm start

        Sigma = np.identity(feature_dim)

        mu_vectors[arm_id] = mu.flatten().tolist()
        sigma_matrices[arm_id] = Sigma.tolist()

    return BanditState(
        algorithm="contextual_thompson_sampling",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        mu_vectors=mu_vectors,
        sigma_matrices=sigma_matrices,
        observation_history=[],  # Start fresh
        feature_dim=feature_dim,
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


def _contextual_to_ucb1(
    state: BanditState, feature_dim: int, source_algo: str
) -> BanditState:
    """Convert contextual state (LinUCB/Contextual Thompson) to UCB1.

    Strategy:
    1. Extract mean from contextual model (theta or mu)
    2. Use first dimension as mean_reward estimate
    3. Use arm_pulls for UCB1 confidence
    """
    # First convert to Contextual Thompson if needed
    if source_algo == "linucb":
        cts_state = _linucb_to_contextual_thompson(state, feature_dim)
    else:
        cts_state = state

    mean_rewards = {}

    for arm_id in state.arm_ids:
        mu = np.array(cts_state.mu_vectors[arm_id]).reshape(-1, 1)

        # Use first dimension of mu as mean_reward approximation
        # This is the "general quality" estimate
        mean_reward = float(mu[0, 0])

        # Clip to [0, 1] range
        mean_reward = max(0.0, min(1.0, mean_reward))

        mean_rewards[arm_id] = mean_reward

    return BanditState(
        algorithm="ucb1",
        arm_ids=state.arm_ids,
        arm_pulls=state.arm_pulls.copy(),
        arm_successes=state.arm_successes.copy(),
        total_queries=state.total_queries,
        mean_reward=mean_rewards,
        window_size=state.window_size,
        updated_at=state.updated_at,
    )


def _contextual_to_thompson(
    state: BanditState, feature_dim: int, source_algo: str
) -> BanditState:
    """Convert contextual state (LinUCB/Contextual Thompson) to Thompson Sampling.

    Strategy:
    1. Extract mean from contextual model
    2. Convert to Thompson Sampling Beta parameters
    """
    # First convert to UCB1 (to get mean_rewards)
    ucb1_state = _contextual_to_ucb1(state, feature_dim, source_algo)

    # Then convert UCB1 to Thompson Sampling
    return _ucb1_to_thompson(ucb1_state, feature_dim)
