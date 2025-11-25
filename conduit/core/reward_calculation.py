"""Reward calculation utilities for multi-objective optimization.

This module provides clear, well-documented functions for calculating
composite rewards from quality, cost, and latency metrics.

Key Concepts:
    - Asymptotic Normalization: Maps unbounded values to [0, 1] without population stats
    - Multi-Objective Reward: Weighted combination of quality, cost, and latency
    - User Preferences: Runtime override of default reward weights

Normalization Functions:
    All normalization maps raw values to [0, 1] where higher is better.

    normalize_quality(score):
        Quality scores are already in [0, 1]. Just clamps to valid range.
        Example: 0.95 -> 0.95, 1.2 -> 1.0, -0.1 -> 0.0

    normalize_cost(cost):
        Uses asymptotic normalization: 1 / (1 + cost)
        Lower cost = higher normalized value
        Example: $0 -> 1.0, $1 -> 0.5, $10 -> 0.09

    normalize_latency(latency):
        Uses asymptotic normalization: 1 / (1 + latency)
        Lower latency = higher normalized value
        Example: 0s -> 1.0, 1s -> 0.5, 10s -> 0.09

Example:
    >>> from conduit.core.reward_calculation import calculate_composite_reward
    >>> reward = calculate_composite_reward(
    ...     quality=0.95,
    ...     cost=0.001,
    ...     latency=1.5
    ... )
    >>> print(f"{reward:.3f}")
    0.869
"""

from conduit.core.defaults import DEFAULT_REWARD_WEIGHTS


def normalize_quality(quality_score: float) -> float:
    """Normalize quality score to [0, 1] range.

    Quality scores are expected to already be in [0, 1].
    This function clamps out-of-range values.

    Args:
        quality_score: Raw quality score (expected 0-1)

    Returns:
        Normalized quality in [0, 1] where higher is better

    Example:
        >>> normalize_quality(0.95)
        0.95
        >>> normalize_quality(1.2)  # Clamp to max
        1.0
        >>> normalize_quality(-0.1)  # Clamp to min
        0.0
    """
    return max(0.0, min(1.0, quality_score))


def normalize_cost(cost: float) -> float:
    """Normalize cost using asymptotic function.

    Uses 1 / (1 + cost) to map unbounded positive values to (0, 1].
    No population statistics required.

    Behavior:
        - cost=0 -> 1.0 (free is best)
        - cost=1 -> 0.5 (moderate penalty)
        - cost=10 -> ~0.09 (high penalty)
        - cost=100 -> ~0.01 (very high penalty)

    Args:
        cost: Raw cost in USD (must be non-negative)

    Returns:
        Normalized cost in (0, 1] where higher is better (lower cost)

    Raises:
        ValueError: If cost is negative

    Example:
        >>> normalize_cost(0.0)
        1.0
        >>> normalize_cost(1.0)
        0.5
        >>> f"{normalize_cost(10.0):.4f}"
        '0.0909'
    """
    if cost < 0:
        raise ValueError(f"Cost cannot be negative, got {cost}")
    return 1.0 / (1.0 + cost)


def normalize_latency(latency: float) -> float:
    """Normalize latency using asymptotic function.

    Uses 1 / (1 + latency) to map unbounded positive values to (0, 1].
    No population statistics required.

    Behavior:
        - latency=0s -> 1.0 (instant is best)
        - latency=1s -> 0.5 (moderate penalty)
        - latency=10s -> ~0.09 (high penalty)
        - latency=100s -> ~0.01 (very high penalty)

    Args:
        latency: Raw latency in seconds (must be non-negative)

    Returns:
        Normalized latency in (0, 1] where higher is better (lower latency)

    Raises:
        ValueError: If latency is negative

    Example:
        >>> normalize_latency(0.0)
        1.0
        >>> normalize_latency(1.0)
        0.5
        >>> f"{normalize_latency(10.0):.4f}"
        '0.0909'
    """
    if latency < 0:
        raise ValueError(f"Latency cannot be negative, got {latency}")
    return 1.0 / (1.0 + latency)


def validate_weights(
    quality_weight: float,
    cost_weight: float,
    latency_weight: float,
    tolerance: float = 0.01,
) -> None:
    """Validate that reward weights sum to 1.0.

    Args:
        quality_weight: Weight for quality component
        cost_weight: Weight for cost component
        latency_weight: Weight for latency component
        tolerance: Acceptable deviation from 1.0 (default: 0.01)

    Raises:
        ValueError: If weights don't sum to 1.0 within tolerance

    Example:
        >>> validate_weights(0.7, 0.2, 0.1)  # OK, sums to 1.0
        >>> validate_weights(0.5, 0.5, 0.5)  # Raises ValueError
        Traceback (most recent call last):
            ...
        ValueError: Reward weights must sum to 1.0, got 1.500
    """
    total = quality_weight + cost_weight + latency_weight
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Reward weights must sum to 1.0, got {total:.3f}")


def calculate_composite_reward(
    quality: float,
    cost: float,
    latency: float,
    quality_weight: float | None = None,
    cost_weight: float | None = None,
    latency_weight: float | None = None,
) -> float:
    """Calculate composite reward from quality, cost, and latency.

    Combines three objectives into a single scalar reward:
    - Quality: Higher is better (response quality)
    - Cost: Lower is better (inverted via asymptotic normalization)
    - Latency: Lower is better (inverted via asymptotic normalization)

    Default weights (from defaults.py):
        - Quality: 70%
        - Cost: 20%
        - Latency: 10%

    Args:
        quality: Quality score (0-1 scale)
        cost: Cost in USD (non-negative)
        latency: Latency in seconds (non-negative)
        quality_weight: Optional override for quality weight
        cost_weight: Optional override for cost weight
        latency_weight: Optional override for latency weight

    Returns:
        Composite reward in [0, 1] range where higher is better

    Raises:
        ValueError: If weights don't sum to 1.0 or inputs are invalid

    Example:
        >>> # High quality, low cost, moderate latency
        >>> reward = calculate_composite_reward(
        ...     quality=0.95,
        ...     cost=0.001,
        ...     latency=1.5
        ... )
        >>> print(f"{reward:.3f}")
        0.869

        >>> # With custom weights (cost-sensitive)
        >>> reward = calculate_composite_reward(
        ...     quality=0.95,
        ...     cost=0.001,
        ...     latency=1.5,
        ...     quality_weight=0.5,
        ...     cost_weight=0.4,
        ...     latency_weight=0.1
        ... )
        >>> print(f"{reward:.3f}")
        0.915
    """
    # Use defaults if not provided
    if quality_weight is None:
        quality_weight = DEFAULT_REWARD_WEIGHTS["quality"]
    if cost_weight is None:
        cost_weight = DEFAULT_REWARD_WEIGHTS["cost"]
    if latency_weight is None:
        latency_weight = DEFAULT_REWARD_WEIGHTS["latency"]

    # Validate weights
    validate_weights(quality_weight, cost_weight, latency_weight)

    # Normalize each component
    quality_norm = normalize_quality(quality)
    cost_norm = normalize_cost(cost)
    latency_norm = normalize_latency(latency)

    # Weighted combination
    reward = (
        quality_weight * quality_norm
        + cost_weight * cost_norm
        + latency_weight * latency_norm
    )

    return reward


def apply_user_preferences(
    base_quality_weight: float,
    base_cost_weight: float,
    base_latency_weight: float,
    quality_preference: float | None = None,
    cost_preference: float | None = None,
    latency_preference: float | None = None,
) -> tuple[float, float, float]:
    """Apply user preference adjustments to base weights.

    User preferences are multipliers (0.0 to 2.0) that adjust base weights:
    - 0.0 = completely deprioritize this objective
    - 1.0 = no change (default)
    - 2.0 = double the weight for this objective

    After applying multipliers, weights are renormalized to sum to 1.0.

    Args:
        base_quality_weight: Starting quality weight
        base_cost_weight: Starting cost weight
        base_latency_weight: Starting latency weight
        quality_preference: Multiplier for quality (default: 1.0)
        cost_preference: Multiplier for cost (default: 1.0)
        latency_preference: Multiplier for latency (default: 1.0)

    Returns:
        Tuple of (quality_weight, cost_weight, latency_weight) normalized to sum to 1.0

    Example:
        >>> # User wants to prioritize cost savings
        >>> weights = apply_user_preferences(
        ...     0.7, 0.2, 0.1,  # Base weights
        ...     cost_preference=2.0  # Double cost importance
        ... )
        >>> print(f"Quality: {weights[0]:.2f}, Cost: {weights[1]:.2f}, Latency: {weights[2]:.2f}")
        Quality: 0.58, Cost: 0.33, Latency: 0.08
    """
    # Apply multipliers (default to 1.0 = no change)
    q_mult = quality_preference if quality_preference is not None else 1.0
    c_mult = cost_preference if cost_preference is not None else 1.0
    l_mult = latency_preference if latency_preference is not None else 1.0

    # Clamp multipliers to valid range
    q_mult = max(0.0, min(2.0, q_mult))
    c_mult = max(0.0, min(2.0, c_mult))
    l_mult = max(0.0, min(2.0, l_mult))

    # Apply multipliers
    adj_quality = base_quality_weight * q_mult
    adj_cost = base_cost_weight * c_mult
    adj_latency = base_latency_weight * l_mult

    # Renormalize to sum to 1.0
    total = adj_quality + adj_cost + adj_latency
    if total == 0:
        # Edge case: all weights zeroed out, return equal weights
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)

    return (adj_quality / total, adj_cost / total, adj_latency / total)
