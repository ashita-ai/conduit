"""Feedback blending for dual-signal feedback loop.

Implements the documented 70/30 explicit/implicit feedback blending formula
for more robust bandit learning. This approach combines:
- Explicit feedback (70%): User-provided signals (thumbs, ratings, quality scores)
- Implicit feedback (30%): System-observed behavioral signals (errors, latency, retries)

The implicit signals provide a behavioral baseline even when explicit feedback
is sparse, improving the antifragility of the learning system.

Reference: AGENTS.md states "Integration: 70% explicit + 30% implicit = final reward"

Example:
    >>> from conduit.feedback.blending import blend_feedback, compute_implicit_score
    >>> from conduit.core.models import ImplicitFeedback
    >>>
    >>> # Compute implicit score from behavioral signals
    >>> implicit_score = compute_implicit_score(implicit_feedback)
    >>>
    >>> # Blend with explicit feedback
    >>> final_quality = blend_feedback(
    ...     explicit_score=0.85,  # User said good
    ...     implicit_score=0.6,   # Some behavioral issues
    ... )
    >>> print(f"{final_quality:.3f}")  # ~0.775 (70% * 0.85 + 30% * 0.6)
"""

from conduit.core.config import load_feedback_config
from conduit.core.models import ImplicitFeedback

# Default blending weights (documented as 70/30 split)
DEFAULT_EXPLICIT_WEIGHT = 0.70
DEFAULT_IMPLICIT_WEIGHT = 0.30


def load_blending_weights() -> tuple[float, float]:
    """Load blending weights from config or use defaults.

    Reads from conduit.yaml feedback.weights section:
        feedback:
          weights:
            explicit: 0.7
            implicit: 0.3

    Returns:
        Tuple of (explicit_weight, implicit_weight) that sum to 1.0
    """
    try:
        config = load_feedback_config()
        weights = config.get("weights", {})
        explicit = weights.get("explicit", DEFAULT_EXPLICIT_WEIGHT)
        implicit = weights.get("implicit", DEFAULT_IMPLICIT_WEIGHT)

        # Validate weights sum to 1.0
        total = explicit + implicit
        if abs(total - 1.0) > 0.01:
            # Normalize if they don't sum to 1.0
            return (explicit / total, implicit / total)
        return (explicit, implicit)
    except (FileNotFoundError, KeyError):
        return (DEFAULT_EXPLICIT_WEIGHT, DEFAULT_IMPLICIT_WEIGHT)


def compute_implicit_score(implicit: ImplicitFeedback) -> float:
    """Compute normalized quality score from implicit behavioral signals.

    Combines three signal categories into a single [0, 1] score:
    1. Error signals (weight: 0.50): Hard failure indicator
    2. Retry signals (weight: 0.30): User dissatisfaction indicator
    3. Latency signals (weight: 0.20): User patience indicator

    Args:
        implicit: ImplicitFeedback with detected behavioral signals

    Returns:
        Normalized implicit quality score in [0.0, 1.0]

    Score Interpretation:
        1.0 = No issues detected (no errors, no retries, fast response)
        0.7-0.9 = Minor issues (slow but acceptable)
        0.4-0.6 = Moderate issues (retry detected or high latency)
        0.0-0.3 = Severe issues (error occurred)

    Example:
        >>> # Perfect response
        >>> score = compute_implicit_score(ImplicitFeedback(
        ...     query_id="q1", model_id="m1", timestamp=0,
        ...     error_occurred=False,
        ...     latency_seconds=1.0, latency_accepted=True, latency_tolerance="high",
        ...     retry_detected=False
        ... ))
        >>> assert score == 1.0

        >>> # Error occurred
        >>> score = compute_implicit_score(ImplicitFeedback(
        ...     query_id="q1", model_id="m1", timestamp=0,
        ...     error_occurred=True, error_type="api_error",
        ...     latency_seconds=30.0, latency_accepted=True, latency_tolerance="low",
        ...     retry_detected=True
        ... ))
        >>> assert score < 0.5  # Severe penalty
    """
    # Signal weights (sum to 1.0)
    error_weight = 0.50  # Errors are most important
    retry_weight = 0.30  # Retries indicate dissatisfaction
    latency_weight = 0.20  # Latency is least important

    # 1. Error score: 0.0 if error, 1.0 if no error
    error_score = 0.0 if implicit.error_occurred else 1.0

    # 2. Retry score: 0.0 if retry detected, 1.0 if no retry
    retry_score = 0.0 if implicit.retry_detected else 1.0

    # 3. Latency score: Based on tolerance level
    # Map tolerance to score: high=1.0, medium=0.6, low=0.3
    latency_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    latency_score = latency_map.get(implicit.latency_tolerance, 0.6)

    # If user didn't wait (latency_accepted=False), penalize further
    if not implicit.latency_accepted:
        latency_score *= 0.5

    # Weighted combination
    implicit_score = (
        error_weight * error_score
        + retry_weight * retry_score
        + latency_weight * latency_score
    )

    return max(0.0, min(1.0, implicit_score))


def blend_feedback(
    explicit_score: float | None,
    implicit_score: float | None,
    explicit_weight: float | None = None,
    implicit_weight: float | None = None,
) -> float:
    """Blend explicit and implicit feedback scores.

    Implements the documented 70/30 blending formula:
        final_score = 0.70 * explicit + 0.30 * implicit

    Handles missing signals gracefully:
    - If only explicit: Use explicit score directly
    - If only implicit: Use implicit score directly
    - If both: Apply weighted blend
    - If neither: Return conservative default (0.5)

    Args:
        explicit_score: User-provided quality score [0, 1] or None
        implicit_score: Computed implicit score [0, 1] or None
        explicit_weight: Override explicit weight (default: 0.70)
        implicit_weight: Override implicit weight (default: 0.30)

    Returns:
        Blended quality score in [0.0, 1.0]

    Example:
        >>> # Both signals available
        >>> score = blend_feedback(explicit_score=0.9, implicit_score=0.6)
        >>> print(f"{score:.3f}")  # 0.81 (0.7*0.9 + 0.3*0.6)

        >>> # Only explicit
        >>> score = blend_feedback(explicit_score=0.8, implicit_score=None)
        >>> assert score == 0.8

        >>> # Only implicit
        >>> score = blend_feedback(explicit_score=None, implicit_score=0.7)
        >>> assert score == 0.7
    """
    # Load weights from config if not provided
    if explicit_weight is None or implicit_weight is None:
        default_explicit, default_implicit = load_blending_weights()
        if explicit_weight is None:
            explicit_weight = default_explicit
        if implicit_weight is None:
            implicit_weight = default_implicit

    # Handle missing signals
    if explicit_score is not None and implicit_score is not None:
        # Both available: apply weighted blend
        return explicit_weight * explicit_score + implicit_weight * implicit_score

    if explicit_score is not None:
        # Only explicit available
        return explicit_score

    if implicit_score is not None:
        # Only implicit available
        return implicit_score

    # Neither available: conservative default
    return 0.5


def compute_blended_confidence(
    explicit_confidence: float | None,
    has_implicit: bool,
    implicit_score: float | None = None,
) -> float:
    """Compute confidence for blended feedback.

    Higher confidence when we have more signal sources:
    - Both explicit + implicit: confidence = explicit_confidence (user rating is ground truth)
    - Only explicit: confidence = explicit_confidence
    - Only implicit: confidence = 0.5 (behavioral signals are softer)
    - Neither: confidence = 0.3 (conservative)

    If implicit score is very low (error/retry detected), we boost confidence
    because behavioral failures are reliable negative signals.

    Args:
        explicit_confidence: Confidence from explicit feedback adapter [0, 1]
        has_implicit: Whether implicit signals are available
        implicit_score: The computed implicit score (for error/retry detection)

    Returns:
        Blended confidence in [0.0, 1.0]
    """
    # Base confidence from explicit feedback
    if explicit_confidence is not None:
        base_confidence = explicit_confidence
    elif has_implicit:
        # Only implicit: softer confidence
        base_confidence = 0.5
    else:
        # No signals: very conservative
        return 0.3

    # Boost confidence if implicit signals strongly indicate failure
    # (errors and retries are reliable negative signals)
    if has_implicit and implicit_score is not None and implicit_score < 0.3:
        # Strong negative implicit signal, boost confidence
        return min(1.0, base_confidence + 0.2)

    return base_confidence
