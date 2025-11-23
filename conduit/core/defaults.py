"""Centralized default values and configuration constants.

All magic numbers and hardcoded thresholds should be defined here
to avoid duplication and ensure consistency across the codebase.
"""

from dataclasses import dataclass

# =============================================================================
# BANDIT ALGORITHM DEFAULTS
# =============================================================================

# Multi-Objective Reward Weights
# Used across all bandit algorithms for composite reward calculation
DEFAULT_REWARD_WEIGHTS = {
    "quality": 0.70,  # 70% weight on response quality
    "cost": 0.20,  # 20% weight on cost efficiency
    "latency": 0.10,  # 10% weight on response speed
}

# Success Threshold
# Minimum reward value to count as "success" for statistics
SUCCESS_THRESHOLD = 0.85

# Exploration Parameters (algorithm-specific defaults)
LINUCB_ALPHA_DEFAULT = 1.0  # LinUCB exploration parameter
EPSILON_GREEDY_DEFAULT = 0.1  # Epsilon-greedy exploration rate (10%)
EPSILON_DECAY_DEFAULT = 1.0  # No decay by default
EPSILON_MIN_DEFAULT = 0.01  # Minimum epsilon after decay
UCB1_C_DEFAULT = 1.5  # UCB1 confidence multiplier

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

# Feature dimensionality
FEATURE_DIM_FULL = 387  # 384 embedding + 3 metadata
FEATURE_DIM_PCA = 67  # 64 PCA components + 3 metadata
EMBEDDING_DIM = 384  # Sentence transformer embedding size

# Feature normalization
TOKEN_COUNT_NORMALIZATION = 1000.0  # Divide token_count by this value

# =============================================================================
# QUALITY ESTIMATION (LiteLLM Feedback)
# =============================================================================


@dataclass
class QualityEstimationConfig:
    """Configuration for automatic quality estimation from response content.

    Used by conduit_litellm/feedback.py for heuristic-based quality scoring.
    """

    # Base quality scores
    base_quality: float = 0.9  # Starting quality for successful responses
    empty_quality: float = 0.1  # Quality for empty/no responses
    failure_quality: float = 0.1  # Quality for failed requests

    # Length thresholds
    min_response_chars: int = 10  # Minimum chars for valid response

    # Quality penalties
    short_response_penalty: float = 0.15  # Penalty for very short responses
    repetition_penalty: float = 0.30  # Penalty for repetitive/looping text
    no_keyword_overlap_penalty: float = 0.20  # No common keywords
    low_keyword_overlap_penalty: float = 0.10  # Few common keywords

    # Keyword overlap thresholds
    keyword_overlap_very_low: float = 0.05  # Almost no common words
    keyword_overlap_low: float = 0.15  # Some common words

    # Repetition detection
    repetition_min_length: int = 20  # Minimum pattern length for repetition
    repetition_occurrence_threshold: int = 3  # Pattern repeats 3+ times

    # Quality bounds
    min_quality: float = 0.1  # Floor for quality scores
    max_quality: float = 0.95  # Ceiling for quality scores

    # Stopwords for keyword overlap (basic set)
    stopwords: frozenset[str] = frozenset({
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "are",
        "was",
        "were",
    })


# Default instance
QUALITY_ESTIMATION_DEFAULTS = QualityEstimationConfig()

# =============================================================================
# IMPLICIT FEEDBACK DETECTION
# =============================================================================


@dataclass
class RetryDetectionConfig:
    """Configuration for retry behavior detection."""

    # Semantic similarity threshold for retry detection
    similarity_threshold: float = 0.85  # 85% similar = likely retry

    # Time window for retry consideration
    time_window_seconds: float = 300.0  # 5 minutes


@dataclass
class LatencyConfig:
    """Configuration for latency tolerance categorization."""

    # Latency thresholds (seconds)
    high_tolerance_max: float = 10.0  # < 10s = fast, high tolerance
    medium_tolerance_max: float = 30.0  # 10-30s = medium tolerance
    # > 30s = slow, low tolerance (user was patient)

    # Reward mapping for implicit feedback
    high_tolerance_reward: float = 0.9  # Fast response
    medium_tolerance_reward: float = 0.7  # Acceptable speed
    low_tolerance_reward: float = 0.5  # Slow but user waited


@dataclass
class ErrorDetectionConfig:
    """Configuration for error signal detection."""

    # Minimum response length
    min_response_chars: int = 10  # Same as quality estimation

    # Error patterns to detect in response text
    error_patterns: tuple[str, ...] = (
        "I apologize, but I",
        "I cannot",
        "I'm unable to",
        "Error:",
        "Exception:",
        "Failed to",
    )


@dataclass
class ImplicitFeedbackConfig:
    """Combined configuration for all implicit feedback signals."""

    retry: RetryDetectionConfig = RetryDetectionConfig()
    latency: LatencyConfig = LatencyConfig()
    error: ErrorDetectionConfig = ErrorDetectionConfig()

    # Implicit feedback reward mapping
    error_reward: float = 0.0  # Complete failure
    retry_reward: float = 0.3  # User dissatisfaction
    # Latency rewards from LatencyConfig

    # Weighting between explicit and implicit feedback
    explicit_weight: float = 0.7  # 70% weight on user ratings
    implicit_weight: float = 0.3  # 30% weight on behavioral signals


# Default instance
IMPLICIT_FEEDBACK_DEFAULTS = ImplicitFeedbackConfig()

# =============================================================================
# HYBRID ROUTING
# =============================================================================

# UCB1 → LinUCB transition point
HYBRID_SWITCH_THRESHOLD = 2000  # Switch after 2000 queries

# =============================================================================
# ARBITER EVALUATION
# =============================================================================

# Sampling and budget defaults
ARBITER_SAMPLE_RATE_DEFAULT = 0.1  # Evaluate 10% of responses
ARBITER_DAILY_BUDGET_DEFAULT = 10.0  # $10/day max spend
ARBITER_MODEL_DEFAULT = "gpt-4o-mini"  # Cheap evaluation model

# Evaluator types
ARBITER_EVALUATORS = [
    "semantic",  # Query-response similarity
    "factuality",  # Factual correctness
]
