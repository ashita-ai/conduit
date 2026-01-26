"""Feedback system for behavior-based and explicit learning.

This module implements both implicit and explicit feedback collection with
automatic blending using the documented 70/30 formula:

    final_reward = 0.70 * explicit_score + 0.30 * implicit_score

Implicit Feedback (behavior-based, 30% weight):
    - Error detection: Capture model failures and quality issues
    - Latency tracking: Monitor response times and user patience
    - Retry detection: Identify repeated queries indicating dissatisfaction

Explicit Feedback (user-provided, 70% weight):
    - Thumbs up/down: Binary approval signals
    - Ratings: Numeric quality scores (1-5 stars, etc.)
    - Task success: Objective completion indicators
    - Quality score: Direct scores from human reviewers or evaluators (Arbiter)

Blending Functions:
    - blend_feedback(): Combine explicit and implicit scores
    - compute_implicit_score(): Convert ImplicitFeedback to [0,1] score
    - compute_blended_confidence(): Determine confidence for blended feedback

Strategic Rationale:
    Dual feedback approach reduces dependency on explicit user feedback
    (antifragile design), captures real behavior signals, and strengthens
    the data moat through continuous learning from usage patterns.

    See: notes/2025-11-18_business_panel_analysis.md (Meadows + Taleb analysis)

Usage - Implicit Feedback:
    >>> from conduit.feedback import (
    ...     ImplicitFeedbackDetector,
    ...     QueryHistoryTracker,
    ... )
    >>> from redis.asyncio import Redis
    >>>
    >>> # Initialize components
    >>> redis = Redis.from_url("redis://localhost:6379")
    >>> history_tracker = QueryHistoryTracker(redis=redis)
    >>> detector = ImplicitFeedbackDetector(history_tracker)
    >>>
    >>> # Detect implicit signals
    >>> implicit_feedback = await detector.detect(
    ...     query="What is Python?",
    ...     query_id="q123",
    ...     features=query_features,
    ...     response_text="Python is...",
    ...     model_id="gpt-4o-mini",
    ...     execution_status="success",
    ...     execution_error=None,
    ...     request_start_time=start_time,
    ...     response_complete_time=end_time,
    ...     user_id="user_abc"
    ... )

Usage - Explicit Feedback:
    >>> from conduit.feedback import (
    ...     FeedbackCollector,
    ...     FeedbackEvent,
    ...     InMemoryFeedbackStore,
    ... )
    >>> from conduit.engines.router import Router
    >>>
    >>> # Setup
    >>> router = Router()
    >>> collector = FeedbackCollector(router)
    >>>
    >>> # Route query
    >>> decision = await router.route(query)
    >>>
    >>> # Track for delayed feedback
    >>> await collector.track(decision, cost=0.001, latency=0.5)
    >>>
    >>> # Later: Record user feedback
    >>> await collector.record(FeedbackEvent(
    ...     query_id=decision.query_id,
    ...     signal_type="thumbs",
    ...     payload={"value": "up"}
    ... ))
"""

# Feedback adapters
from conduit.feedback.adapters import (
    CompletionTimeAdapter,
    FeedbackAdapter,
    QualityScoreAdapter,
    RatingAdapter,
    RegenerationAdapter,
    TaskSuccessAdapter,
    ThumbsAdapter,
)

# Feedback blending (70/30 explicit/implicit)
from conduit.feedback.blending import (
    blend_feedback,
    compute_blended_confidence,
    compute_implicit_score,
    load_blending_weights,
)

# Feedback collector
from conduit.feedback.collector import FeedbackCollector
from conduit.feedback.detector import ImplicitFeedbackDetector
from conduit.feedback.history import QueryHistoryEntry, QueryHistoryTracker

# Explicit feedback models
from conduit.feedback.models import (
    FeedbackEvent,
    PendingQuery,
    RewardMapping,
    SessionFeedback,
)
from conduit.feedback.signals import (
    ErrorSignal,
    LatencySignal,
    RetrySignal,
    SignalDetector,
)

# Feedback stores
from conduit.feedback.stores import (
    FeedbackStore,
    InMemoryFeedbackStore,
    PostgresFeedbackStore,
    RedisFeedbackStore,
)

__all__ = [
    # Feedback Blending (70/30)
    "blend_feedback",
    "compute_implicit_score",
    "compute_blended_confidence",
    "load_blending_weights",
    # Implicit Feedback
    "ImplicitFeedbackDetector",
    "QueryHistoryTracker",
    "QueryHistoryEntry",
    "RetrySignal",
    "LatencySignal",
    "ErrorSignal",
    "SignalDetector",
    # Explicit Feedback Models
    "FeedbackEvent",
    "RewardMapping",
    "PendingQuery",
    "SessionFeedback",
    # Feedback Adapters
    "FeedbackAdapter",
    "ThumbsAdapter",
    "RatingAdapter",
    "TaskSuccessAdapter",
    "QualityScoreAdapter",
    "CompletionTimeAdapter",
    "RegenerationAdapter",
    # Feedback Collector
    "FeedbackCollector",
    # Feedback Stores
    "FeedbackStore",
    "InMemoryFeedbackStore",
    "RedisFeedbackStore",
    "PostgresFeedbackStore",
]
