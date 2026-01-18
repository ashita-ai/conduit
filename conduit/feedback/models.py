"""Explicit feedback models for user-provided signals.

This module defines Pydantic models for explicit feedback:
- FeedbackEvent: User-provided feedback signal (thumbs, rating, task_success)
- RewardMapping: Result of converting feedback to bandit reward
- PendingQuery: Tracked query awaiting delayed feedback

These models complement the implicit feedback signals in signals.py.

Usage:
    >>> from conduit.feedback.models import FeedbackEvent, RewardMapping
    >>>
    >>> # Create feedback event
    >>> event = FeedbackEvent(
    ...     query_id="abc123",
    ...     signal_type="thumbs",
    ...     payload={"value": "up"}
    ... )
    >>>
    >>> # Convert to reward via adapter
    >>> reward = ThumbsAdapter().to_reward(event)
    >>> print(reward.reward)  # 1.0
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class FeedbackEvent(BaseModel):
    """A user feedback signal for a routed query.

    Represents explicit feedback from users (thumbs up/down, ratings,
    task success indicators). Used with FeedbackAdapter to convert
    to bandit rewards.

    Attributes:
        query_id: ID of the query this feedback relates to
        signal_type: Type of feedback (thumbs, rating, task_success, regeneration)
        timestamp: When the feedback was received
        payload: Signal-specific data (e.g., {"value": "up"}, {"rating": 5})
        idempotency_key: Optional client-provided key for deduplication

    Idempotency:
        The feedback system is idempotent: recording the same feedback
        multiple times has the same effect as recording it once. This
        prevents double-counting from retries or duplicate submissions.

        Key strategies:
        - If idempotency_key is provided, it's used directly
        - Otherwise, query_id:signal_type is used as natural key
        - Same query + same signal type = deduplicated
        - Same query + different signal types = both recorded

    Example:
        >>> # Automatic deduplication (natural key)
        >>> event = FeedbackEvent(
        ...     query_id="q123",
        ...     signal_type="thumbs",
        ...     payload={"value": "up"}
        ... )
        >>>
        >>> # Explicit idempotency key (for retries)
        >>> event = FeedbackEvent(
        ...     query_id="q123",
        ...     signal_type="thumbs",
        ...     payload={"value": "up"},
        ...     idempotency_key="client-uuid-12345",
        ... )
    """

    query_id: str = Field(..., description="Query ID this feedback relates to")
    signal_type: str = Field(
        ..., description="Type of feedback: thumbs, rating, task_success, regeneration"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feedback timestamp",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Signal-specific data"
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional client-provided key for deduplication",
    )

    def get_idempotency_key(self) -> str:
        """Get the idempotency key for this event.

        Returns client-provided key if set, otherwise generates
        a natural key from query_id and signal_type.

        Returns:
            Idempotency key string
        """
        if self.idempotency_key:
            return self.idempotency_key
        return f"{self.query_id}:{self.signal_type}"

    def __repr__(self) -> str:
        """Return concise repr for debugging."""
        return f"FeedbackEvent({self.query_id[:8]!r}, {self.signal_type!r})"


class RewardMapping(BaseModel):
    """Result of converting a feedback signal to a bandit reward.

    Adapters convert FeedbackEvent signals to RewardMapping objects
    that can be used to update bandit weights.

    Attributes:
        reward: Normalized reward value (0.0 = bad, 1.0 = good)
        confidence: How certain this signal is (0.0 = uncertain, 1.0 = certain)

    Confidence Interpretation:
        - 1.0: Explicit user feedback (thumbs, rating)
        - 0.8: Implicit but strong signal (regeneration, task failure)
        - 0.5: Uncertain signal (neutral response)

    Example:
        >>> mapping = RewardMapping(reward=1.0, confidence=1.0)  # Thumbs up
        >>> mapping = RewardMapping(reward=0.0, confidence=0.8)  # Regeneration
    """

    reward: float = Field(..., description="Reward value (0.0-1.0)", ge=0.0, le=1.0)
    confidence: float = Field(
        default=1.0, description="Signal confidence (0.0-1.0)", ge=0.0, le=1.0
    )


class PendingQuery(BaseModel):
    """Tracked query awaiting delayed feedback.

    Stores query metadata so feedback can be correlated later.
    Used by FeedbackCollector for delayed feedback scenarios.

    Attributes:
        query_id: Unique query identifier (from RoutingDecision)
        model_id: Model that was selected for this query
        features: QueryFeatures serialized for storage
        cost: Estimated or actual cost of the query
        latency: Response latency in seconds
        created_at: When the query was tracked
        ttl_seconds: Time-to-live before expiry (default 1 hour)
        session_id: Optional session ID for multi-turn tracking

    Example:
        >>> pending = PendingQuery(
        ...     query_id="q123",
        ...     model_id="gpt-4o",
        ...     features={"embedding": [...], "token_count": 50, ...},
        ...     cost=0.001,
        ...     latency=0.5,
        ... )
    """

    query_id: str = Field(..., description="Unique query identifier")
    model_id: str = Field(..., description="Selected model for this query")
    features: dict[str, Any] = Field(
        ..., description="QueryFeatures serialized as dict"
    )
    cost: float = Field(default=0.0, description="Query cost in dollars", ge=0.0)
    latency: float = Field(
        default=0.0, description="Response latency in seconds", ge=0.0
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When query was tracked",
    )
    ttl_seconds: int = Field(default=3600, description="Time-to-live in seconds", gt=0)
    session_id: str | None = Field(
        default=None, description="Optional session ID for multi-turn tracking"
    )

    def is_expired(self) -> bool:
        """Check if this pending query has expired.

        Returns:
            True if TTL has elapsed since created_at
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def __repr__(self) -> str:
        """Return concise repr for debugging."""
        return f"PendingQuery({self.query_id[:8]!r}, {self.model_id!r})"


class SessionFeedback(BaseModel):
    """Session-level feedback for multi-turn conversations.

    When a user provides feedback at the end of a conversation or session,
    this model captures the overall sentiment that should be propagated
    to all queries in the session.

    Attributes:
        session_id: Unique session identifier
        signal_type: Type of feedback signal (thumbs, rating, etc.)
        payload: Signal-specific data
        timestamp: When session feedback was received
        propagation_weight: How much to weight session feedback vs query feedback.
            0.0 = ignore session feedback, 1.0 = full weight (default 0.5)

    Example:
        >>> session_feedback = SessionFeedback(
        ...     session_id="session_abc123",
        ...     signal_type="rating",
        ...     payload={"rating": 4},
        ...     propagation_weight=0.5,
        ... )
    """

    session_id: str = Field(..., description="Session identifier")
    signal_type: str = Field(..., description="Feedback signal type")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Signal-specific data"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When session feedback was received",
    )
    propagation_weight: float = Field(
        default=0.5,
        description="Weight for propagating to individual queries (0-1)",
        ge=0.0,
        le=1.0,
    )
