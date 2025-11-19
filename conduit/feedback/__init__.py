"""Implicit feedback system for behavior-based learning.

This module implements the "Observability Trinity" of implicit signals:
- Error detection: Capture model failures and quality issues
- Latency tracking: Monitor response times and user patience
- Retry detection: Identify repeated/refined queries indicating dissatisfaction

Strategic Rationale:
    Reduces dependency on explicit user feedback (antifragile design),
    captures real behavior signals, and strengthens the data moat through
    continuous learning from usage patterns.

    See: notes/2025-11-18_business_panel_analysis.md (Meadows + Taleb analysis)

Usage:
    >>> from conduit.feedback import (
    ...     ImplicitFeedbackDetector,
    ...     QueryHistoryTracker,
    ...     FeedbackIntegrator,
    ... )
    >>> from redis.asyncio import Redis
    >>>
    >>> # Initialize components
    >>> redis = Redis.from_url("redis://localhost:6379")
    >>> history_tracker = QueryHistoryTracker(redis=redis)
    >>> detector = ImplicitFeedbackDetector(history_tracker)
    >>> integrator = FeedbackIntegrator(bandit)
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
    >>>
    >>> # Update Thompson Sampling bandit
    >>> integrator.update_from_implicit(
    ...     model="gpt-4o-mini",
    ...     features=query_features,
    ...     feedback=implicit_feedback
    ... )
"""

from conduit.feedback.detector import ImplicitFeedbackDetector
from conduit.feedback.history import QueryHistoryEntry, QueryHistoryTracker
from conduit.feedback.integration import FeedbackIntegrator
from conduit.feedback.signals import (
    ErrorSignal,
    LatencySignal,
    RetrySignal,
    SignalDetector,
)

__all__ = [
    # Core Components
    "ImplicitFeedbackDetector",
    "QueryHistoryTracker",
    "FeedbackIntegrator",
    # History Models
    "QueryHistoryEntry",
    # Signal Models
    "RetrySignal",
    "LatencySignal",
    "ErrorSignal",
    "SignalDetector",
]
