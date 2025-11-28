"""OpenTelemetry metrics for Conduit routing system.

Provides custom metrics for cost savings, quality scores, latency,
model selection, and cache performance.

Metrics:
    - conduit.routing.decisions: Counter of routing decisions by model
    - conduit.routing.cost: Histogram of LLM costs in dollars
    - conduit.routing.latency: Histogram of query latency in milliseconds
    - conduit.routing.confidence: Histogram of routing confidence scores
    - conduit.cache.hits: Counter of cache hits
    - conduit.cache.misses: Counter of cache misses
    - conduit.feedback.submissions: Counter of feedback by type
"""

import logging
from typing import Any

from opentelemetry import metrics

from conduit.core.config import settings
from conduit.core.models import RoutingDecision

logger = logging.getLogger(__name__)

# Global meter instance
_meter: metrics.Meter | None = None

# Metric instruments (created on first access)
_routing_decisions_counter: metrics.Counter | None = None
_routing_cost_histogram: metrics.Histogram | None = None
_routing_latency_histogram: metrics.Histogram | None = None
_routing_confidence_histogram: metrics.Histogram | None = None
_cache_hits_counter: metrics.Counter | None = None
_cache_misses_counter: metrics.Counter | None = None
_feedback_submissions_counter: metrics.Counter | None = None


def get_meter(name: str = "conduit") -> metrics.Meter:
    """Get OpenTelemetry meter instance.

    Args:
        name: Meter name

    Returns:
        Meter instance (no-op if telemetry disabled)
    """
    global _meter
    if _meter is None:
        _meter = metrics.get_meter(name)
    return _meter


def _ensure_instruments() -> None:
    """Lazy initialization of metric instruments."""
    if not settings.otel_enabled or not settings.otel_metrics_enabled:
        return

    global _routing_decisions_counter
    global _routing_cost_histogram
    global _routing_latency_histogram
    global _routing_confidence_histogram
    global _cache_hits_counter
    global _cache_misses_counter
    global _feedback_submissions_counter

    meter = get_meter()

    if _routing_decisions_counter is None:
        _routing_decisions_counter = meter.create_counter(
            name="conduit.routing.decisions",
            description="Number of routing decisions made",
            unit="1",
        )

    if _routing_cost_histogram is None:
        _routing_cost_histogram = meter.create_histogram(
            name="conduit.routing.cost",
            description="LLM query cost in dollars",
            unit="USD",
        )

    if _routing_latency_histogram is None:
        _routing_latency_histogram = meter.create_histogram(
            name="conduit.routing.latency",
            description="Query end-to-end latency",
            unit="ms",
        )

    if _routing_confidence_histogram is None:
        _routing_confidence_histogram = meter.create_histogram(
            name="conduit.routing.confidence",
            description="Thompson Sampling confidence score",
            unit="1",
        )

    if _cache_hits_counter is None:
        _cache_hits_counter = meter.create_counter(
            name="conduit.cache.hits",
            description="Number of cache hits",
            unit="1",
        )

    if _cache_misses_counter is None:
        _cache_misses_counter = meter.create_counter(
            name="conduit.cache.misses",
            description="Number of cache misses",
            unit="1",
        )

    if _feedback_submissions_counter is None:
        _feedback_submissions_counter = meter.create_counter(
            name="conduit.feedback.submissions",
            description="Number of feedback submissions",
            unit="1",
        )


def record_routing_decision(
    decision: RoutingDecision,
    cost: float | None = None,
    latency_ms: float | None = None,
) -> None:
    """Record routing decision metrics.

    Args:
        decision: Routing decision with model selection
        cost: LLM query cost in dollars (optional)
        latency_ms: Query latency in milliseconds (optional)
    """
    if not settings.otel_enabled or not settings.otel_metrics_enabled:
        return

    _ensure_instruments()

    attributes = {
        "model": decision.selected_model,
        "domain": getattr(decision.features, "domain", "unknown"),
    }

    # Record decision count
    if _routing_decisions_counter:
        _routing_decisions_counter.add(1, attributes)

    # Record confidence score
    if _routing_confidence_histogram:
        _routing_confidence_histogram.record(decision.confidence, attributes)

    # Record cost if provided
    if cost is not None and _routing_cost_histogram:
        _routing_cost_histogram.record(cost, attributes)

    # Record latency if provided
    if latency_ms is not None and _routing_latency_histogram:
        _routing_latency_histogram.record(latency_ms, attributes)


def record_cache_hit() -> None:
    """Record cache hit metric."""
    if not settings.otel_enabled or not settings.otel_metrics_enabled:
        return

    _ensure_instruments()

    if _cache_hits_counter:
        _cache_hits_counter.add(1)


def record_cache_miss() -> None:
    """Record cache miss metric."""
    if not settings.otel_enabled or not settings.otel_metrics_enabled:
        return

    _ensure_instruments()

    if _cache_misses_counter:
        _cache_misses_counter.add(1)


def record_feedback_submission(feedback_type: str, explicit: bool = True) -> None:
    """Record feedback submission metric.

    Args:
        feedback_type: Type of feedback (positive, negative, neutral)
        explicit: True for explicit user feedback, False for implicit signals
    """
    if not settings.otel_enabled or not settings.otel_metrics_enabled:
        return

    _ensure_instruments()

    if _feedback_submissions_counter:
        attributes = {
            "type": feedback_type,
            "source": "explicit" if explicit else "implicit",
        }
        _feedback_submissions_counter.add(1, attributes)


def get_metrics_summary() -> dict[str, Any]:
    """Get current metrics summary for debugging.

    Returns:
        Dictionary with metric values (not all OTEL providers support reading)

    Note:
        This is for debugging only. Use OTLP backend for production metrics.
    """
    return {
        "otel_enabled": settings.otel_enabled,
        "metrics_enabled": settings.otel_metrics_enabled,
        "service_name": settings.otel_service_name,
        "exporter_endpoint": settings.otel_exporter_otlp_endpoint,
    }
