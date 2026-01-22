"""OpenTelemetry observability for Conduit.

This module provides OpenTelemetry instrumentation for traces and metrics.
Exports to OTLP-compatible backends (Jaeger, Tempo, Prometheus, etc.).

Instrumented Components:
    - FastAPI requests (auto-instrumentation)
    - Redis operations (auto-instrumentation)
    - Custom routing metrics (cost, latency, model selection)
    - Thompson Sampling confidence scores
    - Cache hit rates
    - Feedback integration rates
    - Decision audit logging (compliance and debugging)
    - Structured logging via structlog (JSON/console output)
"""

from conduit.observability.audit import (
    AuditEntry,
    AuditQuery,
    AuditStore,
    InMemoryAuditStore,
    PostgresAuditStore,
    create_audit_entry,
)
from conduit.observability.logging import (
    LogEvents,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)
from conduit.observability.metrics import get_meter, record_routing_decision
from conduit.observability.setup import setup_telemetry, shutdown_telemetry
from conduit.observability.tracing import get_tracer, trace_operation

__all__ = [
    # Telemetry
    "setup_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_meter",
    "trace_operation",
    "record_routing_decision",
    # Audit logging
    "AuditEntry",
    "AuditQuery",
    "AuditStore",
    "PostgresAuditStore",
    "InMemoryAuditStore",
    "create_audit_entry",
    # Structured logging
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
    "LogEvents",
]
