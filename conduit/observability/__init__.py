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
    - Evaluation metrics (regret, convergence, quality)
"""

from conduit.observability.evaluation_exporter import EvaluationMetricsExporter
from conduit.observability.metrics import (
    get_meter,
    record_evaluation_metrics,
    record_routing_decision,
)
from conduit.observability.setup import setup_telemetry, shutdown_telemetry
from conduit.observability.tracing import get_tracer, trace_operation

__all__ = [
    "setup_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_meter",
    "trace_operation",
    "record_routing_decision",
    "record_evaluation_metrics",
    "EvaluationMetricsExporter",
]
