"""OpenTelemetry setup and initialization.

Configures OTLP exporters, trace providers, and metric providers
based on configuration settings.
"""

import logging
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from conduit.core.config import settings

logger = logging.getLogger(__name__)

_instrumented = False


def setup_telemetry(app: Any | None = None) -> None:
    """Initialize OpenTelemetry instrumentation.

    Args:
        app: FastAPI application instance for auto-instrumentation

    Note:
        Only initializes if otel_enabled=True in configuration.
        Safe to call multiple times (idempotent).
    """
    global _instrumented

    if not settings.otel_enabled:
        logger.info("OpenTelemetry disabled by configuration")
        return

    if _instrumented:
        logger.debug("OpenTelemetry already initialized")
        return

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.0.1",
            "deployment.environment": settings.environment,
        }
    )

    # Parse OTLP headers if provided
    headers = None
    if settings.otel_exporter_otlp_headers:
        headers = dict(
            item.split("=", 1)
            for item in settings.otel_exporter_otlp_headers.split(",")
            if "=" in item
        )

    # Setup tracing if enabled
    if settings.otel_traces_enabled:
        trace_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=headers,
        )

        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(trace_provider)

        logger.info(
            f"OpenTelemetry tracing initialized: {settings.otel_exporter_otlp_endpoint}"
        )

    # Setup metrics if enabled
    if settings.otel_metrics_enabled:
        metric_exporter = OTLPMetricExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=headers,
        )

        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=60000,  # Export every 60 seconds
        )

        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)

        logger.info(
            f"OpenTelemetry metrics initialized: {settings.otel_exporter_otlp_endpoint}"
        )

    # Auto-instrument FastAPI if app provided
    if app:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumentation enabled")

    # Auto-instrument Redis
    RedisInstrumentor().instrument()
    logger.info("Redis auto-instrumentation enabled")

    _instrumented = True
    logger.info("OpenTelemetry initialization complete")


def shutdown_telemetry() -> None:
    """Shutdown OpenTelemetry providers gracefully.

    Flushes pending spans and metrics before shutdown.
    """
    if not settings.otel_enabled:
        return

    # Shutdown trace provider
    if settings.otel_traces_enabled:
        trace_provider = trace.get_tracer_provider()
        if hasattr(trace_provider, "shutdown"):
            trace_provider.shutdown()
            logger.info("OpenTelemetry trace provider shutdown")

    # Shutdown meter provider
    if settings.otel_metrics_enabled:
        meter_provider = metrics.get_meter_provider()
        if hasattr(meter_provider, "shutdown"):
            meter_provider.shutdown()
            logger.info("OpenTelemetry meter provider shutdown")

    logger.info("OpenTelemetry shutdown complete")
