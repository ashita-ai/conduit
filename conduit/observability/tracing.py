"""OpenTelemetry tracing utilities for Conduit.

Provides tracer instance and helper decorators for tracing operations.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from opentelemetry import trace

from conduit.core.config import settings

logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


def get_tracer(name: str = "conduit") -> trace.Tracer:
    """Get OpenTelemetry tracer instance.

    Args:
        name: Tracer name (usually module name)

    Returns:
        Tracer instance (no-op if telemetry disabled)
    """
    return trace.get_tracer(name)


def trace_operation(
    operation_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace function execution.

    Args:
        operation_name: Span name (defaults to function name)
        attributes: Additional span attributes

    Example:
        >>> @trace_operation("analyze_query")
        ... async def analyze(query: str) -> QueryFeatures:
        ...     # Function is automatically traced
        ...     return features
    """

    def decorator(func: F) -> F:
        if not settings.otel_enabled or not settings.otel_traces_enabled:
            # Telemetry disabled, return original function
            return func

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(func.__module__)
            span_name = operation_name or func.__name__

            with tracer.start_as_current_span(span_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e))
                    )
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer(func.__module__)
            span_name = operation_name or func.__name__

            with tracer.start_as_current_span(span_name) as span:
                # Add custom attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=str(e))
                    )
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
