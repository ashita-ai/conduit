"""Structured logging configuration for Conduit.

This module provides structured logging using structlog for production-grade
observability. Logs are output as JSON in production for easy parsing by
log aggregators (Datadog, Splunk, CloudWatch, ELK).

Configuration:
    Set via environment variables or conduit.yaml:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - LOG_FORMAT: json, console (default: json in production, console in dev)
    - ENVIRONMENT: development, production (affects format default)

Usage:
    >>> from conduit.observability.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("model_selected", model="gpt-4o-mini", confidence=0.85)

Standard Events:
    Routing:
        - routing_started: Query routing initiated
        - model_selected: Model chosen by bandit
        - routing_completed: Routing finished with timing

    Algorithm:
        - phase_transition: UCB1 to LinUCB switch
        - feedback_received: Bandit update with reward
        - state_persisted: State saved to database

    Cache:
        - cache_hit: Feature vector retrieved from cache
        - cache_miss: Cache miss, computing features
        - cache_error: Cache operation failed
        - circuit_breaker_opened: Too many failures
        - circuit_breaker_closed: Circuit recovered

    Errors:
        - routing_failed: Routing error occurred
        - execution_failed: LLM execution error
        - persistence_failed: State save/load error
"""

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import Processor

# Module-level flag to track initialization
_configured = False


def configure_logging(
    level: str | None = None,
    log_format: str | None = None,
    is_production: bool | None = None,
) -> None:
    """Configure structured logging for the application.

    Should be called once at application startup. Safe to call multiple times
    (subsequent calls are no-ops).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Default from LOG_LEVEL env.
        log_format: Output format (json, console). Default based on environment.
        is_production: Override production detection. Default from ENVIRONMENT env.

    Example:
        >>> configure_logging(level="DEBUG", log_format="console")
        >>> logger = get_logger(__name__)
        >>> logger.info("app_started", version="1.0.0")
    """
    global _configured
    if _configured:
        return

    # Determine configuration from args, env, or defaults
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    if is_production is None:
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

    if log_format is None:
        log_format = os.getenv("LOG_FORMAT", "json" if is_production else "console")

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
    )

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # Production: JSON output for log aggregators
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Auto-configures logging on first call if not already configured.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        Configured structlog BoundLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("model_selected", model="gpt-4o-mini", confidence=0.85)

        Output (JSON):
        {
            "timestamp": "2024-01-15T10:30:00.123Z",
            "level": "info",
            "logger": "conduit.engines.router",
            "event": "model_selected",
            "model": "gpt-4o-mini",
            "confidence": 0.85
        }
    """
    # Auto-configure on first use
    if not _configured:
        configure_logging()

    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log calls in this context.

    Useful for adding request_id, user_id, or other context that should
    appear in all log entries within a request/task scope.

    Args:
        **kwargs: Context variables to bind.

    Example:
        >>> bind_context(request_id="abc123", user_id="user_456")
        >>> logger.info("processing_query")  # Includes request_id and user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Call at the end of a request/task to prevent context leakage.
    """
    structlog.contextvars.clear_contextvars()


# Standard event names for consistency across the codebase
class LogEvents:
    """Standard event names for structured logging.

    Use these constants for consistent event naming across the codebase.
    This enables reliable log filtering and aggregation.

    Example:
        >>> logger.info(LogEvents.MODEL_SELECTED, model="gpt-4o-mini")
    """

    # Routing events
    ROUTING_STARTED = "routing_started"
    MODEL_SELECTED = "model_selected"
    ROUTING_COMPLETED = "routing_completed"
    ROUTING_FAILED = "routing_failed"

    # Algorithm events
    PHASE_TRANSITION = "phase_transition"
    FEEDBACK_RECEIVED = "feedback_received"
    STATE_PERSISTED = "state_persisted"
    STATE_LOADED = "state_loaded"
    STATE_LOAD_FAILED = "state_load_failed"

    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_ERROR = "cache_error"
    CACHE_CLEARED = "cache_cleared"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    CIRCUIT_BREAKER_HALF_OPEN = "circuit_breaker_half_open"

    # Execution events
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    FALLBACK_USED = "fallback_used"

    # API events
    REQUEST_RECEIVED = "request_received"
    REQUEST_COMPLETED = "request_completed"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"

    # Persistence events
    PERSISTENCE_FAILED = "persistence_failed"
    DATABASE_CONNECTED = "database_connected"
    DATABASE_ERROR = "database_error"

    # Lifecycle events
    SERVER_STARTED = "server_started"
    SERVER_SHUTDOWN = "server_shutdown"
    ROUTER_INITIALIZED = "router_initialized"
