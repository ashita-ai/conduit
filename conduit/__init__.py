"""Conduit - ML-powered LLM routing system.

Conduit learns optimal model selection based on cost, latency, and quality
trade-offs using contextual bandits with Thompson Sampling.

Basic usage:
    >>> from conduit import Router, Query
    >>> router = Router()
    >>> query = Query(text="What is 2+2?")
    >>> decision = await router.route(query)
    >>> print(decision.selected_model)
    "o4-mini"
"""

import os

from dotenv import load_dotenv

load_dotenv()

from conduit.core import (
    AnalysisError,
    CircuitBreakerOpenError,
    ConduitError,
    ConfigurationError,
    DatabaseError,
    ExecutionError,
    Query,
    QueryConstraints,
    RateLimitError,
    Response,
    RoutingError,
    RoutingResult,
    ValidationError,
    settings,
)
from conduit.engines import Router

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "Router",
    # Models
    "Query",
    "QueryConstraints",
    "Response",
    "RoutingResult",
    # Configuration
    "settings",
    # Exceptions
    "ConduitError",
    "AnalysisError",
    "RoutingError",
    "ExecutionError",
    "DatabaseError",
    "ValidationError",
    "ConfigurationError",
    "CircuitBreakerOpenError",
    "RateLimitError",
    # Version
    "__version__",
]
