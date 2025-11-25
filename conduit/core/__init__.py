"""Core infrastructure for Conduit routing system."""

from conduit.core.config import (
    APIConfig,
    ArbiterConfig,
    DatabaseConfig,
    MLConfig,
    ProviderConfig,
    RedisConfig,
    Settings,
    TelemetryConfig,
    load_context_priors,
    load_preference_weights,
    settings,
)
from conduit.core.context_detector import ContextDetector
from conduit.core.database import Database
from conduit.core.exceptions import (
    AnalysisError,
    CircuitBreakerOpenError,
    ConduitError,
    ConfigurationError,
    DatabaseError,
    ExecutionError,
    RateLimitError,
    RoutingError,
    ValidationError,
)
from conduit.core.models import (
    Feedback,
    ModelState,
    Query,
    QueryConstraints,
    QueryFeatures,
    Response,
    RoutingDecision,
    RoutingResult,
    UserPreferences,
)
from conduit.core.pricing import ModelPricing
from conduit.core.reward_calculation import (
    apply_user_preferences,
    calculate_composite_reward,
    normalize_cost,
    normalize_latency,
    normalize_quality,
    validate_weights,
)

__all__ = [
    # Config
    "Settings",
    "settings",
    "load_context_priors",
    "load_preference_weights",
    # Nested Config Classes
    "DatabaseConfig",
    "RedisConfig",
    "ProviderConfig",
    "MLConfig",
    "APIConfig",
    "ArbiterConfig",
    "TelemetryConfig",
    # Context Detection
    "ContextDetector",
    # Database
    "Database",
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
    # Models
    "Query",
    "QueryConstraints",
    "QueryFeatures",
    "UserPreferences",
    "RoutingDecision",
    "Response",
    "Feedback",
    "ModelState",
    "RoutingResult",
    # Pricing
    "ModelPricing",
    # Reward Calculation
    "calculate_composite_reward",
    "normalize_quality",
    "normalize_cost",
    "normalize_latency",
    "validate_weights",
    "apply_user_preferences",
]
