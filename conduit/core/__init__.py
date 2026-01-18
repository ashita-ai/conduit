"""Core infrastructure for Conduit routing system."""

from conduit.core.config import (
    Settings,
    load_context_priors,
    load_preference_weights,
    settings,
)
from conduit.core.config_schema import (
    ConduitConfig,
    ConfigValidationError,
    validate_config_dict,
    validate_config_file,
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
from conduit.core.postgres_state_store import (
    PostgresStateStore,
    StateVersionConflictError,
)
from conduit.core.pricing import (
    ModelPricing,
    compute_cost,
    get_all_model_pricing,
    get_model_pricing,
)
from conduit.core.reward_calculation import (
    apply_user_preferences,
    calculate_composite_reward,
    normalize_cost,
    normalize_latency,
    normalize_quality,
    validate_weights,
)
from conduit.core.state_store import (
    BanditState,
    HybridRouterState,
    RouterPhase,
    StateStore,
    StateStoreError,
    deserialize_bandit_matrices,
    list_to_numpy,
    numpy_to_list,
    serialize_bandit_matrices,
)

__all__ = [
    # Config
    "Settings",
    "settings",
    "load_context_priors",
    "load_preference_weights",
    # Config Schema Validation
    "ConduitConfig",
    "ConfigValidationError",
    "validate_config_file",
    "validate_config_dict",
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
    "get_model_pricing",
    "get_all_model_pricing",
    "compute_cost",
    # Reward Calculation
    "calculate_composite_reward",
    "normalize_quality",
    "normalize_cost",
    "normalize_latency",
    "validate_weights",
    "apply_user_preferences",
    # State Persistence
    "StateStore",
    "StateStoreError",
    "StateVersionConflictError",
    "PostgresStateStore",
    "BanditState",
    "HybridRouterState",
    "RouterPhase",
    "numpy_to_list",
    "list_to_numpy",
    "serialize_bandit_matrices",
    "deserialize_bandit_matrices",
]
