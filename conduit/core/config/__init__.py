"""Configuration management for Conduit.

This package provides centralized configuration loading from environment
variables with validation and type safety.

All exports are re-exported here for backward compatibility with existing
imports from `conduit.core.config`.
"""

# Settings class and global instance
# YAML configuration loaders
from conduit.core.config.loaders import (
    load_algorithm_config,
    load_arbiter_config,
    load_cache_config,
    load_context_priors,
    load_cost_config,
    load_feature_dimensions,
    load_feedback_config,
    load_hybrid_routing_config,
    load_litellm_config,
    load_preference_weights,
    load_quality_estimation_config,
    load_routing_config,
)

# Model detection utilities
from conduit.core.config.models import (
    detect_available_models,
    get_arbiter_model,
    get_fallback_model,
    get_models_with_fallback,
)
from conduit.core.config.settings import Settings, settings

# Utility functions
from conduit.core.config.utils import (
    load_default_models,
    load_embeddings_config,
    parse_env_value,
)

__all__ = [
    # Settings
    "Settings",
    "settings",
    # Loaders
    "load_algorithm_config",
    "load_arbiter_config",
    "load_cache_config",
    "load_context_priors",
    "load_cost_config",
    "load_feature_dimensions",
    "load_feedback_config",
    "load_hybrid_routing_config",
    "load_litellm_config",
    "load_preference_weights",
    "load_quality_estimation_config",
    "load_routing_config",
    # Utils
    "load_default_models",
    "load_embeddings_config",
    "parse_env_value",
    # Models
    "detect_available_models",
    "get_arbiter_model",
    "get_fallback_model",
    "get_models_with_fallback",
]
