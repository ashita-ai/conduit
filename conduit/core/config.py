"""Configuration management for Conduit.

This module provides centralized configuration loading from environment
variables with validation and type safety.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Supabase Database
    database_url: str = Field(default="", description="PostgreSQL connection string")
    database_pool_size: int = Field(
        default=20, description="Connection pool size", ge=1, le=100
    )

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_cache_enabled: bool = Field(
        default=True, description="Enable query feature caching"
    )
    redis_cache_ttl: int = Field(
        default=86400, description="Cache TTL in seconds (24 hours)", ge=60, le=86400
    )
    redis_max_retries: int = Field(
        default=3, description="Redis operation max retries", ge=0, le=10
    )
    redis_timeout: int = Field(
        default=5, description="Redis operation timeout seconds", ge=1, le=30
    )
    redis_circuit_breaker_threshold: int = Field(
        default=5, description="Failures before opening circuit", ge=1, le=20
    )
    redis_circuit_breaker_timeout: int = Field(
        default=300,
        description="Circuit breaker timeout seconds (5 min)",
        ge=60,
        le=3600,
    )

    # LLM Provider API Keys (all providers supported by PydanticAI)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    mistral_api_key: str = Field(default="", description="Mistral API key")
    cohere_api_key: str = Field(default="", description="Cohere API key")
    aws_access_key_id: str = Field(
        default="", description="AWS Access Key (for Bedrock)"
    )
    aws_secret_access_key: str = Field(
        default="", description="AWS Secret Key (for Bedrock)"
    )
    aws_region: str = Field(default="us-east-1", description="AWS Region (for Bedrock)")
    huggingface_api_key: str = Field(default="", description="HuggingFace API key")

    # ML Configuration - Embedding Provider
    embedding_provider: str = Field(
        default="huggingface",
        description="Embedding provider type (huggingface, openai, cohere, sentence-transformers)",
    )
    embedding_model: str = Field(
        default="",
        description="Embedding model identifier (provider-specific, empty = use provider default)",
    )
    embedding_api_key: str = Field(
        default="",
        description="API key for embedding provider (if required, defaults to provider-specific env var)",
    )
    default_models: list[str] = Field(
        default=[
            "o4-mini",  # OpenAI - cheap, fast reasoning
            "gpt-5.1",  # OpenAI - latest flagship
            "claude-sonnet-4.5",  # Anthropic - balanced quality
            "claude-opus-4.5",  # Anthropic - premium quality
            "gemini-2.5-pro",  # Google - competitive flagship
        ],
        description="Available models for routing (must match conduit.yaml priors)",
    )

    # Feature Dimension Reduction (PCA)
    use_pca: bool = Field(
        default=False, description="Enable PCA dimensionality reduction for embeddings"
    )
    pca_dimensions: int = Field(
        default=64, description="Target embedding dimensions after PCA", ge=8, le=384
    )
    pca_model_path: str = Field(
        default="models/pca.pkl", description="Path to fitted PCA model"
    )

    # Bandit Algorithm Parameters
    exploration_rate: float = Field(
        default=0.1, description="Exploration rate (epsilon)", ge=0.0, le=1.0
    )
    reward_weight_quality: float = Field(
        default=0.5, description="Quality weight in reward", ge=0.0, le=1.0
    )
    reward_weight_cost: float = Field(
        default=0.3, description="Cost weight in reward", ge=0.0, le=1.0
    )
    reward_weight_latency: float = Field(
        default=0.2, description="Latency weight in reward", ge=0.0, le=1.0
    )
    bandit_window_size: int = Field(
        default=1000,
        description="Sliding window size for non-stationarity (0 = unlimited history)",
        ge=0,
        le=100000,
    )
    bandit_success_threshold: float = Field(
        default=0.85,
        description="Reward threshold for counting a query as 'successful' (statistics only)",
        ge=0.0,
        le=1.0,
    )

    # Hybrid Routing Configuration
    use_hybrid_routing: bool = Field(
        default=False,
        description="Enable hybrid routing (UCB1→LinUCB warm start for 30% faster convergence)",
    )
    hybrid_switch_threshold: int = Field(
        default=2000,
        description="Query count to switch from UCB1 to LinUCB in hybrid mode",
        ge=100,
        le=10000,
    )
    hybrid_ucb1_c: float = Field(
        default=1.5,
        description="UCB1 exploration parameter (c) for hybrid routing phase 1",
        ge=0.1,
        le=10.0,
    )
    hybrid_linucb_alpha: float = Field(
        default=1.0,
        description="LinUCB exploration parameter (alpha) for hybrid routing phase 2",
        ge=0.1,
        le=10.0,
    )

    # API Configuration
    api_rate_limit: int = Field(
        default=100, description="Requests per minute per user", ge=1, le=1000
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port", ge=1, le=65535)
    api_key: str = Field(
        default="", description="API key for authentication (empty = disabled)"
    )
    api_require_auth: bool = Field(
        default=False, description="Require API key authentication for /v1/* endpoints"
    )
    api_max_request_size: int = Field(
        default=10_000,
        description="Maximum request body size in bytes",
        ge=1000,
        le=1_000_000,
    )

    # Execution Timeouts
    llm_timeout_default: float = Field(
        default=60.0,
        description="Default LLM call timeout in seconds",
        ge=1.0,
        le=300.0,
    )
    llm_timeout_fast: float = Field(
        default=30.0,
        description="Timeout for fast models (mini) in seconds",
        ge=1.0,
        le=300.0,
    )
    llm_timeout_premium: float = Field(
        default=90.0,
        description="Timeout for premium models (opus) in seconds",
        ge=1.0,
        le=300.0,
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(
        default="development", description="Environment (development, production)"
    )

    # Arbiter Evaluation Configuration
    arbiter_enabled: bool = Field(
        default=False, description="Enable Arbiter LLM-as-judge evaluation"
    )
    arbiter_sample_rate: float = Field(
        default=0.1,
        description="Fraction of responses to evaluate (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    arbiter_daily_budget: float = Field(
        default=10.0,
        description="Maximum daily evaluation budget (USD)",
        ge=0.0,
        le=1000.0,
    )
    arbiter_model: str = Field(
        default="gpt-4o-mini", description="Model for evaluation (cheap recommended)"
    )

    # OpenTelemetry Configuration
    otel_enabled: bool = Field(
        default=False, description="Enable OpenTelemetry instrumentation"
    )
    otel_service_name: str = Field(
        default="conduit-router", description="Service name for telemetry"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318", description="OTLP gRPC endpoint"
    )
    otel_exporter_otlp_headers: str = Field(
        default="", description="OTLP headers (e.g., 'api-key=xxx')"
    )
    otel_traces_enabled: bool = Field(
        default=True, description="Enable trace collection"
    )
    otel_metrics_enabled: bool = Field(
        default=True, description="Enable metrics collection"
    )

    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "Settings":
        """Validate reward weights sum to approximately 1.0."""
        total = (
            self.reward_weight_quality
            + self.reward_weight_cost
            + self.reward_weight_latency
        )
        if abs(total - 1.0) > 0.01:  # Allow small floating point variance
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.3f} "
                f"(quality={self.reward_weight_quality}, "
                f"cost={self.reward_weight_cost}, "
                f"latency={self.reward_weight_latency})"
            )
        return self

    @property
    def reward_weights(self) -> dict[str, float]:
        """Return reward weights as dictionary."""
        return {
            "quality": self.reward_weight_quality,
            "cost": self.reward_weight_cost,
            "latency": self.reward_weight_latency,
        }

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


def load_preference_weights(
    optimize_for: Literal["balanced", "quality", "cost", "speed"],
) -> dict[str, float]:
    """Load reward weights for user preference preset from conduit.yaml.

    Args:
        optimize_for: User's optimization preference preset

    Returns:
        Dictionary with quality, cost, and latency weights

    Example:
        >>> weights = load_preference_weights("cost")
        >>> print(weights)
        {"quality": 0.4, "cost": 0.5, "latency": 0.1}
    """
    config_path = Path("conduit.yaml")

    # Default weights if config not found
    defaults = {
        "balanced": {"quality": 0.7, "cost": 0.2, "latency": 0.1},
        "quality": {"quality": 0.8, "cost": 0.1, "latency": 0.1},
        "cost": {"quality": 0.4, "cost": 0.5, "latency": 0.1},
        "speed": {"quality": 0.4, "cost": 0.1, "latency": 0.5},
    }

    if not config_path.exists():
        return defaults[optimize_for]

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                return defaults[optimize_for]
            routing = config.get("routing", {})
            if not isinstance(routing, dict):
                return defaults[optimize_for]
            presets = routing.get("presets", {})
            if not isinstance(presets, dict):
                return defaults[optimize_for]
            preset = presets.get(optimize_for)
            if not isinstance(preset, dict):
                return defaults[optimize_for]
            return preset
    except Exception:
        # Fallback to defaults if YAML parsing fails
        return defaults[optimize_for]


def load_context_priors(context: str) -> dict[str, tuple[float, float]]:
    """Load Bayesian priors for Thompson Sampling cold start optimization.

    Context-specific priors encode expected quality levels for different model
    types based on empirical benchmarks and domain expertise. These priors
    help the bandit algorithm make better decisions during cold start.

    Args:
        context: Query context/domain (code, creative, analysis, simple_qa, general)

    Returns:
        Dictionary mapping model_id to (alpha, beta) Beta distribution parameters.
        Alpha represents prior successes, beta represents prior failures.
        Quality = alpha / (alpha + beta).

    Example:
        >>> priors = load_context_priors("code")
        >>> for model, (alpha, beta) in priors.items():
        ...     quality = alpha / (alpha + beta)
        ...     print(f"{model}: {quality:.2%}")
        claude-3-opus: 92%
        gpt-4o: 88%
    """
    config_path = Path("conduit.yaml")

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                return {}

            priors_section = config.get("priors", {})
            if not isinstance(priors_section, dict):
                return {}

            context_priors = priors_section.get(context, {})
            if not isinstance(context_priors, dict):
                return {}

            # Convert to (alpha, beta) tuples with strong prior strength (~10,000 samples)
            result: dict[str, tuple[float, float]] = {}
            for model_id, quality in context_priors.items():
                if isinstance(quality, (int, float)) and 0.0 <= quality <= 1.0:
                    # Strong prior: equivalent to ~10,000 observations
                    prior_strength = 10000
                    alpha = quality * prior_strength
                    beta = (1 - quality) * prior_strength
                    result[model_id] = (alpha, beta)

            return result

    except Exception:
        return {}


def parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate type.

    Args:
        value: String value from environment variable

    Returns:
        Parsed value (int, float, bool, or str)

    Example:
        >>> parse_env_value("1.5")
        1.5
        >>> parse_env_value("true")
        True
        >>> parse_env_value("hello")
        'hello'
    """
    # Try boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def load_routing_config() -> dict[str, Any]:
    """Load routing configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml routing)
        2. Environment variables (ROUTING_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with routing configuration including default_optimization and presets

    Example:
        >>> config = load_routing_config()
        >>> config["default_optimization"]
        'balanced'
        >>> config["presets"]["balanced"]["quality"]
        0.7
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "default_optimization": "balanced",
        "presets": {
            "balanced": {"quality": 0.7, "cost": 0.2, "latency": 0.1},
            "quality": {"quality": 0.8, "cost": 0.1, "latency": 0.1},
            "cost": {"quality": 0.4, "cost": 0.5, "latency": 0.1},
            "speed": {"quality": 0.4, "cost": 0.1, "latency": 0.5},
        },
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    routing_config = config.get("routing", {})
                    if isinstance(routing_config, dict) and routing_config:
                        # Deep merge presets
                        result = defaults.copy()
                        if "default_optimization" in routing_config:
                            result["default_optimization"] = routing_config[
                                "default_optimization"
                            ]
                        if "presets" in routing_config:
                            result["presets"] = {
                                **defaults["presets"],
                                **routing_config["presets"],
                            }
                        return result
        except Exception:
            pass

    # Try environment variables
    env_value = os.getenv("ROUTING_DEFAULT_OPTIMIZATION")
    if env_value is not None:
        defaults["default_optimization"] = env_value

    return defaults


def load_algorithm_config(algorithm: str) -> dict[str, Any]:
    """Load algorithm hyperparameters from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml algorithms.{algorithm})
        2. Environment variables (ALGORITHM_{ALGORITHM}_{KEY})
        3. Hardcoded defaults (embedded below)

    Args:
        algorithm: Algorithm name (linucb, epsilon_greedy, ucb1, thompson_sampling)

    Returns:
        Dict with algorithm hyperparameters

    Example:
        >>> config = load_algorithm_config("linucb")
        >>> config["alpha"]
        1.0
    """
    # Hardcoded defaults per algorithm (ultimate fallback, no imports from defaults.py)
    defaults_map = {
        "linucb": {"alpha": 1.0, "success_threshold": 0.85},
        "epsilon_greedy": {
            "epsilon": 0.1,
            "decay": 1.0,
            "min_epsilon": 0.01,
        },
        "ucb1": {"c": 1.5},
        "thompson_sampling": {"lambda": 1.0},
    }

    defaults = defaults_map.get(algorithm, {})
    config_path = Path("conduit.yaml")

    # Try YAML first
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    algorithms_section = config.get("algorithms", {})
                    if isinstance(algorithms_section, dict):
                        algo_config = algorithms_section.get(algorithm, {})
                        if isinstance(algo_config, dict) and algo_config:
                            # YAML overrides defaults
                            return {**defaults, **algo_config}
        except Exception:
            pass

    # Try environment variables
    env_overrides = {}
    for key in defaults:
        env_key = f"ALGORITHM_{algorithm.upper()}_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                env_overrides[key] = parse_env_value(env_value)
            except ValueError:
                pass

    if env_overrides:
        return {**defaults, **env_overrides}

    # Ultimate fallback
    return defaults


def load_feature_dimensions() -> dict[str, int | float]:
    """Load feature dimension configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml features)
        2. Environment variables (FEATURES_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with feature dimension configuration

    Example:
        >>> config = load_feature_dimensions()
        >>> config["embedding_dim"]
        384
        >>> config["token_count_normalization"]
        1000.0
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "embedding_dim": 384,
        "full_dim": 387,
        "pca_dim": 67,
        "token_count_normalization": 1000.0,
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    features = config.get("features", {})
                    if isinstance(features, dict) and features:
                        return {**defaults, **features}
        except Exception:
            pass

    # Try environment variables
    env_overrides = {}
    for key in defaults:
        env_key = f"FEATURES_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                env_overrides[key] = parse_env_value(env_value)
            except ValueError:
                pass

    if env_overrides:
        return {**defaults, **env_overrides}

    return defaults


def load_quality_estimation_config() -> dict[str, Any]:
    """Load quality estimation configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml quality_estimation)
        2. Environment variables (QUALITY_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with nested quality estimation config

    Example:
        >>> config = load_quality_estimation_config()
        >>> config["base_quality"]
        0.9
        >>> config["penalties"]["short_response"]
        0.15
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "base_quality": 0.9,
        "empty_quality": 0.1,
        "failure_quality": 0.1,
        "min_response_chars": 10,
        "penalties": {
            "short_response": 0.15,
            "repetition": 0.30,
            "no_keyword_overlap": 0.20,
            "low_keyword_overlap": 0.10,
        },
        "thresholds": {
            "keyword_overlap_very_low": 0.05,
            "keyword_overlap_low": 0.15,
            "repetition_min_length": 20,
            "repetition_threshold": 3,
        },
        "bounds": {
            "min_quality": 0.1,
            "max_quality": 0.95,
        },
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    quality_config = config.get("quality_estimation", {})
                    if isinstance(quality_config, dict) and quality_config:
                        # Merge nested dicts
                        result = defaults.copy()
                        for key, value in quality_config.items():
                            if (
                                isinstance(value, dict)
                                and key in result
                                and isinstance(result[key], dict)
                            ):
                                result[key] = {**result[key], **value}
                            else:
                                result[key] = value
                        return result
        except Exception:
            pass

    # Environment variable override for top-level values only
    env_overrides = {}
    simple_keys = [
        "base_quality",
        "empty_quality",
        "failure_quality",
        "min_response_chars",
    ]
    for key in simple_keys:
        env_key = f"QUALITY_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                env_overrides[key] = parse_env_value(env_value)
            except ValueError:
                pass

    if env_overrides:
        return {**defaults, **env_overrides}

    return defaults


def load_feedback_config() -> dict[str, Any]:
    """Load implicit feedback detection configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml feedback)
        2. Environment variables (FEEDBACK_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with nested feedback detection config

    Example:
        >>> config = load_feedback_config()
        >>> config["retry_detection"]["similarity_threshold"]
        0.85
        >>> config["latency_detection"]["high_tolerance_max"]
        10.0
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "retry_detection": {
            "similarity_threshold": 0.85,
            "time_window_seconds": 300.0,
        },
        "latency_detection": {
            "high_tolerance_max": 10.0,
            "medium_tolerance_max": 30.0,
            "high_tolerance_reward": 0.9,
            "medium_tolerance_reward": 0.7,
            "low_tolerance_reward": 0.5,
        },
        "error_detection": {
            "min_response_chars": 10,
            "error_patterns": [
                "I apologize, but I",
                "I cannot",
                "I'm sorry, but",
                "Error:",
                "Exception:",
            ],
        },
        "weights": {
            "explicit": 0.7,
            "implicit": 0.3,
        },
        "rewards": {
            "error": 0.0,
            "retry": 0.3,
        },
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    feedback_config = config.get("feedback", {})
                    if isinstance(feedback_config, dict) and feedback_config:
                        # Deep merge nested dicts
                        result = defaults.copy()
                        for key, value in feedback_config.items():
                            if (
                                isinstance(value, dict)
                                and key in result
                                and isinstance(result[key], dict)
                            ):
                                result[key] = {**result[key], **value}
                            else:
                                result[key] = value
                        return result
        except Exception:
            pass

    return defaults


def load_hybrid_routing_config() -> dict[str, Any]:
    """Load hybrid routing configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml hybrid_routing)
        2. Environment variables (HYBRID_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with hybrid routing parameters

    Example:
        >>> config = load_hybrid_routing_config()
        >>> config["switch_threshold"]
        2000
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "switch_threshold": 2000,
        "ucb1_c": 1.5,
        "linucb_alpha": 1.0,
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    hybrid_config = config.get("hybrid_routing", {})
                    if isinstance(hybrid_config, dict) and hybrid_config:
                        return {**defaults, **hybrid_config}
        except Exception:
            pass

    # Try environment variables
    env_overrides = {}
    for key in defaults:
        env_key = f"HYBRID_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                env_overrides[key] = parse_env_value(env_value)
            except ValueError:
                pass

    if env_overrides:
        return {**defaults, **env_overrides}

    return defaults


def load_arbiter_config() -> dict[str, Any]:
    """Load Arbiter evaluation configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml arbiter)
        2. Environment variables (ARBITER_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with Arbiter configuration

    Example:
        >>> config = load_arbiter_config()
        >>> config["sample_rate"]
        0.1
        >>> config["model"]
        'o4-mini'
    """
    # Hardcoded defaults (ultimate fallback, no imports from defaults.py)
    defaults = {
        "sample_rate": 0.1,
        "daily_budget": 10.0,
        "model": "o4-mini",
        "evaluators": ["semantic", "factuality"],
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    arbiter_config = config.get("arbiter", {})
                    if isinstance(arbiter_config, dict) and arbiter_config:
                        return {**defaults, **arbiter_config}
        except Exception:
            pass

    # Try environment variables
    env_overrides = {}
    simple_keys = ["sample_rate", "daily_budget", "model"]
    for key in simple_keys:
        env_key = f"ARBITER_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                env_overrides[key] = parse_env_value(env_value)
            except ValueError:
                pass

    if env_overrides:
        return {**defaults, **env_overrides}

    return defaults


def load_cache_config() -> dict[str, Any]:
    """Load cache configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml cache)
        2. Environment variables (via Settings class - redis_cache_ttl, etc.)
        3. Hardcoded defaults

    Returns:
        Dict with cache configuration

    Example:
        >>> config = load_cache_config()
        >>> config["ttl"]
        86400
        >>> config["circuit_breaker"]["threshold"]
        5
    """
    defaults = {
        "enabled": settings.redis_cache_enabled,
        "ttl": settings.redis_cache_ttl,
        "max_retries": settings.redis_max_retries,
        "timeout": settings.redis_timeout,
        "circuit_breaker": {
            "threshold": settings.redis_circuit_breaker_threshold,
            "timeout": settings.redis_circuit_breaker_timeout,
        },
        "history_ttl": 300,  # 5 minutes (from history.py default)
    }

    config_path = Path("conduit.yaml")

    # Try YAML (overrides Settings class values)
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    cache_config = config.get("cache", {})
                    if isinstance(cache_config, dict) and cache_config:
                        # Deep merge circuit_breaker
                        result = defaults.copy()
                        for key, value in cache_config.items():
                            if (
                                isinstance(value, dict)
                                and key in result
                                and isinstance(result[key], dict)
                            ):
                                result[key] = {**result[key], **value}
                            else:
                                result[key] = value
                        return result
        except Exception:
            pass

    # Environment variables handled by Settings class
    return defaults


def load_litellm_config() -> dict[str, Any]:
    """Load LiteLLM integration configuration from conduit.yaml.

    3-tier fallback chain:
        1. YAML config (conduit.yaml litellm)
        2. Environment variables (LITELLM_{KEY})
        3. Hardcoded defaults (embedded below)

    Returns:
        Dict with LiteLLM config including model_mappings

    Example:
        >>> config = load_litellm_config()
        >>> config["model_mappings"]["gpt-4o-mini"]
        'o4-mini'
    """
    # Hardcoded defaults (ultimate fallback, no imports)
    defaults = {
        "model_mappings": {
            # OpenAI
            "gpt-4o-mini": "o4-mini",
            "gpt-4o-mini-2024-07-18": "o4-mini",
            "gpt-4o": "gpt-5",
            "gpt-4-turbo": "gpt-5.1",
            "gpt-4": "gpt-5",
            "gpt-3.5-turbo": "gpt-4",
            # Anthropic
            "claude-3-5-sonnet-20241022": "claude-sonnet-4.5",
            "claude-3-5-sonnet-latest": "claude-sonnet-4.5",
            "claude-3-opus-20240229": "claude-opus-4.5",
            "claude-3-opus-latest": "claude-opus-4.5",
            "claude-3-haiku-20240307": "claude-haiku-4.5",
            "claude-3-haiku-latest": "claude-haiku-4.5",
            # Google
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-1.5-pro-latest": "gemini-2.5-pro",
            "gemini-1.5-flash": "gemini-2.0-flash",
            "gemini-1.5-flash-latest": "gemini-2.0-flash",
            "gemini-pro": "gemini-2.5-pro",
        }
    }

    config_path = Path("conduit.yaml")

    # Try YAML
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    litellm_config = config.get("litellm", {})
                    if isinstance(litellm_config, dict) and litellm_config:
                        # Deep merge model_mappings
                        result = defaults.copy()
                        if "model_mappings" in litellm_config:
                            result["model_mappings"] = {
                                **defaults["model_mappings"],
                                **litellm_config["model_mappings"],
                            }
                        return result
        except Exception:
            pass

    return defaults


def load_pricing_config() -> dict[str, dict[str, float]]:
    """Load model pricing from pricing.yaml.

    3-tier fallback chain:
        1. pricing.yaml
        2. Environment variables (PRICING_{MODEL_ID}_INPUT/OUTPUT)
        3. Hardcoded fallback pricing

    Returns:
        Dict mapping model_id to {"input": cost, "output": cost}

    Example:
        >>> pricing = load_pricing_config()
        >>> pricing["o4-mini"]
        {'input': 1.10, 'output': 4.40}
    """
    # Hardcoded fallback (from get_fallback_pricing)
    defaults = {
        # OpenAI
        "o4-mini": {"input": 1.10, "output": 4.40},
        "gpt-5.1": {"input": 2.00, "output": 8.00},
        "gpt-5": {"input": 2.00, "output": 8.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        # Anthropic
        "claude-opus-4-5-20241124": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4-5-20241124": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20241124": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        # Google
        "gemini-3.0-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        # Meta/Llama
        "llama-4-maverick": {"input": 0.20, "output": 0.20},
        "llama-4-scout": {"input": 0.10, "output": 0.10},
        # Mistral
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-small-latest": {"input": 0.20, "output": 0.60},
    }

    pricing_path = Path("pricing.yaml")

    # Try pricing.yaml
    if pricing_path.exists():
        try:
            with open(pricing_path) as f:
                config = yaml.safe_load(f)
                if isinstance(config, dict):
                    pricing_data = config.get("pricing", {})
                    if isinstance(pricing_data, dict) and pricing_data:
                        return pricing_data
        except Exception:
            pass

    return defaults


def get_fallback_pricing() -> dict[str, dict[str, float]]:
    """Get fallback pricing for models when database pricing is unavailable.

    Now loads from pricing.yaml with 3-tier fallback:
        1. pricing.yaml
        2. Environment variables
        3. Hardcoded defaults (embedded in load_pricing_config)

    Returns:
        Dict mapping model_id to input/output costs per million tokens.

    Example:
        >>> pricing = get_fallback_pricing()
        >>> pricing["o4-mini"]
        {'input': 1.10, 'output': 4.40}
    """
    return load_pricing_config()


def get_default_pricing() -> dict[str, float]:
    """Get default pricing for unknown models.

    Returns:
        Dict with input/output costs per million tokens for unknown models.
    """
    # Conservative default: assume mid-range pricing
    return {"input": 1.00, "output": 4.00}


# Global settings instance
settings = Settings()


def get_arbiter_model() -> str:
    """Get the model ID to use for arbiter evaluation.

    Returns the arbiter_model from settings, falling back to 'o4-mini'.

    Returns:
        Model ID string for evaluation.
    """
    return settings.arbiter_model or "o4-mini"


def get_fallback_model() -> str:
    """Get the fallback model ID for routing when no specific model is chosen.

    Returns the first model from default_models, or 'o4-mini' as ultimate fallback.

    Returns:
        Model ID string.
    """
    if settings.default_models:
        return settings.default_models[0]
    return "o4-mini"
