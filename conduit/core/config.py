"""Configuration management for Conduit.

This module provides centralized configuration loading from environment
variables with validation and type safety.

Configuration is organized into nested classes for better separation of concerns:
- DatabaseConfig: PostgreSQL connection settings
- RedisConfig: Redis cache and circuit breaker settings
- ProviderConfig: LLM provider API keys
- MLConfig: Embedding and bandit algorithm settings
- APIConfig: REST API settings
- ArbiterConfig: LLM-as-judge evaluation settings
- TelemetryConfig: OpenTelemetry settings
"""

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Nested Configuration Classes
# =============================================================================


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""

    url: str = Field(default="", description="PostgreSQL connection string")
    pool_size: int = Field(default=20, description="Connection pool size", ge=1, le=100)


class RedisConfig(BaseModel):
    """Redis cache and circuit breaker configuration."""

    url: str = Field(default="redis://localhost:6379", description="Redis URL")
    cache_enabled: bool = Field(default=True, description="Enable query feature caching")
    cache_ttl: int = Field(
        default=86400, description="Cache TTL in seconds (24 hours)", ge=60, le=86400
    )
    max_retries: int = Field(default=3, description="Redis operation max retries", ge=0, le=10)
    timeout: int = Field(default=5, description="Redis operation timeout seconds", ge=1, le=30)
    circuit_breaker_threshold: int = Field(
        default=5, description="Failures before opening circuit", ge=1, le=20
    )
    circuit_breaker_timeout: int = Field(
        default=300, description="Circuit breaker timeout seconds (5 min)", ge=60, le=3600
    )


class ProviderConfig(BaseModel):
    """LLM provider API keys and credentials."""

    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    mistral_api_key: str = Field(default="", description="Mistral API key")
    cohere_api_key: str = Field(default="", description="Cohere API key")
    huggingface_api_key: str = Field(default="", description="HuggingFace API key")
    aws_access_key_id: str = Field(default="", description="AWS Access Key (for Bedrock)")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Key (for Bedrock)")
    aws_region: str = Field(default="us-east-1", description="AWS Region (for Bedrock)")


class MLConfig(BaseModel):
    """Machine learning and bandit algorithm configuration."""

    # Embedding settings
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

    # Default models for routing
    default_models: list[str] = Field(
        default=[
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ],
        description="Available models for routing (must match pricing database IDs)",
    )

    # PCA dimensionality reduction
    use_pca: bool = Field(
        default=False, description="Enable PCA dimensionality reduction for embeddings"
    )
    pca_dimensions: int = Field(
        default=64, description="Target embedding dimensions after PCA", ge=8, le=384
    )
    pca_model_path: str = Field(default="models/pca.pkl", description="Path to fitted PCA model")

    # Bandit algorithm parameters
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

    # Hybrid routing configuration
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

    # Execution timeouts
    llm_timeout_default: float = Field(
        default=60.0, description="Default LLM call timeout in seconds", ge=1.0, le=300.0
    )
    llm_timeout_fast: float = Field(
        default=30.0, description="Timeout for fast models (mini) in seconds", ge=1.0, le=300.0
    )
    llm_timeout_premium: float = Field(
        default=90.0, description="Timeout for premium models (opus) in seconds", ge=1.0, le=300.0
    )

    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "MLConfig":
        """Validate reward weights sum to approximately 1.0."""
        total = self.reward_weight_quality + self.reward_weight_cost + self.reward_weight_latency
        if abs(total - 1.0) > 0.01:
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


class APIConfig(BaseModel):
    """REST API configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port", ge=1, le=65535)
    rate_limit: int = Field(default=100, description="Requests per minute per user", ge=1, le=1000)
    key: str = Field(default="", description="API key for authentication (empty = disabled)")
    require_auth: bool = Field(
        default=False, description="Require API key authentication for /v1/* endpoints"
    )
    max_request_size: int = Field(
        default=10_000, description="Maximum request body size in bytes", ge=1000, le=1_000_000
    )


class ArbiterConfig(BaseModel):
    """Arbiter LLM-as-judge evaluation configuration."""

    enabled: bool = Field(default=False, description="Enable Arbiter LLM-as-judge evaluation")
    sample_rate: float = Field(
        default=0.1, description="Fraction of responses to evaluate (0.0-1.0)", ge=0.0, le=1.0
    )
    daily_budget: float = Field(
        default=10.0, description="Maximum daily evaluation budget (USD)", ge=0.0, le=1000.0
    )
    model: str = Field(default="gpt-4o-mini", description="Model for evaluation (cheap recommended)")


class TelemetryConfig(BaseModel):
    """OpenTelemetry instrumentation configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry instrumentation")
    service_name: str = Field(default="conduit-router", description="Service name for telemetry")
    exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318", description="OTLP gRPC endpoint"
    )
    exporter_otlp_headers: str = Field(default="", description="OTLP headers (e.g., 'api-key=xxx')")
    traces_enabled: bool = Field(default=True, description="Enable trace collection")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")


# =============================================================================
# YAML CONFIG LOADING
# =============================================================================


@lru_cache(maxsize=1)
def _load_conduit_yaml() -> dict[str, Any]:
    """Load and cache conduit.yaml configuration.

    Returns:
        Parsed YAML config or empty dict if file not found/invalid.
    """
    config_path = Path("conduit.yaml")
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def get_default_models() -> list[str]:
    """Get default model pool from conduit.yaml.

    Returns:
        List of model IDs for routing, or hardcoded fallback.
    """
    config = _load_conduit_yaml()
    models = config.get("models", {})
    if isinstance(models, dict) and "default" in models:
        default = models["default"]
        if isinstance(default, list):
            return default
    # Fallback if YAML not available
    return ["o4-mini", "gpt-5.1", "claude-sonnet-4-5-20241124", "gemini-2.5-flash"]


def get_arbiter_model() -> str:
    """Get arbiter evaluation model from conduit.yaml.

    Returns:
        Model ID for arbiter evaluation, or hardcoded fallback.
    """
    config = _load_conduit_yaml()
    models = config.get("models", {})
    if isinstance(models, dict) and "arbiter" in models:
        return str(models["arbiter"])
    return "o4-mini"


def get_fallback_model() -> str:
    """Get global fallback model from conduit.yaml.

    Returns:
        Model ID for global fallback, or hardcoded fallback.
    """
    config = _load_conduit_yaml()
    models = config.get("models", {})
    if isinstance(models, dict) and "fallback" in models:
        return str(models["fallback"])
    return "o4-mini"


def get_provider_fallback(provider: str) -> str:
    """Get provider-specific fallback model from conduit.yaml.

    Args:
        provider: Provider name (openai, anthropic, google, meta, mistral)

    Returns:
        Model ID for provider fallback, or default fallback.
    """
    config = _load_conduit_yaml()
    fallbacks = config.get("provider_fallbacks", {})
    if isinstance(fallbacks, dict):
        if provider in fallbacks:
            return str(fallbacks[provider])
        if "default" in fallbacks:
            return str(fallbacks["default"])
    return "o4-mini"


def get_fallback_pricing() -> dict[str, dict[str, float]]:
    """Get fallback pricing from conduit.yaml.

    Returns:
        Dict mapping model_id to {"input": price, "output": price} per 1M tokens.
    """
    config = _load_conduit_yaml()
    pricing = config.get("pricing", {})
    if not isinstance(pricing, dict):
        return {}

    result: dict[str, dict[str, float]] = {}
    for model_id, prices in pricing.items():
        if isinstance(prices, dict) and "input" in prices and "output" in prices:
            result[model_id] = {
                "input": float(prices["input"]),
                "output": float(prices["output"]),
            }
    return result


def get_default_pricing() -> dict[str, float]:
    """Get default pricing for unknown models from conduit.yaml.

    Returns:
        Dict with input/output prices per 1M tokens.
    """
    config = _load_conduit_yaml()
    pricing = config.get("pricing", {})
    if isinstance(pricing, dict) and "_default" in pricing:
        default = pricing["_default"]
        if isinstance(default, dict):
            return {
                "input": float(default.get("input", 1.0)),
                "output": float(default.get("output", 3.0)),
            }
    return {"input": 1.0, "output": 3.0}


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """Application configuration from environment variables.

    Configuration is organized into nested classes for better maintainability:
    - database: PostgreSQL settings
    - redis: Redis cache settings
    - providers: LLM API keys
    - ml: Machine learning and bandit settings
    - api: REST API settings
    - arbiter: LLM-as-judge evaluation
    - telemetry: OpenTelemetry settings

    For backwards compatibility, flat attribute access is still supported
    via property aliases (e.g., settings.database_url -> settings.database.url).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ==========================================================================
    # Nested Configuration (New Structure)
    # ==========================================================================

    # Note: Pydantic-settings doesn't support nested BaseModel directly,
    # so we use flat fields with env vars and construct nested objects
    # via properties for organizational access.

    # Database Configuration (flat for env var support)
    database_url: str = Field(default="", description="PostgreSQL connection string")
    database_pool_size: int = Field(
        default=20, description="Connection pool size", ge=1, le=100
    )

    # Redis Configuration (flat for env var support)
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
        default=300, description="Circuit breaker timeout seconds (5 min)", ge=60, le=3600
    )

    # LLM Provider API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google/Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    mistral_api_key: str = Field(default="", description="Mistral API key")
    cohere_api_key: str = Field(default="", description="Cohere API key")
    aws_access_key_id: str = Field(default="", description="AWS Access Key (for Bedrock)")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Key (for Bedrock)")
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
        default_factory=get_default_models,
        description="Available models for routing (loaded from conduit.yaml)",
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
    api_key: str = Field(default="", description="API key for authentication (empty = disabled)")
    api_require_auth: bool = Field(
        default=False, description="Require API key authentication for /v1/* endpoints"
    )
    api_max_request_size: int = Field(
        default=10_000, description="Maximum request body size in bytes", ge=1000, le=1_000_000
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
        default=0.1, description="Fraction of responses to evaluate (0.0-1.0)", ge=0.0, le=1.0
    )
    arbiter_daily_budget: float = Field(
        default=10.0, description="Maximum daily evaluation budget (USD)", ge=0.0, le=1000.0
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

    # ==========================================================================
    # Validators
    # ==========================================================================

    @model_validator(mode="after")
    def validate_reward_weights_sum(self) -> "Settings":
        """Validate reward weights sum to approximately 1.0."""
        total = (
            self.reward_weight_quality
            + self.reward_weight_cost
            + self.reward_weight_latency
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.3f} "
                f"(quality={self.reward_weight_quality}, "
                f"cost={self.reward_weight_cost}, "
                f"latency={self.reward_weight_latency})"
            )
        return self

    # ==========================================================================
    # Nested Config Properties (Organizational Access)
    # ==========================================================================

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration as nested object."""
        return DatabaseConfig(
            url=self.database_url,
            pool_size=self.database_pool_size,
        )

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration as nested object."""
        return RedisConfig(
            url=self.redis_url,
            cache_enabled=self.redis_cache_enabled,
            cache_ttl=self.redis_cache_ttl,
            max_retries=self.redis_max_retries,
            timeout=self.redis_timeout,
            circuit_breaker_threshold=self.redis_circuit_breaker_threshold,
            circuit_breaker_timeout=self.redis_circuit_breaker_timeout,
        )

    @property
    def providers(self) -> ProviderConfig:
        """Get provider API keys as nested object."""
        return ProviderConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            mistral_api_key=self.mistral_api_key,
            cohere_api_key=self.cohere_api_key,
            huggingface_api_key=self.huggingface_api_key,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region=self.aws_region,
        )

    @property
    def ml(self) -> MLConfig:
        """Get ML configuration as nested object."""
        return MLConfig(
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_api_key=self.embedding_api_key,
            default_models=self.default_models,
            use_pca=self.use_pca,
            pca_dimensions=self.pca_dimensions,
            pca_model_path=self.pca_model_path,
            exploration_rate=self.exploration_rate,
            reward_weight_quality=self.reward_weight_quality,
            reward_weight_cost=self.reward_weight_cost,
            reward_weight_latency=self.reward_weight_latency,
            bandit_window_size=self.bandit_window_size,
            bandit_success_threshold=self.bandit_success_threshold,
            use_hybrid_routing=self.use_hybrid_routing,
            hybrid_switch_threshold=self.hybrid_switch_threshold,
            hybrid_ucb1_c=self.hybrid_ucb1_c,
            hybrid_linucb_alpha=self.hybrid_linucb_alpha,
            llm_timeout_default=self.llm_timeout_default,
            llm_timeout_fast=self.llm_timeout_fast,
            llm_timeout_premium=self.llm_timeout_premium,
        )

    @property
    def api(self) -> APIConfig:
        """Get API configuration as nested object."""
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            rate_limit=self.api_rate_limit,
            key=self.api_key,
            require_auth=self.api_require_auth,
            max_request_size=self.api_max_request_size,
        )

    @property
    def arbiter(self) -> ArbiterConfig:
        """Get Arbiter configuration as nested object."""
        return ArbiterConfig(
            enabled=self.arbiter_enabled,
            sample_rate=self.arbiter_sample_rate,
            daily_budget=self.arbiter_daily_budget,
            model=self.arbiter_model,
        )

    @property
    def telemetry(self) -> TelemetryConfig:
        """Get telemetry configuration as nested object."""
        return TelemetryConfig(
            enabled=self.otel_enabled,
            service_name=self.otel_service_name,
            exporter_otlp_endpoint=self.otel_exporter_otlp_endpoint,
            exporter_otlp_headers=self.otel_exporter_otlp_headers,
            traces_enabled=self.otel_traces_enabled,
            metrics_enabled=self.otel_metrics_enabled,
        )

    # ==========================================================================
    # Utility Properties
    # ==========================================================================

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


def load_preference_weights(optimize_for: Literal["balanced", "quality", "cost", "speed"]) -> dict[str, float]:
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
        with open(config_path, "r") as f:
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
    """Load industry-wide priors for cold start optimization from conduit.yaml.

    Context-specific priors provide different model performance expectations
    based on query type (code, creative, analysis, simple_qa, general).

    Priors are expressed as Beta distribution parameters (alpha, beta):
    - Quality estimate = alpha / (alpha + beta)
    - Prior strength = alpha + beta (higher = stronger confidence)

    Data sources for priors:
    - Code: SWE-Bench (Vellum), LiveCodeBench (Artificial Analysis)
    - Analysis: GPQA Diamond (Vellum), Intelligence Index (AA)
    - Simple QA: MMLU Pro (Artificial Analysis)
    - Creative: Qualitative community assessments
    - General: Weighted average across benchmarks

    Args:
        context: Query context type (code, creative, analysis, simple_qa, general)

    Returns:
        Dictionary mapping model_id to (alpha, beta) tuples.
        Empty dict if config not found or context not defined.

    Example:
        >>> priors = load_context_priors("code")
        >>> print(priors.get("claude-3-5-sonnet-20241022"))
        (8200, 1800)  # 82% quality estimate for code tasks
    """
    config_path = Path("conduit.yaml")

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                return {}
            priors_section = config.get("priors", {})
            if not isinstance(priors_section, dict):
                return {}
            context_priors = priors_section.get(context, {})
            if not isinstance(context_priors, dict):
                return {}

            # Convert list format [alpha, beta] to tuple format (alpha, beta)
            result: dict[str, tuple[float, float]] = {}
            for model_id, params in context_priors.items():
                if isinstance(params, list) and len(params) == 2:
                    result[model_id] = (float(params[0]), float(params[1]))
            return result
    except Exception:
        # Fallback to empty dict if YAML parsing fails
        return {}


# Global settings instance
settings = Settings()
