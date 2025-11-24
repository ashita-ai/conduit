"""Configuration management for Conduit.

This module provides centralized configuration loading from environment
variables with validation and type safety.
"""

import yaml
from pathlib import Path
from typing import Literal

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
        default=300, description="Circuit breaker timeout seconds (5 min)", ge=60, le=3600
    )

    # LLM Provider API Keys (all providers supported by PydanticAI)
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

    # ML Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )
    default_models: list[str] = Field(
        default=[
            "gpt-4o-mini",                  # OpenAI - cheap, fast, good quality
            "gpt-4o",                       # OpenAI - flagship, balanced
            "claude-3-5-sonnet-20241022",   # Anthropic - current popular
            "claude-3-opus-20240229",       # Anthropic - premium quality
        ],
        description="Available models for routing (must match pricing database IDs)",
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


# Global settings instance
settings = Settings()
