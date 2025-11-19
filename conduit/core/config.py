"""Configuration management for Conduit.

This module provides centralized configuration loading from environment
variables with validation and type safety.
"""

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

    # LLM Provider API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google API key")
    groq_api_key: str = Field(default="", description="Groq API key")

    # ML Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )
    default_models: list[str] = Field(
        default=[
            "gpt-4o-mini",       # OpenAI - cheap, fast, good quality
            "gpt-4o",            # OpenAI - flagship, balanced
            "claude-3.5-sonnet", # Anthropic - current popular
            "claude-opus-4",     # Anthropic - premium quality
        ],
        description="Available models for routing (must match pricing database IDs)",
    )

    # Thompson Sampling Parameters
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

    # API Configuration
    api_rate_limit: int = Field(
        default=100, description="Requests per minute per user", ge=1, le=1000
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port", ge=1, le=65535)

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


# Global settings instance
settings = Settings()
