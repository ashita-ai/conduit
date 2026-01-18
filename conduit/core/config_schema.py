"""Pydantic schema models for conduit.yaml configuration validation.

This module provides strict validation for the YAML configuration file,
catching errors early with clear, actionable error messages.

Usage:
    >>> from conduit.core.config_schema import validate_config_file
    >>>
    >>> # Validate and get typed config
    >>> config = validate_config_file("conduit.yaml")
    >>> print(config.routing.default_optimization)
    'balanced'
    >>>
    >>> # Validation errors raise ConfigValidationError
    >>> try:
    ...     validate_config_file("invalid.yaml")
    ... except ConfigValidationError as e:
    ...     print(e.message)  # Human-readable error

Issue #219: Add validation for conduit.yaml configuration file
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigValidationError(Exception):
    """Raised when configuration validation fails.

    Attributes:
        message: Human-readable error description
        field: Config field that failed validation (if applicable)
        value: The invalid value (if applicable)
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


# Valid algorithm names
VALID_ALGORITHMS = frozenset(
    {
        "thompson_sampling",
        "linucb",
        "ucb1",
        "epsilon_greedy",
        "contextual_thompson_sampling",
        "dueling",
        "random",
        "always_best",
        "always_cheapest",
        "oracle",
        "hybrid_thompson_linucb",
        "hybrid_ucb1_linucb",
    }
)

# Valid embedding providers
VALID_EMBEDDING_PROVIDERS = frozenset(
    {
        "auto",
        "openai",
        "cohere",
        "fastembed",
        "sentence-transformers",
        "huggingface",
    }
)

# Valid optimization presets
VALID_OPTIMIZATION_PRESETS = frozenset({"balanced", "quality", "cost", "speed"})

# Valid context types for priors
VALID_CONTEXTS = frozenset({"code", "creative", "analysis", "simple_qa", "general"})


class PresetWeights(BaseModel):
    """Reward weight preset for routing optimization."""

    quality: float = Field(..., ge=0.0, le=1.0, description="Quality weight")
    cost: float = Field(..., ge=0.0, le=1.0, description="Cost weight")
    latency: float = Field(..., ge=0.0, le=1.0, description="Latency weight")

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "PresetWeights":
        """Validate that weights sum to approximately 1.0."""
        total = self.quality + self.cost + self.latency
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Preset weights must sum to 1.0, got {total:.3f} "
                f"(quality={self.quality}, cost={self.cost}, latency={self.latency})"
            )
        return self


class RoutingConfig(BaseModel):
    """Routing section configuration."""

    default_optimization: Literal["balanced", "quality", "cost", "speed"] = Field(
        default="balanced",
        description="Default optimization strategy",
    )
    presets: dict[str, PresetWeights] = Field(
        default_factory=dict,
        description="Reward weight presets",
    )


class LinUCBConfig(BaseModel):
    """LinUCB algorithm hyperparameters."""

    alpha: float = Field(default=1.0, ge=0.1, le=10.0, description="Exploration param")
    success_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Success threshold"
    )


class EpsilonGreedyConfig(BaseModel):
    """Epsilon-greedy algorithm hyperparameters."""

    epsilon: float = Field(default=0.1, ge=0.0, le=1.0, description="Exploration rate")
    decay: float = Field(default=1.0, ge=0.0, le=1.0, description="Decay rate")
    min_epsilon: float = Field(default=0.01, ge=0.0, le=1.0, description="Min epsilon")


class UCB1Config(BaseModel):
    """UCB1 algorithm hyperparameters."""

    c: float = Field(default=1.5, ge=0.1, le=10.0, description="Confidence multiplier")


class ThompsonSamplingConfig(BaseModel):
    """Thompson Sampling algorithm hyperparameters."""

    # Using 'lambda_' because 'lambda' is reserved in Python
    lambda_: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        alias="lambda",
        description="Regularization parameter",
    )


class AlgorithmsConfig(BaseModel):
    """Algorithm hyperparameters section."""

    linucb: LinUCBConfig = Field(default_factory=LinUCBConfig)
    epsilon_greedy: EpsilonGreedyConfig = Field(default_factory=EpsilonGreedyConfig)
    ucb1: UCB1Config = Field(default_factory=UCB1Config)
    thompson_sampling: ThompsonSamplingConfig = Field(
        default_factory=ThompsonSamplingConfig
    )


class PCAConfig(BaseModel):
    """PCA dimensionality reduction configuration."""

    enabled: bool = Field(default=False, description="Enable PCA compression")
    components: int = Field(
        default=128, ge=8, le=2048, description="Number of PCA components"
    )
    auto_retrain: bool = Field(default=True, description="Auto-retrain PCA")
    retrain_threshold: int = Field(
        default=150, ge=10, le=10000, description="Min queries for retraining"
    )


class EmbeddingsConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = Field(default="auto", description="Embedding provider type")
    model: str | None = Field(default=None, description="Provider-specific model")
    pca: PCAConfig = Field(default_factory=PCAConfig)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate embedding provider is supported."""
        if v not in VALID_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Invalid embedding provider: {v!r}. "
                f"Must be one of: {sorted(VALID_EMBEDDING_PROVIDERS)}"
            )
        return v


class QualityPenalties(BaseModel):
    """Quality estimation penalties."""

    short_response: float = Field(default=0.15, ge=0.0, le=1.0)
    repetition: float = Field(default=0.30, ge=0.0, le=1.0)
    no_keyword_overlap: float = Field(default=0.20, ge=0.0, le=1.0)
    low_keyword_overlap: float = Field(default=0.10, ge=0.0, le=1.0)


class QualityThresholds(BaseModel):
    """Quality estimation detection thresholds."""

    keyword_overlap_very_low: float = Field(default=0.05, ge=0.0, le=1.0)
    keyword_overlap_low: float = Field(default=0.15, ge=0.0, le=1.0)
    repetition_min_length: int = Field(default=20, ge=1, le=1000)
    repetition_threshold: int = Field(default=3, ge=1, le=100)


class QualityBounds(BaseModel):
    """Quality score bounds."""

    min_quality: float = Field(default=0.1, ge=0.0, le=1.0)
    max_quality: float = Field(default=0.95, ge=0.0, le=1.0)


class QualityEstimationConfig(BaseModel):
    """Quality estimation configuration."""

    base_quality: float = Field(default=0.9, ge=0.0, le=1.0)
    empty_quality: float = Field(default=0.1, ge=0.0, le=1.0)
    failure_quality: float = Field(default=0.1, ge=0.0, le=1.0)
    min_response_chars: int = Field(default=10, ge=1, le=10000)
    penalties: QualityPenalties = Field(default_factory=QualityPenalties)
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    bounds: QualityBounds = Field(default_factory=QualityBounds)


class RetryDetectionConfig(BaseModel):
    """Retry detection configuration."""

    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    time_window_seconds: float = Field(default=300.0, ge=1.0, le=3600.0)


class LatencyDetectionConfig(BaseModel):
    """Latency detection configuration."""

    high_tolerance_max: float = Field(default=10.0, ge=0.1, le=300.0)
    medium_tolerance_max: float = Field(default=30.0, ge=0.1, le=300.0)
    high_tolerance_reward: float = Field(default=0.9, ge=0.0, le=1.0)
    medium_tolerance_reward: float = Field(default=0.7, ge=0.0, le=1.0)
    low_tolerance_reward: float = Field(default=0.5, ge=0.0, le=1.0)


class ErrorDetectionConfig(BaseModel):
    """Error detection configuration."""

    min_response_chars: int = Field(default=10, ge=1, le=10000)
    error_patterns: list[str] = Field(default_factory=list)


class FeedbackWeights(BaseModel):
    """Feedback weighting configuration."""

    explicit: float = Field(default=0.7, ge=0.0, le=1.0)
    implicit: float = Field(default=0.3, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "FeedbackWeights":
        """Validate weights sum to 1.0."""
        total = self.explicit + self.implicit
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Feedback weights must sum to 1.0, got {total:.3f} "
                f"(explicit={self.explicit}, implicit={self.implicit})"
            )
        return self


class FeedbackRewards(BaseModel):
    """Implicit feedback reward values."""

    error: float = Field(default=0.0, ge=0.0, le=1.0)
    retry: float = Field(default=0.3, ge=0.0, le=1.0)


class FeedbackConfig(BaseModel):
    """Feedback detection configuration."""

    retry_detection: RetryDetectionConfig = Field(default_factory=RetryDetectionConfig)
    latency_detection: LatencyDetectionConfig = Field(
        default_factory=LatencyDetectionConfig
    )
    error_detection: ErrorDetectionConfig = Field(default_factory=ErrorDetectionConfig)
    weights: FeedbackWeights = Field(default_factory=FeedbackWeights)
    rewards: FeedbackRewards = Field(default_factory=FeedbackRewards)


class HybridRoutingConfig(BaseModel):
    """Hybrid routing configuration."""

    switch_threshold: int = Field(default=2000, ge=100, le=100000)
    ucb1_c: float = Field(default=1.5, ge=0.1, le=10.0)
    linucb_alpha: float = Field(default=1.0, ge=0.1, le=10.0)


class ArbiterConfig(BaseModel):
    """Arbiter LLM-as-judge configuration."""

    sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    daily_budget: float = Field(default=10.0, ge=0.0, le=1000.0)
    model: str = Field(default="o4-mini")
    evaluators: list[str] = Field(default_factory=lambda: ["semantic", "factuality"])


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    threshold: int = Field(default=5, ge=1, le=100)
    timeout: int = Field(default=300, ge=1, le=3600)


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = Field(default=True)
    ttl: int = Field(default=86400, ge=60, le=604800)  # 1 min to 1 week
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout: int = Field(default=5, ge=1, le=60)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    history_ttl: int = Field(default=300, ge=60, le=3600)


class LiteLLMConfig(BaseModel):
    """LiteLLM integration configuration."""

    model_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="Model ID mappings from LiteLLM to Conduit format",
    )


class ConduitConfig(BaseModel):
    """Root configuration model for conduit.yaml.

    Validates the entire configuration structure and provides
    typed access to all configuration sections.
    """

    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    priors: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Context-specific model quality priors",
    )
    algorithms: AlgorithmsConfig = Field(default_factory=AlgorithmsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    quality_estimation: QualityEstimationConfig = Field(
        default_factory=QualityEstimationConfig
    )
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    hybrid_routing: HybridRoutingConfig = Field(default_factory=HybridRoutingConfig)
    arbiter: ArbiterConfig = Field(default_factory=ArbiterConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    litellm: LiteLLMConfig = Field(default_factory=LiteLLMConfig)

    @field_validator("priors")
    @classmethod
    def validate_priors(
        cls, v: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Validate priors structure and quality scores."""
        for context, model_priors in v.items():
            if context not in VALID_CONTEXTS:
                raise ValueError(
                    f"Invalid context in priors: {context!r}. "
                    f"Must be one of: {sorted(VALID_CONTEXTS)}"
                )
            for model_id, quality in model_priors.items():
                if not isinstance(quality, (int, float)):
                    raise ValueError(
                        f"Prior quality for {model_id!r} in {context!r} "
                        f"must be a number, got {type(quality).__name__}"
                    )
                if not 0.0 <= quality <= 1.0:
                    raise ValueError(
                        f"Prior quality for {model_id!r} in {context!r} "
                        f"must be between 0.0 and 1.0, got {quality}"
                    )
        return v


def validate_config_file(path: str | Path) -> ConduitConfig:
    """Load and validate a conduit.yaml configuration file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Validated ConduitConfig instance

    Raises:
        ConfigValidationError: If the file cannot be read or validation fails
        FileNotFoundError: If the file does not exist

    Example:
        >>> config = validate_config_file("conduit.yaml")
        >>> print(config.routing.default_optimization)
        'balanced'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            f"Invalid YAML syntax in {path}: {e}",
            field=None,
            value=None,
        ) from e

    if raw_config is None:
        # Empty file is valid, return defaults
        return ConduitConfig()

    if not isinstance(raw_config, dict):
        raise ConfigValidationError(
            f"Configuration must be a YAML mapping, got {type(raw_config).__name__}",
            field=None,
            value=raw_config,
        )

    try:
        return ConduitConfig.model_validate(raw_config)
    except Exception as e:
        # Convert Pydantic validation errors to ConfigValidationError
        raise ConfigValidationError(
            f"Configuration validation failed: {e}",
            field=None,
            value=raw_config,
        ) from e


def validate_config_dict(config: dict[str, Any]) -> ConduitConfig:
    """Validate a configuration dictionary.

    Args:
        config: Configuration dictionary (typically from yaml.safe_load)

    Returns:
        Validated ConduitConfig instance

    Raises:
        ConfigValidationError: If validation fails

    Example:
        >>> config = validate_config_dict({"routing": {"default_optimization": "cost"}})
        >>> print(config.routing.default_optimization)
        'cost'
    """
    try:
        return ConduitConfig.model_validate(config)
    except Exception as e:
        raise ConfigValidationError(
            f"Configuration validation failed: {e}",
            field=None,
            value=config,
        ) from e
