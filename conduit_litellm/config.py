"""Configuration options for Conduit LiteLLM plugin."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ConduitLiteLLMConfig:
    """Configuration for Conduit routing strategy in LiteLLM.

    This configuration controls how Conduit's ML-based routing integrates
    with LiteLLM's router.

    Attributes:
        embedding_model: Sentence transformer model for query analysis.
        cache_enabled: Enable Redis caching for query feature vectors.
        redis_url: Redis connection URL (if cache_enabled=True).
        cache_ttl: Cache TTL in seconds (default: 3600).

    Note:
        Router always uses hybrid routing (UCB1→LinUCB warm start) by default.
        The use_hybrid parameter has been removed as it's no longer configurable.

    Example:
        >>> config = ConduitLiteLLMConfig(
        ...     cache_enabled=True,
        ...     redis_url="redis://localhost:6379"
        ... )
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    cache_enabled: bool = False
    redis_url: str | None = None
    cache_ttl: int = 3600

    def to_conduit_config(self) -> dict[str, Any]:
        """Convert to Conduit configuration dictionary.

        Returns:
            Dictionary of configuration options for Conduit.
        """
        config = {
            "embedding_model": self.embedding_model,
            "cache_enabled": self.cache_enabled,
        }

        if self.redis_url:
            config["redis_url"] = self.redis_url
            config["cache_ttl"] = self.cache_ttl

        return config


# Default configuration
DEFAULT_CONFIG = ConduitLiteLLMConfig()
