"""Cache configuration and statistics models."""

from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    """Configuration for Redis caching layer.

    Attributes:
        enabled: Whether caching is enabled
        redis_url: Redis connection URL
        ttl: Time-to-live for cache entries in seconds (default 24 hours)
        max_retries: Maximum retry attempts for Redis operations
        timeout: Operation timeout in seconds
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Seconds before attempting recovery
    """

    enabled: bool = Field(default=True, description="Enable/disable caching")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    ttl: int = Field(default=86400, description="Cache TTL in seconds (24 hours)")
    max_retries: int = Field(default=3, description="Max Redis operation retries")
    timeout: int = Field(default=5, description="Redis operation timeout (seconds)")
    circuit_breaker_threshold: int = Field(
        default=5, description="Failures before opening circuit"
    )
    circuit_breaker_timeout: int = Field(
        default=300, description="Circuit breaker timeout (seconds)"
    )


class CacheStats(BaseModel):
    """Cache performance statistics.

    Attributes:
        hits: Number of successful cache hits
        misses: Number of cache misses
        errors: Number of cache operation errors
        hit_rate: Cache hit rate (hits / total requests)
        circuit_state: Current circuit breaker state
    """

    hits: int = Field(default=0, description="Cache hits")
    misses: int = Field(default=0, description="Cache misses")
    errors: int = Field(default=0, description="Cache errors")
    hit_rate: float = Field(default=0.0, description="Hit rate percentage")
    circuit_state: str = Field(
        default="closed", description="Circuit breaker state (closed/open/half_open)"
    )

    def update_hit_rate(self) -> None:
        """Recalculate hit rate based on current stats."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0
