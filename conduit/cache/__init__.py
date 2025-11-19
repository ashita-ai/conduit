"""Caching layer for query feature extraction optimization.

This module provides Redis-based caching for QueryFeatures to optimize
the expensive embedding computation in query analysis. The cache is designed
with fail-safe patterns to ensure the system works perfectly when Redis
is unavailable.

Key Features:
    - MessagePack serialization for compact storage
    - Circuit breaker for automatic failure recovery
    - Graceful degradation (works without Redis)
    - 24-hour TTL with LRU eviction
    - Performance statistics tracking

Usage:
    >>> from conduit.cache import CacheService, CacheConfig
    >>>
    >>> config = CacheConfig(redis_url="redis://localhost:6379")
    >>> cache = CacheService(config)
    >>>
    >>> # Try to get cached features
    >>> features = await cache.get("What is 2+2?")
    >>> if features is None:
    >>>     # Cache miss, compute features
    >>>     features = await analyzer.analyze("What is 2+2?")
    >>>     await cache.set("What is 2+2?", features)
"""

from conduit.cache.models import CacheConfig, CacheStats
from conduit.cache.service import CacheCircuitBreaker, CacheService

__all__ = [
    "CacheService",
    "CacheConfig",
    "CacheStats",
    "CacheCircuitBreaker",
]
