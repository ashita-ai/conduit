"""Integration tests for cache service with real Redis.

Note: These tests require a running Redis instance at redis://localhost:6379
      Skip if Redis is unavailable using: pytest -m "not redis"
"""

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

from conduit.cache import CacheConfig, CacheService
from conduit.core.models import QueryFeatures


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def redis_available():
    """Check if Redis is available for testing."""
    try:
        redis = Redis.from_url("redis://localhost:6379", socket_connect_timeout=1)
        await redis.ping()
        await redis.aclose()
        return True
    except (ConnectionError, Exception):
        pytest.skip("Redis not available at localhost:6379")


@pytest.fixture
async def cache_service(redis_available):
    """Create cache service with real Redis connection."""
    config = CacheConfig(
        enabled=True,
        redis_url="redis://localhost:6379/15",  # Use DB 15 for testing
        ttl=60,  # Short TTL for tests
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=10)

    service = CacheService(config)

    # Clear test database before tests
    if service.redis:
        await service.redis.flushdb()

    yield service

    # Cleanup after tests
    if service.redis:
        await service.redis.flushdb()
    await service.close()


@pytest.fixture
def sample_features():
    """Create sample QueryFeatures for testing."""
    return QueryFeatures(
        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
        token_count=50,
        complexity_score=0.5
    )


class TestCacheIntegrationBasic:
    """Basic integration tests with real Redis."""

    async def test_cache_miss_then_hit(self, cache_service, sample_features):
        """Test full cache workflow: miss → set → hit."""
        query = "What is the capital of France?"

        # Initial cache miss
        result = await cache_service.get(query)
        assert result is None
        assert cache_service.stats.misses == 1

        # Store features
        await cache_service.set(query, sample_features)

        # Cache hit
        result = await cache_service.get(query)
        assert result is not None
        assert result.complexity_score == 0.5
        assert len(result.embedding) == 384
        assert cache_service.stats.hits == 1

    async def test_case_insensitive_cache_key(self, cache_service, sample_features):
        """Test cache keys are normalized for case-insensitive hits."""
        # Store with one case
        await cache_service.set("What is 2+2?", sample_features)

        # Retrieve with different case
        result = await cache_service.get("what is 2+2?")
        assert result is not None

    async def test_whitespace_normalization(self, cache_service, sample_features):
        """Test cache keys normalize whitespace."""
        # Store with extra spaces
        await cache_service.set("What  is   2+2?", sample_features)

        # Retrieve with normal spacing
        result = await cache_service.get("What is 2+2?")
        assert result is not None

    async def test_ttl_expiration(self, cache_service, sample_features):
        """Test cached entries expire after TTL."""
        import asyncio

        # Create cache with 1 second TTL
        config = CacheConfig(
            enabled=True,
            redis_url="redis://localhost:6379/15",
            ttl=1,  # 1 second
        )
        short_ttl_cache = CacheService(config)

        try:
            # Store and retrieve immediately
            await short_ttl_cache.set("test query", sample_features)
            result = await short_ttl_cache.get("test query")
            assert result is not None

            # Wait for expiration
            await asyncio.sleep(1.5)

            # Should be expired
            result = await short_ttl_cache.get("test query")
            assert result is None

        finally:
            await short_ttl_cache.close()

    async def test_multiple_queries_cached(self, cache_service, sample_features):
        """Test multiple different queries can be cached."""
        queries = [
            "What is Python?",
            "Explain machine learning",
            "How does Redis work?",
        ]

        # Cache all queries
        for query in queries:
            await cache_service.set(query, sample_features)

        # Verify all are cached
        for query in queries:
            result = await cache_service.get(query)
            assert result is not None

        assert cache_service.stats.hits == 3
        assert cache_service.stats.misses == 0

    async def test_cache_clear(self, cache_service, sample_features):
        """Test clearing cache removes all entries."""
        # Cache multiple queries
        await cache_service.set("query1", sample_features)
        await cache_service.set("query2", sample_features)
        await cache_service.set("query3", sample_features)

        # Verify cached
        assert await cache_service.get("query1") is not None

        # Clear cache
        await cache_service.clear()

        # Verify all cleared
        assert await cache_service.get("query1") is None
        assert await cache_service.get("query2") is None
        assert await cache_service.get("query3") is None


class TestCacheIntegrationPerformance:
    """Performance and stress tests with real Redis."""

    async def test_large_embedding_serialization(self, cache_service):
        """Test caching large embeddings (realistic size)."""
        # Create features with full-size embedding (384 dimensions)
        large_features = QueryFeatures(
            embedding=[0.123456789] * 384,  # Realistic embedding
            token_count=1000,
            complexity_score=0.85
        )

        # Cache and retrieve
        await cache_service.set("large query", large_features)
        result = await cache_service.get("large query")

        assert result is not None
        assert len(result.embedding) == 384
        assert result.token_count == 1000

    async def test_concurrent_cache_operations(self, cache_service, sample_features):
        """Test concurrent cache operations don't interfere."""
        import asyncio

        # Concurrent writes
        queries = [f"query_{i}" for i in range(10)]
        await asyncio.gather(
            *[cache_service.set(query, sample_features) for query in queries]
        )

        # Concurrent reads
        results = await asyncio.gather(
            *[cache_service.get(query) for query in queries]
        )

        # All should succeed
        assert all(r is not None for r in results)
        assert cache_service.stats.hits == 10


class TestCacheIntegrationFailure:
    """Test failure scenarios and recovery."""

    async def test_invalid_redis_url_fails_gracefully(self):
        """Test invalid Redis URL doesn't break initialization."""
        config = CacheConfig(
            enabled=True,
            redis_url="redis://invalid-host:9999",
            timeout=1)

        service = CacheService(config)

        # Operations should return None gracefully
        result = await service.get("test")
        assert result is None

        await service.close()

    async def test_connection_loss_recovery(self, cache_service, sample_features):
        """Test cache recovers after connection loss."""
        # Initial successful operation
        await cache_service.set("query1", sample_features)
        result = await cache_service.get("query1")
        assert result is not None

        # Simulate connection loss by closing Redis
        if cache_service.redis:
            await cache_service.redis.aclose()

        # Operations should fail gracefully
        result = await cache_service.get("query2")
        assert result is None

        # Circuit breaker should trigger after threshold failures
        for _ in range(cache_service.config.circuit_breaker_threshold):
            await cache_service.get("query")

        assert cache_service.circuit_breaker.state == "open"


class TestCacheIntegrationStats:
    """Test statistics tracking with real operations."""

    async def test_hit_rate_calculation(self, cache_service, sample_features):
        """Test hit rate is calculated correctly."""
        # Initial state
        stats = cache_service.get_stats()
        assert stats.hit_rate == 0.0

        # Mix of hits and misses
        await cache_service.set("cached", sample_features)

        await cache_service.get("cached")  # hit
        await cache_service.get("uncached")  # miss
        await cache_service.get("cached")  # hit
        await cache_service.get("another_miss")  # miss

        # 2 hits, 2 misses = 50% hit rate
        stats = cache_service.get_stats()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 50.0

    async def test_error_tracking(self, cache_service):
        """Test errors are tracked in stats."""
        # Close Redis to force errors
        if cache_service.redis:
            await cache_service.redis.aclose()

        # Try operations
        await cache_service.get("query1")
        await cache_service.get("query2")

        stats = cache_service.get_stats()
        assert stats.errors == 2
