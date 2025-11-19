"""Unit tests for cache service with mocked Redis."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from redis.exceptions import ConnectionError, TimeoutError

from conduit.cache import CacheConfig, CacheService
from conduit.core.models import QueryFeatures


@pytest.fixture
def cache_config():
    """Create cache configuration for testing."""
    return CacheConfig(
        enabled=True,
        redis_url="redis://localhost:6379",
        ttl=86400,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=300,
    )


@pytest.fixture
def sample_features():
    """Create sample QueryFeatures for testing."""
    return QueryFeatures(
        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
        token_count=50,
        complexity_score=0.5,
        domain="code",
        domain_confidence=0.8,
    )


@pytest.fixture
async def cache_service(cache_config):
    """Create cache service with mocked Redis."""
    with patch("conduit.cache.service.Redis") as mock_redis:
        # Mock Redis client
        mock_client = AsyncMock()
        mock_redis.from_url.return_value = mock_client

        service = CacheService(cache_config)
        service.redis = mock_client

        yield service

        await service.close()


class TestCacheService:
    """Test suite for CacheService."""

    async def test_cache_miss(self, cache_service):
        """Test cache miss returns None."""
        cache_service.redis.get = AsyncMock(return_value=None)

        result = await cache_service.get("test query")

        assert result is None
        assert cache_service.stats.misses == 1
        assert cache_service.stats.hits == 0

    async def test_cache_hit(self, cache_service, sample_features):
        """Test cache hit returns deserialized features."""
        import msgpack

        # Mock Redis to return serialized features
        serialized = msgpack.packb(sample_features.model_dump(), use_bin_type=True)
        cache_service.redis.get = AsyncMock(return_value=serialized)

        result = await cache_service.get("test query")

        assert result is not None
        assert result.complexity_score == 0.5
        assert result.domain == "code"
        assert len(result.embedding) == 384
        assert cache_service.stats.hits == 1
        assert cache_service.stats.misses == 0

    async def test_cache_set(self, cache_service, sample_features):
        """Test cache set serializes and stores features."""
        cache_service.redis.set = AsyncMock()

        await cache_service.set("test query", sample_features)

        # Verify Redis set was called
        cache_service.redis.set.assert_called_once()
        call_args = cache_service.redis.set.call_args

        # Verify key format
        key = call_args[0][0]
        assert key.startswith("conduit:query:")

        # Verify TTL was set
        assert call_args[1]["ex"] == 86400

    async def test_cache_key_normalization(self, cache_service):
        """Test cache key normalization for consistent hits."""
        key1 = cache_service._generate_key("What is 2+2?")
        key2 = cache_service._generate_key("what is 2+2?")  # Different case
        key3 = cache_service._generate_key("What   is  2+2?")  # Extra spaces

        # Should generate same key for semantically identical queries
        assert key1 == key2 == key3

    async def test_connection_error_returns_none(self, cache_service):
        """Test connection error is handled gracefully."""
        cache_service.redis.get = AsyncMock(side_effect=ConnectionError("Connection refused"))

        result = await cache_service.get("test query")

        assert result is None
        assert cache_service.stats.errors == 1

    async def test_timeout_error_returns_none(self, cache_service):
        """Test timeout error is handled gracefully."""
        cache_service.redis.get = AsyncMock(side_effect=TimeoutError("Operation timed out"))

        result = await cache_service.get("test query")

        assert result is None
        assert cache_service.stats.errors == 1

    async def test_disabled_cache_returns_none(self):
        """Test disabled cache always returns None."""
        config = CacheConfig(enabled=False)
        service = CacheService(config)

        result = await service.get("test query")

        assert result is None
        assert service.redis is None

    async def test_cache_stats_tracking(self, cache_service, sample_features):
        """Test cache statistics are tracked correctly."""
        import msgpack

        # Mock Redis for hits and misses with valid QueryFeatures
        serialized = msgpack.packb(sample_features.model_dump(), use_bin_type=True)
        cache_service.redis.get = AsyncMock(side_effect=[
            serialized,  # hit
            None,        # miss
            serialized,  # hit
            None,        # miss
        ])

        # Execute operations
        await cache_service.get("query1")  # hit
        await cache_service.get("query2")  # miss
        await cache_service.get("query3")  # hit
        await cache_service.get("query4")  # miss

        stats = cache_service.get_stats()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 50.0

    async def test_clear_cache(self, cache_service):
        """Test cache clear removes all entries."""
        # Mock Redis scan and delete
        cache_service.redis.scan = AsyncMock(side_effect=[
            (0, [b"conduit:query:key1", b"conduit:query:key2"]),  # First batch
        ])
        cache_service.redis.delete = AsyncMock()

        await cache_service.clear()

        # Verify scan was called with correct pattern
        cache_service.redis.scan.assert_called()

        # Verify delete was called
        cache_service.redis.delete.assert_called_once()


class TestCircuitBreaker:
    """Test suite for CacheCircuitBreaker."""

    def test_initial_state_closed(self, cache_service):
        """Test circuit breaker starts in closed state."""
        assert cache_service.circuit_breaker.state == "closed"
        assert cache_service.circuit_breaker.can_attempt()

    def test_opens_after_threshold_failures(self, cache_service):
        """Test circuit breaker opens after threshold failures."""
        breaker = cache_service.circuit_breaker
        threshold = cache_service.config.circuit_breaker_threshold

        # Trigger failures up to threshold
        for _ in range(threshold):
            breaker.on_failure()

        assert breaker.state == "open"
        assert not breaker.can_attempt()

    def test_half_open_after_timeout(self, cache_service):
        """Test circuit breaker transitions to half-open after timeout."""
        import time

        breaker = cache_service.circuit_breaker
        breaker.timeout = 0.1  # Short timeout for testing

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            breaker.on_failure()

        assert breaker.state == "open"

        # Wait for timeout
        time.sleep(0.2)

        # Should transition to half-open
        assert breaker.can_attempt()
        assert breaker.state == "half_open"

    def test_closes_on_success_in_half_open(self, cache_service):
        """Test circuit breaker closes on successful half-open attempt."""
        breaker = cache_service.circuit_breaker

        # Force half-open state
        breaker.state = "half_open"
        breaker.on_success()

        assert breaker.state == "closed"
        assert breaker.failure_count == 0

    def test_reopens_on_failure_in_half_open(self, cache_service):
        """Test circuit breaker reopens on failed half-open attempt."""
        breaker = cache_service.circuit_breaker

        # Force half-open state
        breaker.state = "half_open"

        # Trigger enough failures to open again
        for _ in range(breaker.failure_threshold):
            breaker.on_failure()

        assert breaker.state == "open"

    def test_manual_reset(self, cache_service):
        """Test manual circuit breaker reset."""
        breaker = cache_service.circuit_breaker

        # Open the circuit
        for _ in range(breaker.failure_threshold):
            breaker.on_failure()

        assert breaker.state == "open"

        # Manual reset
        breaker.reset()

        assert breaker.state == "closed"
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None


class TestCacheIntegration:
    """Integration tests for cache with real behavior."""

    async def test_cache_workflow_with_errors(self, cache_service, sample_features):
        """Test full cache workflow with error recovery."""
        # Initial cache miss
        cache_service.redis.get = AsyncMock(return_value=None)
        result = await cache_service.get("query1")
        assert result is None

        # Cache set
        cache_service.redis.set = AsyncMock()
        await cache_service.set("query1", sample_features)

        # Cache hit
        import msgpack
        serialized = msgpack.packb(sample_features.model_dump(), use_bin_type=True)
        cache_service.redis.get = AsyncMock(return_value=serialized)
        result = await cache_service.get("query1")
        assert result is not None
        assert result.domain == "code"

        # Error doesn't break system
        cache_service.redis.get = AsyncMock(side_effect=ConnectionError())
        result = await cache_service.get("query1")
        assert result is None  # Graceful degradation
        assert cache_service.stats.errors == 1

    async def test_circuit_breaker_prevents_redis_calls(self, cache_service):
        """Test circuit breaker prevents Redis calls when open."""
        breaker = cache_service.circuit_breaker

        # Open the circuit by triggering failures
        cache_service.redis.get = AsyncMock(side_effect=ConnectionError())
        for _ in range(breaker.failure_threshold):
            await cache_service.get("query")

        assert breaker.state == "open"

        # New cache get should not call Redis
        cache_service.redis.get.reset_mock()
        result = await cache_service.get("another_query")

        assert result is None
        cache_service.redis.get.assert_not_called()
