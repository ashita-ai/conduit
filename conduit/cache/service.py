"""Redis caching service with circuit breaker pattern."""

import hashlib
import logging
import re
import time

import msgpack
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError

from conduit.cache.models import CacheConfig, CacheStats
from conduit.core.models import QueryFeatures

logger = logging.getLogger(__name__)


class CacheCircuitBreaker:
    """Circuit breaker for cache failures with automatic recovery.

    States:
        closed: Normal operation, cache requests allowed
        open: Circuit tripped, cache bypassed entirely
        half_open: Testing recovery, single request allowed

    Pattern:
        closed -> (failures >= threshold) -> open
        open -> (timeout expired) -> half_open
        half_open -> (success) -> closed
        half_open -> (failure) -> open
    """

    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            timeout: Seconds before attempting recovery from open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = "closed"  # closed, open, half_open
        self.last_failure_time: float | None = None

    def on_success(self) -> None:
        """Record successful operation."""
        if self.state == "half_open":
            logger.info("Circuit breaker recovered, closing circuit")
            self.state = "closed"
            self.failure_count = 0

    def on_failure(self) -> None:
        """Record failed operation and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
            self.state = "open"

    def can_attempt(self) -> bool:
        """Check if cache operation should be attempted.

        Returns:
            True if operation should proceed, False if circuit open
        """
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout expired, transition to half-open
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.timeout
            ):
                logger.info("Circuit breaker timeout expired, entering half-open state")
                self.state = "half_open"
                return True
            return False

        # half_open state: allow single attempt
        return True

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None


class CacheService:
    """Redis-based caching service with fail-safe design.

    Features:
        - MessagePack serialization for compact storage
        - Circuit breaker for automatic failure recovery
        - Graceful degradation when Redis unavailable
        - Performance statistics tracking
        - Configurable TTL and retry behavior

    Design Philosophy:
        Cache is an OPTIONAL optimization. System works perfectly
        when Redis is unavailable - just slower due to recomputation.
    """

    def __init__(self, config: CacheConfig):
        """Initialize cache service.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.redis: Redis[bytes] | None = None
        self.circuit_breaker = CacheCircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout,
        )
        self.stats = CacheStats()

        # Initialize Redis client if caching enabled
        if config.enabled:
            self.redis = Redis.from_url(
                config.redis_url,
                socket_timeout=config.timeout,
                socket_connect_timeout=config.timeout,
                retry_on_timeout=True,
                max_connections=10,
                decode_responses=False,  # We handle bytes for msgpack
            )
            logger.info(f"Cache service initialized with Redis at {config.redis_url}")
        else:
            logger.info("Cache service disabled by configuration")

    async def get(self, query: str) -> QueryFeatures | None:
        """Retrieve cached QueryFeatures for a query.

        Args:
            query: User query text

        Returns:
            Cached QueryFeatures if hit, None if miss or error

        Note:
            All errors are caught and logged. Returns None on any failure,
            allowing caller to compute features normally.
        """
        if not self.config.enabled or not self.redis:
            return None

        if not self.circuit_breaker.can_attempt():
            logger.debug("Circuit breaker open, skipping cache lookup")
            return None

        try:
            cache_key = self._generate_key(query)
            data = await self.redis.get(cache_key)

            if data is None:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                self.circuit_breaker.on_success()  # Successful operation (miss is ok)
                return None

            # Deserialize from MessagePack
            features_dict = msgpack.unpackb(data, raw=False)
            features = QueryFeatures(**features_dict)

            self.stats.hits += 1
            self.stats.update_hit_rate()
            self.circuit_breaker.on_success()
            logger.debug(f"Cache hit for query hash {cache_key[:8]}")

            return features

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache lookup failed (connection): {e}")
            self.stats.errors += 1
            self.circuit_breaker.on_failure()
            return None

        except Exception as e:
            # Catch all other errors (deserialization, etc.)
            logger.error(f"Cache lookup failed (unexpected): {e}")
            self.stats.errors += 1
            return None

    async def set(self, query: str, features: QueryFeatures) -> None:
        """Store QueryFeatures in cache with TTL.

        Args:
            query: User query text
            features: Extracted query features to cache

        Note:
            Failures are logged but don't raise exceptions. Cache writes
            are best-effort - if they fail, routing continues normally.
        """
        if not self.config.enabled or not self.redis:
            return

        if not self.circuit_breaker.can_attempt():
            logger.debug("Circuit breaker open, skipping cache write")
            return

        try:
            cache_key = self._generate_key(query)

            # Serialize to MessagePack
            features_dict = features.model_dump()
            data = msgpack.packb(features_dict, use_bin_type=True)

            await self.redis.set(cache_key, data, ex=self.config.ttl)
            self.circuit_breaker.on_success()
            logger.debug(f"Cached features for query hash {cache_key[:8]}")

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Cache write failed (connection): {e}")
            self.circuit_breaker.on_failure()

        except Exception as e:
            logger.error(f"Cache write failed (unexpected): {e}")
            # Don't trigger circuit breaker for serialization errors

    async def clear(self) -> None:
        """Clear all cache entries (admin operation).

        Note:
            This is expensive and should only be used for maintenance
            or testing. Normal operation uses TTL-based expiry.
        """
        if not self.config.enabled or not self.redis:
            return

        try:
            # Delete all keys matching our pattern
            cursor = 0
            pattern = "conduit:query:*"
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    await self.redis.delete(*keys)
                if cursor == 0:
                    break

            logger.info("Cache cleared successfully")

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self.redis:
            await self.redis.close()
            logger.info("Cache service closed")

    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats with hit/miss/error counts and circuit state
        """
        self.stats.circuit_state = self.circuit_breaker.state
        return self.stats

    def _generate_key(self, query: str) -> str:
        """Generate cache key from query text.

        Args:
            query: User query text

        Returns:
            Cache key string

        Key Generation:
            1. Normalize: lowercase + strip whitespace
            2. Collapse: multiple spaces -> single space
            3. Hash: SHA256 for consistent key length
            4. Prefix: "conduit:query:" for namespace isolation
        """
        # Normalize query text for consistent cache hits
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)  # Collapse whitespace

        # Generate SHA256 hash for consistent key length
        hash_digest = hashlib.sha256(normalized.encode()).hexdigest()

        return f"conduit:query:{hash_digest}"
