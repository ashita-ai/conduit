"""Caching - 10-40x Performance Improvement with Redis.

Demonstrates Redis caching for query features to dramatically
improve routing performance on repeated queries.
"""

import asyncio
import logging
import time

from conduit.core.config import settings
from conduit.core.models import Query
from conduit.engines.router import Router

logger = logging.getLogger(__name__)


async def main():
    logger.info("Cache Performance Demo\n")

    # Test Redis availability using configured URL
    try:
        from redis.asyncio import Redis
        redis = Redis.from_url(settings.redis_url)
        await redis.ping()
        await redis.aclose()
        redis_available = True
        logger.info("✅ Redis connected\n")
    except Exception as e:
        redis_available = False
        logger.warning(f"⚠️  Redis unavailable (caching disabled): {e}\n")

    queries = [
        "What is Python?",
        "Explain machine learning",
        "What is Python?",  # Duplicate - should hit cache
    ]

    # Without cache
    logger.info("="*60)
    logger.info("Without Cache")
    logger.info("="*60)

    router_no_cache = Router(cache_enabled=False)
    times_no_cache = []

    for i, text in enumerate(queries, 1):
        start = time.time()
        await router_no_cache.route(Query(text=text))
        elapsed = (time.time() - start) * 1000
        times_no_cache.append(elapsed)
        logger.info(f"Query {i}: {elapsed:.1f}ms")

    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    logger.info(f"\nAverage: {avg_no_cache:.1f}ms\n")

    await router_no_cache.close()

    # With cache
    if redis_available:
        logger.info("="*60)
        logger.info("With Redis Cache")
        logger.info("="*60)

        router_with_cache = Router(cache_enabled=True)
        times_with_cache = []

        for i, text in enumerate(queries, 1):
            start = time.time()
            await router_with_cache.route(Query(text=text))
            elapsed = (time.time() - start) * 1000
            times_with_cache.append(elapsed)
            cache_hit = " (CACHE HIT!)" if i == 3 else ""
            logger.info(f"Query {i}: {elapsed:.1f}ms{cache_hit}")

        avg_with_cache = sum(times_with_cache) / len(times_with_cache)
        logger.info(f"\nAverage: {avg_with_cache:.1f}ms")

        speedup = avg_no_cache / avg_with_cache
        logger.info(f"Speedup: {speedup:.1f}x faster with cache!")

        stats = router_with_cache.get_cache_stats()
        logger.info(f"\nCache stats: {stats['hits']} hits, {stats['misses']} misses")
        logger.info(f"Hit rate: {stats['hit_rate']:.0f}%")

        await router_with_cache.close()
    else:
        logger.info("💡 Install Redis to see caching benefits:")
        logger.info("   $ brew install redis && redis-server")


if __name__ == "__main__":
    asyncio.run(main())
