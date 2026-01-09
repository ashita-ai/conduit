"""Feedback Loop - Caching, Learning, and State Persistence.

Demonstrates how Conduit learns and improves over time:
1. Caching - 10-40x speedup with Redis
2. Learning - Bandit algorithms improve routing with feedback
3. Persistence - Save/restore state across restarts

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment
    Optional: Redis for caching (brew install redis && redis-server)

Run:
    python examples/feedback_loop.py
"""

import asyncio
import hashlib
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from conduit.core.config import settings
from conduit.core.models import Query
from conduit.core.state_store import BanditState, HybridRouterState, StateStore
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.hybrid_router import HybridRouter
from conduit.engines.router import Router

logger = logging.getLogger(__name__)

# =============================================================================
# Part 1: Caching
# =============================================================================


async def demo_caching() -> None:
    """Demonstrate Redis caching for query features."""
    logger.info("\n" + "=" * 70)
    logger.info("CACHING - 10-40x Speedup with Redis")
    logger.info("=" * 70)

    # Check Redis availability
    redis_available = False
    try:
        from redis.asyncio import Redis

        redis = Redis.from_url(settings.redis_url)
        await redis.ping()
        await redis.aclose()
        redis_available = True
        logger.info("Redis: Connected")
    except Exception as e:
        logger.info(f"Redis: Unavailable ({e})")

    queries = [
        "What is Python?",
        "Explain machine learning",
        "What is Python?",  # Duplicate - cache hit
    ]

    # Without cache
    logger.info("\n[Without Cache]")
    router = Router(cache_enabled=False)
    times = []
    for i, text in enumerate(queries, 1):
        start = time.time()
        await router.route(Query(text=text))
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        logger.info(f"  Query {i}: {elapsed:.1f}ms")
    avg_no_cache = sum(times) / len(times)
    logger.info(f"  Average: {avg_no_cache:.1f}ms")
    await router.close()

    # With cache
    if redis_available:
        logger.info("\n[With Redis Cache]")
        router = Router(cache_enabled=True)
        times = []
        for i, text in enumerate(queries, 1):
            start = time.time()
            await router.route(Query(text=text))
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            hit = " (CACHE HIT)" if i == 3 else ""
            logger.info(f"  Query {i}: {elapsed:.1f}ms{hit}")
        avg_with_cache = sum(times) / len(times)
        logger.info(f"  Average: {avg_with_cache:.1f}ms")
        logger.info(f"  Speedup: {avg_no_cache / avg_with_cache:.1f}x faster")

        stats = router.get_cache_stats()
        logger.info(
            f"  Stats: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.0f}% hit rate)"
        )
        await router.close()
    else:
        logger.info("\nInstall Redis for caching: brew install redis && redis-server")


# =============================================================================
# Part 2: Learning with Feedback
# =============================================================================


async def demo_learning() -> None:
    """Demonstrate learning from feedback."""
    logger.info("\n" + "=" * 70)
    logger.info("LEARNING - Bandit Algorithms Improve Routing")
    logger.info("=" * 70)

    router = Router(cache_enabled=False)
    logger.info("\nSimulating queries with feedback...")

    queries = [
        ("What is 2+2?", "simple"),
        ("Explain quantum mechanics in detail", "complex"),
        ("Translate hello to Spanish", "simple"),
        ("Write a detailed analysis of climate change", "complex"),
        ("What's the capital of France?", "simple"),
    ]

    # Track model usage
    model_usage: dict[str, int] = {}

    logger.info("\nQuery | Selected Model | Confidence")
    logger.info("-" * 50)

    for query_text, query_type in queries:
        decision = await router.route(Query(text=query_text))
        model = decision.selected_model
        model_usage[model] = model_usage.get(model, 0) + 1

        # Display routing decision
        short_query = query_text[:30] + "..." if len(query_text) > 30 else query_text
        logger.info(f"{short_query:35} | {model:15} | {decision.confidence:.2f}")

        # Simulate feedback (in production this comes from actual usage)
        feedback = BanditFeedback(
            model_id=model,
            cost=0.001 if "mini" in model.lower() else 0.01,
            quality_score=0.85 if query_type == "simple" else 0.95,
            latency=0.5 if "mini" in model.lower() else 1.0,
        )
        await router.hybrid_router.update(feedback, decision.features)

    logger.info(f"\nModel Usage: {model_usage}")
    stats = router.hybrid_router.get_stats()
    logger.info(f"Total queries processed: {stats.get('total_queries', 0)}")
    logger.info(f"Current phase: {stats.get('current_phase', 'exploration')}")

    await router.close()

    await router.close()


# =============================================================================
# Part 3: State Persistence
# =============================================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for demos (no API calls)."""

    async def embed(self, text: str) -> list[float]:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= 384:
                break
            val = (int(text_hash[i : i + 2], 16) / 255.0) * 2 - 1
            embedding.append(val)
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return 384

    @property
    def provider_name(self) -> str:
        return "mock"


class InMemoryStateStore(StateStore):
    """In-memory state store for demos (use PostgresStateStore in production)."""

    def __init__(self) -> None:
        self._bandit_states: dict[str, dict[str, BanditState]] = {}
        self._hybrid_states: dict[str, HybridRouterState] = {}

    async def save_bandit_state(
        self, router_id: str, bandit_id: str, state: BanditState
    ) -> None:
        if router_id not in self._bandit_states:
            self._bandit_states[router_id] = {}
        state.updated_at = datetime.now(UTC)
        self._bandit_states[router_id][bandit_id] = state

    async def load_bandit_state(
        self, router_id: str, bandit_id: str
    ) -> BanditState | None:
        return self._bandit_states.get(router_id, {}).get(bandit_id)

    async def save_hybrid_router_state(
        self, router_id: str, state: HybridRouterState
    ) -> None:
        state.updated_at = datetime.now(UTC)
        self._hybrid_states[router_id] = state

    async def load_hybrid_router_state(
        self, router_id: str
    ) -> HybridRouterState | None:
        return self._hybrid_states.get(router_id)

    async def delete_state(self, router_id: str) -> None:
        self._bandit_states.pop(router_id, None)
        self._hybrid_states.pop(router_id, None)

    async def list_router_ids(self) -> list[str]:
        ids = set(self._bandit_states.keys()) | set(self._hybrid_states.keys())
        return sorted(ids)


async def demo_persistence() -> None:
    """Demonstrate state persistence across restarts."""
    logger.info("\n" + "=" * 70)
    logger.info("PERSISTENCE - Save/Restore State Across Restarts")
    logger.info("=" * 70)

    store = InMemoryStateStore()
    router_id = "demo-router"
    models = ["gpt-4o-mini", "gpt-4o"]

    # Create router with mock embeddings
    mock_provider = MockEmbeddingProvider()
    mock_analyzer = QueryAnalyzer(embedding_provider=mock_provider)

    router = HybridRouter(
        models=models,
        switch_threshold=100,
        analyzer=mock_analyzer,
    )

    # Process some queries
    logger.info("\n[1] Processing 20 queries...")
    sample_queries = [
        "What is Python?",
        "Explain machine learning",
        "Write a haiku",
        "Debug this code",
    ]

    for i in range(20):
        query = Query(text=sample_queries[i % len(sample_queries)])
        decision = await router.route(query)
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.9,
            latency=0.5,
        )
        await router.update(feedback, decision.features)

    logger.info(f"    Query count: {router.query_count}")
    logger.info(f"    Phase: {router.current_phase}")

    # Save state
    await router.save_state(store, router_id)
    logger.info("\n[2] State saved to store")

    # Simulate restart by creating new router
    logger.info("\n[3] Simulating restart...")
    del router

    # Create new router and restore state
    mock_provider2 = MockEmbeddingProvider()
    mock_analyzer2 = QueryAnalyzer(embedding_provider=mock_provider2)

    new_router = HybridRouter(
        models=models,
        switch_threshold=100,
        analyzer=mock_analyzer2,
    )

    loaded = await new_router.load_state(store, router_id)
    logger.info(f"\n[4] State restored: {loaded}")
    logger.info(f"    Query count: {new_router.query_count}")
    logger.info(f"    Phase: {new_router.current_phase}")

    # Continue processing
    logger.info("\n[5] Continuing with 10 more queries...")
    for i in range(10):
        query = Query(text=sample_queries[i % len(sample_queries)])
        decision = await new_router.route(query)
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.9,
            latency=0.5,
        )
        await new_router.update(feedback, decision.features)

    logger.info(f"    Total query count: {new_router.query_count}")

    logger.info("\nProduction Usage:")
    logger.info("  Replace InMemoryStateStore with PostgresStateStore:")
    logger.info("    from conduit.core.postgres_state_store import PostgresStateStore")
    logger.info("    store = PostgresStateStore(pool)")


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all feedback loop demos."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    logger.info("=" * 70)
    logger.info("CONDUIT FEEDBACK LOOP")
    logger.info("=" * 70)
    logger.info("\nConduit improves over time through:")
    logger.info("  1. Caching - Faster repeated queries with Redis")
    logger.info("  2. Learning - Bandit algorithms optimize model selection")
    logger.info("  3. Persistence - State survives restarts")

    await demo_caching()
    await demo_learning()
    await demo_persistence()

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(
        """
Key Concepts:

1. CACHING
   router = Router(cache_enabled=True)
   - Requires Redis: brew install redis && redis-server
   - Cache stats: router.get_cache_stats()

2. LEARNING
   # Conduit learns from feedback automatically
   feedback = BanditFeedback(
       model_id=decision.selected_model,
       cost=0.001,
       quality_score=0.95,
       latency=0.5
   )
   await router.hybrid_router.update(feedback, features)

3. PERSISTENCE
   # Save state
   await router.hybrid_router.save_state(store, "my-router")

   # Restore state
   await router.hybrid_router.load_state(store, "my-router")

   # Production: Use PostgresStateStore
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
