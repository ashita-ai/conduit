"""Production Feedback Integration - Pluggable User Feedback System.

Demonstrates how to integrate user feedback signals into Conduit's
bandit learning system for production applications.

Features Demonstrated:
1. Delayed feedback (track query → receive signal later)
2. Immediate feedback (no tracking needed)
3. Multiple signal types (thumbs, ratings, task success)
4. Custom adapter registration
5. Redis-based storage for production

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment
    Optional: Redis for production storage

Run:
    python examples/production_feedback.py

See Also:
    - examples/feedback_loop.py - Implicit feedback (errors, latency, retries)
    - conduit/feedback/__init__.py - Full module documentation
"""

import asyncio
import os
import random
from datetime import datetime, timezone

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit.feedback import (
    FeedbackAdapter,
    FeedbackCollector,
    FeedbackEvent,
    InMemoryFeedbackStore,
    RewardMapping,
    ThumbsAdapter,
)

# =============================================================================
# Part 1: Basic Feedback Collection
# =============================================================================


async def demo_delayed_feedback() -> None:
    """Demonstrate delayed feedback pattern.

    This is the typical production pattern:
    1. Route query and get decision
    2. Execute LLM call
    3. Track query for later feedback
    4. User interacts with response
    5. Record user feedback (async/later)
    """
    print("\n" + "=" * 70)
    print("DELAYED FEEDBACK - Track Now, Record Later")
    print("=" * 70)

    # Setup router and collector
    router = Router(cache_enabled=False)
    collector = FeedbackCollector(router)

    # Simulate a user query
    query = Query(text="What are the best practices for Python error handling?")
    print(f"\nQuery: {query.text[:50]}...")

    # 1. Route the query
    decision = await router.route(query)
    print(f"Routed to: {decision.selected_model}")
    print(f"Query ID: {decision.query_id}")

    # 2. Track for delayed feedback (in production, do this after LLM execution)
    await collector.track(
        decision,
        cost=0.001,  # From actual LLM response
        latency=0.5,  # From actual timing
    )
    print(f"Pending queries: {await collector.get_pending_count()}")

    # 3. Simulate user interaction time...
    print("\n[User reads response and decides to give feedback...]")

    # 4. User provides feedback (could be seconds/minutes later)
    feedback = FeedbackEvent(
        query_id=decision.query_id,
        signal_type="thumbs",
        payload={"value": "up"},  # User liked the response!
    )
    success = await collector.record(feedback)
    print(f"\nFeedback recorded: {success}")
    print(f"Pending queries after: {await collector.get_pending_count()}")

    # Check router stats to see the update
    stats = router.hybrid_router.get_stats()
    print(f"Total queries processed: {stats.get('total_queries', 0)}")

    await router.close()


# =============================================================================
# Part 2: Immediate Feedback
# =============================================================================


async def demo_immediate_feedback() -> None:
    """Demonstrate immediate feedback pattern.

    Use this when feedback is available right after the response:
    - Automated quality checks
    - Task completion verification
    - Programmatic validation
    """
    print("\n" + "=" * 70)
    print("IMMEDIATE FEEDBACK - No Tracking Needed")
    print("=" * 70)

    router = Router(cache_enabled=False)
    collector = FeedbackCollector(router)

    # Simulate multiple queries with immediate task success feedback
    queries = [
        ("Write a Python function to reverse a string", True),  # Code works
        ("Explain quantum computing", True),  # Good explanation
        ("Calculate 2+2", True),  # Correct answer
        ("Generate invalid JSON", False),  # Task failed
    ]

    print("\nQuery | Task Success | Reward")
    print("-" * 50)

    for query_text, success in queries:
        query = Query(text=query_text)
        decision = await router.route(query)

        # Record immediate feedback (no tracking step needed)
        await collector.record_immediate(
            model_id=decision.selected_model,
            features=decision.features,
            signal_type="task_success",
            payload={"success": success},
            cost=0.001,
            latency=0.3,
        )

        reward = 1.0 if success else 0.0
        short_query = query_text[:35] + "..." if len(query_text) > 35 else query_text
        print(f"{short_query:40} | {str(success):5} | {reward:.1f}")

    await router.close()


# =============================================================================
# Part 3: Multiple Signal Types
# =============================================================================


async def demo_signal_types() -> None:
    """Demonstrate different feedback signal types.

    Built-in signals:
    - thumbs: Binary up/down
    - rating: Numeric scale (1-5 stars)
    - task_success: Boolean success/failure
    - quality_score: Direct 0.0-1.0 score from human/evaluator
    - completion_time: How long to complete task
    """
    print("\n" + "=" * 70)
    print("SIGNAL TYPES - Multiple Feedback Patterns")
    print("=" * 70)

    router = Router(cache_enabled=False)
    collector = FeedbackCollector(router)

    print("\nRegistered adapters:", collector.registered_signals)

    # Track a query
    query = Query(text="Help me write a cover letter")
    decision = await router.route(query)
    await collector.track(decision, cost=0.002, latency=1.0)

    # Different signal types for the same query
    signals = [
        ("thumbs", {"value": "up"}, "User liked it"),
        ("rating", {"rating": 4}, "4/5 stars"),
        ("task_success", {"success": True}, "Task completed"),
        ("quality_score", {"score": 0.85}, "Human rated 85%"),
    ]

    print(f"\nQuery ID: {decision.query_id[:12]}...")
    print("\nSimulating different signal types:")

    for signal_type, payload, description in signals:
        # Re-track for each signal (in production you'd pick ONE signal type)
        await collector.track(decision, cost=0.002, latency=1.0)

        event = FeedbackEvent(
            query_id=decision.query_id,
            signal_type=signal_type,
            payload=payload,
        )

        adapter = collector.get_adapter(signal_type)
        if adapter:
            mapping = adapter.to_reward(event)
            print(
                f"  {signal_type:15} | {description:20} | "
                f"reward={mapping.reward:.2f}, confidence={mapping.confidence:.2f}"
            )

    await router.close()


# =============================================================================
# Part 4: Custom Adapter
# =============================================================================


async def demo_custom_adapter() -> None:
    """Demonstrate custom adapter registration.

    Create your own adapters for domain-specific feedback signals.
    """
    print("\n" + "=" * 70)
    print("CUSTOM ADAPTER - Domain-Specific Signals")
    print("=" * 70)

    # Define a custom adapter for code quality feedback
    class CodeQualityAdapter(FeedbackAdapter):
        """Custom adapter for code review feedback."""

        @property
        def signal_type(self) -> str:
            return "code_quality"

        def to_reward(self, event: FeedbackEvent) -> RewardMapping:
            """Convert code quality signals to reward.

            Payload:
                - compiles: bool - Code compiles without errors
                - passes_tests: bool - Code passes unit tests
                - follows_style: bool - Follows coding style guidelines
            """
            compiles = event.payload.get("compiles", False)
            passes_tests = event.payload.get("passes_tests", False)
            follows_style = event.payload.get("follows_style", False)

            # Weighted scoring
            score = 0.0
            if compiles:
                score += 0.5  # Critical: must compile
            if passes_tests:
                score += 0.35  # Important: tests pass
            if follows_style:
                score += 0.15  # Nice to have: style

            return RewardMapping(reward=score, confidence=1.0)

    router = Router(cache_enabled=False)
    collector = FeedbackCollector(router)

    # Register custom adapter
    collector.register(CodeQualityAdapter())
    print(f"\nRegistered signals: {collector.registered_signals}")

    # Use custom adapter
    query = Query(text="Write a Python sorting function")
    decision = await router.route(query)
    await collector.track(decision, cost=0.001, latency=0.5)

    # Code quality feedback
    feedback = FeedbackEvent(
        query_id=decision.query_id,
        signal_type="code_quality",
        payload={
            "compiles": True,
            "passes_tests": True,
            "follows_style": False,
        },
    )

    adapter = collector.get_adapter("code_quality")
    mapping = adapter.to_reward(feedback)
    print(f"\nCode Quality Feedback:")
    print(f"  Compiles: Yes, Passes Tests: Yes, Follows Style: No")
    print(f"  Computed Reward: {mapping.reward:.2f} (0.5 + 0.35 + 0.0 = 0.85)")

    success = await collector.record(feedback)
    print(f"  Feedback recorded: {success}")

    await router.close()


# =============================================================================
# Part 5: Batch Feedback
# =============================================================================


async def demo_batch_feedback() -> None:
    """Demonstrate batch feedback recording.

    Efficient for processing multiple feedback signals at once.
    """
    print("\n" + "=" * 70)
    print("BATCH FEEDBACK - Multiple Signals at Once")
    print("=" * 70)

    router = Router(cache_enabled=False)
    collector = FeedbackCollector(router)

    # Track multiple queries
    query_ids = []
    for i, text in enumerate(
        [
            "What is Python?",
            "Explain machine learning",
            "Write a hello world program",
            "What's the capital of France?",
        ]
    ):
        query = Query(text=text)
        decision = await router.route(query)
        await collector.track(decision, cost=0.001, latency=0.3)
        query_ids.append(decision.query_id)

    print(f"\nTracked {len(query_ids)} queries")

    # Batch feedback
    events = [
        FeedbackEvent(query_id=query_ids[0], signal_type="thumbs", payload={"value": "up"}),
        FeedbackEvent(query_id=query_ids[1], signal_type="rating", payload={"rating": 5}),
        FeedbackEvent(
            query_id=query_ids[2], signal_type="task_success", payload={"success": True}
        ),
        FeedbackEvent(query_id=query_ids[3], signal_type="thumbs", payload={"value": "down"}),
    ]

    results = await collector.record_batch(events)

    print("\nBatch Results:")
    for qid, success in results.items():
        print(f"  {qid[:12]}... : {'OK' if success else 'FAILED'}")

    print(f"\nPending after batch: {await collector.get_pending_count()}")

    await router.close()


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all production feedback demos."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    print("=" * 70)
    print("PRODUCTION FEEDBACK INTEGRATION")
    print("=" * 70)
    print("\nThis demo shows how to integrate user feedback into Conduit's")
    print("bandit learning system for production applications.")

    await demo_delayed_feedback()
    await demo_immediate_feedback()
    await demo_signal_types()
    await demo_custom_adapter()
    await demo_batch_feedback()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        """
Key Patterns:

1. DELAYED FEEDBACK (most common in production)
   decision = await router.route(query)
   await collector.track(decision, cost=cost, latency=latency)
   # ... user interacts ...
   await collector.record(FeedbackEvent(
       query_id=decision.query_id,
       signal_type="thumbs",
       payload={"value": "up"}
   ))

2. IMMEDIATE FEEDBACK (automated checks)
   await collector.record_immediate(
       model_id=model,
       features=features,
       signal_type="task_success",
       payload={"success": True}
   )

3. QUALITY SCORE (human reviewers or Arbiter)
   await collector.record(FeedbackEvent(
       query_id=decision.query_id,
       signal_type="quality_score",
       payload={"score": 0.85}  # Direct 0.0-1.0 score
   ))

4. CUSTOM ADAPTERS
   class MyAdapter(FeedbackAdapter):
       signal_type = "my_signal"
       def to_reward(self, event) -> RewardMapping:
           return RewardMapping(reward=0.9, confidence=1.0)
   collector.register(MyAdapter())

5. PRODUCTION STORAGE
   # Option A: Redis (low-latency, native TTL)
   from conduit.feedback import RedisFeedbackStore
   store = RedisFeedbackStore(redis_client)

   # Option B: PostgreSQL (if you don't have Redis)
   from conduit.feedback import PostgresFeedbackStore
   store = PostgresFeedbackStore(asyncpg_pool)
   # Note: Call cleanup_expired() periodically for PostgreSQL

   collector = FeedbackCollector(router, store=store)
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
