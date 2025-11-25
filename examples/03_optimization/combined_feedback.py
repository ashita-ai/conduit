"""Combined Feedback - Explicit + Implicit Learning.

Demonstrates how Conduit combines explicit user feedback (ratings)
with implicit behavioral signals for optimal learning.

The default weighting is:
- 70% Explicit (quality scores, ratings, expectations)
- 30% Implicit (errors, latency, retries)
"""

import asyncio
import time

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit.engines.bandits.base import BanditFeedback
from conduit.feedback import ImplicitFeedbackDetector, QueryHistoryTracker


async def main():
    print("Combined Feedback Demo - Explicit + Implicit\n")

    # Initialize router
    router = Router()

    # Initialize implicit feedback detector (Redis optional)
    try:
        from redis.asyncio import Redis
        redis = Redis.from_url("redis://localhost:6379")
        await redis.ping()
        print("✅ Redis connected")
        history = QueryHistoryTracker(redis=redis)
    except Exception:
        print("⚠️  Redis unavailable (using in-memory tracking)")
        redis = None
        history = QueryHistoryTracker(redis=None)

    detector = ImplicitFeedbackDetector(history)

    # Scenario: User rates a response after receiving it
    print("\n" + "=" * 60)
    print("Complete Feedback Flow: Query -> Response -> Rating")
    print("=" * 60)

    # Step 1: Route the query
    query = Query(text="Explain machine learning in simple terms")
    decision = await router.route(query)

    print(f"\n1. Query Routed:")
    print(f"   Text: {query.text}")
    print(f"   Model: {decision.selected_model}")
    print(f"   Confidence: {decision.confidence:.0%}")

    # Step 2: Simulate response and collect implicit signals
    start = time.time()
    await asyncio.sleep(0.8)  # Simulate response time
    end = time.time()

    response_text = """
    Machine learning is a type of artificial intelligence where computers
    learn from examples instead of being explicitly programmed. Think of it
    like teaching a child - you show them many pictures of cats and dogs,
    and eventually they learn to tell the difference on their own.
    """

    implicit = await detector.detect(
        query=query.text,
        query_id=query.id,
        features=decision.features,
        response_text=response_text,
        model_id=decision.selected_model,
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\n2. Implicit Signals Collected:")
    print(f"   Error: {implicit.error_occurred}")
    print(f"   Latency: {implicit.latency_seconds:.2f}s")
    print(f"   Tolerance: {implicit.latency_tolerance}")

    # Step 3: User provides explicit rating
    print(f"\n3. User Provides Rating:")
    explicit_quality = 0.9  # User rates 4.5/5 stars
    explicit_met_expectations = True

    print(f"   Quality Score: {explicit_quality:.0%}")
    print(f"   Met Expectations: {explicit_met_expectations}")

    # Step 4: Combine feedback (70% explicit, 30% implicit)
    print(f"\n4. Combined Feedback Calculation:")

    # Implicit quality estimate
    implicit_quality = 0.85 if not implicit.error_occurred else 0.2
    if implicit.latency_tolerance == "slow":
        implicit_quality *= 0.9  # Slight penalty for slow

    # Weighted combination
    explicit_weight = 0.7
    implicit_weight = 0.3
    combined_quality = (explicit_weight * explicit_quality +
                        implicit_weight * implicit_quality)

    print(f"   Explicit Quality: {explicit_quality:.0%} (weight: {explicit_weight})")
    print(f"   Implicit Quality: {implicit_quality:.0%} (weight: {implicit_weight})")
    print(f"   Combined Quality: {combined_quality:.0%}")

    # Step 5: Update router with combined feedback
    feedback = BanditFeedback(
        model_id=decision.selected_model,
        cost=0.0002,  # From response metadata
        quality_score=combined_quality,
        latency=implicit.latency_seconds,
        success=explicit_met_expectations,
    )

    await router.hybrid_router.update(feedback, decision.features)
    print(f"\n5. Router Updated:")
    print(f"   -> Model '{decision.selected_model}' learned from combined feedback")

    # Demonstrate learning effect
    print("\n" + "=" * 60)
    print("Learning Effect - Route Similar Query")
    print("=" * 60)

    query2 = Query(text="What is deep learning in simple terms?")
    decision2 = await router.route(query2)

    print(f"\nSimilar Query: {query2.text}")
    print(f"Routed to: {decision2.selected_model}")
    print(f"Confidence: {decision2.confidence:.0%}")
    print(f"Reasoning: {decision2.reasoning}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary - Combined Feedback Benefits")
    print("=" * 60)

    print("\nWhy Combine Feedback?")
    print("  1. Explicit is accurate but sparse (users don't always rate)")
    print("  2. Implicit is always available but less precise")
    print("  3. Combined approach gets best of both worlds")

    print("\nFeedback Sources:")
    print("  Explicit (70% weight):")
    print("    - Quality scores (0.0-1.0)")
    print("    - Star ratings (1-5)")
    print("    - Met expectations (yes/no)")
    print("  Implicit (30% weight):")
    print("    - Error detection (refusals, failures)")
    print("    - Latency tolerance (fast/acceptable/slow)")
    print("    - Retry patterns (user rephrased query)")

    print("\nReward Composition:")
    print("  Total = 0.7*quality + 0.2*cost_efficiency + 0.1*speed")
    print("  Where quality = 0.7*explicit + 0.3*implicit")

    # Cleanup
    if redis:
        await redis.aclose()
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
