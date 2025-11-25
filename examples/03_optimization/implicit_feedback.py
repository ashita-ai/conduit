"""Implicit Feedback - Learning from Behavioral Signals.

Demonstrates the "Observability Trinity":
- Error detection (model failures, empty responses)
- Latency tracking (user patience analysis)
- Retry detection (semantic similarity-based)

Shows how Conduit learns without explicit user ratings.
"""

import asyncio
import time

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit.engines.bandits.base import BanditFeedback
from conduit.feedback import ImplicitFeedbackDetector, QueryHistoryTracker


async def main():
    print("Implicit Feedback Demo - Learning from Behavior\n")

    # Initialize router
    router = Router()

    # Initialize implicit feedback components
    # Note: Redis is optional - retry detection works better with it
    try:
        from redis.asyncio import Redis
        redis = Redis.from_url("redis://localhost:6379")
        await redis.ping()
        print("✅ Redis connected (retry detection enabled)")
        history = QueryHistoryTracker(redis=redis)
    except Exception:
        print("⚠️  Redis unavailable (retry detection disabled)")
        redis = None
        history = QueryHistoryTracker(redis=None)

    detector = ImplicitFeedbackDetector(history)

    # Scenario 1: Successful Fast Response
    print("\n" + "=" * 60)
    print("Scenario 1: Successful Fast Response")
    print("=" * 60)

    query1 = Query(text="What is Python?")
    decision1 = await router.route(query1)

    print(f"Query: {query1.text}")
    print(f"Routed to: {decision1.selected_model}")

    # Simulate response
    start = time.time()
    await asyncio.sleep(0.2)  # Simulate fast response
    end = time.time()

    # Detect implicit feedback
    implicit1 = await detector.detect(
        query=query1.text,
        query_id=query1.id,
        features=decision1.features,
        response_text="Python is a high-level programming language known for its simplicity...",
        model_id=decision1.selected_model,
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\nImplicit Signals Detected:")
    print(f"  Error Occurred: {implicit1.error_occurred}")
    print(f"  Latency: {implicit1.latency_seconds:.2f}s")
    print(f"  Latency Tolerance: {implicit1.latency_tolerance}")
    print(f"  Retry Detected: {implicit1.retry_detected}")

    # Convert to bandit feedback and update
    # Quality estimated from implicit signals (no errors + fast = good)
    quality_estimate = 0.85 if not implicit1.error_occurred else 0.2
    feedback1 = BanditFeedback(
        model_id=decision1.selected_model,
        cost=0.0001,
        quality_score=quality_estimate,
        latency=implicit1.latency_seconds,
        success=not implicit1.error_occurred,
    )
    await router.hybrid_router.update(feedback1, decision1.features)
    print(f"  -> Router updated (estimated quality: {quality_estimate:.0%})")

    # Scenario 2: Error Response
    print("\n" + "=" * 60)
    print("Scenario 2: Model Returns Error Pattern")
    print("=" * 60)

    query2 = Query(text="Explain quantum physics in detail")
    decision2 = await router.route(query2)

    print(f"Query: {query2.text}")
    print(f"Routed to: {decision2.selected_model}")

    start = time.time()
    await asyncio.sleep(0.1)
    end = time.time()

    # Simulate a refusal/error response
    implicit2 = await detector.detect(
        query=query2.text,
        query_id=query2.id,
        features=decision2.features,
        response_text="I apologize, but I cannot help with that request.",
        model_id=decision2.selected_model,
        execution_status="success",  # HTTP success but content is refusal
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\nImplicit Signals Detected:")
    print(f"  Error Occurred: {implicit2.error_occurred}")
    if implicit2.error_type:
        print(f"  Error Type: {implicit2.error_type}")
    print(f"  Latency: {implicit2.latency_seconds:.2f}s")

    # Quality estimated lower due to refusal pattern
    quality_estimate = 0.3 if implicit2.error_occurred else 0.7
    feedback2 = BanditFeedback(
        model_id=decision2.selected_model,
        cost=0.0002,
        quality_score=quality_estimate,
        latency=implicit2.latency_seconds,
        success=not implicit2.error_occurred,
    )
    await router.hybrid_router.update(feedback2, decision2.features)
    print(f"  -> Router updated (estimated quality: {quality_estimate:.0%})")

    # Scenario 3: Slow Response
    print("\n" + "=" * 60)
    print("Scenario 3: Slow Response (User Patience Test)")
    print("=" * 60)

    query3 = Query(text="Write a comprehensive analysis")
    decision3 = await router.route(query3)

    print(f"Query: {query3.text}")
    print(f"Routed to: {decision3.selected_model}")

    start = time.time()
    await asyncio.sleep(3.0)  # Simulate slow response
    end = time.time()

    implicit3 = await detector.detect(
        query=query3.text,
        query_id=query3.id,
        features=decision3.features,
        response_text="Here is a comprehensive analysis of the topic...",
        model_id=decision3.selected_model,
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\nImplicit Signals Detected:")
    print(f"  Latency: {implicit3.latency_seconds:.2f}s")
    print(f"  Latency Tolerance: {implicit3.latency_tolerance}")
    print(f"  (Slow responses may indicate user impatience)")

    # Quality slightly penalized for slow response
    quality_estimate = 0.75  # Good content but slow
    feedback3 = BanditFeedback(
        model_id=decision3.selected_model,
        cost=0.001,
        quality_score=quality_estimate,
        latency=implicit3.latency_seconds,
        success=True,
    )
    await router.hybrid_router.update(feedback3, decision3.features)
    print(f"  -> Router updated (estimated quality: {quality_estimate:.0%})")

    # Summary
    print("\n" + "=" * 60)
    print("Summary - Implicit Feedback Signals")
    print("=" * 60)

    print("\nThe Observability Trinity:")
    print("  1. Error Detection: Captures refusals, empty responses, API errors")
    print("  2. Latency Tracking: Monitors response time vs user patience")
    print("  3. Retry Detection: Identifies when users rephrase (dissatisfaction)")

    print("\nBenefits:")
    print("  - Works without explicit user ratings")
    print("  - Captures real behavioral signals")
    print("  - Enables continuous learning from usage")

    print("\nIntegration:")
    print("  - Implicit feedback is weighted 30% vs explicit 70%")
    print("  - Composite reward: 0.7*quality + 0.2*cost + 0.1*latency")

    # Cleanup
    if redis:
        await redis.aclose()
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
