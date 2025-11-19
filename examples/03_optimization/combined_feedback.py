"""Combined Feedback - Explicit + Implicit Learning.

Demonstrates how Conduit combines explicit user ratings with
implicit behavioral signals for optimal learning.
"""

import asyncio
import time

from redis.asyncio import Redis

from conduit.core.models import Feedback, Query, QueryFeatures
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.feedback import (
    FeedbackIntegrator,
    ImplicitFeedbackDetector,
    QueryHistoryTracker,
)


async def main():
    print("Combined Feedback Demo - Explicit + Implicit\n")

    # Setup
    try:
        redis = Redis.from_url("redis://localhost:6379")
        await redis.ping()
        print("✅ Redis connected (full feedback enabled)")
    except Exception:
        print("⚠️  Redis unavailable (implicit feedback disabled)")
        redis = None

    analyzer = QueryAnalyzer()
    bandit = ContextualBandit(models=["gpt-4o-mini", "gpt-4o"])
    history = QueryHistoryTracker(redis=redis)
    detector = ImplicitFeedbackDetector(history)
    integrator = FeedbackIntegrator(bandit)

    print("\n" + "="*60)
    print("Scenario 1: High Quality + Fast (Both Positive)")
    print("="*60)

    query1 = Query(text="What is Python?")
    features1 = await analyzer.analyze(query1.text)

    # Simulate fast, quality response
    start = time.time()
    await asyncio.sleep(0.1)  # Fast response
    end = time.time()

    # Explicit: High user rating
    explicit1 = Feedback(
        response_id="resp_001",
        quality_score=0.95,
        user_rating=5,
        met_expectations=True,
    )

    # Implicit: Fast, no errors
    implicit1 = await detector.detect(
        query=query1.text,
        query_id=query1.id,
        features=features1,
        response_text="Python is a programming language...",
        model_id="gpt-4o-mini",
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\n📝 Explicit: {explicit1.quality_score:.0%} quality, {explicit1.user_rating}/5 stars")
    print(f"🔍 Implicit: {implicit1.latency_seconds:.2f}s latency ({implicit1.latency_tolerance}), no errors")

    # Update with both
    integrator.update_from_explicit("gpt-4o-mini", features1, explicit1)
    integrator.update_from_implicit("gpt-4o-mini", features1, implicit1)

    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\n✅ Model success rate: {state.mean_success_rate:.1%}")
    print(f"   Alpha (successes): {state.alpha:.1f}")

    # Scenario 2: Low Quality + Retry (Both Negative)
    print("\n" + "="*60)
    print("Scenario 2: Low Quality + Retry (Both Negative)")
    print("="*60)

    query2 = Query(text="Explain quantum physics in detail")
    features2 = await analyzer.analyze(query2.text)

    start = time.time()
    await asyncio.sleep(0.1)
    end = time.time()

    # Explicit: Low user rating
    explicit2 = Feedback(
        response_id="resp_002",
        quality_score=0.3,
        user_rating=2,
        met_expectations=False,
    )

    # Implicit: Retry detected (simulate by adding to history first)
    if redis:
        await history.add_query(
            query_id="prev_query",
            query_text=query2.text,
            features=features2,
            user_id="demo_user",
            model_used="gpt-4o-mini",
        )
        await asyncio.sleep(0.1)

    implicit2 = await detector.detect(
        query=query2.text,
        query_id=query2.id,
        features=features2,
        response_text="Brief unhelpful response",
        model_id="gpt-4o-mini",
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\n📝 Explicit: {explicit2.quality_score:.0%} quality, {explicit2.user_rating}/5 stars")
    print(f"🔍 Implicit: Retry={'Yes' if implicit2.retry_detected else 'No'}")

    # Update with both
    integrator.update_from_explicit("gpt-4o-mini", features2, explicit2)
    integrator.update_from_implicit("gpt-4o-mini", features2, implicit2)

    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\n❌ Model success rate: {state.mean_success_rate:.1%} (decreased)")
    print(f"   Beta (failures): {state.beta:.1f}")

    # Scenario 3: Mixed Signals (Explicit positive, Implicit negative)
    print("\n" + "="*60)
    print("Scenario 3: Mixed Signals (High quality but slow)")
    print("="*60)

    query3 = Query(text="Write a comprehensive essay on AI")
    features3 = await analyzer.analyze(query3.text)

    start = time.time()
    await asyncio.sleep(15.0)  # Slow response
    end = time.time()

    # Explicit: User satisfied with quality
    explicit3 = Feedback(
        response_id="resp_003",
        quality_score=0.9,
        user_rating=5,
        met_expectations=True,
    )

    # Implicit: Slow latency
    implicit3 = await detector.detect(
        query=query3.text,
        query_id=query3.id,
        features=features3,
        response_text="Comprehensive detailed essay...",
        model_id="gpt-4o",
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"\n📝 Explicit: {explicit3.quality_score:.0%} quality (positive)")
    print(f"🔍 Implicit: {implicit3.latency_seconds:.1f}s latency ({implicit3.latency_tolerance}) (negative)")

    # Update with both
    integrator.update_from_explicit("gpt-4o", features3, explicit3)
    integrator.update_from_implicit("gpt-4o", features3, implicit3)

    state = bandit.get_model_state("gpt-4o")
    print(f"\n⚖️  Weighted result: {state.mean_success_rate:.1%}")
    print(f"   Explicit (70%) dominated over implicit (30%)")

    # Summary
    print("\n" + "="*60)
    print("Summary: Weighted Feedback System")
    print("="*60)
    print("\n💡 Key Insights:")
    print("   1. Explicit feedback (70%) dominates - user ratings matter most")
    print("   2. Implicit signals (30%) add behavioral context")
    print("   3. Mixed signals show realistic user behavior")
    print("   4. System learns from BOTH to optimize routing")

    print("\n📊 Final Model States:")
    for model_id in ["gpt-4o-mini", "gpt-4o"]:
        state = bandit.get_model_state(model_id)
        print(f"   {model_id}: {state.mean_success_rate:.0%} success rate")

    # Cleanup
    if redis:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
