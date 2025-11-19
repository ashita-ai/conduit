"""Implicit Feedback - Learning from Behavioral Signals.

Demonstrates the "Observability Trinity":
- Error detection (model failures, empty responses)
- Latency tracking (user patience analysis)
- Retry detection (semantic similarity-based)

Shows how Conduit learns without explicit user ratings.
"""

import asyncio
import time

from redis.asyncio import Redis

from conduit.core.models import Query, QueryFeatures
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.feedback import (
    FeedbackIntegrator,
    ImplicitFeedbackDetector,
    QueryHistoryTracker,
)


async def main():
    print("Implicit Feedback Demo - Learning from Behavior\n")

    # Setup
    try:
        redis = Redis.from_url("redis://localhost:6379")
        await redis.ping()
        print(" Redis connected (retry detection enabled)")
    except Exception:
        print("   Redis unavailable (retry detection disabled)")
        redis = None

    analyzer = QueryAnalyzer()
    bandit = ContextualBandit(models=["gpt-4o-mini", "gpt-4o"])
    history = QueryHistoryTracker(redis=redis)
    detector = ImplicitFeedbackDetector(history)
    integrator = FeedbackIntegrator(bandit)

    # Scenario 1: Success
    print("\n" + "="*60)
    print("Scenario 1: Successful Fast Response")
    print("="*60)

    query1 = Query(text="What is Python?")
    features1 = await analyzer.analyze(query1.text)

    start = time.time()
    await asyncio.sleep(0.2)  # Simulate fast response
    end = time.time()

    feedback = await detector.detect(
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

    print(f"Error: {feedback.error_occurred}")
    print(f"Latency: {feedback.latency_seconds:.2f}s ({feedback.latency_tolerance})")
    print(f"Retry: {feedback.retry_detected}")

    integrator.update_from_implicit("gpt-4o-mini", features1, feedback)
    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\nModel success rate: {state.mean_success_rate:.1%}")

    # Scenario 2: Error
    print("\n" + "="*60)
    print("Scenario 2: Model Error")
    print("="*60)

    query2 = Query(text="Explain quantum physics")
    features2 = await analyzer.analyze(query2.text)

    start = time.time()
    await asyncio.sleep(0.1)
    end = time.time()

    feedback = await detector.detect(
        query=query2.text,
        query_id=query2.id,
        features=features2,
        response_text="I apologize, but I cannot help with that.",
        model_id="gpt-4o-mini",
        execution_status="success",
        execution_error=None,
        request_start_time=start,
        response_complete_time=end,
        user_id="demo_user",
    )

    print(f"L Error: {feedback.error_occurred} ({feedback.error_type})")
    print(f"Latency: {feedback.latency_seconds:.2f}s")

    integrator.update_from_implicit("gpt-4o-mini", features2, feedback)
    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\nModel success rate: {state.mean_success_rate:.1%} (decreased!)")

    # Cleanup
    if redis:
        await redis.aclose()

    print("\n=¡ Learning happens automatically from behavioral signals!")


if __name__ == "__main__":
    asyncio.run(main())
