"""Explicit Feedback - User Ratings and Quality Scores.

Demonstrates how to submit explicit user feedback (ratings,
quality scores, expectations) to improve model selection.

Conduit's hybrid router learns from feedback to improve routing
decisions over time using contextual bandit algorithms.
"""

import asyncio

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit.engines.bandits.base import BanditFeedback


async def main():
    print("Explicit Feedback Demo - User Ratings\n")

    # Initialize router (uses hybrid routing by default)
    router = Router()

    # Track model performance across queries
    model_stats: dict[str, dict] = {}

    # Scenario 1: High Quality Response
    print("=" * 60)
    print("Scenario 1: Excellent Response - High Quality Feedback")
    print("=" * 60)

    query1 = Query(text="Explain Python in simple terms")
    decision1 = await router.route(query1)

    print(f"Query: {query1.text}")
    print(f"Routed to: {decision1.selected_model}")
    print(f"Confidence: {decision1.confidence:.0%}")

    # User provides positive feedback
    feedback1 = BanditFeedback(
        model_id=decision1.selected_model,
        cost=0.0001,  # Actual cost from response
        quality_score=0.95,  # High quality - user satisfied
        latency=0.5,  # Response time in seconds
        success=True,
    )

    print(f"\nUser Feedback:")
    print(f"  Quality Score: {feedback1.quality_score:.0%}")
    print(f"  Success: {feedback1.success}")

    # Update the router with feedback
    await router.hybrid_router.update(feedback1, decision1.features)
    print("  -> Router updated with positive feedback")

    # Track stats
    model_stats[decision1.selected_model] = {
        "queries": 1,
        "avg_quality": feedback1.quality_score,
        "successes": 1,
    }

    # Scenario 2: Poor Quality Response
    print("\n" + "=" * 60)
    print("Scenario 2: Poor Response - Low Quality Feedback")
    print("=" * 60)

    query2 = Query(text="Write a detailed essay on AI ethics")
    decision2 = await router.route(query2)

    print(f"Query: {query2.text}")
    print(f"Routed to: {decision2.selected_model}")

    # User provides negative feedback
    feedback2 = BanditFeedback(
        model_id=decision2.selected_model,
        cost=0.0005,
        quality_score=0.3,  # Low quality - user unsatisfied
        latency=2.0,
        success=False,  # Did not meet expectations
    )

    print(f"\nUser Feedback:")
    print(f"  Quality Score: {feedback2.quality_score:.0%}")
    print(f"  Success: {feedback2.success}")

    await router.hybrid_router.update(feedback2, decision2.features)
    print("  -> Router updated with negative feedback")

    # Update stats
    if decision2.selected_model in model_stats:
        stats = model_stats[decision2.selected_model]
        stats["queries"] += 1
        stats["avg_quality"] = (stats["avg_quality"] + feedback2.quality_score) / 2
        if not feedback2.success:
            stats["successes"] = stats.get("successes", 0)
    else:
        model_stats[decision2.selected_model] = {
            "queries": 1,
            "avg_quality": feedback2.quality_score,
            "successes": 0 if not feedback2.success else 1,
        }

    # Scenario 3: Another similar query - should route differently
    print("\n" + "=" * 60)
    print("Scenario 3: Similar Query After Learning")
    print("=" * 60)

    query3 = Query(text="Write an analysis of machine learning trends")
    decision3 = await router.route(query3)

    print(f"Query: {query3.text}")
    print(f"Routed to: {decision3.selected_model}")
    print(f"Confidence: {decision3.confidence:.0%}")
    print(f"Reasoning: {decision3.reasoning}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary - How Feedback Improves Routing")
    print("=" * 60)

    print("\nKey Points:")
    print("  1. Explicit feedback directly updates the bandit's model weights")
    print("  2. High quality scores increase a model's selection probability")
    print("  3. Low quality scores decrease selection probability")
    print("  4. The router learns which models work best for query types")

    print("\nModel Statistics from this session:")
    for model_id, stats in model_stats.items():
        print(f"  {model_id}:")
        print(f"    Queries: {stats['queries']}")
        print(f"    Avg Quality: {stats['avg_quality']:.0%}")
        print(f"    Successes: {stats['successes']}")

    print("\nFeedback Integration:")
    print("  - Composite reward = 0.7*quality + 0.2*cost_efficiency + 0.1*speed")
    print("  - Use UserPreferences(optimize_for='cost'|'quality'|'speed') to adjust")

    # Cleanup
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
