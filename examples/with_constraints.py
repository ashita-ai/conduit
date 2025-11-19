"""Routing with constraints example.

This example demonstrates routing with cost, latency, and quality constraints.
"""

import asyncio
import os
from conduit.engines.router import Router
from conduit.core.models import Query, QueryConstraints


async def main() -> None:
    """Run constrained routing example."""
    router = Router()

    # Example 1: Cost-constrained query
    print("=" * 60)
    print("Example 1: Cost-Constrained Routing")
    print("=" * 60)

    query_cheap = Query(
        text="Write a short poem about the ocean",
        user_id="budget_user",
        constraints=QueryConstraints(
            max_cost=0.001,  # Prefer cheaper models
            min_quality=0.7,  # But maintain reasonable quality
        ),
    )

    result_cheap = await router.route(query_cheap)
    print(f"Query: {query_cheap.text}")
    print(f"Selected Model: {result_cheap.selected_model}")
    print(f"Confidence: {result_cheap.confidence:.2f}")
    print(f"Reasoning: {result_cheap.reasoning}")

    # Example 2: Quality-constrained query
    print("\n" + "=" * 60)
    print("Example 2: Quality-Constrained Routing")
    print("=" * 60)

    query_quality = Query(
        text="Explain quantum entanglement in detail with mathematical formulations",
        user_id="quality_user",
        constraints=QueryConstraints(
            min_quality=0.95,  # Require high quality
            max_latency=10.0,  # But reasonable latency
        ),
    )

    result_quality = await router.route(query_quality)
    print(f"Query: {query_quality.text}")
    print(f"Selected Model: {result_quality.selected_model}")
    print(f"Confidence: {result_quality.confidence:.2f}")
    print(f"Reasoning: {result_quality.reasoning}")

    # Example 3: Latency-constrained query
    print("\n" + "=" * 60)
    print("Example 3: Latency-Constrained Routing")
    print("=" * 60)

    query_fast = Query(
        text="Quick: What's the capital of France?",
        user_id="speed_user",
        constraints=QueryConstraints(
            max_latency=2.0,  # Fast response required
            preferred_provider="groq",  # Prefer fast provider
        ),
    )

    result_fast = await router.route(query_fast)
    print(f"Query: {query_fast.text}")
    print(f"Selected Model: {result_fast.selected_model}")
    print(f"Confidence: {result_fast.confidence:.2f}")
    print(f"Reasoning: {result_fast.reasoning}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        exit(1)

    asyncio.run(main())
