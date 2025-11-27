"""Routing with constraints example.

This example demonstrates routing with cost, latency, and quality constraints.
"""

import asyncio
import os

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.router import Router


async def main() -> None:
    """Run constrained routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        return

    print("Initializing Conduit...")
    router = Router()

    # Example 1: Cost-constrained query
    print("\n" + "=" * 60)
    print("Example 1: Cost-Constrained Routing")
    print("=" * 60)

    constraints_cheap = QueryConstraints(
        max_cost=0.001,  # Prefer cheaper models
        min_quality=0.7,  # But maintain reasonable quality
    )
    query_cheap = Query(
        text="Write a short poem about the ocean",
        constraints=constraints_cheap,
        preferences=UserPreferences(optimize_for="cost"),
    )

    try:
        decision = await router.route(query_cheap)
        print(f"Query: {query_cheap.text}")
        print(f"Constraints: max_cost=${constraints_cheap.max_cost}, min_quality={constraints_cheap.min_quality}")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Quality-constrained query
    print("\n" + "=" * 60)
    print("Example 2: Quality-Constrained Routing")
    print("=" * 60)

    constraints_quality = QueryConstraints(
        min_quality=0.95,  # Require high quality
        max_latency=10.0,  # But reasonable latency
    )
    query_quality = Query(
        text="Explain quantum entanglement in detail with mathematical formulations",
        constraints=constraints_quality,
        preferences=UserPreferences(optimize_for="quality"),
    )

    try:
        decision = await router.route(query_quality)
        print(f"Query: {query_quality.text[:50]}...")
        print(f"Constraints: min_quality={constraints_quality.min_quality}, max_latency={constraints_quality.max_latency}s")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Speed-optimized query
    print("\n" + "=" * 60)
    print("Example 3: Speed-Optimized Routing")
    print("=" * 60)

    constraints_fast = QueryConstraints(
        max_latency=2.0,  # Fast response required
        preferred_provider="openai",  # Prefer fast provider
    )
    query_fast = Query(
        text="Quick: What's the capital of France?",
        constraints=constraints_fast,
        preferences=UserPreferences(optimize_for="speed"),
    )

    try:
        decision = await router.route(query_fast)
        print(f"Query: {query_fast.text}")
        print(f"Constraints: max_latency={constraints_fast.max_latency}s, preferred_provider={constraints_fast.preferred_provider}")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Constraint Types Summary")
    print("=" * 60)
    print("\nQueryConstraints fields:")
    print("  - max_cost: Maximum cost in dollars (e.g., 0.001)")
    print("  - max_latency: Maximum latency in seconds (e.g., 2.0)")
    print("  - min_quality: Minimum quality score 0.0-1.0 (e.g., 0.9)")
    print("  - preferred_provider: Provider preference (openai, anthropic, google, groq)")

    print("\nUserPreferences optimize_for options:")
    print("  - balanced: Default (70% quality, 20% cost, 10% latency)")
    print("  - quality: Maximize quality (80% quality, 10% cost, 10% latency)")
    print("  - cost: Minimize cost (40% quality, 50% cost, 10% latency)")
    print("  - speed: Minimize latency (40% quality, 10% cost, 50% latency)")

    # Cleanup
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
