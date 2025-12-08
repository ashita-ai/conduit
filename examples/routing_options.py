"""Routing Options - Constraints, Preferences, and Algorithms.

Demonstrates all routing configuration options in one file:
1. Constraints (max_cost, max_latency, min_quality, preferred_provider)
2. User Preferences (optimize_for: balanced/quality/cost/speed)
3. Algorithm Selection (thompson_sampling, linucb, ucb1, etc.)

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Run:
    python examples/routing_options.py
"""

import asyncio
import os

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.router import Router


async def demo_constraints(router: Router) -> None:
    """Demonstrate constraint-based routing."""
    print("\n" + "=" * 70)
    print("CONSTRAINTS - Control model selection limits")
    print("=" * 70)

    # Cost-constrained: prefer cheaper models
    print("\n[1] Cost-Constrained (max $0.001, min quality 0.7)")
    constraints = QueryConstraints(max_cost=0.001, min_quality=0.7)
    query = Query(
        text="Write a haiku about programming",
        constraints=constraints,
        preferences=UserPreferences(optimize_for="cost"),
    )
    decision = await router.route(query)
    print(f"    Model: {decision.selected_model}")
    print(f"    Confidence: {decision.confidence:.2f}")

    # Quality-constrained: require high quality models
    print("\n[2] Quality-Constrained (min quality 0.95)")
    constraints = QueryConstraints(min_quality=0.95, max_latency=10.0)
    query = Query(
        text="Explain quantum entanglement with mathematical formulations",
        constraints=constraints,
        preferences=UserPreferences(optimize_for="quality"),
    )
    decision = await router.route(query)
    print(f"    Model: {decision.selected_model}")
    print(f"    Confidence: {decision.confidence:.2f}")

    # Speed-constrained: require fast response
    print("\n[3] Speed-Constrained (max 2s latency, prefer OpenAI)")
    constraints = QueryConstraints(max_latency=2.0, preferred_provider="openai")
    query = Query(
        text="What is the capital of France?",
        constraints=constraints,
        preferences=UserPreferences(optimize_for="speed"),
    )
    decision = await router.route(query)
    print(f"    Model: {decision.selected_model}")
    print(f"    Confidence: {decision.confidence:.2f}")


async def demo_preferences(router: Router) -> None:
    """Demonstrate preference-based routing."""
    print("\n" + "=" * 70)
    print("PREFERENCES - Adjust quality/cost/latency tradeoffs")
    print("=" * 70)
    print("\nPresets and their weights:")
    print("  balanced: 70% quality, 20% cost, 10% latency (default)")
    print("  quality:  80% quality, 10% cost, 10% latency")
    print("  cost:     40% quality, 50% cost, 10% latency")
    print("  speed:    40% quality, 10% cost, 50% latency")

    test_query = "Explain machine learning in simple terms"
    presets = ["balanced", "quality", "cost", "speed"]

    for preset in presets:
        query = Query(
            text=test_query,
            preferences=UserPreferences(optimize_for=preset),  # type: ignore[arg-type]
        )
        decision = await router.route(query)
        print(
            f"\n[{preset}] Model: {decision.selected_model}, Confidence: {decision.confidence:.2f}"
        )


async def demo_algorithms() -> None:
    """Demonstrate different routing algorithms."""
    print("\n" + "=" * 70)
    print("ALGORITHMS - Different learning strategies")
    print("=" * 70)
    print("\nAvailable algorithms:")
    print("  thompson_sampling (default) - Bayesian bandit, best cold-start")
    print("  linucb                      - Contextual linear bandit")
    print("  ucb1                        - Upper confidence bound")
    print("  epsilon_greedy              - Epsilon-greedy exploration")
    print("  random                      - Random selection (baseline)")

    algorithms = ["thompson_sampling", "linucb", "ucb1"]
    query = Query(text="Write a function to sort a list")

    for algo in algorithms:
        router = Router(algorithm=algo)
        decision = await router.route(query)
        print(
            f"\n[{algo}] Model: {decision.selected_model}, Confidence: {decision.confidence:.2f}"
        )
        await router.close()


async def demo_model_selection() -> None:
    """Demonstrate custom model selection."""
    print("\n" + "=" * 70)
    print("MODEL SELECTION - Choose which models to route between")
    print("=" * 70)

    # Default models (uses conduit.yaml configuration)
    print("\n[1] Default models (from conduit.yaml)")
    router = Router()
    decision = await router.route(Query(text="Hello world"))
    print(f"    Available models: {len(router.hybrid_router.arms)}")
    print(f"    Selected: {decision.selected_model}")
    await router.close()

    # Custom model list
    print("\n[2] Custom model list")
    router = Router(models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"])
    decision = await router.route(Query(text="Hello world"))
    print("    Models: gpt-4o-mini, gpt-4o, claude-sonnet-4-20250514")
    print(f"    Selected: {decision.selected_model}")
    await router.close()


async def main() -> None:
    """Run all routing option demos."""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    print("=" * 70)
    print("CONDUIT ROUTING OPTIONS")
    print("=" * 70)

    router = Router()

    await demo_constraints(router)
    await demo_preferences(router)
    await router.close()

    await demo_algorithms()
    await demo_model_selection()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Configuration Reference")
    print("=" * 70)
    print(
        """
QueryConstraints:
  max_cost: float        Maximum cost in dollars (e.g., 0.001)
  max_latency: float     Maximum latency in seconds (e.g., 2.0)
  min_quality: float     Minimum quality 0.0-1.0 (e.g., 0.9)
  preferred_provider: str  Provider preference (openai, anthropic, google, groq)

UserPreferences:
  optimize_for: str      Preset: balanced, quality, cost, or speed

Router:
  algorithm: str         Routing algorithm (thompson_sampling, linucb, ucb1, etc.)
  models: list[str]      Custom model list to route between
  cache_enabled: bool    Enable Redis caching (default: True)
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
