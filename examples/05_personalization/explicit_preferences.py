"""User Preferences Example.

Demonstrates how to use explicit user preferences to control routing behavior.
Conduit supports 4 optimization presets: balanced, quality, cost, and speed.
"""

import asyncio
from conduit.core import Query, UserPreferences
from conduit.engines import Router


async def main() -> None:
    """Demonstrate routing with different user preferences."""
    print("🎯 User Preferences Example\n")
    print("Conduit supports 4 optimization presets:")
    print("  - balanced: Default (70% quality, 20% cost, 10% latency)")
    print("  - quality:  Maximize quality (80% quality, 10% cost, 10% latency)")
    print("  - cost:     Minimize cost (40% quality, 50% cost, 10% latency)")
    print("  - speed:    Minimize latency (40% quality, 10% cost, 50% latency)\n")

    # Initialize router (no special setup needed for preferences)
    router = Router(
        models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"],
        cache_enabled=False  # Disable caching for clearer routing demonstration
    )

    # Test queries demonstrating each preference
    test_cases = [
        ("balanced", "What is the capital of France?"),
        ("quality", "Explain the theory of relativity in detail."),
        ("cost", "What is 2+2?"),
        ("speed", "Quick: What time is it in Tokyo?"),
    ]

    for optimize_for, query_text in test_cases:
        print(f"\n{'='*70}")
        print(f"Preference: {optimize_for}")
        print(f"Query: {query_text}")
        print(f"{'='*70}")

        # Create query with specific preference
        query = Query(
            text=query_text,
            preferences=UserPreferences(optimize_for=optimize_for)  # type: ignore[arg-type]
        )

        # Route query
        decision = await router.route(query)

        # Show results
        print(f"✓ Selected Model: {decision.selected_model}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Reward Weights:")
        print(f"    - Quality: {router.hybrid_router.ucb1.reward_weights['quality']:.1f}")
        print(f"    - Cost: {router.hybrid_router.ucb1.reward_weights['cost']:.1f}")
        print(f"    - Latency: {router.hybrid_router.ucb1.reward_weights['latency']:.1f}")

    # Demonstrate default behavior (balanced)
    print(f"\n{'='*70}")
    print("Default Behavior (no preference specified → balanced)")
    print(f"{'='*70}")

    default_query = Query(text="Tell me about machine learning.")
    default_decision = await router.route(default_query)

    print(f"✓ Selected Model: {default_decision.selected_model}")
    print(f"  Confidence: {default_decision.confidence:.2f}")
    print(f"  Reward Weights (default to balanced):")
    print(f"    - Quality: {router.hybrid_router.ucb1.reward_weights['quality']:.1f}")
    print(f"    - Cost: {router.hybrid_router.ucb1.reward_weights['cost']:.1f}")
    print(f"    - Latency: {router.hybrid_router.ucb1.reward_weights['latency']:.1f}")

    # Custom configuration via conduit.yaml
    print(f"\n{'='*70}")
    print("Custom Configuration")
    print(f"{'='*70}")
    print("\nYou can customize preset weights in conduit.yaml:")
    print("""
routing:
  presets:
    cost:
      quality: 0.3  # Even lower quality tolerance
      cost: 0.6     # Even higher cost focus
      latency: 0.1

    custom_preset:
      quality: 0.5
      cost: 0.3
      latency: 0.2
""")
    print("See conduit.yaml in the project root for the full configuration.")

    # Cleanup
    await router.close()
    print("\n✨ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
