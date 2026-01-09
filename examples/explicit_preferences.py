"""User Preferences Example.

Demonstrates how to use explicit user preferences to control routing behavior.
Conduit supports 4 optimization presets: balanced, quality, cost, and speed.
"""

import asyncio
import logging

from conduit.core import Query, UserPreferences
from conduit.engines import Router

logger = logging.getLogger(__name__)


async def main() -> None:
    """Demonstrate routing with different user preferences."""
    logger.info("🎯 User Preferences Example\n")
    logger.info("Conduit supports 4 optimization presets:")
    logger.info("  - balanced: Default (70% quality, 20% cost, 10% latency)")
    logger.info("  - quality:  Maximize quality (80% quality, 10% cost, 10% latency)")
    logger.info("  - cost:     Minimize cost (40% quality, 50% cost, 10% latency)")
    logger.info("  - speed:    Minimize latency (40% quality, 10% cost, 50% latency)\n")

    # Initialize router (no special setup needed for preferences)
    router = Router(
        models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
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
        logger.info(f"\n{'='*70}")
        logger.info(f"Preference: {optimize_for}")
        logger.info(f"Query: {query_text}")
        logger.info(f"{'='*70}")

        # Create query with specific preference
        query = Query(
            text=query_text,
            preferences=UserPreferences(optimize_for=optimize_for)  # type: ignore[arg-type]
        )

        # Route query
        decision = await router.route(query)

        # Show results
        logger.info(f"✓ Selected Model: {decision.selected_model}")
        logger.info(f"  Confidence: {decision.confidence:.2f}")
        logger.info(f"  Reasoning: {decision.reasoning}")
        logger.info(f"  Reward Weights:")
        logger.info(f"    - Quality: {router.hybrid_router.ucb1.reward_weights['quality']:.1f}")
        logger.info(f"    - Cost: {router.hybrid_router.ucb1.reward_weights['cost']:.1f}")
        logger.info(f"    - Latency: {router.hybrid_router.ucb1.reward_weights['latency']:.1f}")

    # Demonstrate default behavior (balanced)
    logger.info(f"\n{'='*70}")
    logger.info("Default Behavior (no preference specified → balanced)")
    logger.info(f"{'='*70}")

    default_query = Query(text="Tell me about machine learning.")
    default_decision = await router.route(default_query)

    logger.info(f"✓ Selected Model: {default_decision.selected_model}")
    logger.info(f"  Confidence: {default_decision.confidence:.2f}")
    logger.info(f"  Reward Weights (default to balanced):")
    logger.info(f"    - Quality: {router.hybrid_router.ucb1.reward_weights['quality']:.1f}")
    logger.info(f"    - Cost: {router.hybrid_router.ucb1.reward_weights['cost']:.1f}")
    logger.info(f"    - Latency: {router.hybrid_router.ucb1.reward_weights['latency']:.1f}")

    # Custom configuration via conduit.yaml
    logger.info(f"\n{'='*70}")
    logger.info("Custom Configuration")
    logger.info(f"{'='*70}")
    logger.info("\nYou can customize preset weights in conduit.yaml:")
    logger.info("""
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
    logger.info("See conduit.yaml in the project root for the full configuration.")

    # Cleanup
    await router.close()
    logger.info("\n✨ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
