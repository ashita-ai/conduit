"""Routing with constraints example.

This example demonstrates routing with cost, latency, and quality constraints.

Example usage:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info(f"Query: {query_cheap.text}")
"""

import asyncio
import logging
import os

from conduit.core.models import Query, QueryConstraints, UserPreferences
from conduit.engines.router import Router

logger = logging.getLogger(__name__)


async def main() -> None:
    """Run constrained routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        logger.info("Example: export OPENAI_API_KEY=sk-...")
        return

    logger.info("Initializing Conduit...")
    router = Router()

    # Example 1: Cost-constrained query
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Cost-Constrained Routing")
    logger.info("=" * 60)

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
        logger.info(f"Query: {query_cheap.text}")
        logger.info(f"Constraints: max_cost=${constraints_cheap.max_cost}, min_quality={constraints_cheap.min_quality}")
        logger.info(f"Selected Model: {decision.selected_model}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        logger.info(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        logger.error(f"Error: {e}")

    # Example 2: Quality-constrained query
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Quality-Constrained Routing")
    logger.info("=" * 60)

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
        logger.info(f"Query: {query_quality.text[:50]}...")
        logger.info(f"Constraints: min_quality={constraints_quality.min_quality}, max_latency={constraints_quality.max_latency}s")
        logger.info(f"Selected Model: {decision.selected_model}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        logger.info(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        logger.error(f"Error: {e}")

    # Example 3: Speed-optimized query
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Speed-Optimized Routing")
    logger.info("=" * 60)

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
        logger.info(f"Query: {query_fast.text}")
        logger.info(f"Constraints: max_latency={constraints_fast.max_latency}s, preferred_provider={constraints_fast.preferred_provider}")
        logger.info(f"Selected Model: {decision.selected_model}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        logger.info(f"Reasoning: {decision.reasoning}")
    except Exception as e:
        logger.error(f"Error: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Constraint Types Summary")
    logger.info("=" * 60)
    logger.info("\nQueryConstraints fields:")
    logger.info("  - max_cost: Maximum cost in dollars (e.g., 0.001)")
    logger.info("  - max_latency: Maximum latency in seconds (e.g., 2.0)")
    logger.info("  - min_quality: Minimum quality score 0.0-1.0 (e.g., 0.9)")
    logger.info("  - preferred_provider: Provider preference (openai, anthropic, google, groq)")

    logger.info("\nUserPreferences optimize_for options:")
    logger.info("  - balanced: Default (70% quality, 20% cost, 10% latency)")
    logger.info("  - quality: Maximize quality (80% quality, 10% cost, 10% latency)")
    logger.info("  - cost: Minimize cost (40% quality, 50% cost, 10% latency)")
    logger.info("  - speed: Minimize latency (40% quality, 10% cost, 50% latency)")

    # Cleanup
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
