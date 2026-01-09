"""Basic routing example using Conduit.

This example demonstrates simple query routing with ML-powered model selection.

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Expected output:
    Initializing Conduit...

    Query: What is 2+2? Explain your reasoning.

    Routing to optimal model...

    ============================================================
    Routing Results
    ============================================================
    Selected Model: o4-mini
    Confidence: 0.73
    Reasoning: UCB selection with exploration bonus

    Features:
      - Token count: 12
      - Complexity: 0.25
"""

import asyncio
import logging
import os

from conduit.core.models import Query
from conduit.engines.router import Router

logger = logging.getLogger(__name__)

async def main() -> None:
    """Run basic routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        logger.info("Example: export OPENAI_API_KEY=sk-...")
        return

    logger.info("Initializing Conduit...")
    router = Router()

    # Create a simple query
    query_text = "What is 2+2? Explain your reasoning."
    query = Query(text=query_text)

    logger.info(f"\nQuery: {query_text}")
    logger.info("\nRouting to optimal model...")

    try:
        # Route query to optimal model
        decision = await router.route(query)

        logger.info(f"\n{'='*60}")
        logger.info("Routing Results")
        logger.info(f"{'='*60}")
        logger.info(f"Selected Model: {decision.selected_model}")
        logger.info(f"Confidence: {decision.confidence:.2f}")
        logger.info(f"Reasoning: {decision.reasoning}")
        logger.info(f"\nFeatures:")
        logger.info(f"  - Token count: {decision.features.token_count}")
        logger.info(f"  - Complexity: {decision.features.complexity_score:.2f}")

    except Exception as e:
        logger.error(f"\nError: {e}")
        return
    finally:
        # Cleanup
        await router.close()

if __name__ == "__main__":
    asyncio.run(main())
