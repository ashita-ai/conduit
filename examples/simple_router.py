"""Simple router example using the new Router class.

This example demonstrates the basic Router interface as shown in the README.

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Expected output:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Initializing Conduit...")
    >>> logger.info(f"Query: {query.text}")
    >>> logger.info("Routing to optimal model...")
    >>> logger.info(f"Selected Model: {decision.selected_model}")
    >>> logger.info(f"Confidence: {decision.confidence:.2f}")
    >>> logger.info(f"Reasoning: {decision.reasoning}")
    >>> logger.info(f"  - Token count: {decision.features.token_count}")
    >>> logger.info(f"  - Complexity: {decision.features.complexity_score:.2f}")
"""

import asyncio
import logging
import os
from conduit.engines import Router
from conduit.core.models import Query

logger = logging.getLogger(__name__)

async def main() -> None:
    """Run simple router example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        logger.info("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    logger.info("Initializing Conduit...")
    router = Router()

    # Create a simple query
    query = Query(text="What is 2+2?")

    logger.info(f"\nQuery: {query.text}")
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
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
