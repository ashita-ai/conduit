"""Quickstart - Basic Conduit Routing.

The simplest way to use Conduit for ML-powered LLM routing.

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Run:
    python examples/quickstart.py

Expected output:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info(f'Routing query: "{query.text}"')
    >>> logger.info(f"Selected: {decision.selected_model} (confidence: {decision.confidence:.0%})")
    >>> logger.info(f"Features: {decision.features.token_count} tokens, complexity {decision.features.complexity_score:.2f}")
"""

import asyncio
import logging
import os

from conduit.core.models import Query
from conduit.engines.router import Router

logger = logging.getLogger(__name__)


async def main() -> None:
    """Basic routing example - 10 essential lines."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    # Core routing - 5 lines
    router = Router()
    query = Query(text="What is 2+2?")
    decision = await router.route(query)

    # Show results
    logger.info(f'Routing query: "{query.text}"')
    logger.info(
        f"Selected: {decision.selected_model} (confidence: {decision.confidence:.0%})"
    )
    logger.info(
        f"Features: {decision.features.token_count} tokens, complexity {decision.features.complexity_score:.2f}"
    )

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
