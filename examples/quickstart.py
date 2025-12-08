"""Quickstart - Basic Conduit Routing.

The simplest way to use Conduit for ML-powered LLM routing.

Requirements:
    OPENAI_API_KEY or ANTHROPIC_API_KEY in environment

Run:
    python examples/quickstart.py

Expected output:
    Routing query: "What is 2+2?"
    Selected: gpt-4o-mini (confidence: 73%)
    Features: 8 tokens, complexity 0.25
"""

import asyncio
import os

from conduit.core.models import Query
from conduit.engines.router import Router


async def main() -> None:
    """Basic routing example - 10 essential lines."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    # Core routing - 5 lines
    router = Router()
    query = Query(text="What is 2+2?")
    decision = await router.route(query)

    # Show results
    print(f'Routing query: "{query.text}"')
    print(
        f"Selected: {decision.selected_model} (confidence: {decision.confidence:.0%})"
    )
    print(
        f"Features: {decision.features.token_count} tokens, complexity {decision.features.complexity_score:.2f}"
    )

    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
