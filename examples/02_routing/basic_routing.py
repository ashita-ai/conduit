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
      - Domain: general (0.82)
"""

import asyncio
import os

from conduit.core.models import Query
from conduit.engines.router import Router

async def main() -> None:
    """Run basic routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        return

    print("Initializing Conduit...")
    router = Router()

    # Create a simple query
    query_text = "What is 2+2? Explain your reasoning."
    query = Query(text=query_text)

    print(f"\nQuery: {query_text}")
    print("\nRouting to optimal model...")

    try:
        # Route query to optimal model
        decision = await router.route(query)

        print(f"\n{'='*60}")
        print("Routing Results")
        print(f"{'='*60}")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"\nFeatures:")
        print(f"  - Token count: {decision.features.token_count}")
        print(f"  - Complexity: {decision.features.complexity_score:.2f}")
        print(f"  - Domain: {decision.features.domain} ({decision.features.domain_confidence:.2f})")

    except Exception as e:
        print(f"\nError: {e}")
        return
    finally:
        # Cleanup
        await router.close()

if __name__ == "__main__":
    asyncio.run(main())
