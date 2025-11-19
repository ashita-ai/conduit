"""Simple router example using the new Router class.

This example demonstrates the basic Router interface as shown in the README.
"""

import asyncio
import os
from conduit.engines import Router
from conduit.core.models import Query


async def main() -> None:
    """Run simple router example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    print("Initializing Conduit Router...")
    router = Router()

    # Create a simple query
    query = Query(text="What is 2+2?")

    print(f"\nQuery: {query.text}")
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
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
