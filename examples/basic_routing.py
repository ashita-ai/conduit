"""Basic routing example using Conduit.

This example demonstrates simple query routing with ML-powered model selection.
"""

import asyncio
import os
from pydantic import BaseModel
from conduit.engines.router import Router
from conduit.core.models import Query


class AnalysisResult(BaseModel):
    """Example result type for structured outputs."""

    answer: str
    confidence: float
    reasoning: str


async def main() -> None:
    """Run basic routing example."""
    # Initialize router with available models
    router = Router()

    # Create a simple query
    query = Query(
        text="What is 2+2?",
        user_id="example_user",
    )

    print(f"Query: {query.text}")
    print("\nRouting to optimal model...")

    # Route query to best model
    result = await router.route(query)

    print(f"\nSelected Model: {result.selected_model}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"\nFeatures:")
    print(f"  - Token count: {result.features.token_count}")
    print(f"  - Complexity: {result.features.complexity_score:.2f}")
    print(f"  - Domain: {result.features.domain} ({result.features.domain_confidence:.2f})")


if __name__ == "__main__":
    # Ensure environment is configured
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        exit(1)

    asyncio.run(main())
