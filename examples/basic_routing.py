"""Basic routing example using Conduit.

This example demonstrates simple query routing with ML-powered model selection.
"""

import asyncio
import logging
import os
from pydantic import BaseModel

from conduit.utils.service_factory import create_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Example result type for structured outputs."""

    content: str


async def main() -> None:
    """Run basic routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    print("Initializing Conduit routing service...")
    service = await create_service(default_result_type=AnalysisResult)

    # Create a simple query
    query_text = "What is 2+2? Explain your reasoning."

    print(f"\nQuery: {query_text}")
    print("\nRouting to optimal model and executing...")

    try:
        # Route and execute query
        result = await service.complete(
            prompt=query_text,
            user_id="example_user",
        )

        print(f"\n{'='*60}")
        print("Routing Results")
        print(f"{'='*60}")
        print(f"Selected Model: {result.model}")
        print(f"Confidence: {result.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Reasoning: {result.metadata.get('reasoning', 'N/A')}")
        print(f"\nExecution Metrics:")
        print(f"  - Cost: ${result.metadata.get('cost', 0.0):.6f}")
        print(f"  - Latency: {result.metadata.get('latency', 0.0):.2f}s")
        print(f"  - Tokens: {result.metadata.get('tokens', 0)}")
        print(f"\nResponse:")
        print(f"  {result.data.get('content', 'N/A')}")

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        print(f"\nError: {e}")
        exit(1)
    finally:
        # Cleanup
        await service.database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
