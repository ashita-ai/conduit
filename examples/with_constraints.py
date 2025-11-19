"""Routing with constraints example.

This example demonstrates routing with cost, latency, and quality constraints.
"""

import asyncio
import logging
import os
from pydantic import BaseModel

from conduit.utils.service_factory import create_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleResult(BaseModel):
    """Simple result type for examples."""

    content: str


async def main() -> None:
    """Run constrained routing example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    print("Initializing Conduit routing service...")
    service = await create_service(default_result_type=SimpleResult)

    # Example 1: Cost-constrained query
    print("\n" + "=" * 60)
    print("Example 1: Cost-Constrained Routing")
    print("=" * 60)

    try:
        result_cheap = await service.complete(
            prompt="Write a short poem about the ocean",
            user_id="budget_user",
            constraints={
                "max_cost": 0.001,  # Prefer cheaper models
                "min_quality": 0.7,  # But maintain reasonable quality
            },
        )
        print(f"Query: Write a short poem about the ocean")
        print(f"Selected Model: {result_cheap.model}")
        print(f"Cost: ${result_cheap.metadata.get('cost', 0.0):.6f}")
        print(f"Confidence: {result_cheap.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Reasoning: {result_cheap.metadata.get('reasoning', 'N/A')}")
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        print(f"Error: {e}")

    # Example 2: Quality-constrained query
    print("\n" + "=" * 60)
    print("Example 2: Quality-Constrained Routing")
    print("=" * 60)

    try:
        result_quality = await service.complete(
            prompt="Explain quantum entanglement in detail with mathematical formulations",
            user_id="quality_user",
            constraints={
                "min_quality": 0.95,  # Require high quality
                "max_latency": 10.0,  # But reasonable latency
            },
        )
        print(f"Query: Explain quantum entanglement...")
        print(f"Selected Model: {result_quality.model}")
        print(f"Latency: {result_quality.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_quality.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Reasoning: {result_quality.metadata.get('reasoning', 'N/A')}")
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
        print(f"Error: {e}")

    # Example 3: Latency-constrained query
    print("\n" + "=" * 60)
    print("Example 3: Latency-Constrained Routing")
    print("=" * 60)

    try:
        result_fast = await service.complete(
            prompt="Quick: What's the capital of France?",
            user_id="speed_user",
            constraints={
                "max_latency": 2.0,  # Fast response required
                "preferred_provider": "openai",  # Prefer fast provider
            },
        )
        print(f"Query: Quick: What's the capital of France?")
        print(f"Selected Model: {result_fast.model}")
        print(f"Latency: {result_fast.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_fast.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Reasoning: {result_fast.metadata.get('reasoning', 'N/A')}")
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
        print(f"Error: {e}")

    # Cleanup
    await service.database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
