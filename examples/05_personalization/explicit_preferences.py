"""User preferences example.

This example demonstrates how to use explicit user preferences to control
routing optimization priorities (cost, speed, quality, or balanced).

User preferences allow you to customize the routing behavior for different
use cases without changing the ML model. The system learns which models
are best for each preference profile.
"""

import asyncio
import logging
import os
from pydantic import BaseModel

from conduit.core.models import Query, UserPreferences
from conduit.utils.service_factory import create_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleResult(BaseModel):
    """Simple result type for examples."""

    content: str


async def main() -> None:
    """Run user preferences example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        exit(1)

    print("Initializing Conduit routing service...")
    service = await create_service(default_result_type=SimpleResult)

    # Example 1: Cost-optimized routing
    print("\n" + "=" * 60)
    print("Example 1: Cost-Optimized Routing")
    print("=" * 60)
    print("Priority: Minimize cost (50% cost, 40% quality, 10% latency)")

    try:
        query_cost = Query(
            text="Summarize the main benefits of renewable energy",
            user_id="cost_conscious_user",
            preferences=UserPreferences(optimize_for="cost"),
        )
        result_cost = await service.complete_with_query(query_cost)
        
        print(f"\nQuery: {query_cost.text}")
        print(f"Preference: {query_cost.preferences.optimize_for}")
        print(f"Selected Model: {result_cost.model}")
        print(f"Cost: ${result_cost.metadata.get('cost', 0.0):.6f}")
        print(f"Latency: {result_cost.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_cost.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Response preview: {result_cost.data.get('content', '')[:150]}...")
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        print(f"Error: {e}")

    # Example 2: Speed-optimized routing
    print("\n" + "=" * 60)
    print("Example 2: Speed-Optimized Routing")
    print("=" * 60)
    print("Priority: Minimize latency (50% latency, 40% quality, 10% cost)")

    try:
        query_speed = Query(
            text="What's the capital of France?",
            user_id="speed_user",
            preferences=UserPreferences(optimize_for="speed"),
        )
        result_speed = await service.complete_with_query(query_speed)
        
        print(f"\nQuery: {query_speed.text}")
        print(f"Preference: {query_speed.preferences.optimize_for}")
        print(f"Selected Model: {result_speed.model}")
        print(f"Cost: ${result_speed.metadata.get('cost', 0.0):.6f}")
        print(f"Latency: {result_speed.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_speed.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Response: {result_speed.data.get('content', '')}")
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
        print(f"Error: {e}")

    # Example 3: Quality-optimized routing
    print("\n" + "=" * 60)
    print("Example 3: Quality-Optimized Routing")
    print("=" * 60)
    print("Priority: Maximize quality (80% quality, 10% cost, 10% latency)")

    try:
        query_quality = Query(
            text="Explain the difference between supervised and unsupervised learning "
            "in machine learning, with examples and use cases for each approach.",
            user_id="quality_user",
            preferences=UserPreferences(optimize_for="quality"),
        )
        result_quality = await service.complete_with_query(query_quality)
        
        print(f"\nQuery: {query_quality.text}")
        print(f"Preference: {query_quality.preferences.optimize_for}")
        print(f"Selected Model: {result_quality.model}")
        print(f"Cost: ${result_quality.metadata.get('cost', 0.0):.6f}")
        print(f"Latency: {result_quality.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_quality.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Response preview: {result_quality.data.get('content', '')[:200]}...")
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
        print(f"Error: {e}")

    # Example 4: Balanced routing (default)
    print("\n" + "=" * 60)
    print("Example 4: Balanced Routing (Default)")
    print("=" * 60)
    print("Priority: Balanced optimization (70% quality, 20% cost, 10% latency)")

    try:
        # When not specified, defaults to balanced
        query_balanced = Query(
            text="What are the key principles of sustainable software development?",
            user_id="balanced_user",
            # preferences defaults to UserPreferences(optimize_for="balanced")
        )
        result_balanced = await service.complete_with_query(query_balanced)
        
        print(f"\nQuery: {query_balanced.text}")
        print(f"Preference: {query_balanced.preferences.optimize_for} (default)")
        print(f"Selected Model: {result_balanced.model}")
        print(f"Cost: ${result_balanced.metadata.get('cost', 0.0):.6f}")
        print(f"Latency: {result_balanced.metadata.get('latency', 0.0):.2f}s")
        print(f"Confidence: {result_balanced.metadata.get('routing_confidence', 0.0):.2f}")
        print(f"Response preview: {result_balanced.data.get('content', '')[:200]}...")
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")
        print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: How Preferences Work")
    print("=" * 60)
    print("""
User preferences adjust the reward function weights used by the bandit algorithms
to learn which models perform best for different optimization goals:

1. **Cost** (optimize_for="cost"):
   - 50% weight on cost, 40% on quality, 10% on latency
   - Prefers cheaper models, learns which ones maintain acceptable quality

2. **Speed** (optimize_for="speed"):
   - 50% weight on latency, 40% on quality, 10% on cost
   - Prefers faster models, learns which ones respond quickest

3. **Quality** (optimize_for="quality"):
   - 80% weight on quality, 10% on cost, 10% on latency
   - Prefers highest quality models, less concerned about cost/speed

4. **Balanced** (optimize_for="balanced", default):
   - 70% weight on quality, 20% on cost, 10% on latency
   - Good default that balances all three factors

The system learns from feedback which models best satisfy each preference profile,
enabling personalized routing without manual model selection.
    """)

    # Cleanup
    await service.database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
