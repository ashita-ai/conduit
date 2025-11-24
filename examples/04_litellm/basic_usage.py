"""Basic LiteLLM integration with Conduit ML routing.

Demonstrates the simplest possible setup - just 3 steps:
1. Create LiteLLM Router with model list
2. Create ConduitRoutingStrategy
3. Make requests (Conduit learns automatically)

Features:
- Automatic bandit learning from every request
- Cost and latency tracking
- Zero manual feedback required
- Works with any LiteLLM-supported provider

Requirements:
- pip install conduit[litellm]
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate basic Conduit + LiteLLM integration."""

    # Check prerequisites
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        logger.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        from litellm import Router

        from conduit_litellm import ConduitRoutingStrategy
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}\nInstall: pip install conduit[litellm]")
        return

    logger.info("=" * 80)
    logger.info("Conduit + LiteLLM: Basic Usage Demo")
    logger.info("=" * 80)
    logger.info("")

    # Step 1: Create LiteLLM Router with your models
    logger.info("Step 1: Creating LiteLLM router with 2 models...")
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            },
        ]
    )

    # Step 2: Enable Conduit ML routing
    logger.info("Step 2: Enabling Conduit ML routing strategy...")
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    logger.info("Step 3: Making requests (Conduit learns from each one)...")
    logger.info("")

    # Step 3: Make requests - Conduit learns automatically!
    queries = [
        "What is 2+2?",
        "Explain quantum computing in simple terms",
        "Write a Python function to reverse a string",
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}/{len(queries)}: {query[:50]}...")

        response = await router.acompletion(
            model="gpt-4o-mini",  # Model group (Conduit picks specific deployment)
            messages=[{"role": "user", "content": query}],
        )

        # Extract response details
        model_used = response.model
        content = response.choices[0].message.content
        cost = response._hidden_params.get("response_cost", 0)

        logger.info(f"  Model: {model_used}")
        logger.info(f"  Cost: ${cost:.6f}")
        logger.info(f"  Response: {content[:80]}...")
        logger.info("")

        # Feedback is captured automatically - no manual work needed!
        # Conduit learns:
        # - Which model was used
        # - How much it cost
        # - How long it took
        # - Estimated quality from response content

    logger.info("=" * 80)
    logger.info("Success! Key Points:")
    logger.info("=" * 80)
    logger.info("✅ Conduit automatically learns from every request")
    logger.info("✅ No manual feedback required (cost/latency tracked automatically)")
    logger.info("✅ Quality estimated from response content")
    logger.info("✅ Bandits improve routing over time (exploration → exploitation)")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("- Try custom_config.py to configure bandit algorithms")
    logger.info("- Try multi_provider.py to route across 5+ providers")
    logger.info("- Try arbiter_quality_measurement.py for LLM-as-judge evaluation")
    logger.info("")

    # Clean up
    strategy.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
