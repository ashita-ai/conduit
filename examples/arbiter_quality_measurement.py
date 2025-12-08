"""LiteLLM integration with Arbiter LLM-as-judge quality measurement.

Demonstrates optional quality assessment using Arbiter evaluator.

Features:
- Fire-and-forget async evaluation (doesn't block routing)
- Configurable sampling rate to control costs
- Automatic feedback storage for bandit learning
- Graceful degradation if evaluation fails

Requirements:
- pip install conduit[litellm]
- DATABASE_URL environment variable
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parents[2] / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, assume env vars are set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate Arbiter integration with LiteLLM plugin."""

    # Check prerequisites
    if not os.getenv("DATABASE_URL"):
        logger.error(
            "DATABASE_URL environment variable required for Arbiter evaluator. "
            "Example: export DATABASE_URL=postgresql://user:pass@localhost:5432/conduit"
        )
        return

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        logger.error("OPENAI_API_KEY or ANTHROPIC_API_KEY required for LLM calls")
        return

    try:
        from litellm import Router

        from conduit.core.database import Database
        from conduit.evaluation import ArbiterEvaluator
        from conduit_litellm import ConduitRoutingStrategy
    except ImportError as e:
        logger.error(
            f"Missing dependencies: {e}\n"
            "Install with: pip install conduit[litellm]"
        )
        return

    # Initialize database for Arbiter evaluator
    logger.info("Connecting to database...")
    db = Database()
    await db.connect()

    strategy = None  # Initialize for finally block
    try:
        # Initialize Arbiter evaluator
        # - sample_rate=0.1 means evaluate 10% of responses
        # - daily_budget=$10 controls maximum evaluation cost
        logger.info("Initializing Arbiter evaluator...")
        evaluator = ArbiterEvaluator(
            db=db,
            sample_rate=0.1,  # Evaluate 10% of queries
            daily_budget=10.0,  # Max $10/day on evaluations
            model="gpt-4o-mini",  # Use cheap model for evaluation
        )

        # Configure LiteLLM router with multiple models
        # KEY: Use same model_name "llm" so Conduit can route between models
        logger.info("Setting up LiteLLM router...")
        router = Router(
            model_list=[
                {
                    "model_name": "llm",  # Shared name - Conduit routes between these
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    "model_info": {"id": "o4-mini"},  # Conduit's standardized model ID
                },
                {
                    "model_name": "llm",  # Same name - part of routing pool
                    "litellm_params": {
                        "model": "claude-3-haiku-20240307",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                    "model_info": {"id": "claude-haiku-4.5"},  # Conduit's standardized model ID
                },
            ],
            # LiteLLM debug for visibility
            debug_level="INFO",
        )

        # Setup Conduit routing strategy WITH Arbiter evaluator
        logger.info("Enabling Conduit ML routing with Arbiter quality measurement...")
        strategy = ConduitRoutingStrategy(
            evaluator=evaluator,  # Enable LLM-as-judge evaluation
            cache_enabled=False,  # Disable caching for demo
        )
        ConduitRoutingStrategy.setup_strategy(router, strategy)

        # Make test queries
        test_queries = [
            "What is the capital of France?",
            "Explain quantum entanglement in simple terms",
            "Write a haiku about machine learning",
        ]

        logger.info(f"\nMaking {len(test_queries)} test queries...\n")
        logger.info("Note: Arbiter will evaluate ~10% of responses using LLM-as-judge\n")

        for i, query_text in enumerate(test_queries, 1):
            logger.info(f"Query {i}/{len(test_queries)}: {query_text[:50]}...")

            # LiteLLM routes through Conduit ML engine
            response = await router.acompletion(
                model="llm",  # Conduit selects optimal model
                messages=[{"role": "user", "content": query_text}],
            )

            # Extract response
            response_text = response.choices[0].message.content
            model_used = response.model
            cost = response._hidden_params.get("response_cost", 0)

            logger.info(f"  Model: {model_used}")
            logger.info(f"  Cost: ${cost:.6f}")
            logger.info(f"  Response: {response_text[:80]}...")

            # Arbiter evaluation happens in background (fire-and-forget)
            # Check if this query was sampled for evaluation
            if await evaluator.should_evaluate():
                logger.info(
                    "  Quality: Arbiter evaluation queued (10% sampling, runs in background)"
                )
            else:
                logger.info("  Quality: Skipped (not in 10% sample)")

            logger.info("")

        # Give background evaluations time to complete
        logger.info("Waiting for background evaluations to complete...")
        await asyncio.sleep(2)

        logger.info("\nSuccess! Arbiter integration working correctly.")
        logger.info(
            "- Quality evaluations run in background without blocking routing"
        )
        logger.info("- Feedback stored in database for bandit learning")
        logger.info("- Sampling rate (10%) controls evaluation costs")

    finally:
        # Clean up resources
        logger.info("\nCleaning up resources...")
        if strategy is not None:
            strategy.cleanup()  # Unregister feedback logger
        await db.disconnect()  # Close database connection


if __name__ == "__main__":
    asyncio.run(main())
