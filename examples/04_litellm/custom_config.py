"""Advanced LiteLLM configuration with custom Conduit settings.

Demonstrates how to customize Conduit's ML routing behavior:
- Cache configuration (Redis)
- Custom embedding models
- Explicit Router pre-configuration
- Resource cleanup

Use Cases:
- Production deployments requiring caching
- Custom embedding models for domain-specific routing
- Fine-tuning exploration/exploitation trade-offs

Requirements:
- pip install conduit[litellm]
- OPENAI_API_KEY or ANTHROPIC_API_KEY
- Redis (optional, for caching)
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate custom Conduit configuration with LiteLLM."""

    # Check prerequisites
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        logger.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        from litellm import Router

        from conduit.engines.router import Router as ConduitRouter
        from conduit_litellm import ConduitRoutingStrategy
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}\nInstall: pip install conduit[litellm]")
        return

    logger.info("=" * 80)
    logger.info("Conduit + LiteLLM: Custom Configuration Demo")
    logger.info("=" * 80)
    logger.info("")

    # Configure LiteLLM Router
    logger.info("Creating LiteLLM router...")
    litellm_router = Router(
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

    # Option 1: Use conduit_config to pass settings
    logger.info("\n=== Option 1: Configure via conduit_config ===")
    logger.info("Passing configuration to auto-created Router...")

    cache_enabled = os.getenv("REDIS_URL") is not None

    strategy1 = ConduitRoutingStrategy(
        # Conduit configuration passed through
        cache_enabled=cache_enabled,  # Enable Redis caching
        embedding_model="all-MiniLM-L6-v2",  # Fast, lightweight embeddings
    )

    logger.info(f"  Cache enabled: {cache_enabled}")
    logger.info(f"  Embedding model: all-MiniLM-L6-v2 (384 dimensions)")
    logger.info("")

    # Option 2: Pre-configure Conduit Router with full control
    logger.info("=== Option 2: Pre-configured Conduit Router ===")
    logger.info("Creating Conduit Router with explicit settings...")

    # Create Conduit Router with explicit configuration
    conduit_router = ConduitRouter(
        models=["gpt-4o-mini", "claude-3-haiku-20240307"],
        embedding_model="all-MiniLM-L6-v2",
        cache_enabled=cache_enabled,
    )

    # Pass pre-configured router to strategy
    strategy2 = ConduitRoutingStrategy(conduit_router=conduit_router)

    logger.info("  Using pre-configured Router")
    logger.info(f"  Models: {len(conduit_router.hybrid_router.bandit.arms)} arms")
    logger.info(f"  Feature dimensions: {conduit_router.analyzer.feature_dim}")
    logger.info("")

    # Use Option 2 for demonstration
    logger.info("=== Setting up ML routing with custom config ===")
    ConduitRoutingStrategy.setup_strategy(litellm_router, strategy2)

    # Make test queries
    logger.info("Making test queries...\n")

    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How do transformers work?",
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}/{len(queries)}: {query}")

        response = await litellm_router.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
        )

        model_used = response.model
        cost = response._hidden_params.get("response_cost", 0)

        logger.info(f"  → Model: {model_used}")
        logger.info(f"  → Cost: ${cost:.6f}\n")

    # Check cache stats (if enabled)
    if cache_enabled and conduit_router.cache:
        stats = conduit_router.get_cache_stats()
        logger.info("=" * 80)
        logger.info("Cache Statistics:")
        logger.info("=" * 80)
        logger.info(f"  Hits: {stats['hits']}")
        logger.info(f"  Misses: {stats['misses']}")
        logger.info(f"  Hit rate: {stats['hit_rate']:.1f}%")
        logger.info(f"  Circuit state: {stats['circuit_state']}")
        logger.info("")

    logger.info("=" * 80)
    logger.info("Configuration Options Summary:")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Cache Settings:")
    logger.info("  cache_enabled: Enable Redis caching (default: False)")
    logger.info("  redis_url: Redis connection URL")
    logger.info("")
    logger.info("Embedding Models:")
    logger.info("  all-MiniLM-L6-v2: Fast, 384 dims (default)")
    logger.info("  all-mpnet-base-v2: Better quality, 768 dims")
    logger.info("  paraphrase-multilingual: Multi-language, 384 dims")
    logger.info("")
    logger.info("Router Pre-configuration:")
    logger.info("  Pass conduit_router= for full control")
    logger.info("  Pass **conduit_config for auto-creation")
    logger.info("")

    # Clean up
    logger.info("Cleaning up resources...")
    strategy2.cleanup()
    if conduit_router.cache:
        await conduit_router.close()


if __name__ == "__main__":
    asyncio.run(main())
