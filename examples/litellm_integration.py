"""LiteLLM Integration - ML-Powered Routing for LiteLLM.

Demonstrates how to use Conduit as an intelligent routing strategy for LiteLLM:
1. Basic usage - Route between models with the same name
2. Multi-provider - Route across OpenAI, Anthropic, Google, Groq
3. Custom configuration - Caching, hybrid routing, embeddings

Requirements:
    pip install conduit[litellm]
    OPENAI_API_KEY environment variable (or other provider keys)

Run:
    python examples/litellm_integration.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit_litellm import ConduitRoutingStrategy

# Suppress noisy feedback warnings in demo (these are expected when switching strategies)
logging.getLogger("conduit_litellm.feedback").setLevel(logging.ERROR)


async def demo_basic_usage() -> ConduitRoutingStrategy | None:
    """Basic Conduit + LiteLLM integration."""
    logger.info("\n" + "=" * 70)
    logger.info("BASIC USAGE - Route Between Models")
    logger.info("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        logger.info("Skipping: OPENAI_API_KEY not set")
        return None

    from litellm import Router

    from conduit_litellm import ConduitRoutingStrategy

    # KEY: Use same model_name for multiple deployments so Conduit can choose
    model_list = [
        {
            "model_name": "gpt",  # Shared name - Conduit picks between these
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "o4-mini"},
        },
        {
            "model_name": "gpt",  # Same name - part of routing pool
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-5"},
        },
    ]

    router = Router(model_list=model_list)
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    logger.info("Conduit routing activated")
    logger.info("Models: gpt-4o-mini (cheap), gpt-4o (capable)")

    queries = [
        ("What is 2+2?", "simple"),
        ("Explain quantum mechanics in detail", "complex"),
        ("Translate hello to Spanish", "simple"),
    ]

    logger.info("\nQuery | Selected Model")
    logger.info("-" * 50)

    for query, _query_type in queries:
        response = await router.acompletion(
            model="gpt",  # Conduit chooses optimal model
            messages=[{"role": "user", "content": query}],
        )
        short_query = query[:35] + "..." if len(query) > 35 else query
        logger.info(f"{short_query:40} | {response.model}")

    return strategy


async def demo_multi_provider() -> ConduitRoutingStrategy | None:
    """Route across multiple providers."""
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-PROVIDER - Route Across Providers")
    logger.info("=" * 70)

    # Check available providers
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available = [k for k, v in api_keys.items() if v]

    if len(available) < 2:
        logger.info(f"Skipping: Need 2+ providers (found: {', '.join(available) or 'none'})")
        logger.info("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY")
        return None

    from litellm import Router

    from conduit_litellm import ConduitRoutingStrategy

    # Build model list from available providers
    model_list = []

    if api_keys["OpenAI"]:
        model_list.extend(
            [
                {
                    "model_name": "llm",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "api_key": api_keys["OpenAI"],
                    },
                    "model_info": {"id": "o4-mini"},
                },
                {
                    "model_name": "llm",
                    "litellm_params": {
                        "model": "gpt-4o",
                        "api_key": api_keys["OpenAI"],
                    },
                    "model_info": {"id": "gpt-5"},
                },
            ]
        )

    if api_keys["Anthropic"]:
        model_list.extend(
            [
                {
                    "model_name": "llm",
                    "litellm_params": {
                        "model": "claude-sonnet-4-20250514",
                        "api_key": api_keys["Anthropic"],
                    },
                    "model_info": {"id": "claude-sonnet-4"},
                },
                {
                    "model_name": "llm",
                    "litellm_params": {
                        "model": "claude-3-5-haiku-20241022",
                        "api_key": api_keys["Anthropic"],
                    },
                    "model_info": {"id": "claude-haiku-3.5"},
                },
            ]
        )

    if api_keys["Google"]:
        model_list.append(
            {
                "model_name": "llm",
                "litellm_params": {
                    "model": "gemini/gemini-2.0-flash",
                    "api_key": api_keys["Google"],
                },
                "model_info": {"id": "gemini-2.0-flash"},
            }
        )

    if api_keys["Groq"]:
        model_list.append(
            {
                "model_name": "llm",
                "litellm_params": {
                    "model": "groq/llama-3.3-70b-versatile",
                    "api_key": api_keys["Groq"],
                },
                "model_info": {"id": "llama-3.3-70b-versatile"},
            }
        )

    logger.info(f"Found {len(available)} providers: {', '.join(available)}")
    logger.info(f"Configured {len(model_list)} models")

    router = Router(model_list=model_list)
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    logger.info("\nConduit multi-provider routing activated")
    logger.info("\nQuery | Selected Model")
    logger.info("-" * 50)

    queries = [
        "What is 2+2?",
        "Write a haiku about AI",
        "Explain photosynthesis",
    ]

    for query in queries:
        try:
            response = await router.acompletion(
                model="llm",  # Conduit selects from all providers
                messages=[{"role": "user", "content": query}],
            )
            short_query = query[:35] + "..." if len(query) > 35 else query
            logger.info(f"{short_query:40} | {response.model}")
        except Exception as e:
            short_query = query[:35] + "..." if len(query) > 35 else query
            logger.error(f"{short_query:40} | Error: {type(e).__name__}")

    return strategy


async def demo_custom_config() -> ConduitRoutingStrategy | None:
    """Custom Conduit configuration."""
    logger.info("\n" + "=" * 70)
    logger.info("CUSTOM CONFIG - Caching and Hybrid Routing")
    logger.info("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        logger.info("Skipping: OPENAI_API_KEY not set")
        return None

    from litellm import Router

    from conduit_litellm import ConduitRoutingStrategy

    model_list = [
        {
            "model_name": "gpt",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "o4-mini"},
        },
        {
            "model_name": "gpt",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-5"},
        },
    ]

    router = Router(model_list=model_list)

    # Custom configuration
    strategy = ConduitRoutingStrategy(
        cache_enabled=bool(os.getenv("REDIS_URL")),  # Enable caching if Redis available
    )
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    logger.info("Configuration:")
    logger.info("  - Hybrid routing: UCB1 -> LinUCB (30% faster convergence)")
    logger.info(
        f"  - Redis caching: {'Enabled' if os.getenv('REDIS_URL') else 'Disabled (set REDIS_URL)'}"
    )
    logger.info("  - Learning: Automatic from usage patterns")

    logger.info("\nHybrid Routing Phases:")
    logger.info("  Phase 1 (0-100 queries): UCB1 - Fast exploration")
    logger.info("  Phase 2 (100+ queries):  LinUCB - Contextual optimization")

    logger.info("\nQuery | Model | Confidence")
    logger.info("-" * 50)

    queries = [
        "Simple question: What is 2+2?",
        "Complex: Explain machine learning algorithms",
        "Creative: Write a poem",
    ]

    for query in queries:
        response = await router.acompletion(
            model="gpt",
            messages=[{"role": "user", "content": query}],
        )
        short_query = query[:30] + "..." if len(query) > 30 else query
        logger.info(f"{short_query:35} | {response.model}")

    return strategy


async def main() -> None:
    """Run all LiteLLM integration demos."""
    logger.info("=" * 70)
    logger.info("CONDUIT + LITELLM INTEGRATION")
    logger.info("=" * 70)
    logger.info("\nConduit provides ML-powered routing for LiteLLM.")
    logger.info("It learns which model works best for each query type.")

    # Run demos with proper cleanup between each
    # Each demo creates its own strategy with isolated model registry
    strategy1 = await demo_basic_usage()
    if strategy1:
        strategy1.cleanup()

    strategy2 = await demo_multi_provider()
    if strategy2:
        strategy2.cleanup()

    strategy3 = await demo_custom_config()
    if strategy3:
        strategy3.cleanup()

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY - LiteLLM Integration")
    logger.info("=" * 70)
    logger.info(
        """
Setup:
    pip install conduit[litellm]

    from litellm import Router
    from conduit_litellm import ConduitRoutingStrategy

    # KEY: Use same model_name for all models Conduit should route between
    model_list = [
        {"model_name": "llm", "litellm_params": {"model": "gpt-4o-mini"}, ...},
        {"model_name": "llm", "litellm_params": {"model": "gpt-4o"}, ...},
    ]

    router = Router(model_list=model_list)
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    # Conduit automatically routes to optimal model
    response = await router.acompletion(model="llm", messages=[...])

Features:
    - Automatic model selection based on query features
    - Multi-provider routing (OpenAI, Anthropic, Google, Groq)
    - Hybrid UCB1 -> LinUCB for 30% faster convergence
    - Optional Redis caching (set REDIS_URL)
    - Learns from usage patterns over time
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
