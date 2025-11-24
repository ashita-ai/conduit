"""Multi-provider routing with Conduit ML optimization.

Demonstrates Conduit's strength: learning optimal routing across 100+ providers.

This example shows routing across 5 different LLM providers:
- OpenAI (gpt-4o-mini)
- Anthropic (claude-3-haiku)
- Google (gemini-pro)
- Groq (llama-3.1-8b, mixtral-8x7b)

Conduit learns:
- Which provider is best for different query types
- Cost/quality/latency trade-offs per provider
- Optimal model selection based on context

Requirements:
- pip install conduit[litellm]
- API keys for providers you want to use (at least 2):
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - GOOGLE_API_KEY (or GEMINI_API_KEY)
  - GROQ_API_KEY
"""

import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate multi-provider routing with ML optimization."""

    # Check which providers are available
    providers = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available = [name for name, key in providers.items() if key]

    if len(available) < 2:
        logger.error(
            f"Need at least 2 providers. Available: {available}\n"
            f"Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY"
        )
        return

    try:
        from litellm import Router

        from conduit_litellm import ConduitRoutingStrategy
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}\nInstall: pip install conduit[litellm]")
        return

    logger.info("=" * 80)
    logger.info("Conduit Multi-Provider Routing Demo")
    logger.info("=" * 80)
    logger.info(f"Available providers: {', '.join(available)}")
    logger.info("")

    # Build model list based on available providers
    model_list = []

    if "OpenAI" in available:
        model_list.append({
            "model_name": "openai-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        })

    if "Anthropic" in available:
        model_list.append({
            "model_name": "anthropic-haiku",
            "litellm_params": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
        })

    if "Google" in available:
        model_list.append({
            "model_name": "google-gemini",
            "litellm_params": {
                "model": "gemini-1.5-flash",
                "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            },
        })

    if "Groq" in available:
        # Add 2 Groq models for variety
        model_list.extend([
            {
                "model_name": "groq-llama",
                "litellm_params": {
                    "model": "groq/llama-3.1-8b-instant",
                    "api_key": os.getenv("GROQ_API_KEY"),
                },
            },
            {
                "model_name": "groq-mixtral",
                "litellm_params": {
                    "model": "groq/mixtral-8x7b-32768",
                    "api_key": os.getenv("GROQ_API_KEY"),
                },
            },
        ])

    logger.info(f"Configured {len(model_list)} models across {len(available)} providers")
    logger.info("")

    # Create router with all available models
    router = Router(model_list=model_list)

    # Enable Conduit ML routing
    logger.info("Enabling Conduit ML routing...")
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    # Diverse test queries to see routing decisions
    queries = [
        "What is 2+2?",  # Simple math
        "Write a Python function to sort a list",  # Code
        "Explain quantum entanglement in simple terms",  # Complex explanation
        "Translate 'hello' to Spanish",  # Translation
        "What is the capital of France?",  # Factual
        "Write a haiku about coding",  # Creative
        "Debug this code: def foo(): return x",  # Technical
        "What are the best practices for REST APIs?",  # Technical knowledge
    ]

    logger.info(f"Testing with {len(queries)} diverse queries...\n")

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}/{len(queries)}: {query[:60]}...")

        try:
            # Use first model group as default (Conduit picks the deployment)
            response = await router.acompletion(
                model=model_list[0]["model_name"],
                messages=[{"role": "user", "content": query}],
                timeout=30,
            )

            model_used = response.model
            cost = response._hidden_params.get("response_cost", 0)
            content = response.choices[0].message.content

            # Determine provider from model
            provider = "Unknown"
            if "gpt" in model_used.lower():
                provider = "OpenAI"
            elif "claude" in model_used.lower():
                provider = "Anthropic"
            elif "gemini" in model_used.lower():
                provider = "Google"
            elif "llama" in model_used.lower() or "mixtral" in model_used.lower():
                provider = "Groq"

            logger.info(f"  → Provider: {provider}")
            logger.info(f"  → Model: {model_used}")
            logger.info(f"  → Cost: ${cost:.6f}")
            logger.info(f"  → Response: {content[:60]}...")
            logger.info("")

            results.append({"query": query, "provider": provider, "cost": cost})

        except Exception as e:
            logger.warning(f"  ⚠️  Failed: {e}")
            logger.info("")

    # Show routing distribution
    logger.info("=" * 80)
    logger.info("Routing Distribution:")
    logger.info("=" * 80)

    provider_counts = {}
    total_cost = 0
    for result in results:
        provider = result["provider"]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        total_cost += result["cost"]

    for provider in sorted(provider_counts.keys()):
        count = provider_counts[provider]
        pct = (count / len(results)) * 100
        logger.info(f"  {provider}: {count}/{len(results)} queries ({pct:.1f}%)")

    logger.info(f"\n  Total cost: ${total_cost:.6f}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("Key Insights:")
    logger.info("=" * 80)
    logger.info("✅ Conduit routes across multiple providers automatically")
    logger.info("✅ Learns cost/quality/latency trade-offs per provider")
    logger.info("✅ Adapts routing based on query context")
    logger.info("✅ No manual configuration needed - ML does the work")
    logger.info("")
    logger.info("With more queries, Conduit will:")
    logger.info("  - Identify which providers excel at code vs creative tasks")
    logger.info("  - Learn speed/cost trade-offs per provider")
    logger.info("  - Optimize routing for your specific workload")
    logger.info("")

    # Clean up
    strategy.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
