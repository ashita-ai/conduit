"""Multi-Provider Routing with Conduit + LiteLLM.

Demonstrates intelligent routing across multiple LLM providers
(OpenAI, Anthropic, Google, Groq) using Conduit's ML-based selection.

Requirements:
    - At least 2 provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    - pip install conduit[litellm]

Run:
    python examples/04_litellm/multi_provider.py
"""

import asyncio
import logging
import os

from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

logger = logging.getLogger(__name__)


async def main() -> None:
    """Multi-provider routing example."""

    logger.info("🚀 Multi-Provider Routing with Conduit\n")

    # Check available API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available_providers = [k for k, v in api_keys.items() if v]

    if len(available_providers) < 2:
        logger.error("❌ Need at least 2 API keys for multi-provider demo")
        logger.info("   Set any combination of:")
        logger.info("   - OPENAI_API_KEY")
        logger.info("   - ANTHROPIC_API_KEY")
        logger.info("   - GOOGLE_API_KEY")
        logger.info("   - GROQ_API_KEY")
        return

    logger.info(f"✅ Found {len(available_providers)} providers: {', '.join(available_providers)}\n")

    # Configure models from available providers
    # KEY: Use same model_name "llm" for all models so Conduit can route between them
    model_list = []

    if api_keys["OpenAI"]:
        model_list.extend([
            {
                "model_name": "llm",  # Shared name - Conduit routes across all providers
                "litellm_params": {"model": "gpt-4o-mini", "api_key": api_keys["OpenAI"]},
                "model_info": {"id": "o4-mini"},  # Conduit's standardized model ID
            },
            {
                "model_name": "llm",  # Same name - part of routing pool
                "litellm_params": {"model": "gpt-4o", "api_key": api_keys["OpenAI"]},
                "model_info": {"id": "gpt-5"},  # Conduit's standardized model ID
            },
        ])

    if api_keys["Anthropic"]:
        model_list.extend([
            {
                "model_name": "llm",  # Same name - Conduit picks best
                "litellm_params": {"model": "claude-sonnet-4-20250514", "api_key": api_keys["Anthropic"]},
                "model_info": {"id": "claude-sonnet-4"},  # Conduit's standardized model ID
            },
            {
                "model_name": "llm",  # Same name - part of pool
                "litellm_params": {"model": "claude-3-5-haiku-20241022", "api_key": api_keys["Anthropic"]},
                "model_info": {"id": "claude-haiku-3.5"},  # Conduit's standardized model ID
            },
        ])

    if api_keys["Google"]:
        model_list.append({
            "model_name": "llm",  # Same name - Conduit can choose Gemini
            "litellm_params": {"model": "gemini/gemini-2.0-flash", "api_key": api_keys["Google"]},
            "model_info": {"id": "gemini-2.0-flash"},  # Conduit's standardized model ID
        })

    if api_keys["Groq"]:
        model_list.append({
            "model_name": "llm",  # Same name - Conduit can choose Llama
            "litellm_params": {"model": "groq/llama-3.1-70b-versatile", "api_key": api_keys["Groq"]},
            "model_info": {"id": "llama-3.1-70b-versatile"},  # No mapping, use LiteLLM ID
        })

    logger.info(f"📋 Configured {len(model_list)} models:")
    for model in model_list:
        logger.info(f"   - {model['model_info']['id']}")
    logger.info("")

    # Initialize LiteLLM router
    router = Router(model_list=model_list)

    # Set up Conduit routing (hybrid UCB1→LinUCB enabled by default)
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    logger.info("✅ Conduit multi-provider routing activated")
    logger.info(f"🤖 Conduit will intelligently choose between {len(model_list)} models\n")
    logger.info("=" * 70)

    # Test diverse queries
    queries = [
        ("Math", "Calculate the derivative of x^2 + 3x + 5"),
        ("Code", "Write a Python function to find prime numbers"),
        ("Creative", "Write a short poem about artificial intelligence"),
        ("Analysis", "Compare capitalism and socialism in 3 paragraphs"),
        ("Science", "Explain photosynthesis in simple terms"),
    ]

    for category, query in queries:
        logger.info(f"\n[{category}] {query}")
        logger.info("-" * 70)

        try:
            response = await router.acompletion(
                model="llm",  # Conduit selects optimal model from all providers
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
            )

            model_used = response.model
            cost = response._hidden_params.get("response_cost", 0.0)
            content = response.choices[0].message.content

            logger.info(f"✅ Conduit selected: {model_used}")
            logger.info(f"💰 Cost: ~${cost:.6f}")
            logger.info(f"📝 Response: {content[:150]}...")

        except Exception as e:
            logger.error(f"❌ Error: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("✨ Multi-Provider Routing Complete!")
    logger.info("")
    logger.info("Key Benefits:")
    logger.info("  ✅ Automatic provider selection based on query type")
    logger.info("  ✅ Cost optimization across providers")
    logger.info("  ✅ Quality maximization through ML learning")
    logger.info("  ✅ No manual routing rules needed")
    logger.info("")
    logger.info("Conduit learns which providers excel at which tasks!")


if __name__ == "__main__":
    asyncio.run(main())
