"""Multi-Provider Routing with Conduit + LiteLLM.

Demonstrates intelligent routing across multiple LLM providers
(OpenAI, Anthropic, Google, Groq) using Conduit's ML-based selection.

Requirements:
    - At least 2 provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    - pip install conduit[litellm]

Run:
    python examples/multi_provider.py
"""

import asyncio
import os

from litellm import Router
from conduit_litellm import ConduitRoutingStrategy


async def main() -> None:
    """Multi-provider routing example."""

    print("🚀 Multi-Provider Routing with Conduit\n")

    # Check available API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available_providers = [k for k, v in api_keys.items() if v]

    if len(available_providers) < 2:
        print("❌ Need at least 2 API keys for multi-provider demo")
        print("   Set any combination of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        print("   - GROQ_API_KEY")
        return

    print(f"✅ Found {len(available_providers)} providers: {', '.join(available_providers)}\n")

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

    print(f"📋 Configured {len(model_list)} models:")
    for model in model_list:
        print(f"   - {model['model_info']['id']}")
    print()

    # Initialize LiteLLM router
    router = Router(model_list=model_list)

    # Set up Conduit routing (hybrid UCB1→LinUCB enabled by default)
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    print("✅ Conduit multi-provider routing activated")
    print(f"🤖 Conduit will intelligently choose between {len(model_list)} models\n")
    print("=" * 70)

    # Test diverse queries
    queries = [
        ("Math", "Calculate the derivative of x^2 + 3x + 5"),
        ("Code", "Write a Python function to find prime numbers"),
        ("Creative", "Write a short poem about artificial intelligence"),
        ("Analysis", "Compare capitalism and socialism in 3 paragraphs"),
        ("Science", "Explain photosynthesis in simple terms"),
    ]

    for category, query in queries:
        print(f"\n[{category}] {query}")
        print("-" * 70)

        try:
            response = await router.acompletion(
                model="llm",  # Conduit selects optimal model from all providers
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
            )

            model_used = response.model
            cost = response._hidden_params.get("response_cost", 0.0)
            content = response.choices[0].message.content

            print(f"✅ Conduit selected: {model_used}")
            print(f"💰 Cost: ~${cost:.6f}")
            print(f"📝 Response: {content[:150]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✨ Multi-Provider Routing Complete!")
    print()
    print("Key Benefits:")
    print("  ✅ Automatic provider selection based on query type")
    print("  ✅ Cost optimization across providers")
    print("  ✅ Quality maximization through ML learning")
    print("  ✅ No manual routing rules needed")
    print()
    print("Conduit learns which providers excel at which tasks!")


if __name__ == "__main__":
    asyncio.run(main())
