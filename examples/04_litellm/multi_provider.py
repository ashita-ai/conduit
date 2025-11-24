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
    model_list = []

    if api_keys["OpenAI"]:
        model_list.extend([
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "gpt-4o-mini", "api_key": api_keys["OpenAI"]},
                "model_info": {"id": "gpt-4o-mini"},
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "gpt-4o", "api_key": api_keys["OpenAI"]},
                "model_info": {"id": "gpt-4o"},
            },
        ])

    if api_keys["Anthropic"]:
        model_list.extend([
            {
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {"model": "claude-3-5-sonnet-20241022", "api_key": api_keys["Anthropic"]},
                "model_info": {"id": "claude-3-5-sonnet"},
            },
            {
                "model_name": "claude-3-5-haiku",
                "litellm_params": {"model": "claude-3-5-haiku-20241022", "api_key": api_keys["Anthropic"]},
                "model_info": {"id": "claude-3-5-haiku"},
            },
        ])

    if api_keys["Google"]:
        model_list.append({
            "model_name": "gemini-1.5-flash",
            "litellm_params": {"model": "gemini/gemini-1.5-flash", "api_key": api_keys["Google"]},
            "model_info": {"id": "gemini-1.5-flash"},
        })

    if api_keys["Groq"]:
        model_list.append({
            "model_name": "llama-3.1-70b",
            "litellm_params": {"model": "groq/llama-3.1-70b-versatile", "api_key": api_keys["Groq"]},
            "model_info": {"id": "llama-3.1-70b"},
        })

    print(f"📋 Configured {len(model_list)} models:")
    for model in model_list:
        print(f"   - {model['model_info']['id']}")
    print()

    # Initialize LiteLLM router
    router = Router(model_list=model_list)

    # Set up Conduit with hybrid routing
    strategy = ConduitRoutingStrategy(use_hybrid=True)
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    print("✅ Conduit multi-provider routing activated\n")
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
                model=model_list[0]["model_name"],  # Conduit selects optimal model
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
            )

            model_used = response.model
            cost = response._hidden_params.get("response_cost", 0.0)
            content = response.choices[0].message.content

            print(f"Model: {model_used}")
            print(f"Cost: ~${cost:.6f}")
            print(f"Response: {content[:150]}...")

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
