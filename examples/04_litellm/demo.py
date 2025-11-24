"""LiteLLM Integration Demo - Conduit ML Routing with LiteLLM.

This example demonstrates how to use Conduit's ML-powered routing
as a custom strategy for LiteLLM, enabling intelligent model selection
across 100+ LLM providers.

Requirements:
    - OpenAI API key (or other provider keys)
    - pip install conduit[litellm]

Run:
    python examples/04_litellm/demo.py
"""

import asyncio
import os

try:
    from litellm import Router
    LITELLM_AVAILABLE = True
except ImportError:
    print("❌ LiteLLM not installed. Install with: pip install conduit[litellm]")
    exit(1)

from conduit_litellm import ConduitRoutingStrategy


async def main() -> None:
    """Demonstrate Conduit + LiteLLM integration."""

    print("=" * 70)
    print("🚀 Conduit + LiteLLM Integration Demo")
    print("=" * 70)
    print()

    # Check for API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "Groq": os.getenv("GROQ_API_KEY"),
    }

    available_providers = [k for k, v in api_keys.items() if v]

    if not available_providers:
        print("❌ No API keys found. Set at least one:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        print("   - GROQ_API_KEY")
        return

    print(f"✅ Found API keys for: {', '.join(available_providers)}")
    print()

    # Configure LiteLLM model list
    model_list = []

    # OpenAI models
    if api_keys["OpenAI"]:
        model_list.extend([
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": api_keys["OpenAI"],
                },
                "model_info": {"id": "gpt-4o-mini-openai"},
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": api_keys["OpenAI"],
                },
                "model_info": {"id": "gpt-4o-openai"},
            },
        ])

    # Anthropic models
    if api_keys["Anthropic"]:
        model_list.extend([
            {
                "model_name": "claude-3-5-sonnet",
                "litellm_params": {
                    "model": "claude-3-5-sonnet-20241022",
                    "api_key": api_keys["Anthropic"],
                },
                "model_info": {"id": "claude-3-5-sonnet"},
            },
            {
                "model_name": "claude-3-5-haiku",
                "litellm_params": {
                    "model": "claude-3-5-haiku-20241022",
                    "api_key": api_keys["Anthropic"],
                },
                "model_info": {"id": "claude-3-5-haiku"},
            },
        ])

    # Google models
    if api_keys["Google"]:
        model_list.extend([
            {
                "model_name": "gemini-1.5-flash",
                "litellm_params": {
                    "model": "gemini/gemini-1.5-flash",
                    "api_key": api_keys["Google"],
                },
                "model_info": {"id": "gemini-1.5-flash"},
            },
        ])

    # Groq models
    if api_keys["Groq"]:
        model_list.extend([
            {
                "model_name": "llama-3.1-70b",
                "litellm_params": {
                    "model": "groq/llama-3.1-70b-versatile",
                    "api_key": api_keys["Groq"],
                },
                "model_info": {"id": "llama-3.1-70b-groq"},
            },
        ])

    print(f"📋 Configured {len(model_list)} models for routing:")
    for model in model_list:
        print(f"   - {model['model_info']['id']}")
    print()

    # Initialize LiteLLM router
    print("🔧 Initializing LiteLLM router...")
    router = Router(
        model_list=model_list,
        routing_strategy="usage-based-routing",  # Will be overridden by Conduit
    )

    # Create Conduit routing strategy
    print("🧠 Setting up Conduit ML routing strategy...")
    conduit_strategy = ConduitRoutingStrategy(
        use_hybrid=True,  # UCB1→LinUCB warm start
        cache_enabled=bool(os.getenv("REDIS_CACHE_ENABLED")),
    )

    # Set Conduit as custom routing strategy using helper method
    ConduitRoutingStrategy.setup_strategy(router, conduit_strategy)
    print("✅ Conduit routing strategy activated!")
    print()

    # Run test queries
    test_queries = [
        "What is 2+2?",  # Simple math
        "Explain quantum entanglement in detail.",  # Complex physics
        "Write a haiku about AI.",  # Creative writing
        "What are the best practices for Python async/await?",  # Technical
    ]

    print("=" * 70)
    print("🎯 Running Test Queries")
    print("=" * 70)
    print()

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query}")
        print("-" * 70)

        try:
            # Make request through LiteLLM with Conduit routing
            response = await router.acompletion(
                model=model_list[0]["model_name"],  # Model group (Conduit will select specific one)
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
            )

            # Extract response
            content = response.choices[0].message.content
            model_used = response.model

            print(f"✅ Model Selected: {model_used}")
            print(f"📝 Response: {content[:150]}{'...' if len(content) > 150 else ''}")
            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    print("=" * 70)
    print("✨ Demo Complete!")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("  ✅ LiteLLM router with Conduit ML strategy")
    print("  ✅ Intelligent model selection based on query features")
    print("  ✅ Multi-provider support (OpenAI, Anthropic, Google, Groq)")
    print("  ✅ Hybrid routing (UCB1→LinUCB warm start)")
    print()
    print("Next Steps:")
    print("  - Issue #13: Add feedback loop for learning")
    print("  - Issue #14: Add comprehensive tests")
    print("  - Issue #15: Create more examples")


if __name__ == "__main__":
    asyncio.run(main())
