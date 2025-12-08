"""Custom Conduit Configuration with LiteLLM.

Shows how to customize Conduit's bandit algorithm, hybrid routing,
and caching when using with LiteLLM.

Requirements:
    - OPENAI_API_KEY environment variable
    - pip install conduit[litellm]
    - Optional: Redis for caching

Run:
    python examples/custom_config.py
"""

import asyncio
import os

from litellm import Router
from conduit_litellm import ConduitRoutingStrategy


async def main() -> None:
    """Demonstrate custom Conduit configuration."""

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("🚀 Custom Conduit Configuration Example\n")

    # Configure LiteLLM model list
    # KEY: Use same model_name "gpt" for both models so Conduit can route between them
    model_list = [
        {
            "model_name": "gpt",  # Shared name - Conduit picks between these
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "o4-mini"},  # Conduit's standardized model ID
        },
        {
            "model_name": "gpt",  # Same name - part of routing pool
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-5"},  # Conduit's standardized model ID
        },
    ]

    # Initialize LiteLLM router
    router = Router(model_list=model_list)

    # Custom Conduit configuration
    print("⚙️  Configuration:")
    print("   - Hybrid routing: UCB1 → LinUCB (30% faster convergence)")
    print("   - Redis caching: Enabled (if REDIS_URL set)")
    print("   - Embedding model: all-MiniLM-L6-v2")
    print("   - Models: gpt-4o-mini (cheap), gpt-4o (capable)")
    print()

    strategy = ConduitRoutingStrategy(
        cache_enabled=bool(os.getenv("REDIS_URL")),  # Enable Redis caching if URL present
    )
    # Note: Hybrid routing (UCB1→LinUCB) is always enabled by default
    # Redis URL configured via REDIS_URL environment variable

    ConduitRoutingStrategy.setup_strategy(router, strategy)

    print("✅ Custom strategy activated\n")

    # Run test queries
    queries = [
        ("Simple", "What is 2+2?"),
        ("Complex", "Explain the theory of relativity in detail."),
        ("Code", "Write a Python decorator for retry logic."),
        ("Creative", "Write a haiku about machine learning."),
    ]

    for label, query in queries:
        print(f"[{label}] {query}")

        response = await router.acompletion(
            model="gpt",  # Conduit selects optimal model
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
        )

        print(f"   → Conduit selected: {response.model}")
        print(f"   💰 Cost: ~${response._hidden_params.get('response_cost', 0.0):.6f}")
        print(f"   📝 Response: {response.choices[0].message.content[:80]}...")
        print()

    print("✨ Hybrid routing learns quickly from early queries!")
    print("   First ~100 queries: UCB1 (fast exploration)")
    print("   After ~100 queries: LinUCB (contextual optimization)")


if __name__ == "__main__":
    asyncio.run(main())
