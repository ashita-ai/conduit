"""Custom Conduit Configuration with LiteLLM.

Shows how to customize Conduit's bandit algorithm, hybrid routing,
and caching when using with LiteLLM.

Requirements:
    - OPENAI_API_KEY environment variable
    - pip install conduit[litellm]
    - Optional: Redis for caching

Run:
    python examples/04_litellm/custom_config.py
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
    model_list = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-4o-mini"},
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-4o"},
        },
    ]

    # Initialize LiteLLM router
    router = Router(model_list=model_list)

    # Custom Conduit configuration
    print("⚙️  Configuration:")
    print("   - Hybrid routing: UCB1 → LinUCB (30% faster convergence)")
    print("   - Redis caching: Enabled (if REDIS_URL set)")
    print("   - Embedding model: all-MiniLM-L6-v2")
    print()

    strategy = ConduitRoutingStrategy(
        use_hybrid=True,  # Enable UCB1 → LinUCB warm start
        cache_enabled=bool(os.getenv("REDIS_URL")),  # Enable Redis caching if URL present
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )

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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
        )

        print(f"   Model: {response.model}")
        print(f"   Cost: ~${response._hidden_params.get('response_cost', 0.0):.6f}")
        print(f"   Response: {response.choices[0].message.content[:80]}...")
        print()

    print("✨ Hybrid routing learns quickly from early queries!")
    print("   First ~100 queries: UCB1 (fast exploration)")
    print("   After ~100 queries: LinUCB (contextual optimization)")


if __name__ == "__main__":
    asyncio.run(main())
