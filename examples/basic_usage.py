"""Basic LiteLLM + Conduit Integration.

Simplest example of using Conduit's ML routing with LiteLLM.

Requirements:
    - OPENAI_API_KEY environment variable
    - pip install conduit[litellm]

Run:
    python examples/04_litellm/basic_usage.py

Expected output:
    Basic Conduit + LiteLLM Integration

    Conduit routing strategy activated
    Available models: o4-mini (cheap), gpt-5 (capable)
    Conduit will learn which model is best for each query type

    [1/5] Query (simple): What is 2+2?...
          Conduit selected: gpt-4o-mini
          Response: 2 + 2 equals 4...

    [2/5] Query (complex): Explain quantum mechanics in detail....
          Conduit selected: gpt-4o
          Response: Quantum mechanics is a fundamental theory...

    ... (3 more queries)

    Done! Conduit learned from these requests and will:
       Route simple queries to o4-mini (cheaper)
       Route complex queries to gpt-5 (better quality)
       Continuously improve routing decisions over time
"""

import asyncio
import os

from litellm import Router
from conduit_litellm import ConduitRoutingStrategy


async def main() -> None:
    """Basic Conduit + LiteLLM example."""

    # 1. Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("🚀 Basic Conduit + LiteLLM Integration\n")

    # 2. Configure LiteLLM model list
    # KEY: Use same model_name for multiple deployments so Conduit can choose
    model_list = [
        {
            "model_name": "gpt",  # Shared name - Conduit picks between these
            "litellm_params": {
                "model": "gpt-4o-mini",  # Fast, cheap model
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "o4-mini"},  # Conduit's standardized model ID
        },
        {
            "model_name": "gpt",  # Same name - part of routing pool
            "litellm_params": {
                "model": "gpt-4o",  # Slower, more capable model
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            "model_info": {"id": "gpt-5"},  # Conduit's standardized model ID
        },
    ]

    # 3. Initialize LiteLLM router
    router = Router(model_list=model_list)

    # 4. Set up Conduit ML routing strategy
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    print("✅ Conduit routing strategy activated")
    print("📊 Available models: o4-mini (cheap), gpt-5 (capable)")
    print("🤖 Conduit will learn which model is best for each query type\n")

    # 5. Make requests - Conduit learns as it goes
    queries = [
        ("What is 2+2?", "simple"),
        ("Explain quantum mechanics in detail.", "complex"),
        ("Translate 'hello' to Spanish", "simple"),
        ("Write a detailed analysis of climate change impacts", "complex"),
        ("What's 5 + 7?", "simple"),
    ]

    for i, (query, query_type) in enumerate(queries, 1):
        print(f"[{i}/5] Query ({query_type}): {query[:50]}...")

        response = await router.acompletion(
            model="gpt",  # Conduit chooses between gpt-4o-mini and gpt-4o
            messages=[{"role": "user", "content": query}]
        )

        # Show which model Conduit selected
        selected = response.model
        print(f"      → Conduit selected: {selected}")
        print(f"      → Response: {response.choices[0].message.content[:80]}...\n")

    print("✨ Done! Conduit learned from these requests and will:")
    print("   • Route simple queries to o4-mini (cheaper)")
    print("   • Route complex queries to gpt-5 (better quality)")
    print("   • Continuously improve routing decisions over time")


if __name__ == "__main__":
    asyncio.run(main())
