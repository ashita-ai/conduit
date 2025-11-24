"""Basic LiteLLM + Conduit Integration.

Simplest example of using Conduit's ML routing with LiteLLM.

Requirements:
    - OPENAI_API_KEY environment variable
    - pip install conduit[litellm]

Run:
    python examples/04_litellm/basic_usage.py
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

    # 3. Initialize LiteLLM router
    router = Router(model_list=model_list)

    # 4. Set up Conduit ML routing strategy
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)

    print("✅ Conduit routing strategy activated\n")

    # 5. Make requests (Conduit automatically selects best model)
    queries = [
        "What is 2+2?",
        "Explain quantum mechanics in detail."
    ]

    for query in queries:
        print(f"Query: {query}")

        response = await router.acompletion(
            model="gpt-4o-mini",  # Model group (Conduit selects specific one)
            messages=[{"role": "user", "content": query}]
        )

        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content[:100]}...\n")

    print("✨ Done! Conduit learned from these requests and will improve over time.")


if __name__ == "__main__":
    asyncio.run(main())
