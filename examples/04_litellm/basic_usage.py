"""Basic LiteLLM Integration - Simple ML-Powered Routing.

This example demonstrates the simplest way to use Conduit's ML routing
with LiteLLM's router for intelligent model selection across providers.

Key Features:
- Automatic model selection using contextual bandits
- Automatic feedback learning from every request
- Support for 100+ LLM providers via LiteLLM
- Zero-config intelligent routing
"""

import asyncio
import os

from litellm import Router

from conduit_litellm import ConduitRoutingStrategy


async def main():
    """Run basic LiteLLM + Conduit routing example."""
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY=sk-...")
        return
    
    print("=" * 80)
    print("Basic LiteLLM + Conduit Routing Example")
    print("=" * 80)
    print()
    
    # Configure LiteLLM router with multiple models
    model_list = [
        {
            "model_name": "gpt-4",  # Model group name
            "litellm_params": {
                "model": "gpt-4o-mini",  # Actual model
            },
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4o",  # More capable but expensive
            },
        },
        {
            "model_name": "gpt-3.5",
            "litellm_params": {
                "model": "gpt-3.5-turbo",  # Fast and cheap
            },
        },
    ]
    
    # Create LiteLLM router
    router = Router(model_list=model_list)
    
    # Setup Conduit ML routing strategy
    # This replaces LiteLLM's default routing with intelligent ML-based selection
    strategy = ConduitRoutingStrategy()
    ConduitRoutingStrategy.setup_strategy(router, strategy)
    
    print("✅ Conduit routing strategy initialized")
    print("   - Feedback loop: ENABLED (automatic learning)")
    print("   - Algorithm: LinUCB (contextual bandit)")
    print("   - Models: gpt-4o-mini, gpt-4o, gpt-3.5-turbo")
    print()
    
    # Test queries - Conduit will learn which model works best for each type
    queries = [
        "What is 2+2?",  # Simple query - should route to cheap model
        "Explain quantum entanglement in detail",  # Complex - may route to capable model
        "Translate 'hello' to Spanish",  # Simple - cheap model
    ]
    
    try:
        for i, query in enumerate(queries, 1):
            print("-" * 80)
            print(f"Query {i}: {query}")
            print()
            
            # LiteLLM uses Conduit to select the optimal model
            response = await router.acompletion(
                model="gpt-4",  # Model group (Conduit selects specific deployment)
                messages=[{"role": "user", "content": query}],
            )
            
            # Extract response details
            model_used = response.model
            content = response.choices[0].message.content
            
            print(f"Selected Model: {model_used}")
            print(f"Response: {content[:100]}...")
            
            # Feedback is captured automatically by ConduitFeedbackLogger
            # Cost, latency, and quality are fed back to the bandit for learning
            print("✅ Feedback captured automatically (cost, latency, quality)")
            print()
            
    finally:
        # Clean up resources
        strategy.cleanup()
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        print("Conduit learned from all queries and will improve routing over time:")
        print("  - Fast queries → cheaper models (gpt-3.5-turbo, gpt-4o-mini)")
        print("  - Complex queries → capable models (gpt-4o)")
        print("  - Learning happens automatically with every request")
        print()
        print("Next steps:")
        print("  - Try custom_config.py for hybrid routing and caching")
        print("  - Try multi_provider.py for cross-provider routing")


if __name__ == "__main__":
    asyncio.run(main())
