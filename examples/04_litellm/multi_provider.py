"""Multi-Provider Routing - Intelligent Selection Across 5+ Providers.

This example demonstrates Conduit's ability to intelligently route across
multiple LLM providers (OpenAI, Anthropic, Google, etc.) with:
- Automatic failover and fallback handling
- Cost-aware provider selection
- Quality-based routing decisions
- Cross-provider performance learning

Real-World Benefits:
- Avoid provider outages with automatic failover
- Optimize costs across providers (e.g., use cheaper Gemini for simple tasks)
- Leverage provider strengths (e.g., Claude for analysis, GPT for code)
"""

import asyncio
import os

from litellm import Router

from conduit_litellm import ConduitRoutingStrategy


async def main():
    """Demonstrate multi-provider routing with Conduit."""
    
    # Check for API keys (at least one required)
    providers = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    }
    
    available_providers = [name for name, key in providers.items() if key]
    
    if not available_providers:
        print("Error: Set at least one API key:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GEMINI_API_KEY or GOOGLE_API_KEY")
        return
    
    print("=" * 80)
    print("Multi-Provider Routing Example")
    print("=" * 80)
    print()
    print(f"Available providers: {', '.join(available_providers)}")
    print()
    
    # Configure LiteLLM with models from multiple providers
    model_list = []
    
    # OpenAI models
    if "OpenAI" in available_providers:
        model_list.extend([
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o-mini"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4o"},
            },
        ])
    
    # Anthropic models
    if "Anthropic" in available_providers:
        model_list.extend([
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "claude-3-haiku-20240307"},
            },
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "claude-3-5-sonnet-20241022"},
            },
        ])
    
    # Google models
    if "Google" in available_providers:
        model_list.extend([
            {
                "model_name": "gemini",
                "litellm_params": {"model": "gemini/gemini-1.5-flash"},
            },
            {
                "model_name": "gemini",
                "litellm_params": {"model": "gemini/gemini-1.5-pro"},
            },
        ])
    
    print(f"Configured {len(model_list)} models across {len(available_providers)} providers")
    print()
    
    # Create router with fallback support
    router = Router(
        model_list=model_list,
        # LiteLLM fallback configuration (Conduit chooses primary)
        fallbacks=[
            {"gpt-4": ["claude-3", "gemini"]} if len(available_providers) > 1 else {},
        ],
        # Timeout and retry settings
        timeout=30,
        num_retries=2,
    )
    
    # Setup Conduit routing strategy
    strategy = ConduitRoutingStrategy(use_hybrid=True)
    ConduitRoutingStrategy.setup_strategy(router, strategy)
    
    print("✅ Multi-provider routing initialized")
    print("   - Conduit selects optimal model from all providers")
    print("   - LiteLLM handles fallback if primary fails")
    print("   - Learning happens across all providers")
    print()
    
    # Test queries that may benefit from different providers
    test_cases = [
        {
            "query": "What is 2+2?",
            "expected": "Simple math - likely cheap model (Gemini Flash, GPT-4o-mini)",
        },
        {
            "query": "Analyze the economic implications of climate change policy",
            "expected": "Complex analysis - likely capable model (Claude Sonnet, GPT-4o)",
        },
        {
            "query": "Write a Python function to reverse a string",
            "expected": "Code generation - GPT models often preferred",
        },
        {
            "query": "Explain quantum entanglement to a 5-year-old",
            "expected": "Explanation - Claude often excels at teaching",
        },
    ]
    
    try:
        for i, test in enumerate(test_cases, 1):
            query = test["query"]
            expected = test["expected"]
            
            print("-" * 80)
            print(f"Query {i}: {query}")
            print(f"Expected: {expected}")
            print()
            
            try:
                # Primary routing via Conduit
                response = await router.acompletion(
                    model="gpt-4" if "OpenAI" in available_providers else "claude-3",
                    messages=[{"role": "user", "content": query}],
                )
                
                model_used = response.model
                content = response.choices[0].message.content
                
                # Determine provider from model name
                provider = (
                    "OpenAI" if "gpt" in model_used.lower()
                    else "Anthropic" if "claude" in model_used.lower()
                    else "Google" if "gemini" in model_used.lower()
                    else "Unknown"
                )
                
                print(f"✅ Selected: {model_used} ({provider})")
                print(f"   Response: {content[:100]}...")
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"   Fallback mechanism would try alternative providers")
                print()
        
        print()
        print("=" * 80)
        print("Multi-Provider Benefits")
        print("=" * 80)
        print()
        print("Reliability:")
        print("  - Automatic failover if primary provider down")
        print("  - Reduced downtime with cross-provider redundancy")
        print("  - LiteLLM + Conduit handle routing and fallback")
        print()
        print("Cost Optimization:")
        print("  - Conduit learns which provider is cheapest for each task")
        print("  - Example: Gemini Flash for simple queries (1/10th GPT cost)")
        print("  - Automatic cost-aware routing over time")
        print()
        print("Quality Optimization:")
        print("  - Route complex tasks to capable models (GPT-4o, Claude Sonnet)")
        print("  - Route simple tasks to fast/cheap models (Gemini Flash, GPT-4o-mini)")
        print("  - Learning improves routing quality over time")
        print()
        print("Provider Strengths:")
        print("  - Claude: Analysis, reasoning, teaching")
        print("  - GPT: Code generation, general purpose")
        print("  - Gemini: Cost-effective, fast responses")
        print("  - Conduit learns these strengths automatically")
        print()
        
    finally:
        strategy.cleanup()
        print("✅ Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
