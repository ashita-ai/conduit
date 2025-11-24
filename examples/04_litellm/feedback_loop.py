"""Feedback Loop - Automatic Learning from Every Request.

This example demonstrates Conduit's automatic feedback loop that learns
from every LiteLLM request without manual intervention:

Automatic Feedback Captured:
- Cost: From LiteLLM response metadata
- Latency: Request duration (start_time to end_time)
- Quality: Estimated from success/failure (0.9 success, 0.1 failure)

Learning Process:
1. LiteLLM completes request → ConduitFeedbackLogger callback triggered
2. Extract cost, latency from response metadata
3. Regenerate query features from request messages
4. Update bandit algorithm with composite reward
5. Future queries benefit from learned performance

Benefits:
- Zero manual feedback required
- Continuous learning from production traffic
- Automatic cost and latency optimization
- Quality improvement over time
"""

import asyncio
import os
import time

from litellm import Router

from conduit_litellm import ConduitRoutingStrategy


async def demonstrate_learning():
    """Show how Conduit learns and improves routing over requests."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return
    
    print("=" * 80)
    print("Automatic Feedback Loop - Learning from Every Request")
    print("=" * 80)
    print()
    
    # Configure router with models of different performance characteristics
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-3.5-turbo"},  # Fast, cheap, lower quality
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o-mini"},  # Balanced
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o"},  # Slow, expensive, high quality
        },
    ]
    
    router = Router(model_list=model_list)
    strategy = ConduitRoutingStrategy(use_hybrid=True)
    ConduitRoutingStrategy.setup_strategy(router, strategy)
    
    print("Model Performance Characteristics:")
    print("  gpt-3.5-turbo:  Fast (0.5s), Cheap ($0.0005), Quality: 7/10")
    print("  gpt-4o-mini:    Medium (1.0s), Medium ($0.002), Quality: 8/10")
    print("  gpt-4o:         Slow (2.0s), Expensive ($0.01), Quality: 10/10")
    print()
    print("Conduit will learn to route based on query complexity...")
    print()
    
    # Test with queries of varying complexity
    test_queries = [
        # Simple queries - should learn to use cheap models
        ("What is 2+2?", "simple"),
        ("Translate 'hello' to French", "simple"),
        ("What's the capital of France?", "simple"),
        
        # Complex queries - should learn to use capable models
        ("Explain the Riemann hypothesis and its implications", "complex"),
        ("Analyze the economic impact of quantum computing", "complex"),
        ("Design a distributed system for real-time analytics", "complex"),
    ]
    
    try:
        print("-" * 80)
        print("Learning Phase - Routing Queries")
        print("-" * 80)
        print()
        
        model_usage = {}
        
        for i, (query, complexity) in enumerate(test_queries, 1):
            start = time.time()
            
            response = await router.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": query}],
            )
            
            elapsed = (time.time() - start) * 1000
            model_used = response.model
            
            # Track model usage
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
            
            print(f"Query {i} ({complexity}): {query[:50]}...")
            print(f"  → Model: {model_used}")
            print(f"  → Latency: {elapsed:.0f}ms")
            
            # Show that feedback is captured automatically
            print(f"  ✅ Feedback captured: cost=${response._hidden_params.get('response_cost', 0):.6f}")
            print()
            
            # Small delay to let feedback processing complete
            await asyncio.sleep(0.5)
        
        print()
        print("=" * 80)
        print("Learning Results")
        print("=" * 80)
        print()
        
        print("Model Selection Summary:")
        for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(test_queries)) * 100
            print(f"  {model}: {count} queries ({percentage:.0f}%)")
        print()
        
        print("Expected Learning Behavior:")
        print("  - Simple queries: Increasingly routed to gpt-3.5-turbo (cheap, sufficient)")
        print("  - Complex queries: Increasingly routed to gpt-4o (expensive, necessary)")
        print("  - Mixed queries: gpt-4o-mini as balanced middle ground")
        print()
        
        print("Feedback Loop Components:")
        print("  1. LiteLLM completes request")
        print("  2. ConduitFeedbackLogger.async_log_success_event() called")
        print("  3. Extract: cost, latency, model_id from response")
        print("  4. Regenerate: query features from messages")
        print("  5. Update: bandit algorithm with composite reward")
        print("  6. Next query: Benefits from learned performance")
        print()
        
        print("Composite Reward Calculation:")
        print("  reward = quality_weight * quality")
        print("         - cost_weight * normalized_cost")
        print("         - latency_weight * normalized_latency")
        print()
        print("  Default weights:")
        print("    - Quality: 0.7 (primary optimization)")
        print("    - Cost: 0.2 (secondary optimization)")
        print("    - Latency: 0.1 (tertiary optimization)")
        print()
        
        print("Production Impact:")
        print("  - Automatic learning from ALL production traffic")
        print("  - No manual feedback collection required")
        print("  - Continuous optimization of cost/latency/quality trade-offs")
        print("  - Model performance learned contextually per query type")
        print()
        
    finally:
        strategy.cleanup()
        print("✅ Resources cleaned up")


async def main():
    """Run feedback loop demonstration."""
    await demonstrate_learning()


if __name__ == "__main__":
    asyncio.run(main())
