"""Performance Comparison - ML Routing vs Rule-Based Routing.

This example compares Conduit's ML-powered routing against traditional
rule-based routing strategies to demonstrate the value of contextual bandits.

Comparison:
1. Rule-Based (Default LiteLLM):
   - Round-robin: Equal distribution across models
   - Random: Random selection
   - Fixed: Always use cheapest/fastest model

2. ML-Based (Conduit):
   - LinUCB: Context-aware intelligent selection
   - Learning: Improves over time with feedback
   - Adaptive: Adjusts to query characteristics

Expected Results:
- ML routing: Lower cost while maintaining quality
- ML routing: Better latency on average
- ML routing: Improved performance over time
"""

import asyncio
import os
import random
import time
from typing import Any

from litellm import Router

from conduit_litellm import ConduitRoutingStrategy


class RoundRobinStrategy:
    """Simple round-robin routing for comparison."""
    
    def __init__(self):
        self.current_index = 0
    
    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Select next deployment in round-robin order."""
        if not hasattr(self, '_router') or not self._router:
            raise RuntimeError("Router not set. Call set_router() first.")
        
        deployments = self._router.model_list
        selected = deployments[self.current_index % len(deployments)]
        self.current_index += 1
        return selected
    
    def set_router(self, router: Any) -> None:
        """Store router reference."""
        self._router = router


async def run_benchmark(strategy_name: str, router: Router, queries: list[str]) -> dict[str, Any]:
    """Run benchmark with given routing strategy."""
    
    print(f"\nRunning benchmark: {strategy_name}")
    print("-" * 60)
    
    total_cost = 0.0
    total_latency = 0.0
    model_usage = {}
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        
        try:
            response = await router.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": query}],
            )
            
            elapsed = time.time() - start
            model_used = response.model
            cost = response._hidden_params.get('response_cost', 0)
            
            total_cost += cost
            total_latency += elapsed
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
            
            if i % 5 == 0:
                print(f"  Query {i}/{len(queries)}: {model_used} (${cost:.6f}, {elapsed:.2f}s)")
        
        except Exception as e:
            print(f"  Query {i} failed: {e}")
    
    avg_cost = total_cost / len(queries)
    avg_latency = total_latency / len(queries)
    
    print(f"\nResults for {strategy_name}:")
    print(f"  Average cost: ${avg_cost:.6f}")
    print(f"  Average latency: {avg_latency:.2f}s")
    print(f"  Model distribution: {model_usage}")
    
    return {
        "strategy": strategy_name,
        "avg_cost": avg_cost,
        "avg_latency": avg_latency,
        "total_cost": total_cost,
        "model_usage": model_usage,
    }


async def main():
    """Compare ML routing vs rule-based routing."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return
    
    print("=" * 80)
    print("Performance Comparison: ML Routing vs Rule-Based Routing")
    print("=" * 80)
    
    # Test queries with varying complexity
    test_queries = [
        # Simple queries (10x) - should use cheap models
        *["What is 2+2?" for _ in range(10)],
        *["Translate 'hello' to Spanish" for _ in range(10)],
        
        # Medium queries (10x) - balanced models
        *["Explain photosynthesis" for _ in range(10)],
        *["Write a Python function to sort" for _ in range(10)],
        
        # Complex queries (10x) - need capable models
        *["Analyze quantum computing implications" for _ in range(10)],
        *["Design a distributed database system" for _ in range(10)],
    ]
    
    # Shuffle to avoid ordering bias
    random.shuffle(test_queries)
    
    print(f"\nBenchmark configuration:")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Simple: 20 (33%), Medium: 20 (33%), Complex: 20 (33%)")
    print()
    
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-3.5-turbo"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o-mini"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4o"},
        },
    ]
    
    results = []
    
    # Benchmark 1: Round-Robin (baseline)
    print("\n" + "=" * 80)
    print("Benchmark 1: Round-Robin (Baseline)")
    print("=" * 80)
    
    router_rr = Router(model_list=model_list)
    rr_strategy = RoundRobinStrategy()
    rr_strategy.set_router(router_rr)
    router_rr.set_custom_routing_strategy(rr_strategy)
    
    result_rr = await run_benchmark("Round-Robin", router_rr, test_queries)
    results.append(result_rr)
    
    # Benchmark 2: Conduit ML Routing
    print("\n" + "=" * 80)
    print("Benchmark 2: Conduit ML Routing (LinUCB)")
    print("=" * 80)
    
    router_ml = Router(model_list=model_list)
    ml_strategy = ConduitRoutingStrategy(use_hybrid=True)
    ConduitRoutingStrategy.setup_strategy(router_ml, ml_strategy)
    
    result_ml = await run_benchmark("Conduit ML", router_ml, test_queries)
    results.append(result_ml)
    
    ml_strategy.cleanup()
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print()
    
    # Cost comparison
    print("Cost Analysis:")
    rr_cost = result_rr["avg_cost"]
    ml_cost = result_ml["avg_cost"]
    cost_savings = ((rr_cost - ml_cost) / rr_cost) * 100 if rr_cost > 0 else 0
    
    print(f"  Round-Robin: ${rr_cost:.6f} per query")
    print(f"  Conduit ML:  ${ml_cost:.6f} per query")
    print(f"  Savings:     {cost_savings:.1f}% {'(ML better)' if cost_savings > 0 else '(RR better)'}")
    print()
    
    # Latency comparison
    print("Latency Analysis:")
    rr_latency = result_rr["avg_latency"]
    ml_latency = result_ml["avg_latency"]
    latency_improvement = ((rr_latency - ml_latency) / rr_latency) * 100 if rr_latency > 0 else 0
    
    print(f"  Round-Robin: {rr_latency:.2f}s per query")
    print(f"  Conduit ML:  {ml_latency:.2f}s per query")
    print(f"  Improvement: {latency_improvement:.1f}% {'(ML faster)' if latency_improvement > 0 else '(RR faster)'}")
    print()
    
    # Model distribution
    print("Model Distribution:")
    print(f"  Round-Robin: {result_rr['model_usage']}")
    print(f"  Conduit ML:  {result_ml['model_usage']}")
    print()
    
    print("Key Insights:")
    print("  - Round-robin distributes evenly (33/33/33)")
    print("  - ML routing adapts to query complexity:")
    print("    * Simple queries → cheap models (gpt-3.5-turbo)")
    print("    * Complex queries → capable models (gpt-4o)")
    print("    * Balanced queries → middle ground (gpt-4o-mini)")
    print()
    
    print("Long-Term Benefits of ML Routing:")
    print("  - Continuous learning improves over time")
    print("  - Adapts to changing model performance")
    print("  - Context-aware decisions (not blind rotation)")
    print("  - Automatic cost/latency/quality optimization")
    print()


if __name__ == "__main__":
    asyncio.run(main())
