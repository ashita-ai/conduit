"""Cost savings benchmark: Conduit vs. Static Routing.

This benchmark compares Conduit's adaptive routing against static routing strategies
to demonstrate cost savings while maintaining quality.

Run with:
    python benchmarks/cost_savings_comparison.py

Output:
    - Console summary of cost savings
    - CSV file with detailed results: cost_savings_results.csv
    - PNG graph: cost_savings_graph.png
"""

import asyncio
import csv
import random
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from conduit.core.config import get_fallback_pricing, get_default_pricing
from conduit.core.models import Query
from conduit.engines.router import Router


class StaticRouter:
    """Static router that always uses the same model."""

    def __init__(self, model_id: str):
        """Initialize static router.

        Args:
            model_id: Model to always use
        """
        self.model_id = model_id
        self.total_cost = 0.0
        self.total_queries = 0

    async def route(self, query: Query) -> str:
        """Route query (always returns same model).

        Args:
            query: Query to route

        Returns:
            Model ID (always the same)
        """
        self.total_queries += 1
        return self.model_id


class CostSavingsBenchmark:
    """Benchmark comparing Conduit vs. static routing."""

    def __init__(self):
        """Initialize benchmark."""
        # Use OpenAI embeddings (requires OPENAI_API_KEY environment variable)
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable required for cost savings benchmark. "
                "Set it with: export OPENAI_API_KEY=sk-..."
            )

        self.conduit_router = Router(
            embedding_provider_type="openai",
            embedding_model="text-embedding-3-small",  # Fast, cheap embeddings
            embedding_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.static_always_best = StaticRouter("gpt-4o")  # Always use best model
        self.static_always_cheap = StaticRouter("gpt-4o-mini")  # Always use cheapest
        self.results: List[dict] = []
        
        # Load real pricing from pricing.yaml
        self.pricing = get_fallback_pricing()
        self.default_pricing = get_default_pricing()

    def generate_queries(self, count: int = 1000) -> List[dict]:
        """Generate diverse queries.

        Args:
            count: Number of queries to generate

        Returns:
            List of query dictionaries
        """
        simple_queries = [
            "What is 2+2?",
            "What is the capital of France?",
            "What is Python?",
            "Hello, how are you?",
            "What is the weather?",
            "What is 5*5?",
            "What is a variable?",
            "What is a function?",
            "What is a list?",
            "What is a dictionary?",
        ]

        complex_queries = [
            "Explain quantum computing and its applications in cryptography",
            "Write a comprehensive guide to machine learning model evaluation",
            "Design a distributed system architecture for handling 1M requests/second",
            "Explain the mathematical foundations of neural networks",
            "Compare and contrast different database architectures",
            "Write a detailed analysis of the CAP theorem",
            "Explain how transformers work in natural language processing",
            "Design a microservices architecture for an e-commerce platform",
            "Explain the principles of functional programming",
            "Write a guide to optimizing database queries",
        ]

        queries = []
        for i in range(count):
            if random.random() < 0.6:  # 60% simple queries
                text = random.choice(simple_queries)
                complexity = "simple"
            else:  # 40% complex queries
                text = random.choice(complex_queries)
                complexity = "complex"

            queries.append({"text": text, "complexity": complexity})

        return queries

    def simulate_cost(self, model_id: str, complexity: str) -> float:
        """Simulate cost for a model and query complexity using real pricing.

        Args:
            model_id: Model identifier
            complexity: Query complexity ("simple" or "complex")

        Returns:
            Cost in dollars
        """
        # Use real pricing from pricing.yaml
        # Assumes ~500 input + 200 output tokens for simple queries
        # Assumes ~1000 input + 500 output tokens for complex queries
        
        # Try to find pricing for this model (handle aliases)
        model_pricing = None
        
        # Direct match
        if model_id in self.pricing:
            model_pricing = self.pricing[model_id]
        # Try common aliases
        elif model_id == "claude-opus-4.5":
            model_pricing = self.pricing.get("claude-opus-4-5-20241124")
        elif model_id == "claude-sonnet-4.5":
            model_pricing = self.pricing.get("claude-sonnet-4-5-20241124")
        elif model_id == "gpt-5.1":
            model_pricing = self.pricing.get("gpt-5.1") or self.pricing.get("gpt-5")
        
        # Fallback to default pricing
        if model_pricing is None:
            model_pricing = self.default_pricing
        
        # Calculate cost based on token usage
        if complexity == "simple":
            input_tokens = 500
            output_tokens = 200
        else:  # complex
            input_tokens = 1000
            output_tokens = 500
        
        # Pricing is per million tokens
        input_cost = (model_pricing["input"] * input_tokens) / 1_000_000
        output_cost = (model_pricing["output"] * output_tokens) / 1_000_000
        
        return input_cost + output_cost

    def simulate_quality(self, model_id: str, complexity: str) -> float:
        """Simulate quality for a model and query complexity.

        Args:
            model_id: Model identifier
            complexity: Query complexity ("simple" or "complex")

        Returns:
            Quality score (0.0-1.0)
        """
        # Simplified quality model
        if "mini" in model_id.lower() or "haiku" in model_id.lower():
            # Cheaper models work well for simple queries, struggle with complex
            return 0.90 if complexity == "simple" else 0.65
        else:
            # Premium models work well for both
            return 0.95 if complexity == "simple" else 0.90

    async def run_benchmark(self, num_queries: int = 1000):
        """Run the benchmark.

        Args:
            num_queries: Number of queries to simulate
        """
        print(f"Running benchmark with {num_queries} queries...")
        queries = self.generate_queries(num_queries)

        conduit_costs = []
        static_best_costs = []
        static_cheap_costs = []

        conduit_quality = []
        static_best_quality = []
        static_cheap_quality = []

        for i, query_info in enumerate(queries, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{num_queries} queries...")

            query = Query(text=query_info["text"])
            complexity = query_info["complexity"]

            # Conduit routing
            decision = await self.conduit_router.route(query)
            conduit_model = decision.selected_model
            conduit_cost = self.simulate_cost(conduit_model, complexity)
            conduit_quality_score = self.simulate_quality(conduit_model, complexity)
            conduit_costs.append(conduit_cost)
            conduit_quality.append(conduit_quality_score)

            # Static: Always best
            static_best_model = await self.static_always_best.route(query)
            static_best_cost = self.simulate_cost(static_best_model, complexity)
            static_best_quality_score = self.simulate_quality(static_best_model, complexity)
            static_best_costs.append(static_best_cost)
            static_best_quality.append(static_best_quality_score)

            # Static: Always cheap
            static_cheap_model = await self.static_always_cheap.route(query)
            static_cheap_cost = self.simulate_cost(static_cheap_model, complexity)
            static_cheap_quality_score = self.simulate_quality(static_cheap_model, complexity)
            static_cheap_costs.append(static_cheap_cost)
            static_cheap_quality.append(static_cheap_quality_score)

            # Update Conduit with feedback
            from conduit.engines.bandits.base import BanditFeedback

            feedback = BanditFeedback(
                model_id=conduit_model,
                cost=conduit_cost,
                quality_score=conduit_quality_score,
                latency=1.0,
            )
            features = await self.conduit_router.analyzer.analyze(query)
            await self.conduit_router.hybrid_router.update(feedback, features)

            # Store result
            self.results.append(
                {
                    "query_num": i,
                    "complexity": complexity,
                    "conduit_model": conduit_model,
                    "conduit_cost": conduit_cost,
                    "conduit_quality": conduit_quality_score,
                    "static_best_cost": static_best_cost,
                    "static_best_quality": static_best_quality_score,
                    "static_cheap_cost": static_cheap_cost,
                    "static_cheap_quality": static_cheap_quality_score,
                }
            )

        # Calculate cumulative costs
        conduit_cumulative = np.cumsum(conduit_costs)
        static_best_cumulative = np.cumsum(static_best_costs)
        static_cheap_cumulative = np.cumsum(static_cheap_costs)

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        total_conduit = conduit_cumulative[-1]
        total_static_best = static_best_cumulative[-1]
        total_static_cheap = static_cheap_cumulative[-1]

        avg_conduit_quality = np.mean(conduit_quality)
        avg_static_best_quality = np.mean(static_best_quality)
        avg_static_cheap_quality = np.mean(static_cheap_quality)

        print(f"\nTotal Cost (1000 queries):")
        print(f"  Conduit (Adaptive):     ${total_conduit:.4f}")
        print(f"  Static (Always Best):    ${total_static_best:.4f}")
        print(f"  Static (Always Cheap):  ${total_static_cheap:.4f}")

        print(f"\nCost Savings:")
        savings_vs_best = ((total_static_best - total_conduit) / total_static_best) * 100
        savings_vs_cheap = ((total_static_cheap - total_conduit) / total_static_cheap) * 100
        print(f"  vs. Always Best:  {savings_vs_best:.1f}% (${total_static_best - total_conduit:.4f})")
        print(f"  vs. Always Cheap:  {savings_vs_cheap:.1f}% (${total_static_cheap - total_conduit:.4f})")

        print(f"\nAverage Quality:")
        print(f"  Conduit (Adaptive):     {avg_conduit_quality:.3f}")
        print(f"  Static (Always Best):    {avg_static_best_quality:.3f}")
        print(f"  Static (Always Cheap):  {avg_static_cheap_quality:.3f}")

        # Save CSV
        self.save_csv()

        # Generate graph
        self.generate_graph(conduit_cumulative, static_best_cumulative, static_cheap_cumulative)

    def save_csv(self):
        """Save results to CSV file."""
        filename = "cost_savings_results.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to: {filename}")

    def generate_graph(
        self,
        conduit_cumulative: np.ndarray,
        static_best_cumulative: np.ndarray,
        static_cheap_cumulative: np.ndarray,
        conduit_quality: List[float],
        static_best_quality: List[float],
        static_cheap_quality: List[float],
    ):
        """Generate cost savings graph with dual-axis showing cost and quality.

        Args:
            conduit_cumulative: Cumulative costs for Conduit
            static_best_cumulative: Cumulative costs for static (always best)
            static_cheap_cumulative: Cumulative costs for static (always cheap)
            conduit_quality: Quality scores for Conduit (per query)
            static_best_quality: Quality scores for static best (per query)
            static_cheap_quality: Quality scores for static cheap (per query)
        """
        fig, ax1 = plt.subplots(figsize=(14, 7))

        x = np.arange(1, len(conduit_cumulative) + 1)

        # Left axis: Cumulative Cost
        ax1.plot(x, conduit_cumulative, label="Conduit Cost", linewidth=2.5, color="#10b981")
        ax1.plot(x, static_best_cumulative, label="Static Best Cost", linewidth=2, color="#ef4444", linestyle="--", alpha=0.8)
        ax1.plot(x, static_cheap_cumulative, label="Static Cheap Cost", linewidth=2, color="#f59e0b", linestyle="--", alpha=0.8)
        ax1.set_xlabel("Query Number", fontsize=12)
        ax1.set_ylabel("Cumulative Cost ($)", fontsize=12, color="#1f2937")
        ax1.tick_params(axis="y", labelcolor="#1f2937")
        ax1.grid(True, alpha=0.3)

        # Right axis: Average Quality (rolling average)
        ax2 = ax1.twinx()
        
        # Calculate rolling average quality (window of 50 queries)
        window = 50
        conduit_quality_rolling = []
        static_best_quality_rolling = []
        static_cheap_quality_rolling = []
        
        for i in range(len(conduit_quality)):
            start = max(0, i - window + 1)
            conduit_qual = np.mean(conduit_quality[start : i + 1])
            static_best_qual = np.mean(static_best_quality[start : i + 1])
            static_cheap_qual = np.mean(static_cheap_quality[start : i + 1])
            conduit_quality_rolling.append(conduit_qual)
            static_best_quality_rolling.append(static_best_qual)
            static_cheap_quality_rolling.append(static_cheap_qual)
        
        ax2.plot(x, conduit_quality_rolling, label="Conduit Quality", linewidth=2, color="#10b981", linestyle=":", alpha=0.7)
        ax2.plot(x, static_best_quality_rolling, label="Static Best Quality", linewidth=1.5, color="#ef4444", linestyle=":", alpha=0.6)
        ax2.plot(x, static_cheap_quality_rolling, label="Static Cheap Quality", linewidth=1.5, color="#f59e0b", linestyle=":", alpha=0.6)
        ax2.set_ylabel("Average Quality (rolling 50 queries)", fontsize=12, color="#6366f1")
        ax2.tick_params(axis="y", labelcolor="#6366f1")
        ax2.set_ylim([0.6, 1.0])  # Quality range

        # Title and legend
        plt.title("Cost & Quality Comparison: Conduit vs. Static Routing", fontsize=14, fontweight="bold", pad=20)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

        # Add savings annotation
        final_savings = ((static_best_cumulative[-1] - conduit_cumulative[-1]) / static_best_cumulative[-1]) * 100
        final_conduit_quality = conduit_quality_rolling[-1] if conduit_quality_rolling else 0.0
        final_best_quality = static_best_quality_rolling[-1] if static_best_quality_rolling else 1.0
        quality_retention = (final_conduit_quality / final_best_quality) * 100 if final_best_quality > 0 else 0
        
        ax1.annotate(
            f"Savings: {final_savings:.1f}%\nQuality: {quality_retention:.1f}%",
            xy=(len(conduit_cumulative), conduit_cumulative[-1]),
            xytext=(len(conduit_cumulative) * 0.65, conduit_cumulative[-1] * 1.15),
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#10b981"),
        )

        plt.tight_layout()
        filename = "cost_savings_graph.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Graph saved to: {filename}")


async def main():
    """Run the benchmark."""
    import sys
    # Allow smaller test runs: python cost_savings_comparison.py 100
    num_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    benchmark = CostSavingsBenchmark()
    await benchmark.run_benchmark(num_queries=num_queries)


if __name__ == "__main__":
    asyncio.run(main())

