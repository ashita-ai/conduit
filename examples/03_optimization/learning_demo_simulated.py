#!/usr/bin/env python3
"""Conduit Learning Demo - No API Keys Required

Demonstrates how Conduit learns optimal routing through feedback.
Uses simulated LLM responses to show the learning curve clearly.

This demo shows Conduit's unique value proposition:
- ML-based online learning (not static rules)
- Adapts to YOUR workload through feedback
- Learns which model is best for each query type

Run: uv run python examples/03_optimization/learning_demo_simulated.py
"""

import asyncio
import random
from dataclasses import dataclass

import numpy as np

from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.base import ModelArm, BanditFeedback
from conduit.core.models import QueryFeatures


@dataclass
class SimulatedModel:
    """Simulated LLM model with known characteristics."""

    model_id: str
    provider: str

    # Performance characteristics (0-1 scale)
    simple_qa_quality: float
    code_quality: float
    creative_quality: float

    # Cost per query (simulated)
    cost: float

    # Latency in seconds (simulated)
    latency: float


# Define simulated models with known characteristics
SIMULATED_MODELS = [
    SimulatedModel(
        model_id="fast-cheap",
        provider="simulated",
        simple_qa_quality=0.90,  # Excellent for simple queries
        code_quality=0.55,       # Mediocre for code
        creative_quality=0.50,   # Poor for creative
        cost=0.001,
        latency=0.5,
    ),
    SimulatedModel(
        model_id="balanced",
        provider="simulated",
        simple_qa_quality=0.80,
        code_quality=0.75,
        creative_quality=0.70,
        cost=0.01,
        latency=1.5,
    ),
    SimulatedModel(
        model_id="premium",
        provider="simulated",
        simple_qa_quality=0.85,  # Overkill for simple
        code_quality=0.95,       # Excellent for code
        creative_quality=0.92,   # Excellent for creative
        cost=0.05,
        latency=3.0,
    ),
]


@dataclass
class Query:
    """Simulated query with known optimal model."""

    text: str
    domain: str  # "simple_qa", "code", "creative"

    def get_optimal_model(self) -> str:
        """Return the optimal model for this query type."""
        if self.domain == "simple_qa":
            return "fast-cheap"  # Cheap model is best
        elif self.domain == "code":
            return "premium"     # Premium model needed
        else:  # creative
            return "premium"     # Premium model needed


def generate_query_batch(n: int = 100) -> list[Query]:
    """Generate a batch of queries with known optimal models."""
    queries = []

    simple_qa_examples = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "How many days in a week?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
    ]

    code_examples = [
        "Write a function to sort a list in Python",
        "Implement binary search in TypeScript",
        "Create a REST API endpoint in FastAPI",
        "Debug this async/await race condition",
        "Optimize this SQL query for performance",
    ]

    creative_examples = [
        "Write a haiku about machine learning",
        "Create a story about a robot finding love",
        "Compose a marketing tagline for an AI product",
        "Write a persuasive essay on climate change",
        "Generate creative names for a tech startup",
    ]

    for _ in range(n):
        # Distribute: 50% simple, 30% code, 20% creative
        r = random.random()
        if r < 0.50:
            text = random.choice(simple_qa_examples)
            domain = "simple_qa"
        elif r < 0.80:
            text = random.choice(code_examples)
            domain = "code"
        else:
            text = random.choice(creative_examples)
            domain = "creative"

        queries.append(Query(text=text, domain=domain))

    return queries


def simulate_response(model: SimulatedModel, query: Query) -> tuple[float, float, float]:
    """Simulate model response and return (quality, cost, latency).

    Quality is based on model's strength for the query domain + noise.
    """
    # Get base quality for this query type
    if query.domain == "simple_qa":
        base_quality = model.simple_qa_quality
    elif query.domain == "code":
        base_quality = model.code_quality
    else:
        base_quality = model.creative_quality

    # Add realistic noise (models aren't perfectly consistent)
    noise = np.random.normal(0, 0.05)
    quality = np.clip(base_quality + noise, 0.0, 1.0)

    # Add small variation to cost and latency
    cost = model.cost * (1 + np.random.normal(0, 0.1))
    latency = model.latency * (1 + np.random.normal(0, 0.1))

    return float(quality), float(cost), float(latency)


def create_features(query: Query) -> QueryFeatures:
    """Create query features for the bandit (simplified for demo)."""
    # Create a simple embedding based on domain
    # In real usage, this would be sentence-transformers
    embedding = [0.0] * 384

    if query.domain == "simple_qa":
        # Simple queries have high values in first 128 dims
        for i in range(128):
            embedding[i] = 0.5 + random.random() * 0.5
    elif query.domain == "code":
        # Code queries have high values in middle 128 dims
        for i in range(128, 256):
            embedding[i] = 0.5 + random.random() * 0.5
    else:  # creative
        # Creative queries have high values in last 128 dims
        for i in range(256, 384):
            embedding[i] = 0.5 + random.random() * 0.5

    # Add some noise for realism
    for i in range(384):
        embedding[i] += random.gauss(0, 0.1)

    return QueryFeatures(
        embedding=embedding,
        token_count=len(query.text.split()) * 2,
        complexity_score=0.3 if query.domain == "simple_qa" else 0.7,
        domain=query.domain,
        domain_confidence=0.9,
    )


async def run_learning_demo():
    """Run the learning demonstration."""
    print("=" * 70)
    print("CONDUIT LEARNING DEMO - Simulated (No API Keys Required)")
    print("=" * 70)
    print()
    print("This demo shows how Conduit learns optimal routing through feedback.")
    print("We simulate 3 models with known strengths:")
    print()
    print("  fast-cheap: Best for simple Q&A (cost: $0.001, latency: 0.5s)")
    print("  balanced:   Medium performance across all tasks")
    print("  premium:    Best for code and creative (cost: $0.05, latency: 3s)")
    print()
    print("Query distribution: 50% simple Q&A, 30% code, 20% creative")
    print("Optimal routing: simple_qa->fast-cheap, code/creative->premium")
    print()
    print("-" * 70)

    # Create model arms for the bandit
    arms = [
        ModelArm(
            model_id=m.model_id,
            provider=m.provider,
            model_name=m.model_id,
            cost_per_input_token=m.cost / 1000,   # Cost per 1K tokens
            cost_per_output_token=m.cost / 1000,  # Cost per 1K tokens
        )
        for m in SIMULATED_MODELS
    ]

    # Initialize LinUCB bandit
    bandit = LinUCBBandit(arms=arms, alpha=1.0, feature_dim=387)

    # Generate queries
    queries = generate_query_batch(100)

    # Track metrics
    optimal_choices = []
    cumulative_cost = 0.0
    cumulative_quality = 0.0

    # Process in batches for cleaner output
    batch_size = 20

    for batch_idx in range(0, len(queries), batch_size):
        batch_queries = queries[batch_idx:batch_idx + batch_size]
        batch_optimal = 0
        batch_cost = 0.0
        batch_quality = 0.0

        for query in batch_queries:
            # Create features
            features = create_features(query)

            # Get bandit's selection
            selected_arm = await bandit.select_arm(features)

            # Get corresponding model
            model = next(m for m in SIMULATED_MODELS if m.model_id == selected_arm.model_id)

            # Simulate response
            quality, cost, latency = simulate_response(model, query)

            # Track if optimal choice was made
            optimal_model = query.get_optimal_model()
            is_optimal = selected_arm.model_id == optimal_model
            batch_optimal += 1 if is_optimal else 0

            # Track cost and quality
            batch_cost += cost
            batch_quality += quality

            # Provide feedback to bandit
            feedback = BanditFeedback(
                model_id=selected_arm.model_id,
                cost=cost,
                quality_score=quality,
                latency=latency,
            )
            await bandit.update(feedback, features)

        # Record batch metrics
        optimal_choices.append(batch_optimal / len(batch_queries))
        cumulative_cost += batch_cost
        cumulative_quality += batch_quality

        # Print progress
        queries_processed = batch_idx + len(batch_queries)
        optimal_rate = batch_optimal / len(batch_queries) * 100
        avg_quality = batch_quality / len(batch_queries)

        print(f"Queries {batch_idx+1:3d}-{queries_processed:3d}: "
              f"Optimal: {optimal_rate:5.1f}% | "
              f"Quality: {avg_quality:.3f} | "
              f"Cost: ${batch_cost:.4f}")

    # Print summary
    print()
    print("-" * 70)
    print("LEARNING SUMMARY")
    print("-" * 70)
    print()

    # Show learning curve
    print("Optimal Selection Rate by Phase:")
    print(f"  Queries   1-20  (exploring): {optimal_choices[0]*100:.1f}%")
    print(f"  Queries  21-40  (learning):  {optimal_choices[1]*100:.1f}%")
    print(f"  Queries  41-60  (improving): {optimal_choices[2]*100:.1f}%")
    print(f"  Queries  61-80  (converging):{optimal_choices[3]*100:.1f}%")
    print(f"  Queries  81-100 (optimal):   {optimal_choices[4]*100:.1f}%")
    print()

    avg_optimal = sum(optimal_choices) / len(optimal_choices) * 100
    final_optimal = optimal_choices[-1] * 100

    print(f"Average optimal selection: {avg_optimal:.1f}%")
    print(f"Final phase optimal rate:  {final_optimal:.1f}%")
    print(f"Total cost: ${cumulative_cost:.4f}")
    print(f"Average quality: {cumulative_quality/100:.3f}")
    print()

    # Show model statistics
    print("Model Selection Statistics:")
    stats = bandit.get_stats()
    arm_pulls = stats.get("arm_pulls", {})
    arm_success_rates = stats.get("arm_success_rates", {})
    for arm_id in arm_pulls:
        pulls = arm_pulls.get(arm_id, 0)
        success_rate = arm_success_rates.get(arm_id, 0)
        print(f"  {arm_id:12s}: {pulls:3d} queries, success rate: {success_rate:.3f}")

    print()
    print("=" * 70)
    print("KEY INSIGHT: Conduit learned which model works best for each query type")
    print("through real feedback - no manual rules needed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_learning_demo())
