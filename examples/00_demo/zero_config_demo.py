#!/usr/bin/env python3
"""Zero-Config Demo: Conduit Learning in Action.

Demonstrates Conduit's bandit learning with zero external dependencies.
No API keys, no database, no Redis - just pure Python.

Run:
    python examples/00_demo/zero_config_demo.py

This demo shows:
    - Phase 1 (Exploration): Random-ish model selection while gathering data
    - Phase 2 (Exploitation): Learned routing based on feedback
    - Measurable improvement: Phase 2 achieves higher rewards than Phase 1

How it works:
    Uses Thompson Sampling, the same algorithm Conduit uses for cold-start
    routing in production. Simulates LLM responses where expensive models
    outperform cheap ones on complex queries.

Example usage:
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("CONDUIT ZERO-CONFIG DEMO")
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from conduit.core.models import QueryFeatures
from conduit.engines.bandits import BanditFeedback, ModelArm, ThompsonSamplingBandit

logger = logging.getLogger(__name__)

# Simulated model definitions with realistic pricing
SIMULATED_MODELS: list[dict[str, Any]] = [
    {
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "cost_per_input_token": 0.00000015,  # $0.15 per 1M tokens
        "cost_per_output_token": 0.0000006,  # $0.60 per 1M tokens
        "quality_simple": 0.88,  # Quality on simple tasks
        "quality_complex": 0.72,  # Quality on complex tasks
    },
    {
        "model_id": "gpt-4o",
        "provider": "openai",
        "cost_per_input_token": 0.0000025,  # $2.50 per 1M tokens
        "cost_per_output_token": 0.00001,  # $10 per 1M tokens
        "quality_simple": 0.90,  # Slight advantage on simple tasks
        "quality_complex": 0.95,  # Significantly better on complex tasks
    },
    {
        "model_id": "claude-3-5-haiku",
        "provider": "anthropic",
        "cost_per_input_token": 0.0000008,  # $0.80 per 1M tokens
        "cost_per_output_token": 0.000004,  # $4 per 1M tokens
        "quality_simple": 0.86,
        "quality_complex": 0.78,
    },
]


# Demo queries with varying complexity
DEMO_QUERIES: list[dict[str, Any]] = [
    # Simple queries (complexity < 0.4)
    {"text": "What is 2 + 2?", "complexity": 0.1},
    {"text": "What color is the sky?", "complexity": 0.15},
    {"text": "How many days in a week?", "complexity": 0.1},
    {"text": "Capital of France?", "complexity": 0.2},
    {"text": "What year is it?", "complexity": 0.1},
    {"text": "Is water wet?", "complexity": 0.15},
    # Medium complexity (0.4-0.7)
    {"text": "Explain photosynthesis briefly", "complexity": 0.5},
    {"text": "What causes seasons?", "complexity": 0.45},
    {"text": "Summarize the water cycle", "complexity": 0.5},
    {"text": "How do vaccines work?", "complexity": 0.55},
    # Complex queries (complexity >= 0.7)
    {"text": "Analyze economic implications of AI automation", "complexity": 0.85},
    {"text": "Compare quantum computing paradigms", "complexity": 0.9},
    {"text": "Design a distributed system architecture", "complexity": 0.95},
    {"text": "Prove the halting problem is undecidable", "complexity": 0.9},
    {
        "text": "Explain consciousness from a neuroscience perspective",
        "complexity": 0.85,
    },
]


def simulate_llm_response(
    model_id: str, complexity: float
) -> tuple[float, float, float]:
    """Simulate LLM response quality, cost, and latency.

    This function models realistic LLM behavior:
    - Simple queries: All models perform similarly well
    - Complex queries: Expensive models significantly outperform cheap ones

    Args:
        model_id: Model to simulate
        complexity: Query complexity (0.0-1.0)

    Returns:
        Tuple of (quality_score, cost, latency)
    """
    model = next((m for m in SIMULATED_MODELS if m["model_id"] == model_id), None)
    if model is None:
        return 0.5, 0.001, 1.0

    # Calculate quality based on complexity
    if complexity < 0.4:
        # Simple query: use simple quality
        base_quality = model["quality_simple"]
    elif complexity >= 0.7:
        # Complex query: use complex quality
        base_quality = model["quality_complex"]
    else:
        # Medium complexity: linear interpolation
        t = (complexity - 0.4) / 0.3
        base_quality = model["quality_simple"] * (1 - t) + model["quality_complex"] * t

    # Add small random noise for realism
    quality = max(0.0, min(1.0, base_quality + random.gauss(0, 0.02)))

    # Simulate cost (assume 100 input tokens, 50 output tokens)
    cost = model["cost_per_input_token"] * 100 + model["cost_per_output_token"] * 50

    # Simulate latency (more expensive models slightly slower)
    base_latency = 0.5 if "mini" in model_id or "haiku" in model_id else 0.8
    latency = base_latency + random.uniform(0, 0.2)

    return quality, cost, latency


def create_mock_features(complexity: float) -> QueryFeatures:
    """Create mock QueryFeatures for the demo.

    Thompson Sampling is non-contextual, so embeddings are not used.
    We only need token_count and complexity_score for display purposes.
    """
    return QueryFeatures(
        embedding=[0.0] * 384,  # Not used by Thompson Sampling
        token_count=int(complexity * 100 + 20),
        complexity_score=complexity,
        query_text=None,
    )


async def run_demo() -> None:
    """Run the zero-config demo showing bandit learning in action."""
    logger.info("=" * 70)
    logger.info("CONDUIT ZERO-CONFIG DEMO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This demo shows how Conduit learns to route queries optimally.")
    logger.info("No API keys, no database, no external dependencies required.")
    logger.info("")

    # Create model arms
    logger.info("-" * 70)
    logger.info("SETUP: Creating bandit with 3 models")
    logger.info("-" * 70)

    arms = [
        ModelArm(
            model_id=m["model_id"],
            provider=m["provider"],
            model_name=m["model_id"],
            cost_per_input_token=m["cost_per_input_token"],
            cost_per_output_token=m["cost_per_output_token"],
            expected_quality=0.5,  # Neutral prior
        )
        for m in SIMULATED_MODELS
    ]

    for arm in arms:
        cost_per_1m = arm.cost_per_input_token * 1_000_000
        logger.info(f"  - {arm.model_id} ({arm.provider}): ${cost_per_1m:.2f}/1M tokens")

    # Initialize Thompson Sampling bandit
    bandit = ThompsonSamplingBandit(
        arms=arms,
        prior_alpha=1.0,  # Uniform prior (no initial preference)
        prior_beta=1.0,
        random_seed=42,  # Reproducible results
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: EXPLORATION (Learning from feedback)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The bandit explores different models and learns from feedback.")
    logger.info("Early routing is more random as the system gathers data.")
    logger.info("")

    # Phase 1: Exploration (30 queries)
    phase1_rewards: list[float] = []
    phase1_costs: list[float] = []
    phase1_selections: dict[str, int] = {m["model_id"]: 0 for m in SIMULATED_MODELS}

    random.seed(42)  # For reproducible simulations

    for i in range(30):
        # Pick a random query
        query = random.choice(DEMO_QUERIES)
        features = create_mock_features(query["complexity"])

        # Select model using bandit
        arm = await bandit.select_arm(features)
        phase1_selections[arm.model_id] += 1

        # Simulate LLM response
        quality, cost, latency = simulate_llm_response(
            arm.model_id, query["complexity"]
        )

        # Provide feedback to bandit
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=cost,
            quality_score=quality,
            latency=latency,
        )
        await bandit.update(feedback, features)

        # Track metrics
        reward = feedback.calculate_reward()
        phase1_rewards.append(reward)
        phase1_costs.append(cost)

        # Show progress every 10 queries
        if (i + 1) % 10 == 0:
            avg_reward = sum(phase1_rewards[-10:]) / 10
            logger.info(f"  Queries 1-{i+1}: Avg reward = {avg_reward:.3f}")

    phase1_avg_reward = sum(phase1_rewards) / len(phase1_rewards)
    phase1_total_cost = sum(phase1_costs)

    logger.info("")
    logger.info("Phase 1 Summary:")
    logger.info(f"  Average reward: {phase1_avg_reward:.3f}")
    logger.info(f"  Total cost: ${phase1_total_cost:.6f}")
    logger.info(f"  Model selections: {phase1_selections}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: EXPLOITATION (Using learned preferences)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The bandit now exploits what it learned, making smarter selections.")
    logger.info("Watch for higher rewards as it picks better models.")
    logger.info("")

    # Phase 2: Exploitation (30 queries)
    phase2_rewards: list[float] = []
    phase2_costs: list[float] = []
    phase2_selections: dict[str, int] = {m["model_id"]: 0 for m in SIMULATED_MODELS}

    for i in range(30):
        # Pick a random query
        query = random.choice(DEMO_QUERIES)
        features = create_mock_features(query["complexity"])

        # Select model using bandit
        arm = await bandit.select_arm(features)
        phase2_selections[arm.model_id] += 1

        # Simulate LLM response
        quality, cost, latency = simulate_llm_response(
            arm.model_id, query["complexity"]
        )

        # Provide feedback to bandit
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=cost,
            quality_score=quality,
            latency=latency,
        )
        await bandit.update(feedback, features)

        # Track metrics
        reward = feedback.calculate_reward()
        phase2_rewards.append(reward)
        phase2_costs.append(cost)

        # Show progress every 10 queries
        if (i + 1) % 10 == 0:
            avg_reward = sum(phase2_rewards[-10:]) / 10
            logger.info(f"  Queries 31-{30+i+1}: Avg reward = {avg_reward:.3f}")

    phase2_avg_reward = sum(phase2_rewards) / len(phase2_rewards)
    phase2_total_cost = sum(phase2_costs)

    logger.info("")
    logger.info("Phase 2 Summary:")
    logger.info(f"  Average reward: {phase2_avg_reward:.3f}")
    logger.info(f"  Total cost: ${phase2_total_cost:.6f}")
    logger.info(f"  Model selections: {phase2_selections}")

    # Final comparison
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS: Learning Improvement")
    logger.info("=" * 70)
    logger.info("")

    improvement = ((phase2_avg_reward - phase1_avg_reward) / phase1_avg_reward) * 100
    cost_change = ((phase2_total_cost - phase1_total_cost) / phase1_total_cost) * 100

    logger.info(f"  Phase 1 avg reward: {phase1_avg_reward:.3f}")
    logger.info(f"  Phase 2 avg reward: {phase2_avg_reward:.3f}")
    logger.info(f"  Improvement: {improvement:+.1f}%")
    logger.info("")
    logger.info(f"  Phase 1 total cost: ${phase1_total_cost:.6f}")
    logger.info(f"  Phase 2 total cost: ${phase2_total_cost:.6f}")
    logger.info(f"  Cost change: {cost_change:+.1f}%")
    logger.info("")

    # Show learned model preferences
    logger.info("Learned Model Preferences (Beta distribution parameters):")
    stats = bandit.get_stats()
    for model_id, dist in stats["arm_distributions"].items():
        mean = dist["mean"]
        alpha = dist["alpha"]
        beta = dist["beta"]
        logger.info(f"  {model_id}: mean={mean:.3f} (alpha={alpha:.1f}, beta={beta:.1f})")

    logger.info("")
    logger.info("=" * 70)
    logger.info("WHAT THIS MEANS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The bandit learned to prefer models that provide better rewards.")
    logger.info("This is how Conduit optimizes LLM routing in production:")
    logger.info("  1. Start with neutral priors (no assumptions)")
    logger.info("  2. Explore different models via Thompson Sampling")
    logger.info("  3. Learn from feedback (quality, cost, latency)")
    logger.info("  4. Converge to optimal routing policy")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - See examples/quickstart.py for real LLM routing")
    logger.info("  - See examples/feedback_loop.py for production feedback patterns")
    logger.info("")


if __name__ == "__main__":
    asyncio.run(run_demo())
