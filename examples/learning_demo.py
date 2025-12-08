#!/usr/bin/env python3
"""Conduit Learning Demo - Real API Calls

Demonstrates Conduit learning with actual LLM providers.
Requires at least one API key (OPENAI_API_KEY recommended).

This demo shows:
- Real routing decisions using Conduit
- Actual cost and latency tracking
- Live feedback loop with real LLM responses

Run: uv run python examples/learning_demo.py

Environment variables (set at least one):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
"""

import asyncio
import os
import time
from typing import Any

from pydantic import BaseModel

from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.base import ModelArm, BanditFeedback
from conduit.core.models import QueryFeatures
from conduit.core.pricing import get_model_pricing, compute_cost

# Check for available API keys
def get_available_models() -> list[dict[str, Any]]:
    """Get list of available models based on API keys."""
    models = []

    if os.getenv("OPENAI_API_KEY"):
        models.extend([
            {
                "model_id": "gpt-4o-mini",
                "provider": "openai",
                "description": "Fast, cheap, good for simple tasks",
            },
            {
                "model_id": "gpt-4o",
                "provider": "openai",
                "description": "Balanced performance and cost",
            },
        ])

    if os.getenv("ANTHROPIC_API_KEY"):
        models.extend([
            {
                "model_id": "claude-3-5-haiku-20241022",
                "provider": "anthropic",
                "description": "Fast and affordable",
            },
            {
                "model_id": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "description": "Excellent for complex tasks",
            },
        ])

    if os.getenv("GOOGLE_API_KEY"):
        models.extend([
            {
                "model_id": "gemini-2.0-flash",
                "provider": "google",
                "description": "Fast and affordable",
            },
        ])

    return models

class Answer(BaseModel):
    """Simple answer model for LLM responses."""

    answer: str
    confidence: float

# Test queries covering different complexity levels
TEST_QUERIES = [
    # Simple Q&A (should favor fast/cheap models)
    {"text": "What is 2 + 2?", "domain": "simple_qa", "expected_answer_contains": "4"},
    {"text": "What is the capital of France?", "domain": "simple_qa", "expected_answer_contains": "Paris"},
    {"text": "How many days are in a week?", "domain": "simple_qa", "expected_answer_contains": "7"},

    # Code questions (may benefit from stronger models)
    {"text": "Write a Python one-liner to sum a list of numbers", "domain": "code", "expected_answer_contains": "sum"},
    {"text": "What does the Python 'zip' function do?", "domain": "code", "expected_answer_contains": ""},

    # Reasoning (benefits from stronger models)
    {"text": "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?", "domain": "reasoning", "expected_answer_contains": ""},

    # More simple Q&A
    {"text": "What color is grass?", "domain": "simple_qa", "expected_answer_contains": "green"},
    {"text": "How many legs does a spider have?", "domain": "simple_qa", "expected_answer_contains": "8"},
]

def create_features(query: dict) -> QueryFeatures:
    """Create query features for routing decision."""
    # Simple embedding based on domain (in production, use sentence-transformers)
    import random
    embedding = [random.gauss(0, 0.1) for _ in range(384)]

    # Add domain-specific signal
    domain = query.get("domain", "general")
    if domain == "simple_qa":
        for i in range(0, 128):
            embedding[i] += 0.5
    elif domain == "code":
        for i in range(128, 256):
            embedding[i] += 0.5
    else:  # reasoning
        for i in range(256, 384):
            embedding[i] += 0.5

    return QueryFeatures(query_text="test query", 
        embedding=embedding,
        token_count=len(query["text"].split()) * 2,
        complexity_score=0.3 if domain == "simple_qa" else 0.7,
        domain=domain,
        domain_confidence=0.9,
    )

async def call_llm(model_id: str, provider: str, prompt: str) -> tuple[str, float, float]:
    """Call LLM and return (response, cost, latency).

    Uses litellm for unified API access.
    """
    import litellm

    start_time = time.time()

    try:
        # Format model for litellm
        litellm_model = f"{provider}/{model_id}"
        if provider == "google":
            litellm_model = f"gemini/{model_id}"

        response = await litellm.acompletion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        latency = time.time() - start_time
        text = response.choices[0].message.content

        # Get cost from response metadata
        cost = response._hidden_params.get("response_cost", 0.0) or 0.0

        # If cost not available, estimate from tokens using LiteLLM pricing
        if cost == 0.0:
            usage = response.usage
            cost = compute_cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                model_id=model_id,
            )

        return text, cost, latency

    except Exception as e:
        latency = time.time() - start_time
        return f"Error: {e}", 0.0, latency

def evaluate_response(response: str, expected: str) -> float:
    """Simple quality evaluation (0-1 scale).

    In production, use LLM-as-judge (Arbiter) for better evaluation.
    """
    if expected and expected.lower() in response.lower():
        return 0.95  # Contains expected answer
    elif "error" in response.lower():
        return 0.1   # Error response
    else:
        return 0.7   # Plausible response (can't verify without expected)

async def run_real_demo():
    """Run the real API learning demonstration."""
    print("=" * 70)
    print("CONDUIT LEARNING DEMO - Real API Calls")
    print("=" * 70)
    print()

    # Check for available models
    available_models = get_available_models()

    if not available_models:
        print("ERROR: No API keys found!")
        print()
        print("Please set at least one of these environment variables:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export GOOGLE_API_KEY=...")
        print()
        print("Then run this demo again.")
        return

    print(f"Found {len(available_models)} available models:")
    for m in available_models:
        print(f"  - {m['model_id']} ({m['provider']}): {m['description']}")
    print()

    # Create model arms using LiteLLM pricing
    arms = []
    for m in available_models:
        pricing = get_model_pricing(m["model_id"])
        # Use pricing if available, otherwise use defaults
        input_cost = pricing.input_cost_per_token if pricing else 1.0 / 1_000_000
        output_cost = pricing.output_cost_per_token if pricing else 1.0 / 1_000_000
        arms.append(ModelArm(
            model_id=m["model_id"],
            provider=m["provider"],
            model_name=m["model_id"],
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
        ))

    # Initialize bandit
    bandit = LinUCBBandit(arms=arms, alpha=1.5, feature_dim=386)

    print(f"Initialized LinUCB bandit with {len(arms)} arms")
    print()
    print("-" * 70)
    print("Running queries (this will make real API calls)...")
    print("-" * 70)
    print()

    # Track metrics
    total_cost = 0.0
    total_quality = 0.0
    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        # Get features for routing
        features = create_features(query)

        # Get bandit's selection
        selected_arm = await bandit.select_arm(features)

        print(f"Query {i}: \"{query['text'][:50]}...\"")
        print(f"  Domain: {query['domain']}")
        print(f"  Selected: {selected_arm.model_id}")

        # Call the LLM
        response_text, cost, latency = await call_llm(
            selected_arm.model_id,
            selected_arm.provider,
            query["text"],
        )

        # Evaluate quality
        quality = evaluate_response(response_text, query.get("expected_answer_contains", ""))

        print(f"  Response: \"{response_text[:60]}...\"")
        print(f"  Cost: ${cost:.6f} | Latency: {latency:.2f}s | Quality: {quality:.2f}")
        print()

        # Update bandit with feedback
        feedback = BanditFeedback(
            model_id=selected_arm.model_id,
            cost=cost,
            quality_score=quality,
            latency=latency,
        )
        await bandit.update(feedback, features)

        # Track totals
        total_cost += cost
        total_quality += quality
        results.append({
            "query": query["text"],
            "domain": query["domain"],
            "model": selected_arm.model_id,
            "cost": cost,
            "latency": latency,
            "quality": quality,
        })

        # Small delay between API calls
        await asyncio.sleep(0.5)

    # Print summary
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print()
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Average quality: {total_quality/len(TEST_QUERIES):.2f}")
    print()

    # Model distribution
    print("Model Selection Distribution:")
    model_counts: dict[str, int] = {}
    model_costs: dict[str, float] = {}
    for r in results:
        model = r["model"]
        model_counts[model] = model_counts.get(model, 0) + 1
        model_costs[model] = model_costs.get(model, 0) + r["cost"]

    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        cost = model_costs[model]
        print(f"  {model}: {count} queries ({pct:.0f}%), ${cost:.6f}")

    print()
    print("Bandit Statistics:")
    stats = bandit.get_stats()
    arm_pulls = stats.get("arm_pulls", {})
    arm_success_rates = stats.get("arm_success_rates", {})
    for arm_id in arm_pulls:
        pulls = arm_pulls.get(arm_id, 0)
        success_rate = arm_success_rates.get(arm_id, 0)
        print(f"  {arm_id}: {pulls} pulls, success rate: {success_rate:.3f}")

    print()
    print("=" * 70)
    print("Conduit learned from real feedback to optimize model selection!")
    print("Run more queries to see the bandit converge to optimal routing.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_real_demo())
