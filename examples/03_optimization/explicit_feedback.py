"""Explicit Feedback - User Ratings and Quality Scores.

Demonstrates how to submit explicit user feedback (ratings,
quality scores, expectations) to improve model selection.
"""

import asyncio

from conduit.core.models import Feedback, Query, QueryFeatures
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.feedback import FeedbackIntegrator


async def main():
    print("Explicit Feedback Demo - User Ratings\n")

    # Setup
    analyzer = QueryAnalyzer()
    bandit = ContextualBandit(models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"])
    integrator = FeedbackIntegrator(bandit)

    # Scenario 1: High Quality + Met Expectations
    print("="*60)
    print("Scenario 1: Excellent Response")
    print("="*60)

    query1 = Query(text="Explain Python in simple terms")
    features1 = await analyzer.analyze(query1.text)

    # Simulate excellent response
    feedback1 = Feedback(
        response_id="resp_001",
        quality_score=0.95,  # 0-1 scale
        user_rating=5,  # 1-5 stars
        met_expectations=True,
        comments="Very clear explanation!",
    )

    print(f"Quality score: {feedback1.quality_score:.0%}")
    print(f"User rating: {feedback1.user_rating}/5 stars")
    print(f"Met expectations: {feedback1.met_expectations}")

    integrator.update_from_explicit("gpt-4o-mini", features1, feedback1)
    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\nModel success rate: {state.mean_success_rate:.1%}")
    print(f"Alpha (successes): {state.alpha:.0f}")

    # Scenario 2: Low Quality + Unmet Expectations
    print("\n" + "="*60)
    print("Scenario 2: Poor Response")
    print("="*60)

    query2 = Query(text="Write a detailed essay on AI")
    features2 = await analyzer.analyze(query2.text)

    # Simulate poor response
    feedback2 = Feedback(
        response_id="resp_002",
        quality_score=0.3,
        user_rating=2,
        met_expectations=False,
        comments="Too brief, not helpful",
    )

    print(f"Quality score: {feedback2.quality_score:.0%}")
    print(f"User rating: {feedback2.user_rating}/5 stars")
    print(f"Met expectations: {feedback2.met_expectations}")

    integrator.update_from_explicit("gpt-4o-mini", features2, feedback2)
    state = bandit.get_model_state("gpt-4o-mini")
    print(f"\nModel success rate: {state.mean_success_rate:.1%}")
    print(f"Beta (failures): {state.beta:.0f}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\n💡 Explicit feedback directly updates Thompson Sampling:")
    print("   - High quality → More likely to select this model")
    print("   - Low quality → Less likely to select this model")
    print("   - Weighted 70% (explicit) vs 30% (implicit)")

    print("\n📊 Final Model States:")
    for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]:
        state = bandit.get_model_state(model_id)
        print(f"   {model_id}: {state.mean_success_rate:.0%} success rate")


if __name__ == "__main__":
    asyncio.run(main())
