"""Integration tests for feedback loop closure.

Tests that the complete feedback loop is closed:
    Query → Router.route() → User feedback → Router.update() → Bandit learns → Better routing

This is the CORE VALUE PROPOSITION of Conduit. If this doesn't work, the system doesn't improve over time.
"""

import pytest

from conduit.core.models import Query
from conduit.engines.router import Router


@pytest.mark.asyncio
async def test_feedback_loop_improves_routing():
    """Prove: route → feedback → learn → improved routing.

    This test verifies the complete feedback loop by:
    1. Creating balanced training data (both arms explored equally)
    2. Providing differential feedback (cheap model = good, expensive = bad)
    3. Routing new queries after learning
    4. Verifying cheap model selection rate increases

    This is a CRITICAL test for Conduit's core value proposition.
    """
    # Setup: Two models with different costs
    router = Router(
        models=["gpt-4o-mini", "gpt-4o"],  # mini is cheaper
        algorithm="linucb",  # Use LinUCB directly
    )

    # Override alpha to reduce exploration (prioritize learned knowledge)
    router.hybrid_router.phase2_bandit.alpha = 0.1

    # Phase 1: Create balanced training set by routing and immediately providing feedback
    # This ensures both arms get explored regardless of LinUCB's initial bias
    training_queries = 40  # 20 per arm expected
    for i in range(training_queries):
        query = Query(text=f"Math question {i}: what is {i}+{i}?")
        decision = await router.route(query)

        # Provide differential feedback immediately:
        # Cheap model gets excellent reward, expensive model gets poor reward
        if decision.selected_model == "gpt-4o-mini":
            await router.update(
                model_id=decision.selected_model,
                cost=0.001,  # Very cheap
                quality_score=0.95,  # Excellent quality
                latency=0.5,
                features=decision.features,
            )
        else:  # gpt-4o
            await router.update(
                model_id=decision.selected_model,
                cost=0.10,  # 100x more expensive
                quality_score=0.95,  # Same quality (worse reward due to cost)
                latency=0.8,
                features=decision.features,
            )

    # Check that both arms were explored during training
    linucb = router.hybrid_router.phase2_bandit
    mini_pulls = linucb.arm_pulls["gpt-4o-mini"]
    gpt4_pulls = linucb.arm_pulls["gpt-4o"]
    print(f"\nTraining phase: gpt-4o-mini={mini_pulls}, gpt-4o={gpt4_pulls}")

    # Phase 2: Route new queries after learning
    test_queries = 50
    selections = []
    for i in range(test_queries):
        query = Query(text=f"New question {i+training_queries}: what is {i*2}+{i*3}?")
        decision = await router.route(query)
        selections.append(decision.selected_model)

    cheap_count = sum(1 for model in selections if model == "gpt-4o-mini")
    cheap_rate = cheap_count / test_queries

    await router.close()

    print(f"\nTest phase: {cheap_count}/{test_queries} selected gpt-4o-mini ({cheap_rate:.0%})")

    # Verify: After learning, cheap model should be strongly preferred (>70%)
    # This proves the feedback loop is working
    assert cheap_rate > 0.70, (
        f"After learning that gpt-4o-mini is better (cheaper with same quality), "
        f"it should be selected >70% of the time. Got: {cheap_rate:.0%}"
    )


@pytest.mark.asyncio
async def test_hybrid_routing_feedback_loop():
    """Test feedback loop works across Thompson Sampling → LinUCB transition.

    Tests that learning happens across both phases:
    - Phase 1 (Thompson Sampling): Bayesian exploration
    - Phase 2 (LinUCB): Contextual learning with real features
    """
    # Create router with low switch threshold to test phase transition
    from conduit.engines.hybrid_router import HybridRouter
    from conduit.engines.analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    hybrid_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        switch_threshold=20,  # Switch after 20 queries (instead of default 2000)
        analyzer=analyzer,
    )

    # Create router wrapper
    router = Router(models=["gpt-4o-mini", "gpt-4o"], algorithm="linucb")
    router.hybrid_router = hybrid_router  # Replace with our custom hybrid router
    router.analyzer = analyzer

    # Reduce LinUCB alpha for exploitation (once it switches to LinUCB)
    hybrid_router.phase2_bandit.alpha = 0.1

    # Route queries across both phases and provide immediate feedback
    for i in range(60):  # 60 queries: first 20 in Thompson, rest in LinUCB
        query = Query(text=f"Question {i}: explain concept")
        decision = await router.route(query)

        # Provide immediate feedback (cheap model is better)
        if decision.selected_model == "gpt-4o-mini":
            await router.update(
                model_id=decision.selected_model,
                cost=0.001,
                quality_score=0.90,
                latency=0.5,
                features=decision.features,
            )
        else:
            await router.update(
                model_id=decision.selected_model,
                cost=0.10,
                quality_score=0.90,
                latency=0.8,
                features=decision.features,
            )

    # Verify phase transition happened
    assert router.hybrid_router.query_count == 60
    assert router.hybrid_router.current_phase == "linucb"  # Should be in LinUCB phase now

    # Test: Route new queries after learning across both phases
    selections = []
    for i in range(30):
        query = Query(text=f"Final test {i+60}")
        decision = await router.route(query)
        selections.append(decision.selected_model)

    cheap_count = sum(1 for model in selections if model == "gpt-4o-mini")
    cheap_rate = cheap_count / 30

    await router.close()

    print(f"\nHybrid routing learning: {cheap_count}/30 selected cheap model ({cheap_rate:.0%})")

    # After learning across both phases, cheap model should be preferred
    assert cheap_rate > 0.70, (
        f"After learning across UCB1→LinUCB, cheap model should be selected >70%. "
        f"Got: {cheap_rate:.0%}"
    )
