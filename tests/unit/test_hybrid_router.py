"""Tests for HybridRouter (UCB1→LinUCB warm start routing)."""

import pytest

from conduit.core.models import Query, QueryFeatures
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.engines.hybrid_router import HybridRouter


@pytest.fixture
def test_models():
    """Test model IDs."""
    return ["gpt-4o-mini", "claude-3-5-sonnet", "gpt-4o"]


@pytest.fixture
def hybrid_router(test_models, test_analyzer):
    """Create hybrid router with test configuration."""
    return HybridRouter(
        models=test_models,
        switch_threshold=10,  # Low threshold for testing
        analyzer=test_analyzer,
        feature_dim=386,
        phase1_algorithm="ucb1",  # Use UCB1 for these UCB1-specific tests
        ucb1_c=1.5,
        linucb_alpha=1.0,
    )


@pytest.mark.asyncio
async def test_initialization(hybrid_router, test_models):
    """Test hybrid router initializes correctly."""
    assert hybrid_router.current_phase == "ucb1"
    assert hybrid_router.query_count == 0
    assert hybrid_router.switch_threshold == 10
    assert len(hybrid_router.models) == 3
    assert len(hybrid_router.arms) == 3


@pytest.mark.asyncio
async def test_ucb1_phase_routing(hybrid_router):
    """Test routing uses UCB1 in phase 1."""
    query = Query(text="What is 2+2?")

    # Route 5 queries in UCB1 phase
    for i in range(5):
        decision = await hybrid_router.route(query)

        assert decision.selected_model in hybrid_router.models
        assert hybrid_router.current_phase == "ucb1"
        assert hybrid_router.query_count == i + 1
        assert decision.metadata["phase"] == "ucb1"
        assert decision.metadata["query_count"] == i + 1


@pytest.mark.asyncio
async def test_automatic_transition_to_linucb(hybrid_router):
    """Test automatic transition from UCB1 to LinUCB at threshold."""
    query = Query(text="What is 2+2?")

    # Route queries up to threshold - 1
    for _ in range(9):
        decision = await hybrid_router.route(query)
        assert hybrid_router.current_phase == "ucb1"

    # This query should trigger transition
    decision = await hybrid_router.route(query)
    assert hybrid_router.current_phase == "linucb"
    assert hybrid_router.query_count == 10
    assert decision.metadata["phase"] == "linucb"


@pytest.mark.asyncio
async def test_linucb_phase_routing(hybrid_router):
    """Test routing uses LinUCB in phase 2."""
    query = Query(text="What is 2+2?")

    # Get to LinUCB phase
    for _ in range(10):
        await hybrid_router.route(query)

    assert hybrid_router.current_phase == "linucb"

    # Route additional queries in LinUCB phase
    for _i in range(5):
        decision = await hybrid_router.route(query)

        assert decision.selected_model in hybrid_router.models
        assert hybrid_router.current_phase == "linucb"
        assert decision.metadata["phase"] == "linucb"
        assert "queries_since_transition" in decision.metadata


@pytest.mark.asyncio
async def test_knowledge_transfer_at_transition(hybrid_router):
    """Test UCB1 knowledge transfers to LinUCB at transition."""
    query = Query(text="What is 2+2?")

    # Route and provide feedback in UCB1 phase
    for _ in range(9):
        decision = await hybrid_router.route(query)

        # Provide different rewards for different models
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.9 if "gpt-4o" in decision.selected_model else 0.7,
            latency=1.0,
        )
        await hybrid_router.update(feedback, decision.features)

    # Get UCB1 stats before transition
    ucb1_stats_before = hybrid_router.ucb1.get_stats()

    # Trigger transition
    await hybrid_router.route(query)

    assert hybrid_router.current_phase == "linucb"

    # Verify LinUCB received knowledge transfer
    # Check that LinUCB's b vectors are non-zero for models that were pulled
    for model_id in hybrid_router.models:
        pulls = ucb1_stats_before["arm_pulls"].get(model_id, 0)
        if pulls > 0:
            # LinUCB should have initialized b vector from UCB1 knowledge
            # First dimension of b should be non-zero
            assert hybrid_router.linucb.b[model_id][0] != 0.0


@pytest.mark.asyncio
async def test_update_routes_to_correct_bandit(hybrid_router):
    """Test updates route to UCB1 or LinUCB based on current phase."""
    query = Query(text="What is 2+2?")

    # Phase 1: Updates go to UCB1
    decision = await hybrid_router.route(query)
    feedback = BanditFeedback(
        model_id=decision.selected_model,
        cost=0.001,
        quality_score=0.9,
        latency=1.0,
    )

    initial_ucb1_pulls = hybrid_router.ucb1.arm_pulls[decision.selected_model]
    await hybrid_router.update(feedback, decision.features)

    assert hybrid_router.ucb1.arm_pulls[decision.selected_model] == initial_ucb1_pulls + 1

    # Transition to phase 2
    for _ in range(9):
        await hybrid_router.route(query)

    # Phase 2: Updates go to LinUCB
    decision = await hybrid_router.route(query)
    feedback = BanditFeedback(
        model_id=decision.selected_model,
        cost=0.001,
        quality_score=0.9,
        latency=1.0,
    )

    initial_linucb_pulls = hybrid_router.linucb.arm_pulls[decision.selected_model]
    await hybrid_router.update(feedback, decision.features)

    assert hybrid_router.linucb.arm_pulls[decision.selected_model] == initial_linucb_pulls + 1


@pytest.mark.asyncio
async def test_update_requires_features_in_linucb_phase(hybrid_router):
    """Test update raises error if features not provided in LinUCB phase."""
    query = Query(text="What is 2+2?")

    # Get to LinUCB phase
    for _ in range(10):
        await hybrid_router.route(query)

    decision = await hybrid_router.route(query)
    feedback = BanditFeedback(
        model_id=decision.selected_model,
        cost=0.001,
        quality_score=0.9,
        latency=1.0,
    )

    # Should raise error if features not provided
    with pytest.raises(ValueError, match="Features required for LinUCB update"):
        await hybrid_router.update(feedback, None)


@pytest.mark.asyncio
async def test_get_stats_shows_current_phase(hybrid_router):
    """Test get_stats returns correct phase information."""
    query = Query(text="What is 2+2?")

    # Phase 1
    await hybrid_router.route(query)
    stats = hybrid_router.get_stats()

    assert stats["phase"] == "ucb1"
    assert stats["query_count"] == 1
    assert stats["queries_until_transition"] == 9

    # Transition to phase 2
    for _ in range(9):
        await hybrid_router.route(query)

    stats = hybrid_router.get_stats()

    assert stats["phase"] == "linucb"
    assert stats["query_count"] == 10
    assert stats["queries_until_transition"] == 0


@pytest.mark.asyncio
async def test_reset_returns_to_initial_state(hybrid_router):
    """Test reset returns router to initial UCB1 state."""
    query = Query(text="What is 2+2?")

    # Route past transition
    for _ in range(15):
        decision = await hybrid_router.route(query)
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        await hybrid_router.update(feedback, decision.features)

    assert hybrid_router.current_phase == "linucb"
    assert hybrid_router.query_count == 15

    # Reset
    hybrid_router.reset()

    assert hybrid_router.current_phase == "ucb1"
    assert hybrid_router.query_count == 0

    # Verify bandits were reset
    stats = hybrid_router.get_stats()
    assert stats["total_queries"] == 0


@pytest.mark.asyncio
async def test_confidence_calculation_ucb1_phase(hybrid_router):
    """Test confidence increases with pulls in UCB1 phase."""
    query = Query(text="What is 2+2?")

    # Complete exploration phase by routing to all 3 models
    decisions = []
    for _i in range(3):
        decision = await hybrid_router.route(query)
        decisions.append(decision)
        # Give feedback to complete exploration
        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
        )
        await hybrid_router.update(feedback, decision.features)

    # Now in exploitation phase - give extra feedback to first model
    target_model = decisions[0].selected_model
    feedback = BanditFeedback(
        model_id=target_model,
        cost=0.001,
        quality_score=0.95,  # High quality
        latency=1.0,
    )
    await hybrid_router.update(feedback, decisions[0].features)

    # Route again - should prefer high-quality model with increased confidence
    decision = await hybrid_router.route(query)

    # Confidence should be higher than initial 0.1 (pulls >= 1)
    assert decision.confidence > 0.1


@pytest.mark.asyncio
async def test_provider_inference():
    """Test provider inference from model IDs."""
    router = HybridRouter(
        models=[
            "gpt-4o",
            "claude-3-5-sonnet",
            "gemini-1.5-pro",
            "llama-3.1-70b",
            "mistral-large",
            "command-r-plus",
        ],
        switch_threshold=10,
        feature_dim=386,
    )

    assert router._infer_provider("gpt-4o") == "openai"
    assert router._infer_provider("claude-3-5-sonnet") == "anthropic"
    assert router._infer_provider("gemini-1.5-pro") == "google-gla"  # pydantic_ai expects google-gla
    assert router._infer_provider("llama-3.1-70b") == "groq"
    assert router._infer_provider("mistral-large") == "mistral"
    assert router._infer_provider("command-r-plus") == "cohere"
    assert router._infer_provider("unknown-model") == "unknown"


@pytest.mark.asyncio
async def test_custom_switch_threshold(test_analyzer):
    """Test custom switch threshold configuration."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        switch_threshold=100,
        analyzer=test_analyzer,
        feature_dim=386,
        phase1_algorithm="ucb1",  # Test UCB1 → LinUCB transition
    )

    assert router.switch_threshold == 100

    # Route 99 queries - should stay in UCB1
    query = Query(text="test")
    for _ in range(99):
        await router.route(query)

    assert router.current_phase == "ucb1"

    # 100th query triggers transition
    await router.route(query)
    assert router.current_phase == "linucb"


@pytest.mark.asyncio
async def test_custom_exploration_parameters(test_analyzer):
    """Test custom UCB1 and LinUCB exploration parameters."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        switch_threshold=10,
        analyzer=test_analyzer,
        feature_dim=386,
        phase1_algorithm="ucb1",  # Test UCB1-specific parameters
        ucb1_c=2.0,
        linucb_alpha=0.5,
    )

    assert router.ucb1.c == 2.0
    assert router.linucb.alpha == 0.5


@pytest.mark.asyncio
async def test_reward_weights_propagation(test_models):
    """Test custom reward weights propagate to both bandits."""
    custom_weights = {"quality": 0.6, "cost": 0.3, "latency": 0.1}

    router = HybridRouter(
        models=test_models,
        switch_threshold=10,
        reward_weights=custom_weights,
        feature_dim=386,
    )

    assert router.ucb1.reward_weights == custom_weights
    assert router.linucb.reward_weights == custom_weights

def test_ucb1_property_emits_deprecation_warning(hybrid_router):
    with pytest.warns(DeprecationWarning, match="phase1_bandit"):
        _ = hybrid_router.ucb1


def test_linucb_property_emits_deprecation_warning(hybrid_router):
    with pytest.warns(DeprecationWarning, match="phase2_bandit"):
        _ = hybrid_router.linucb
