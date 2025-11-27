"""Integration tests for HybridRouter UCB1→LinUCB transition.

Tests the complete lifecycle of hybrid routing including:
- UCB1 exploration phase (queries 0-threshold)
- Automatic transition at threshold
- Knowledge transfer from UCB1 to LinUCB
- State persistence across restarts
- Feedback learning in both phases

These tests require:
- DATABASE_URL environment variable (postgresql://...)
- Database schema migrated (run alembic upgrade head)
"""

import os
from uuid import uuid4

import numpy as np
import pytest

from conduit.core.models import Query, QueryFeatures
from conduit.core.postgres_state_store import PostgresStateStore
from conduit.core.state_store import RouterPhase
from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.hybrid_router import HybridRouter


# Skip all tests if DATABASE_URL not available
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured",
)


@pytest.fixture
async def postgres_store():
    """Create PostgreSQL state store for persistence testing."""
    from conduit.core.database import Database

    db = Database()
    await db.connect()
    store = PostgresStateStore(db.pool)
    yield store
    await db.disconnect()


@pytest.fixture
def small_threshold_router(test_arms):
    """Create router with small threshold for fast testing."""
    return HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=10,  # Small threshold for fast tests
    )


@pytest.mark.asyncio
async def test_transition_trigger_at_threshold(small_threshold_router):
    """Test that transition from UCB1 to LinUCB occurs at correct query count.

    Verifies:
    - Router starts in UCB1 phase
    - Transition happens exactly at switch_threshold
    - Phase changes to LinUCB after transition
    """
    router = small_threshold_router
    threshold = router.switch_threshold

    # Verify starting state
    assert router.current_phase == "ucb1"
    assert router.query_count == 0

    # Route queries up to threshold - 1
    for i in range(threshold - 1):
        query = Query(text=f"Query {i}")
        decision = await router.route(query)

        assert router.current_phase == "ucb1", f"Should stay UCB1 at query {i + 1}"
        assert decision.metadata["phase"] == "ucb1"
        assert router.query_count == i + 1

    # Verify still in UCB1 phase
    assert router.current_phase == "ucb1"
    assert router.query_count == threshold - 1

    # Next query should trigger transition
    query = Query(text=f"Transition query {threshold}")
    decision = await router.route(query)

    # Verify transition occurred
    assert router.current_phase == "linucb", "Should transition to LinUCB at threshold"
    assert decision.metadata["phase"] == "linucb"
    assert router.query_count == threshold


@pytest.mark.asyncio
async def test_knowledge_transfer_from_ucb1_to_linucb(test_arms, test_features):
    """Test that UCB1 quality estimates are transferred to LinUCB initialization.

    Verifies:
    - LinUCB's b vector starts at zero
    - After transition, b[0] contains UCB1 mean rewards
    - Knowledge transfer scales with number of pulls
    """
    router = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=21,  # Set to 21 so 20 queries stay in UCB1
    )

    # Verify LinUCB starts with zero knowledge
    for model_id in router.models:
        assert np.allclose(
            router.linucb.b[model_id], 0
        ), f"LinUCB b should start at 0 for {model_id}"

    # Provide feedback during UCB1 phase (varying quality by model)
    for i in range(20):
        query = Query(text=f"Training query {i}")
        decision = await router.route(query)

        # Simulate different quality scores for different models
        quality_map = {
            "o4-mini": 0.7,
            "gpt-5.1": 0.9,
            "claude-haiku-4-5": 0.6,
        }
        quality_score = quality_map.get(decision.selected_model, 0.5)

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=quality_score,
            latency=1.0,
        )
        await router.update(feedback, decision.features)

    # Verify still in UCB1
    assert router.current_phase == "ucb1"

    # Capture UCB1 statistics before transition
    ucb1_stats = router.ucb1.get_stats()
    ucb1_pulls = ucb1_stats["arm_pulls"]
    ucb1_rewards = ucb1_stats.get("arm_mean_rewards", {})

    # Trigger transition
    query = Query(text="Transition query")
    await router.route(query)

    # Verify transition occurred
    assert router.current_phase == "linucb"

    # Verify knowledge transfer: LinUCB's b[0] should contain UCB1 knowledge
    for model_id in router.models:
        pulls = ucb1_pulls.get(model_id, 0)

        if pulls > 0:
            # b[0] should be non-zero (contains transferred knowledge)
            assert (
                router.linucb.b[model_id][0] != 0
            ), f"LinUCB b[0] should contain UCB1 knowledge for {model_id}"

            # Verify scaling: more pulls = stronger transfer
            mean_reward = ucb1_rewards.get(model_id, 0.5)
            scaling_factor = min(10.0, pulls / 100.0)

            # Allow for floating point precision
            expected_b0 = mean_reward * scaling_factor
            actual_b0 = router.linucb.b[model_id][0]
            assert np.isclose(
                actual_b0, expected_b0, rtol=1e-5
            ), f"LinUCB b[0] mismatch for {model_id}: expected {expected_b0}, got {actual_b0}"


@pytest.mark.asyncio
async def test_phase_detection_accuracy(small_threshold_router):
    """Test that current_phase property accurately reflects routing phase.

    Verifies:
    - current_phase starts as "ucb1"
    - current_phase changes to "linucb" after transition
    - Routing decisions include correct phase metadata
    """
    router = small_threshold_router
    threshold = router.switch_threshold

    # Phase 1: UCB1
    for i in range(threshold - 1):
        query = Query(text=f"UCB1 query {i}")
        decision = await router.route(query)

        assert router.current_phase == "ucb1"
        assert decision.metadata["phase"] == "ucb1"
        assert "query_count" in decision.metadata
        assert "switch_threshold" in decision.metadata

    # Transition
    query = Query(text="Transition query")
    decision = await router.route(query)

    # Phase 2: LinUCB
    assert router.current_phase == "linucb"
    assert decision.metadata["phase"] == "linucb"
    assert "queries_since_transition" in decision.metadata

    # Continue in LinUCB
    for i in range(5):
        query = Query(text=f"LinUCB query {i}")
        decision = await router.route(query)

        assert router.current_phase == "linucb"
        assert decision.metadata["phase"] == "linucb"


@pytest.mark.asyncio
async def test_stateful_persistence_across_restarts(
    test_arms, postgres_store, test_features
):
    """Test that routing state persists across router restarts.

    Verifies:
    - State can be saved during UCB1 phase
    - State can be saved after transition to LinUCB
    - Restored router continues from saved state
    - Query count, phase, and bandit state are preserved
    """
    router_id = f"test-router-{uuid4()}"

    # Create router and route some queries in UCB1
    router1 = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=15,
    )

    for i in range(10):
        query = Query(text=f"Query {i}")
        decision = await router1.route(query)

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
        )
        await router1.update(feedback, decision.features)

    # Save state during UCB1 phase
    await router1.save_state(postgres_store, router_id)

    # Create new router and load state
    router2 = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=15,
    )

    loaded = await router2.load_state(postgres_store, router_id)
    assert loaded, "State should be loaded successfully"

    # Verify UCB1 state restored
    assert router2.current_phase == "ucb1"
    assert router2.query_count == 10
    assert router2.switch_threshold == 15

    # Continue routing to trigger transition
    for i in range(10, 20):
        query = Query(text=f"Query {i}")
        decision = await router2.route(query)

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.75,
            latency=1.0,
        )
        await router2.update(feedback, decision.features)

    # Verify transition occurred
    assert router2.current_phase == "linucb"
    assert router2.query_count > 15

    # Save state after transition
    await router2.save_state(postgres_store, router_id)

    # Create third router and load LinUCB state
    router3 = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=15,
    )

    loaded = await router3.load_state(postgres_store, router_id)
    assert loaded, "LinUCB state should be loaded successfully"

    # Verify LinUCB state restored
    assert router3.current_phase == "linucb"
    assert router3.query_count == router2.query_count

    # Verify LinUCB can continue routing
    query = Query(text="Post-restore query")
    decision = await router3.route(query)
    assert decision.metadata["phase"] == "linucb"


@pytest.mark.asyncio
async def test_feedback_learning_in_both_phases(test_arms):
    """Test that both UCB1 and LinUCB learn from feedback correctly.

    Verifies:
    - UCB1 updates arm statistics during phase 1
    - LinUCB updates A matrix and b vector during phase 2
    - Knowledge accumulates across both phases
    """
    router = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=25,
    )

    # Phase 1: UCB1 Learning
    ucb1_initial_pulls = {model: 0 for model in router.models}

    for i in range(25):
        query = Query(text=f"UCB1 query {i}")
        decision = await router.route(query)

        # Provide varying feedback
        quality_score = 0.6 + (i % 3) * 0.1  # Varies: 0.6, 0.7, 0.8

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=quality_score,
            latency=1.0,
        )
        await router.update(feedback, decision.features)

    # Verify UCB1 learned
    ucb1_stats = router.ucb1.get_stats()
    ucb1_final_pulls = ucb1_stats["arm_pulls"]

    for model_id in router.models:
        assert (
            ucb1_final_pulls.get(model_id, 0) > ucb1_initial_pulls[model_id]
        ), f"UCB1 should have pulled {model_id} at least once"

    # Transition should have occurred
    assert router.current_phase == "linucb"

    # Capture LinUCB initial state
    linucb_initial_A = {
        model_id: router.linucb.A[model_id].copy() for model_id in router.models
    }

    # Phase 2: LinUCB Learning
    for i in range(20):
        query = Query(text=f"LinUCB query {i}")
        decision = await router.route(query)

        quality_score = 0.7 + (i % 2) * 0.15  # Varies: 0.7, 0.85

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=quality_score,
            latency=1.0,
        )
        await router.update(feedback, decision.features)

    # Verify LinUCB learned (A matrix updated)
    for model_id in router.models:
        pulls = router.linucb.arm_pulls.get(model_id, 0)

        if pulls > 0:
            # A matrix should have changed from initial state
            current_A = router.linucb.A[model_id]
            initial_A = linucb_initial_A[model_id]

            assert not np.allclose(
                current_A, initial_A, rtol=1e-5
            ), f"LinUCB A matrix should have updated for {model_id}"

            # A matrix should still be symmetric
            assert np.allclose(
                current_A, current_A.T, rtol=1e-10
            ), f"A matrix should be symmetric for {model_id}"

            # A matrix should still be positive definite
            eigenvalues = np.linalg.eigvalsh(current_A)
            assert np.all(
                eigenvalues > 0
            ), f"A matrix should be positive definite for {model_id}"


@pytest.mark.asyncio
async def test_full_lifecycle_with_multiple_models(test_arms):
    """Test complete lifecycle with 3+ models.

    Verifies:
    - All models get explored during UCB1 phase
    - Transition handles multiple models correctly
    - LinUCB continues to route between all models
    """
    router = HybridRouter(
        models=[arm.model_id for arm in test_arms],  # 3 models
        switch_threshold=30,
    )

    # Track which models were selected
    ucb1_selections = {model: 0 for model in router.models}
    linucb_selections = {model: 0 for model in router.models}

    # Phase 1: UCB1 (30 queries)
    for i in range(30):
        query = Query(text=f"Query {i}")
        decision = await router.route(query)

        ucb1_selections[decision.selected_model] += 1

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.75,
            latency=1.0,
        )
        await router.update(feedback, decision.features)

    # Verify all models were tried during UCB1
    for model in router.models:
        assert (
            ucb1_selections[model] > 0
        ), f"UCB1 should have explored {model} at least once"

    # Verify transition
    assert router.current_phase == "linucb"

    # Phase 2: LinUCB (30 queries)
    for i in range(30):
        query = Query(text=f"LinUCB query {i}")
        decision = await router.route(query)

        linucb_selections[decision.selected_model] += 1

        feedback = BanditFeedback(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
        )
        await router.update(feedback, decision.features)

    # Verify LinUCB routed to at least 2 models (may converge to best)
    models_used = sum(1 for count in linucb_selections.values() if count > 0)
    assert models_used >= 2, "LinUCB should route to multiple models"


@pytest.mark.asyncio
async def test_zero_threshold_starts_with_linucb(test_arms):
    """Test that switch_threshold=0 starts directly in LinUCB phase.

    Verifies:
    - Router starts in LinUCB when threshold is 0
    - No UCB1 phase occurs
    - First query uses contextual features
    """
    router = HybridRouter(
        models=[arm.model_id for arm in test_arms],
        switch_threshold=0,  # Start directly in LinUCB
    )

    # First query should use LinUCB
    query = Query(text="First query")
    decision = await router.route(query)

    assert router.current_phase == "linucb", "Should start in LinUCB with threshold=0"
    assert decision.metadata["phase"] == "linucb"
    assert router.query_count == 1


@pytest.mark.asyncio
async def test_reset_returns_to_ucb1(small_threshold_router):
    """Test that reset() returns router to initial UCB1 state.

    Verifies:
    - Router can be reset after transition to LinUCB
    - Query count returns to 0
    - Phase returns to UCB1
    - Bandit states are cleared
    """
    router = small_threshold_router

    # Route past threshold to trigger transition
    for i in range(15):
        query = Query(text=f"Query {i}")
        await router.route(query)

    # Verify in LinUCB
    assert router.current_phase == "linucb"
    assert router.query_count > router.switch_threshold

    # Reset
    router.reset()

    # Verify back to initial state
    assert router.current_phase == "ucb1"
    assert router.query_count == 0

    # Verify can route again from UCB1
    query = Query(text="Post-reset query")
    decision = await router.route(query)
    assert decision.metadata["phase"] == "ucb1"
