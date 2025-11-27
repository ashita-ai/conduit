"""Tests for configurable hybrid routing algorithms and state conversion.

Tests all 4 algorithm combinations and optimistic state conversion logic.
"""

import pytest

from conduit.core.models import Query, QueryFeatures
from conduit.core.state_store import BanditState, HybridRouterState, RouterPhase
from conduit.engines.bandits import convert_bandit_state
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.engines.hybrid_router import HybridRouter


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def test_arms():
    """Create test model arms."""
    return [
        ModelArm(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0003,
            expected_quality=0.7,
        ),
        ModelArm(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            cost_per_input_token=0.0005,
            cost_per_output_token=0.0015,
            expected_quality=0.9,
        ),
    ]


@pytest.fixture
def test_features():
    """Create test query features."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )


# ============================================================================
# ALGORITHM CONFIGURATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_thompson_to_linucb_default():
    """Test default Thompson Sampling → LinUCB configuration."""
    router = HybridRouter(models=["gpt-4o-mini", "gpt-4o"])

    assert router.phase1_algorithm == "thompson_sampling"
    assert router.phase2_algorithm == "linucb"
    assert router.current_phase == "thompson_sampling"

    # Verify bandit types
    from conduit.engines.bandits import LinUCBBandit, ThompsonSamplingBandit

    assert isinstance(router.phase1_bandit, ThompsonSamplingBandit)
    assert isinstance(router.phase2_bandit, LinUCBBandit)


@pytest.mark.asyncio
async def test_thompson_to_linucb():
    """Test Thompson Sampling → LinUCB configuration."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="linucb",
    )

    assert router.phase1_algorithm == "thompson_sampling"
    assert router.phase2_algorithm == "linucb"
    assert router.current_phase == "thompson_sampling"

    # Verify bandit types
    from conduit.engines.bandits import LinUCBBandit, ThompsonSamplingBandit

    assert isinstance(router.phase1_bandit, ThompsonSamplingBandit)
    assert isinstance(router.phase2_bandit, LinUCBBandit)


@pytest.mark.asyncio
async def test_ucb1_to_contextual_thompson():
    """Test UCB1 → Contextual Thompson Sampling configuration."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="ucb1",
        phase2_algorithm="contextual_thompson_sampling",
    )

    assert router.phase1_algorithm == "ucb1"
    assert router.phase2_algorithm == "contextual_thompson_sampling"

    # Verify bandit types
    from conduit.engines.bandits import (
        ContextualThompsonSamplingBandit,
        UCB1Bandit,
    )

    assert isinstance(router.phase1_bandit, UCB1Bandit)
    assert isinstance(router.phase2_bandit, ContextualThompsonSamplingBandit)


@pytest.mark.asyncio
async def test_thompson_to_contextual_thompson():
    """Test Thompson Sampling → Contextual Thompson Sampling (full Bayesian)."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="contextual_thompson_sampling",
    )

    assert router.phase1_algorithm == "thompson_sampling"
    assert router.phase2_algorithm == "contextual_thompson_sampling"

    # Verify bandit types
    from conduit.engines.bandits import (
        ContextualThompsonSamplingBandit,
        ThompsonSamplingBandit,
    )

    assert isinstance(router.phase1_bandit, ThompsonSamplingBandit)
    assert isinstance(router.phase2_bandit, ContextualThompsonSamplingBandit)


@pytest.mark.asyncio
async def test_invalid_phase1_algorithm():
    """Test error on invalid phase1_algorithm."""
    with pytest.raises(ValueError, match="Invalid phase1_algorithm"):
        HybridRouter(
            models=["gpt-4o-mini"],
            phase1_algorithm="invalid",
            phase2_algorithm="linucb",
        )


@pytest.mark.asyncio
async def test_invalid_phase2_algorithm():
    """Test error on invalid phase2_algorithm."""
    with pytest.raises(ValueError, match="Invalid phase2_algorithm"):
        HybridRouter(
            models=["gpt-4o-mini"],
            phase1_algorithm="ucb1",
            phase2_algorithm="invalid",
        )


# ============================================================================
# STATE CONVERSION TESTS
# ============================================================================


def test_ucb1_to_thompson_conversion():
    """Test UCB1 → Thompson Sampling state conversion."""
    # Create UCB1 state
    ucb1_state = BanditState(
        algorithm="ucb1",
        arm_ids=["gpt-4o-mini", "gpt-4o"],
        arm_pulls={"gpt-4o-mini": 100, "gpt-4o": 50},
        arm_successes={"gpt-4o-mini": 75, "gpt-4o": 40},
        total_queries=150,
        mean_reward={"gpt-4o-mini": 0.75, "gpt-4o": 0.80},  # Fixed: mean_reward not mean_rewards
    )

    # Convert to Thompson Sampling
    thompson_state = convert_bandit_state(ucb1_state, "thompson_sampling")

    assert thompson_state.algorithm == "thompson_sampling"
    assert thompson_state.arm_ids == ["gpt-4o-mini", "gpt-4o"]
    assert thompson_state.total_queries == 150

    # Check Beta parameters (alpha = 1 + mean_reward * pulls)
    alpha_mini = thompson_state.alpha_params["gpt-4o-mini"]
    beta_mini = thompson_state.beta_params["gpt-4o-mini"]
    assert alpha_mini == pytest.approx(1.0 + 0.75 * 100)  # 76.0
    assert beta_mini == pytest.approx(1.0 + 0.25 * 100)  # 26.0

    # Verify mean approximately preserved (slight difference due to Beta(1,1) prior)
    mean_recovered = alpha_mini / (alpha_mini + beta_mini)
    # 76/(76+26) = 0.745, close to original 0.75
    assert mean_recovered == pytest.approx(76.0 / 102.0, rel=1e-3)


def test_thompson_to_ucb1_conversion():
    """Test Thompson Sampling → UCB1 state conversion."""
    # Create Thompson Sampling state
    thompson_state = BanditState(
        algorithm="thompson_sampling",
        arm_ids=["gpt-4o-mini", "gpt-4o"],
        arm_pulls={"gpt-4o-mini": 100, "gpt-4o": 50},
        arm_successes={"gpt-4o-mini": 75, "gpt-4o": 40},
        total_queries=150,
        alpha_params={"gpt-4o-mini": 76.0, "gpt-4o": 41.0},
        beta_params={"gpt-4o-mini": 26.0, "gpt-4o": 11.0},
    )

    # Convert to UCB1
    ucb1_state = convert_bandit_state(thompson_state, "ucb1")

    assert ucb1_state.algorithm == "ucb1"
    assert ucb1_state.arm_ids == ["gpt-4o-mini", "gpt-4o"]
    assert ucb1_state.total_queries == 150

    # Check mean rewards recovered
    mean_mini = ucb1_state.mean_reward["gpt-4o-mini"]  # Fixed: mean_reward not mean_rewards
    assert mean_mini == pytest.approx(76.0 / (76.0 + 26.0), rel=1e-3)  # ~0.745

    # Check pull counts (alpha + beta - 2)
    pulls_mini = ucb1_state.arm_pulls["gpt-4o-mini"]
    assert pulls_mini == 100  # (76 + 26) - 2


def test_linucb_to_contextual_thompson_conversion(test_arms):
    """Test LinUCB → Contextual Thompson Sampling conversion."""
    import numpy as np

    from conduit.core.state_store import serialize_bandit_matrices

    # Create LinUCB state
    feature_dim = 387
    A_matrices = {}
    b_vectors = {}

    for arm in test_arms:
        # Simple diagonal A matrix and b vector
        A = np.identity(feature_dim) * 2.0
        b = np.ones((feature_dim, 1)) * 0.5
        A_matrices[arm.model_id] = A
        b_vectors[arm.model_id] = b

    A_serial, b_serial = serialize_bandit_matrices(A_matrices, b_vectors)

    linucb_state = BanditState(
        algorithm="linucb",
        arm_ids=[arm.model_id for arm in test_arms],
        arm_pulls={"gpt-4o-mini": 50, "gpt-4o": 30},
        arm_successes={"gpt-4o-mini": 40, "gpt-4o": 25},
        total_queries=80,
        A_matrices=A_serial,
        b_vectors=b_serial,
        feature_dim=feature_dim,
    )

    # Convert to Contextual Thompson
    cts_state = convert_bandit_state(
        linucb_state, "contextual_thompson_sampling", feature_dim=feature_dim
    )

    assert cts_state.algorithm == "contextual_thompson_sampling"
    assert cts_state.feature_dim == feature_dim
    assert cts_state.total_queries == 80

    # Verify mu and Sigma dimensions
    mu_mini = np.array(cts_state.mu_vectors["gpt-4o-mini"])
    sigma_mini = np.array(cts_state.sigma_matrices["gpt-4o-mini"])

    assert mu_mini.shape == (feature_dim,)
    assert sigma_mini.shape == (feature_dim, feature_dim)


def test_contextual_thompson_to_linucb_conversion(test_arms):
    """Test Contextual Thompson Sampling → LinUCB conversion."""
    import numpy as np

    # Create Contextual Thompson Sampling state
    feature_dim = 387
    mu_vectors = {}
    sigma_matrices = {}

    for arm in test_arms:
        mu = np.random.randn(feature_dim) * 0.1
        sigma = np.identity(feature_dim) * 0.5
        mu_vectors[arm.model_id] = mu.tolist()
        sigma_matrices[arm.model_id] = sigma.tolist()

    cts_state = BanditState(
        algorithm="contextual_thompson_sampling",
        arm_ids=[arm.model_id for arm in test_arms],
        arm_pulls={"gpt-4o-mini": 50, "gpt-4o": 30},
        arm_successes={"gpt-4o-mini": 40, "gpt-4o": 25},
        total_queries=80,
        mu_vectors=mu_vectors,
        sigma_matrices=sigma_matrices,
        feature_dim=feature_dim,
    )

    # Convert to LinUCB
    linucb_state = convert_bandit_state(cts_state, "linucb", feature_dim=feature_dim)

    assert linucb_state.algorithm == "linucb"
    assert linucb_state.feature_dim == feature_dim
    assert linucb_state.total_queries == 80

    # Verify A and b matrices exist
    assert linucb_state.A_matrices is not None
    assert linucb_state.b_vectors is not None


def test_no_conversion_needed():
    """Test that same algorithm returns unchanged state."""
    state = BanditState(
        algorithm="ucb1",
        arm_ids=["gpt-4o-mini"],
        arm_pulls={"gpt-4o-mini": 10},
        arm_successes={"gpt-4o-mini": 8},
        total_queries=10,
        mean_reward={"gpt-4o-mini": 0.8},  # Fixed
    )

    # Convert to same algorithm
    same_state = convert_bandit_state(state, "ucb1")

    assert same_state.algorithm == "ucb1"
    assert same_state.total_queries == 10


def test_unsupported_conversion():
    """Test error on unsupported conversion."""
    state = BanditState(
        algorithm="ucb1",
        arm_ids=["gpt-4o-mini"],
        arm_pulls={"gpt-4o-mini": 10},
        arm_successes={"gpt-4o-mini": 8},
        total_queries=10,
        mean_reward={"gpt-4o-mini": 0.8},  # Fixed
    )

    # Try unsupported conversion
    with pytest.raises(ValueError, match="Conversion not supported"):
        convert_bandit_state(state, "unsupported_algorithm")


# ============================================================================
# HYBRID ROUTER STATE LOADING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_state_save_and_load_thompson_to_linucb():
    """Test state persistence for Thompson → LinUCB configuration."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="linucb",
    )

    # Simulate some routing
    router.query_count = 500

    # Serialize state
    state = router.to_state()

    assert state.phase1_algorithm == "thompson_sampling"
    assert state.phase2_algorithm == "linucb"
    assert state.query_count == 500
    assert state.phase1_state is not None
    assert state.phase2_state is not None

    # Create new router and restore state
    new_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="linucb",
    )

    new_router.from_state(state, allow_conversion=True)

    assert new_router.query_count == 500
    assert new_router.current_phase == "thompson_sampling"


@pytest.mark.asyncio
async def test_optimistic_conversion_on_load():
    """Test optimistic state conversion when algorithms mismatch."""
    # Create router with UCB1 → LinUCB
    old_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="ucb1",
        phase2_algorithm="linucb",
    )
    old_router.query_count = 1000

    # Save state
    old_state = old_router.to_state()
    assert old_state.phase1_algorithm == "ucb1"

    # Create new router with Thompson → LinUCB
    new_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",  # Different!
        phase2_algorithm="linucb",
    )

    # Load with optimistic conversion
    new_router.from_state(old_state, allow_conversion=True)

    assert new_router.query_count == 1000
    # State should be converted from UCB1 to Thompson Sampling


@pytest.mark.asyncio
async def test_strict_mode_rejects_mismatch():
    """Test that strict mode (allow_conversion=False) raises error on mismatch."""
    # Create state with UCB1 → LinUCB
    old_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="ucb1",
        phase2_algorithm="linucb",
    )
    old_state = old_router.to_state()

    # Create router with different algorithms
    new_router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",  # Mismatch!
        phase2_algorithm="linucb",
    )

    # Strict mode should error
    with pytest.raises(ValueError, match="Algorithm mismatch"):
        new_router.from_state(old_state, allow_conversion=False)


@pytest.mark.asyncio
async def test_backward_compatibility_with_old_state():
    """Test loading old state format (ucb1_state/linucb_state fields)."""
    # Create old-format state (before algorithm configuration)
    old_state = HybridRouterState(
        query_count=1500,
        current_phase=RouterPhase.UCB1,
        transition_threshold=2000,
        ucb1_state=BanditState(
            algorithm="ucb1",
            arm_ids=["gpt-4o-mini", "gpt-4o"],
            arm_pulls={"gpt-4o-mini": 100, "gpt-4o": 50},
            arm_successes={"gpt-4o-mini": 75, "gpt-4o": 40},
            total_queries=150,
            mean_reward={"gpt-4o-mini": 0.75, "gpt-4o": 0.80},  # Fixed
        ),
        linucb_state=None,
        # Old state doesn't have phase1_algorithm/phase2_algorithm fields
    )

    # Load into new router (should default to ucb1/linucb)
    router = HybridRouter(models=["gpt-4o-mini", "gpt-4o"])
    router.from_state(old_state, allow_conversion=True)

    assert router.query_count == 1500
    # Should successfully load despite missing algorithm identifiers


# ============================================================================
# TRANSITION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_thompson_to_linucb_transition(test_features):
    """Test transition from Thompson Sampling to LinUCB with knowledge transfer."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="linucb",
        switch_threshold=10,  # Small threshold for testing
    )

    # Route queries in phase1 (Thompson Sampling)
    for i in range(15):
        await router.route(Query(text=f"Query {i}"))

    # Should have transitioned to LinUCB
    assert router.current_phase == "linucb"
    assert router.query_count == 15

    # Phase2 bandit should have received converted state from Thompson Sampling
    stats = router.phase2_bandit.get_stats()
    # State was transferred during transition, so total_queries matches phase1
    assert stats["total_queries"] == 15


@pytest.mark.asyncio
async def test_all_combinations_transition():
    """Test that all 4 algorithm combinations can transition successfully."""
    combinations = [
        ("ucb1", "linucb"),
        ("thompson_sampling", "linucb"),
        ("ucb1", "contextual_thompson_sampling"),
        ("thompson_sampling", "contextual_thompson_sampling"),
    ]

    for phase1, phase2 in combinations:
        router = HybridRouter(
            models=["gpt-4o-mini", "gpt-4o"],
            phase1_algorithm=phase1,
            phase2_algorithm=phase2,
            switch_threshold=5,
        )

        # Route enough queries to trigger transition
        for i in range(10):
            await router.route(Query(text=f"Test query {i}"))

        # Verify transition occurred
        assert router.current_phase == phase2, f"Failed to transition: {phase1} → {phase2}"


# ============================================================================
# STATISTICS TESTS
# ============================================================================


def test_get_stats_includes_algorithm_info():
    """Test that get_stats() includes algorithm identifiers."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="contextual_thompson_sampling",
    )

    stats = router.get_stats()

    assert stats["phase1_algorithm"] == "thompson_sampling"
    assert stats["phase2_algorithm"] == "contextual_thompson_sampling"
    assert stats["phase"] == "thompson_sampling"  # Current phase


def test_reset_preserves_algorithm_config():
    """Test that reset() preserves algorithm configuration."""
    router = HybridRouter(
        models=["gpt-4o-mini", "gpt-4o"],
        phase1_algorithm="thompson_sampling",
        phase2_algorithm="contextual_thompson_sampling",
    )

    router.query_count = 500
    router.reset()

    assert router.query_count == 0
    assert router.phase1_algorithm == "thompson_sampling"
    assert router.phase2_algorithm == "contextual_thompson_sampling"
    assert router.current_phase == "thompson_sampling"  # Reset to phase1
