"""Tests for state persistence across bandit algorithms.

Tests serialization (to_state) and deserialization (from_state) for:
- UCB1Bandit
- LinUCBBandit
- ThompsonSamplingBandit
- EpsilonGreedyBandit
- ContextualThompsonSamplingBandit
- HybridRouter
"""

import numpy as np
import pytest

from conduit.core.state_store import BanditState, HybridRouterState, RouterPhase
from conduit.engines.bandits import (
    ContextualThompsonSamplingBandit,
    EpsilonGreedyBandit,
    LinUCBBandit,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from conduit.engines.bandits.base import ModelArm
from conduit.engines.hybrid_router import HybridRouter


@pytest.fixture
def test_arms():
    """Create test model arms."""
    return [
        ModelArm(
            model_id="test-model-1",
            provider="test",
            model_name="test-1",
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
            expected_quality=0.8,
        ),
        ModelArm(
            model_id="test-model-2",
            provider="test",
            model_name="test-2",
            cost_per_input_token=0.002,
            cost_per_output_token=0.004,
            expected_quality=0.9,
        ),
    ]


class TestUCB1StatePersistence:
    """Tests for UCB1Bandit state persistence."""

    def test_to_state_basic(self, test_arms):
        """Test basic state serialization."""
        bandit = UCB1Bandit(test_arms)
        bandit.total_queries = 100
        bandit.arm_pulls["test-model-1"] = 60
        bandit.arm_pulls["test-model-2"] = 40
        bandit.mean_reward["test-model-1"] = 0.85
        bandit.mean_reward["test-model-2"] = 0.75
        bandit.explored_arms = {"test-model-1", "test-model-2"}

        state = bandit.to_state()

        assert state.algorithm == "ucb1"
        assert state.total_queries == 100
        assert state.arm_pulls["test-model-1"] == 60
        assert state.mean_reward["test-model-1"] == 0.85
        assert set(state.explored_arms) == {"test-model-1", "test-model-2"}

    def test_from_state_restores_correctly(self, test_arms):
        """Test state restoration."""
        # Create and populate first bandit
        bandit1 = UCB1Bandit(test_arms)
        bandit1.total_queries = 200
        bandit1.arm_pulls["test-model-1"] = 100
        bandit1.arm_pulls["test-model-2"] = 100
        bandit1.mean_reward["test-model-1"] = 0.9
        bandit1.explored_arms = {"test-model-1", "test-model-2"}

        # Serialize and restore
        state = bandit1.to_state()
        bandit2 = UCB1Bandit(test_arms)
        bandit2.from_state(state)

        assert bandit2.total_queries == 200
        assert bandit2.arm_pulls["test-model-1"] == 100
        assert bandit2.mean_reward["test-model-1"] == 0.9
        assert bandit2.explored_arms == {"test-model-1", "test-model-2"}

    def test_from_state_rejects_wrong_algorithm(self, test_arms):
        """Test that from_state rejects wrong algorithm type."""
        bandit = UCB1Bandit(test_arms)
        state = BanditState(
            algorithm="linucb",  # Wrong algorithm
            arm_ids=["test-model-1", "test-model-2"],
        )

        with pytest.raises(ValueError, match="algorithm.*!= 'ucb1'"):
            bandit.from_state(state)

    def test_from_state_rejects_mismatched_arms(self, test_arms):
        """Test that from_state rejects mismatched arms."""
        bandit = UCB1Bandit(test_arms)
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["different-model"],  # Wrong arms
        )

        with pytest.raises(ValueError, match="don't match"):
            bandit.from_state(state)


class TestLinUCBStatePersistence:
    """Tests for LinUCBBandit state persistence."""

    def test_to_state_preserves_matrices(self, test_arms):
        """Test that A matrices and b vectors are preserved."""
        bandit = LinUCBBandit(test_arms, feature_dim=10)
        bandit.total_queries = 50

        # Modify A matrix and b vector
        bandit.A["test-model-1"][0, 0] = 2.5
        bandit.b["test-model-1"][0, 0] = 1.5

        state = bandit.to_state()

        assert state.algorithm == "linucb"
        assert state.feature_dim == 10
        # Check matrix preserved
        assert state.A_matrices["test-model-1"][0][0] == 2.5
        assert state.b_vectors["test-model-1"][0] == 1.5

    def test_from_state_restores_matrices(self, test_arms):
        """Test that A matrices and b vectors are restored correctly."""
        # Create first bandit with modified state
        bandit1 = LinUCBBandit(test_arms, feature_dim=10)
        bandit1.A["test-model-1"][0, 0] = 3.0
        bandit1.A["test-model-1"][1, 1] = 2.0
        bandit1.b["test-model-1"][0, 0] = 1.0
        bandit1.total_queries = 100
        bandit1.arm_pulls["test-model-1"] = 50

        # Serialize and restore
        state = bandit1.to_state()
        bandit2 = LinUCBBandit(test_arms, feature_dim=10)
        bandit2.from_state(state)

        # Verify matrices restored
        assert bandit2.A["test-model-1"][0, 0] == 3.0
        assert bandit2.A["test-model-1"][1, 1] == 2.0
        assert bandit2.b["test-model-1"][0, 0] == 1.0
        assert bandit2.total_queries == 100
        assert bandit2.arm_pulls["test-model-1"] == 50

    def test_a_inv_recomputed_on_restore(self, test_arms):
        """Test that A_inv is recomputed from A on restore."""
        bandit1 = LinUCBBandit(test_arms, feature_dim=10)
        # Modify A
        bandit1.A["test-model-1"][0, 0] = 2.0

        state = bandit1.to_state()
        bandit2 = LinUCBBandit(test_arms, feature_dim=10)
        bandit2.from_state(state)

        # A_inv should be inverse of A
        identity = bandit2.A["test-model-1"] @ bandit2.A_inv["test-model-1"]
        np.testing.assert_array_almost_equal(identity, np.eye(10), decimal=5)

    def test_from_state_rejects_wrong_feature_dim(self, test_arms):
        """Test that from_state rejects mismatched feature dimensions."""
        bandit = LinUCBBandit(test_arms, feature_dim=10)
        state = BanditState(
            algorithm="linucb",
            arm_ids=["test-model-1", "test-model-2"],
            feature_dim=20,  # Wrong dimension
            A_matrices={},
            b_vectors={},
        )

        with pytest.raises(ValueError, match="feature_dim"):
            bandit.from_state(state)


class TestThompsonSamplingStatePersistence:
    """Tests for ThompsonSamplingBandit state persistence."""

    def test_to_state_preserves_distribution(self, test_arms):
        """Test that alpha and beta parameters are preserved."""
        bandit = ThompsonSamplingBandit(test_arms)
        bandit.alpha["test-model-1"] = 10.0
        bandit.beta["test-model-1"] = 2.0
        bandit.total_queries = 50

        state = bandit.to_state()

        assert state.algorithm == "thompson_sampling"
        assert state.alpha_params["test-model-1"] == 10.0
        assert state.beta_params["test-model-1"] == 2.0

    def test_from_state_restores_distribution(self, test_arms):
        """Test that distribution parameters are restored."""
        bandit1 = ThompsonSamplingBandit(test_arms)
        bandit1.alpha["test-model-1"] = 15.0
        bandit1.beta["test-model-1"] = 5.0
        bandit1.total_queries = 100

        state = bandit1.to_state()
        bandit2 = ThompsonSamplingBandit(test_arms)
        bandit2.from_state(state)

        assert bandit2.alpha["test-model-1"] == 15.0
        assert bandit2.beta["test-model-1"] == 5.0
        assert bandit2.total_queries == 100


class TestEpsilonGreedyStatePersistence:
    """Tests for EpsilonGreedyBandit state persistence."""

    def test_to_state_preserves_epsilon(self, test_arms):
        """Test that current epsilon is preserved."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.5)
        bandit.epsilon = 0.3  # Decayed epsilon

        state = bandit.to_state()

        assert state.algorithm == "epsilon_greedy"
        assert state.epsilon == 0.3

    def test_from_state_restores_epsilon(self, test_arms):
        """Test that epsilon is restored."""
        bandit1 = EpsilonGreedyBandit(test_arms, epsilon=0.5)
        bandit1.epsilon = 0.2
        bandit1.mean_reward["test-model-1"] = 0.85

        state = bandit1.to_state()
        bandit2 = EpsilonGreedyBandit(test_arms, epsilon=0.5)
        bandit2.from_state(state)

        assert bandit2.epsilon == 0.2
        assert bandit2.mean_reward["test-model-1"] == 0.85


class TestContextualThompsonSamplingStatePersistence:
    """Tests for ContextualThompsonSamplingBandit state persistence."""

    def test_to_state_preserves_posterior(self, test_arms):
        """Test that mu and Sigma are preserved."""
        bandit = ContextualThompsonSamplingBandit(test_arms, feature_dim=10)
        bandit.mu["test-model-1"][0, 0] = 0.5
        bandit.Sigma["test-model-1"][0, 0] = 2.0

        state = bandit.to_state()

        assert state.algorithm == "contextual_thompson_sampling"
        assert state.mu_vectors["test-model-1"][0] == 0.5
        assert state.sigma_matrices["test-model-1"][0][0] == 2.0

    def test_from_state_restores_posterior(self, test_arms):
        """Test that posterior parameters are restored."""
        bandit1 = ContextualThompsonSamplingBandit(test_arms, feature_dim=10)
        bandit1.mu["test-model-1"][0, 0] = 1.0
        bandit1.Sigma["test-model-1"][0, 0] = 0.5

        state = bandit1.to_state()
        bandit2 = ContextualThompsonSamplingBandit(test_arms, feature_dim=10)
        bandit2.from_state(state)

        assert bandit2.mu["test-model-1"][0, 0] == 1.0
        assert bandit2.Sigma["test-model-1"][0, 0] == 0.5


class TestHybridRouterStatePersistence:
    """Tests for HybridRouter state persistence."""

    def test_to_state_preserves_phase(self):
        """Test that current phase is preserved."""
        router = HybridRouter(
            models=["test-model-1", "test-model-2"],
            switch_threshold=2000,
        )
        router.query_count = 1500
        router.current_phase = "ucb1"

        state = router.to_state()

        assert state.current_phase == RouterPhase.UCB1
        assert state.query_count == 1500
        assert state.ucb1_state is not None
        assert state.linucb_state is not None

    def test_from_state_restores_phase(self):
        """Test that phase is restored."""
        router1 = HybridRouter(
            models=["test-model-1", "test-model-2"],
            switch_threshold=2000,
        )
        router1.query_count = 2500
        router1.current_phase = "linucb"
        router1.ucb1.total_queries = 2000
        router1.linucb.total_queries = 500

        state = router1.to_state()
        router2 = HybridRouter(
            models=["test-model-1", "test-model-2"],
            switch_threshold=2000,
        )
        router2.from_state(state)

        assert router2.query_count == 2500
        assert router2.current_phase == "linucb"
        assert router2.ucb1.total_queries == 2000
        assert router2.linucb.total_queries == 500

    def test_to_state_preserves_bandit_states(self):
        """Test that embedded bandit states are preserved."""
        router = HybridRouter(
            models=["test-model-1", "test-model-2"],
            switch_threshold=2000,
        )
        router.ucb1.arm_pulls["test-model-1"] = 100
        router.linucb.arm_pulls["test-model-2"] = 50

        state = router.to_state()

        assert state.ucb1_state.arm_pulls["test-model-1"] == 100
        assert state.linucb_state.arm_pulls["test-model-2"] == 50

    def test_full_roundtrip(self):
        """Test complete serialize-deserialize roundtrip."""
        # Create router with complex state
        router1 = HybridRouter(
            models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
            switch_threshold=2000,
        )
        router1.query_count = 1000
        router1.ucb1.total_queries = 1000
        router1.ucb1.arm_pulls["gpt-4o-mini"] = 400
        router1.ucb1.arm_pulls["gpt-4o"] = 350
        router1.ucb1.arm_pulls["claude-3-5-sonnet"] = 250
        router1.ucb1.mean_reward["gpt-4o-mini"] = 0.88
        router1.ucb1.mean_reward["gpt-4o"] = 0.92
        router1.ucb1.mean_reward["claude-3-5-sonnet"] = 0.90

        # Serialize
        state = router1.to_state()

        # Create new router and restore
        router2 = HybridRouter(
            models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
            switch_threshold=2000,
        )
        router2.from_state(state)

        # Verify all state restored
        assert router2.query_count == 1000
        assert router2.current_phase == "ucb1"
        assert router2.ucb1.arm_pulls["gpt-4o-mini"] == 400
        assert router2.ucb1.arm_pulls["gpt-4o"] == 350
        assert router2.ucb1.mean_reward["gpt-4o-mini"] == 0.88
        assert router2.ucb1.mean_reward["gpt-4o"] == 0.92


class TestStateStoreModels:
    """Tests for state store data models."""

    def test_bandit_state_serialization(self):
        """Test BanditState JSON serialization."""
        state = BanditState(
            algorithm="ucb1",
            arm_ids=["arm1", "arm2"],
            arm_pulls={"arm1": 10, "arm2": 20},
            total_queries=30,
        )

        # Should serialize to JSON without error
        json_str = state.model_dump_json()
        assert "ucb1" in json_str
        assert "arm1" in json_str

    def test_hybrid_router_state_serialization(self):
        """Test HybridRouterState JSON serialization."""
        state = HybridRouterState(
            query_count=1000,
            current_phase=RouterPhase.LINUCB,
            transition_threshold=2000,
        )

        # Should serialize to JSON without error
        json_str = state.model_dump_json()
        assert "linucb" in json_str
        assert "1000" in json_str
