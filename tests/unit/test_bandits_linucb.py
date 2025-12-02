"""Unit tests for LinUCB bandit algorithm.

Uses shared fixtures from tests/conftest.py: test_arms, test_features
"""

import numpy as np
import pytest

from conduit.core.models import QueryFeatures
from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.bandits.linucb import LinUCBBandit

# test_arms and test_features fixtures imported from conftest.py


class TestLinUCBBandit:
    """Tests for LinUCBBandit."""

    def test_initialization_default_alpha(self, test_arms):
        """Test LinUCB bandit initializes with default parameters."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        assert bandit.name == "linucb"
        assert len(bandit.arms) == 3
        assert bandit.total_queries == 0
        assert bandit.alpha == 1.0
        assert bandit.feature_dim == 386

        # Check initial state - A should be identity, b should be zeros
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert bandit.arm_pulls[model_id] == 0
            # A should be identity matrix
            assert np.allclose(bandit.A[model_id], np.identity(386))
            # b should be zero vector
            assert np.allclose(bandit.b[model_id], np.zeros((386, 1)))

    def test_initialization_custom_alpha(self, test_arms):
        """Test LinUCB bandit with custom exploration parameter."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=2.0)

        assert bandit.alpha == 2.0

    def test_feature_extraction(self, test_arms, test_features):
        """Test feature vector extraction from QueryFeatures."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        x = bandit._extract_features(test_features)

        # Check shape
        assert x.shape == (386, 1)

        # Check embedding values (first 384 dims)
        assert np.allclose(x[:384, 0], 0.1)

        # Check metadata (last 2 dims)
        assert np.isclose(x[384, 0], 50.0 / 1000.0)  # normalized token_count
        assert np.isclose(x[385, 0], 0.5)  # complexity_score
        assert np.isclose(
            x[385, 0], test_features.complexity_score
        )  # complexity_score (last metadata)

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm(self, test_arms, test_features):
        """Test arm selection returns valid arm from available arms."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert arm.model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]
        assert bandit.total_queries == 1

    @pytest.mark.asyncio
    async def test_update_with_feedback(self, test_arms, test_features):
        """Test update correctly updates A and b matrices."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        arm = await bandit.select_arm(test_features)

        # Store initial A and b
        A_before = bandit.A[arm.model_id].copy()
        b_before = bandit.b[arm.model_id].copy()

        feedback = BanditFeedback(
            model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
        )

        await bandit.update(feedback, test_features)

        # A and b should have changed
        assert not np.allclose(bandit.A[arm.model_id], A_before)
        assert not np.allclose(bandit.b[arm.model_id], b_before)

        # Arm pulls should increment
        assert bandit.arm_pulls[arm.model_id] == 1

    @pytest.mark.asyncio
    async def test_matrix_updates(self, test_arms, test_features):
        """Test A and b update formulas are correct."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        arm = test_arms[0]
        model_id = arm.model_id

        # Get feature vector
        x = bandit._extract_features(test_features)

        # Store initial values
        A_initial = bandit.A[model_id].copy()
        b_initial = bandit.b[model_id].copy()

        # Apply update manually
        feedback = BanditFeedback(
            model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
        )

        await bandit.update(feedback, test_features)

        # Verify A = A + x @ x.T
        expected_A = A_initial + x @ x.T
        assert np.allclose(bandit.A[model_id], expected_A)

        # Verify b = b + composite_reward * x
        # Composite: 0.9*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8798...
        composite_reward = 0.9 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        expected_b = b_initial + composite_reward * x
        assert np.allclose(bandit.b[model_id], expected_b)

    @pytest.mark.asyncio
    async def test_learning_with_different_contexts(self, test_arms):
        """Test bandit learns different rewards for different contexts."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=0.5)

        # Context 1: Simple query (low complexity)
        context1 = QueryFeatures(
            embedding=[0.1] * 384, token_count=10, complexity_score=0.1
        )

        # Context 2: Complex query (high complexity)
        context2 = QueryFeatures(
            embedding=[0.9] * 384, token_count=100, complexity_score=0.9
        )

        # Train on context1: o4-mini performs well
        for _ in range(5):
            arm = await bandit.select_arm(context1)
            if arm.model_id == "o4-mini":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=quality, latency=1.0
            )
            await bandit.update(feedback, context1)

        # Train on context2: gpt-5.1 performs well
        for _ in range(5):
            arm = await bandit.select_arm(context2)
            if arm.model_id == "gpt-5.1":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=quality, latency=1.0
            )
            await bandit.update(feedback, context2)

        # After learning, bandit should have updated its models
        assert bandit.arm_pulls["o4-mini"] >= 1
        assert bandit.arm_pulls["gpt-5.1"] >= 1

    @pytest.mark.asyncio
    async def test_exploration_parameter_effect(self, test_arms, test_features):
        """Test alpha parameter controls exploration vs exploitation."""
        # Low alpha = more exploitation
        bandit_low = LinUCBBandit(test_arms, feature_dim=386, alpha=0.1)

        # High alpha = more exploration
        bandit_high = LinUCBBandit(test_arms, feature_dim=386, alpha=5.0)

        # Verify alpha is stored correctly
        assert bandit_low.alpha == 0.1
        assert bandit_high.alpha == 5.0

        # Both should be able to select arms
        arm_low = await bandit_low.select_arm(test_features)
        arm_high = await bandit_high.select_arm(test_features)

        assert arm_low in test_arms
        assert arm_high in test_arms

    @pytest.mark.asyncio
    async def test_reset(self, test_arms, test_features):
        """Test reset restores initial state."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Do some updates
        for _ in range(5):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, test_features)

        assert bandit.total_queries == 5

        # Reset
        bandit.reset()

        # Check all restored to initial state
        assert bandit.total_queries == 0
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert bandit.arm_pulls[model_id] == 0
            # A should be identity
            assert np.allclose(bandit.A[model_id], np.identity(386))
            # b should be zeros
            assert np.allclose(bandit.b[model_id], np.zeros((386, 1)))

    @pytest.mark.asyncio
    async def test_get_stats(self, test_arms, test_features):
        """Test get_stats returns comprehensive statistics."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5)

        # Do some updates
        arm = await bandit.select_arm(test_features)
        feedback = BanditFeedback(
            model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
        )
        await bandit.update(feedback, test_features)

        stats = bandit.get_stats()

        # Check required fields
        assert "total_queries" in stats
        assert "alpha" in stats
        assert "feature_dim" in stats
        assert "arm_pulls" in stats
        assert "arm_success_rates" in stats
        assert "arm_theta_norms" in stats

        assert stats["alpha"] == 1.5
        assert stats["feature_dim"] == 386
        assert stats["total_queries"] == 1

    @pytest.mark.asyncio
    async def test_contextual_learning(self, test_arms):
        """Test LinUCB learns context-dependent preferences."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.0)

        # Scenario: Model A good for short queries, Model B good for long queries

        # Short queries (token_count < 50) -> o4-mini performs well
        for _ in range(10):
            features = QueryFeatures(
                embedding=[np.random.rand() for _ in range(384)],
                token_count=20,
                complexity_score=0.3,
            )

            arm = await bandit.select_arm(features)

            if arm.model_id == "o4-mini":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=quality, latency=1.0
            )
            await bandit.update(feedback, features)

        # Long queries (token_count > 200) -> gpt-5.1 performs well
        for _ in range(10):
            features = QueryFeatures(
                embedding=[np.random.rand() for _ in range(384)],
                token_count=300,
                complexity_score=0.8,
            )

            arm = await bandit.select_arm(features)

            if arm.model_id == "gpt-5.1":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=quality, latency=1.0
            )
            await bandit.update(feedback, features)

        # Both models should have been tried
        assert bandit.arm_pulls["o4-mini"] > 0
        assert bandit.arm_pulls["gpt-5.1"] > 0

        # Total should be 20
        assert bandit.total_queries == 20

    @pytest.mark.asyncio
    async def test_ucb_computation(self, test_arms, test_features):
        """Test UCB value computation is reasonable."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.0)

        # Select an arm and update it
        arm = await bandit.select_arm(test_features)
        feedback = BanditFeedback(
            model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
        )
        await bandit.update(feedback, test_features)

        # Manually compute UCB for the updated arm
        x = bandit._extract_features(test_features)
        A_inv = np.linalg.inv(bandit.A[arm.model_id])
        theta = A_inv @ bandit.b[arm.model_id]

        mean_reward = (theta.T @ x)[0, 0]
        uncertainty = np.sqrt((x.T @ A_inv @ x)[0, 0])
        expected_ucb = mean_reward + bandit.alpha * uncertainty

        # UCB should be finite and reasonable
        assert np.isfinite(expected_ucb)
        assert -1.0 <= expected_ucb <= 2.0  # Reasonable range for normalized rewards

    @pytest.mark.asyncio
    async def test_sherman_morrison_incremental_update(self, test_arms, test_features):
        """Test Sherman-Morrison incremental A_inv update (no sliding window)."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, window_size=0)  # No sliding window

        arm = test_arms[0]
        model_id = arm.model_id

        # Store initial A_inv
        A_inv_initial = bandit.A_inv[model_id].copy()

        # Apply feedback
        feedback = BanditFeedback(
            model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
        )
        await bandit.update(feedback, test_features)

        # Verify A_inv was updated (should differ from initial identity)
        assert not np.allclose(bandit.A_inv[model_id], A_inv_initial)

        # Verify A_inv is actually the inverse of A (A @ A_inv = I)
        product = bandit.A[model_id] @ bandit.A_inv[model_id]
        assert np.allclose(product, np.identity(bandit.feature_dim), atol=1e-10)

        # Apply more updates to test incremental nature
        for i in range(5):
            features_i = QueryFeatures(
                embedding=[0.1 + i * 0.01] * 384,
                token_count=50 + i * 10,
                complexity_score=0.5 + i * 0.05,
            )
            feedback_i = BanditFeedback(
                model_id=model_id,
                cost=0.001 * (i + 1),
                quality_score=0.9 - i * 0.02,
                latency=1.0 + i * 0.1,
            )
            await bandit.update(feedback_i, features_i)

            # Verify A_inv remains valid after each update
            product = bandit.A[model_id] @ bandit.A_inv[model_id]
            assert np.allclose(product, np.identity(bandit.feature_dim), atol=1e-9)

    @pytest.mark.asyncio
    async def test_sherman_morrison_vs_direct_inversion(self, test_arms, test_features):
        """Test Sherman-Morrison gives same result as direct matrix inversion."""
        # Set random seed for reproducibility
        np.random.seed(42)

        # Create two bandits: one with Sherman-Morrison, one using direct inversion for comparison
        bandit_sm = LinUCBBandit(test_arms, feature_dim=386, window_size=0, alpha=1.0)

        # Simulate multiple updates
        for i in range(10):
            features_i = QueryFeatures(
                embedding=[np.random.rand() for _ in range(384)],
                token_count=50 + i * 10,
                complexity_score=0.5 + i * 0.05,
            )

            arm = await bandit_sm.select_arm(features_i)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001 * (i + 1),
                quality_score=0.85 + np.random.rand() * 0.1,
                latency=1.0 + i * 0.1,
            )
            await bandit_sm.update(feedback, features_i)

        # For each arm, verify that cached A_inv matches direct inversion of A
        for model_id in bandit_sm.arms:
            direct_inv = np.linalg.inv(bandit_sm.A[model_id])
            assert np.allclose(bandit_sm.A_inv[model_id], direct_inv, atol=1e-10)

    @pytest.mark.asyncio
    async def test_sliding_window_incremental_updates(self, test_arms, test_features):
        """Test sliding window mode uses incremental Woodbury updates for O(d²) complexity."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, window_size=5)  # Small window

        arm = test_arms[0]
        model_id = arm.model_id

        # Add more updates than window size
        for i in range(10):
            features_i = QueryFeatures(
                embedding=[0.1 + i * 0.01] * 384,
                token_count=50 + i * 10,
                complexity_score=0.5,
            )
            feedback = BanditFeedback(
                model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, features_i)

            # Verify A_inv is correct inverse of A
            product = bandit.A[model_id] @ bandit.A_inv[model_id]
            assert np.allclose(product, np.identity(bandit.feature_dim), atol=1e-9)

        # Window should only contain last 5 observations
        assert len(bandit.observation_history[model_id]) == 5

    @pytest.mark.asyncio
    async def test_a_inv_initialization(self, test_arms):
        """Test A_inv is initialized to identity matrix."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert np.allclose(bandit.A_inv[model_id], np.identity(386))
            # Verify it's the inverse of A (which is also identity initially)
            product = bandit.A[model_id] @ bandit.A_inv[model_id]
            assert np.allclose(product, np.identity(386))

    @pytest.mark.asyncio
    async def test_a_inv_reset(self, test_arms, test_features):
        """Test reset() restores A_inv to identity."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Do some updates
        for _ in range(5):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, test_features)

        # Verify A_inv changed
        for model_id in bandit.arms:
            if bandit.arm_pulls[model_id] > 0:
                # This arm was updated, A_inv should differ from identity
                assert not np.allclose(bandit.A_inv[model_id], np.identity(386))

        # Reset
        bandit.reset()

        # Check A_inv restored to identity
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert np.allclose(bandit.A_inv[model_id], np.identity(386))

    @pytest.mark.asyncio
    async def test_numerical_stability_fallback(self, test_arms):
        """Test numerical stability fallback when denominator is too small."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, window_size=0)

        # This test verifies the code path exists, though triggering it is difficult
        # The denominator = 1 + x^T @ A_inv @ x should always be > 1 for positive definite A
        # Just verify normal operation doesn't hit fallback
        arm = test_arms[0]
        model_id = arm.model_id

        features = QueryFeatures(
            embedding=[0.1] * 384, token_count=50, complexity_score=0.5
        )

        feedback = BanditFeedback(
            model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
        )

        await bandit.update(feedback, features)

        # Verify A_inv is still valid
        product = bandit.A[model_id] @ bandit.A_inv[model_id]
        assert np.allclose(product, np.identity(bandit.feature_dim), atol=1e-10)

    def test_custom_reward_weights(self, test_arms):
        """Test initialization with custom reward weights."""
        custom_weights = {"quality": 0.5, "cost": 0.3, "latency": 0.2}
        bandit = LinUCBBandit(test_arms, feature_dim=386, reward_weights=custom_weights)

        assert bandit.reward_weights == custom_weights
        assert bandit.reward_weights["quality"] == 0.5
        assert bandit.reward_weights["cost"] == 0.3
        assert bandit.reward_weights["latency"] == 0.2

    def test_random_seed(self, test_arms):
        """Test initialization with random seed for reproducibility."""
        bandit1 = LinUCBBandit(test_arms, feature_dim=386, random_seed=42)
        bandit2 = LinUCBBandit(test_arms, feature_dim=386, random_seed=42)

        # Both bandits should be in the same state after seeding
        assert bandit1.alpha == bandit2.alpha
        assert bandit1.feature_dim == bandit2.feature_dim


class TestLinUCBStatePersistence:
    """Tests for LinUCB state persistence with observation history."""

    @pytest.mark.asyncio
    async def test_to_state_basic(self, test_arms, test_features):
        """Test basic to_state serialization."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5, window_size=10)

        # Build some state
        for i in range(3):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, test_features)

        state = bandit.to_state()

        assert state.algorithm == "linucb"
        assert state.alpha == 1.5
        assert state.feature_dim == 386
        assert state.total_queries == 3
        assert set(state.arm_ids) == {"o4-mini", "gpt-5.1", "claude-haiku-4-5"}

    @pytest.mark.asyncio
    async def test_to_state_with_observation_history(self, test_arms, test_features):
        """Test to_state includes observation history."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, window_size=5)

        # Build state with multiple observations
        for i in range(3):
            features_i = QueryFeatures(
                embedding=[0.1 + i * 0.01] * 384,
                token_count=50 + i * 10,
                complexity_score=0.5,
            )
            arm = await bandit.select_arm(features_i)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.8 + i * 0.05, latency=1.0
            )
            await bandit.update(feedback, features_i)

        state = bandit.to_state()

        # Should have observation history entries
        assert len(state.observation_history) == 3
        for entry in state.observation_history:
            assert "arm_id" in entry
            assert "features" in entry
            assert "reward" in entry
            assert len(entry["features"]) == 386

    @pytest.mark.asyncio
    async def test_from_state_restores_counters(self, test_arms, test_features):
        """Test from_state restores arm pulls and query counts."""
        from conduit.core.state_store import BanditState

        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5)

        # Build some state
        for _ in range(5):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, test_features)

        # Serialize state
        state = bandit.to_state()

        # Create new bandit and restore
        bandit2 = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5)
        bandit2.from_state(state)

        assert bandit2.total_queries == bandit.total_queries
        assert bandit2.arm_pulls == bandit.arm_pulls
        assert bandit2.arm_successes == bandit.arm_successes

    @pytest.mark.asyncio
    async def test_from_state_restores_matrices(self, test_arms, test_features):
        """Test from_state restores A and b matrices correctly."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Build some state
        for i in range(5):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, test_features)

        state = bandit.to_state()

        # Create new bandit and restore
        bandit2 = LinUCBBandit(test_arms, feature_dim=386)
        bandit2.from_state(state)

        # Verify A and b are restored
        for model_id in bandit.arms:
            assert np.allclose(bandit2.A[model_id], bandit.A[model_id])
            assert np.allclose(bandit2.b[model_id], bandit.b[model_id])

    @pytest.mark.asyncio
    async def test_from_state_restores_observation_history(self, test_arms, test_features):
        """Test from_state restores observation history."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, window_size=10)

        # Build state with observations on specific arm
        model_id = test_arms[0].model_id
        for i in range(3):
            features_i = QueryFeatures(
                embedding=[0.1 + i * 0.01] * 384,
                token_count=50,
                complexity_score=0.5,
            )
            feedback = BanditFeedback(
                model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
            )
            await bandit.update(feedback, features_i)

        state = bandit.to_state()

        # Create new bandit and restore
        bandit2 = LinUCBBandit(test_arms, feature_dim=386, window_size=10)
        bandit2.from_state(state)

        # Verify observation history restored
        assert len(bandit2.observation_history[model_id]) == 3

    @pytest.mark.asyncio
    async def test_from_state_rejects_wrong_algorithm(self, test_arms):
        """Test from_state raises error for wrong algorithm."""
        from conduit.core.state_store import BanditState

        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Create state with wrong algorithm
        state = BanditState(
            algorithm="thompson_sampling",  # Wrong algorithm
            arm_ids=["o4-mini", "gpt-5.1", "claude-haiku-4-5"],
            arm_pulls={"o4-mini": 0, "gpt-5.1": 0, "claude-haiku-4-5": 0},
            arm_successes={"o4-mini": 0, "gpt-5.1": 0, "claude-haiku-4-5": 0},
            total_queries=0,
        )

        with pytest.raises(ValueError, match="'thompson_sampling' != 'linucb'"):
            bandit.from_state(state)

    @pytest.mark.asyncio
    async def test_from_state_rejects_mismatched_arms(self, test_arms):
        """Test from_state raises error for mismatched arms."""
        from conduit.core.state_store import BanditState

        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Create state with different arms
        state = BanditState(
            algorithm="linucb",
            arm_ids=["model-a", "model-b"],  # Different arms
            arm_pulls={"model-a": 0, "model-b": 0},
            arm_successes={"model-a": 0, "model-b": 0},
            total_queries=0,
        )

        with pytest.raises(ValueError, match="don't match current arms"):
            bandit.from_state(state)

    @pytest.mark.asyncio
    async def test_from_state_rejects_mismatched_feature_dim(self, test_arms):
        """Test from_state raises error for mismatched feature dimension."""
        from conduit.core.state_store import BanditState

        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Create state with different feature dimension
        state = BanditState(
            algorithm="linucb",
            arm_ids=["o4-mini", "gpt-5.1", "claude-haiku-4-5"],
            arm_pulls={"o4-mini": 0, "gpt-5.1": 0, "claude-haiku-4-5": 0},
            arm_successes={"o4-mini": 0, "gpt-5.1": 0, "claude-haiku-4-5": 0},
            total_queries=0,
            feature_dim=128,  # Different dimension
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            bandit.from_state(state)

    @pytest.mark.asyncio
    async def test_roundtrip_state_persistence(self, test_arms, test_features):
        """Test full roundtrip: build state, serialize, deserialize, verify."""
        bandit = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5, window_size=10)

        # Build state
        for i in range(10):
            features_i = QueryFeatures(
                embedding=[0.1 + i * 0.005] * 384,
                token_count=50 + i * 5,
                complexity_score=0.5,
            )
            arm = await bandit.select_arm(features_i)
            feedback = BanditFeedback(
                model_id=arm.model_id, cost=0.001, quality_score=0.85 + i * 0.01, latency=1.0
            )
            await bandit.update(feedback, features_i)

        # Serialize
        state = bandit.to_state()

        # Deserialize into new bandit
        bandit2 = LinUCBBandit(test_arms, feature_dim=386, alpha=1.5, window_size=10)
        bandit2.from_state(state)

        # Verify everything matches
        assert bandit2.total_queries == bandit.total_queries
        assert bandit2.arm_pulls == bandit.arm_pulls
        assert bandit2.arm_successes == bandit.arm_successes
        assert bandit2.alpha == bandit.alpha

        for model_id in bandit.arms:
            assert np.allclose(bandit2.A[model_id], bandit.A[model_id])
            assert np.allclose(bandit2.b[model_id], bandit.b[model_id])
            assert np.allclose(bandit2.A_inv[model_id], bandit.A_inv[model_id], atol=1e-10)
