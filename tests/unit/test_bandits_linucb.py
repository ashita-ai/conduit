"""Unit tests for LinUCB bandit algorithm."""

import numpy as np
import pytest

from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.core.models import QueryFeatures


@pytest.fixture
def test_arms():
    """Create test model arms."""
    return [
        ModelArm(
            model_id="gpt-4o-mini",
            model_name="gpt-4o-mini",
            provider="openai",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="gpt-4o",
            model_name="gpt-4o",
            provider="openai",
            cost_per_input_token=0.0025,
            cost_per_output_token=0.010,
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-3-haiku",
            model_name="claude-3-haiku",
            provider="anthropic",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
            expected_quality=0.80,
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


class TestLinUCBBandit:
    """Tests for LinUCBBandit."""

    def test_initialization_default_alpha(self, test_arms):
        """Test LinUCB bandit initializes with default parameters."""
        bandit = LinUCBBandit(test_arms)

        assert bandit.name == "linucb"
        assert len(bandit.arms) == 3
        assert bandit.total_queries == 0
        assert bandit.alpha == 1.0
        assert bandit.feature_dim == 387

        # Check initial state - A should be identity, b should be zeros
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            # A should be identity matrix
            assert np.allclose(bandit.A[model_id], np.identity(387))
            # b should be zero vector
            assert np.allclose(bandit.b[model_id], np.zeros((387, 1)))

    def test_initialization_custom_alpha(self, test_arms):
        """Test LinUCB bandit with custom exploration parameter."""
        bandit = LinUCBBandit(test_arms, alpha=2.0)

        assert bandit.alpha == 2.0

    def test_feature_extraction(self, test_arms, test_features):
        """Test feature vector extraction from QueryFeatures."""
        bandit = LinUCBBandit(test_arms)

        x = bandit._extract_features(test_features)

        # Check shape
        assert x.shape == (387, 1)

        # Check embedding values (first 384 dims)
        assert np.allclose(x[:384, 0], 0.1)

        # Check metadata (last 3 dims)
        assert np.isclose(x[384, 0], 50.0 / 1000.0)  # normalized token_count
        assert np.isclose(x[385, 0], 0.5)  # complexity_score
        assert np.isclose(x[386, 0], 0.8)  # domain_confidence

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm(self, test_arms, test_features):
        """Test arm selection returns valid arm from available arms."""
        bandit = LinUCBBandit(test_arms)

        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert arm.model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]
        assert bandit.total_queries == 1

    @pytest.mark.asyncio
    async def test_update_with_feedback(self, test_arms, test_features):
        """Test update correctly updates A and b matrices."""
        bandit = LinUCBBandit(test_arms)

        arm = await bandit.select_arm(test_features)

        # Store initial A and b
        A_before = bandit.A[arm.model_id].copy()
        b_before = bandit.b[arm.model_id].copy()

        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
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
        bandit = LinUCBBandit(test_arms)

        arm = test_arms[0]
        model_id = arm.model_id

        # Get feature vector
        x = bandit._extract_features(test_features)

        # Store initial values
        A_initial = bandit.A[model_id].copy()
        b_initial = bandit.b[model_id].copy()

        # Apply update manually
        feedback = BanditFeedback(
            model_id=model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )

        await bandit.update(feedback, test_features)

        # Verify A = A + x @ x.T
        expected_A = A_initial + x @ x.T
        assert np.allclose(bandit.A[model_id], expected_A)

        # Verify b = b + reward * x
        expected_b = b_initial + 0.9 * x
        assert np.allclose(bandit.b[model_id], expected_b)

    @pytest.mark.asyncio
    async def test_learning_with_different_contexts(self, test_arms):
        """Test bandit learns different rewards for different contexts."""
        bandit = LinUCBBandit(test_arms, alpha=0.5)

        # Context 1: Simple query (low complexity)
        context1 = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=10,
            complexity_score=0.1,
            domain="general",
            domain_confidence=0.9,
        )

        # Context 2: Complex query (high complexity)
        context2 = QueryFeatures(
            embedding=[0.9] * 384,
            token_count=100,
            complexity_score=0.9,
            domain="technical",
            domain_confidence=0.8,
        )

        # Train on context1: gpt-4o-mini performs well
        for _ in range(5):
            arm = await bandit.select_arm(context1)
            if arm.model_id == "gpt-4o-mini":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, context1)

        # Train on context2: gpt-4o performs well
        for _ in range(5):
            arm = await bandit.select_arm(context2)
            if arm.model_id == "gpt-4o":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, context2)

        # After learning, bandit should have updated its models
        assert bandit.arm_pulls["gpt-4o-mini"] >= 1
        assert bandit.arm_pulls["gpt-4o"] >= 1

    @pytest.mark.asyncio
    async def test_exploration_parameter_effect(self, test_arms, test_features):
        """Test alpha parameter controls exploration vs exploitation."""
        # Low alpha = more exploitation
        bandit_low = LinUCBBandit(test_arms, alpha=0.1)

        # High alpha = more exploration
        bandit_high = LinUCBBandit(test_arms, alpha=5.0)

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
        bandit = LinUCBBandit(test_arms)

        # Do some updates
        for _ in range(5):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.9,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        assert bandit.total_queries == 5

        # Reset
        bandit.reset()

        # Check all restored to initial state
        assert bandit.total_queries == 0
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            # A should be identity
            assert np.allclose(bandit.A[model_id], np.identity(387))
            # b should be zeros
            assert np.allclose(bandit.b[model_id], np.zeros((387, 1)))

    @pytest.mark.asyncio
    async def test_get_stats(self, test_arms, test_features):
        """Test get_stats returns comprehensive statistics."""
        bandit = LinUCBBandit(test_arms, alpha=1.5)

        # Do some updates
        arm = await bandit.select_arm(test_features)
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
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
        assert stats["feature_dim"] == 387
        assert stats["total_queries"] == 1

    @pytest.mark.asyncio
    async def test_contextual_learning(self, test_arms):
        """Test LinUCB learns context-dependent preferences."""
        bandit = LinUCBBandit(test_arms, alpha=1.0)

        # Scenario: Model A good for short queries, Model B good for long queries

        # Short queries (token_count < 50) -> gpt-4o-mini performs well
        for _ in range(10):
            features = QueryFeatures(
                embedding=[np.random.rand() for _ in range(384)],
                token_count=20,
                complexity_score=0.3,
                domain="general",
                domain_confidence=0.8,
            )

            arm = await bandit.select_arm(features)

            if arm.model_id == "gpt-4o-mini":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, features)

        # Long queries (token_count > 200) -> gpt-4o performs well
        for _ in range(10):
            features = QueryFeatures(
                embedding=[np.random.rand() for _ in range(384)],
                token_count=300,
                complexity_score=0.8,
                domain="technical",
                domain_confidence=0.7,
            )

            arm = await bandit.select_arm(features)

            if arm.model_id == "gpt-4o":
                quality = 0.95
            else:
                quality = 0.6

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, features)

        # Both models should have been tried
        assert bandit.arm_pulls["gpt-4o-mini"] > 0
        assert bandit.arm_pulls["gpt-4o"] > 0

        # Total should be 20
        assert bandit.total_queries == 20

    @pytest.mark.asyncio
    async def test_ucb_computation(self, test_arms, test_features):
        """Test UCB value computation is reasonable."""
        bandit = LinUCBBandit(test_arms, alpha=1.0)

        # Select an arm and update it
        arm = await bandit.select_arm(test_features)
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
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
