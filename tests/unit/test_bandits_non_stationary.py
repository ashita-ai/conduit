"""Tests for non-stationary bandit algorithms with sliding windows (Phase 3).

These tests verify that bandit algorithms can adapt to distribution shifts
(e.g., model quality/cost changes over time) using sliding window mechanisms.

Uses shared fixtures from tests/conftest.py: test_arms, test_features
"""

import numpy as np
import pytest

from conduit.core.models import QueryFeatures
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.engines.bandits.epsilon_greedy import EpsilonGreedyBandit
from conduit.engines.bandits.linucb import LinUCBBandit
from conduit.engines.bandits.thompson_sampling import ThompsonSamplingBandit
from conduit.engines.bandits.ucb import UCB1Bandit

# test_arms and test_features fixtures imported from conftest.py


class TestThompsonSamplingNonStationary:
    """Tests for Thompson Sampling with sliding window."""

    @pytest.mark.asyncio
    async def test_sliding_window_initialization(
        self, test_arms: list[ModelArm]
    ) -> None:
        """Test that sliding window is initialized correctly."""
        bandit = ThompsonSamplingBandit(arms=test_arms, window_size=100, random_seed=42)

        assert bandit.window_size == 100
        assert len(bandit.reward_history) == 3  # Three arms from fixture
        assert all(history.maxlen == 100 for history in bandit.reward_history.values())

    @pytest.mark.asyncio
    async def test_unlimited_history(self, test_arms: list[ModelArm]) -> None:
        """Test that window_size=0 creates unlimited history."""
        bandit = ThompsonSamplingBandit(
            arms=test_arms, window_size=0, random_seed=42  # Unlimited
        )

        assert bandit.window_size == 0
        # Deque without maxlen = unlimited
        assert all(history.maxlen is None for history in bandit.reward_history.values())

    @pytest.mark.asyncio
    async def test_window_drops_oldest_observations(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test that sliding window drops oldest observations when full."""
        bandit = ThompsonSamplingBandit(
            arms=test_arms, window_size=3, random_seed=42  # Very small window
        )

        arm = test_arms[0]
        model_id = arm.model_id

        # Add 5 observations (window only keeps last 3)
        for i in range(5):
            reward = (i + 1) / 10.0  # 0.1, 0.2, 0.3, 0.4, 0.5
            feedback = BanditFeedback(
                model_id=model_id,
                cost=0.001,
                quality_score=reward,  # Varying quality
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Window should only have last 3 observations
        assert len(bandit.reward_history[model_id]) == 3

        # Check that alpha/beta recalculated from last 3 rewards only
        # Last 3 quality_scores: 0.3, 0.4, 0.5
        # Calculate what the composite rewards would be using bandit's reward weights
        last_3_qualities = [0.3, 0.4, 0.5]
        expected_rewards = []
        for q in last_3_qualities:
            fb = BanditFeedback(
                model_id=model_id, cost=0.001, quality_score=q, latency=1.0
            )
            # Use bandit's reward weights for consistency
            expected_rewards.append(
                fb.calculate_reward(
                    quality_weight=bandit.reward_weights["quality"],
                    cost_weight=bandit.reward_weights["cost"],
                    latency_weight=bandit.reward_weights["latency"],
                )
            )

        expected_alpha = bandit.prior_alpha + sum(expected_rewards)
        expected_beta = bandit.prior_beta + sum(1.0 - r for r in expected_rewards)

        assert abs(bandit.alpha[model_id] - expected_alpha) < 0.001
        assert abs(bandit.beta[model_id] - expected_beta) < 0.001

    @pytest.mark.asyncio
    async def test_adapts_to_distribution_shift(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test that algorithm adapts when model quality changes."""
        bandit = ThompsonSamplingBandit(arms=test_arms, window_size=50, random_seed=42)

        arm1_id = test_arms[0].model_id
        arm2_id = test_arms[1].model_id

        # Phase 1: Arm 1 is better (50 observations)
        for _ in range(50):
            feedback1 = BanditFeedback(
                model_id=arm1_id,
                cost=0.0001,
                quality_score=0.95,  # High quality
                latency=1.0,
            )
            await bandit.update(feedback1, test_features)

            feedback2 = BanditFeedback(
                model_id=arm2_id,
                cost=0.0002,
                quality_score=0.70,  # Low quality
                latency=1.2,
            )
            await bandit.update(feedback2, test_features)

        # At this point, arm1 should have higher alpha (better expected reward)
        alpha1_phase1 = bandit.alpha[arm1_id]
        alpha2_phase1 = bandit.alpha[arm2_id]
        assert alpha1_phase1 > alpha2_phase1

        # Phase 2: Distribution shift - arm 2 becomes better (100 observations)
        # This will push out all Phase 1 observations (window=50)
        for _ in range(100):
            feedback1 = BanditFeedback(
                model_id=arm1_id,
                cost=0.0001,
                quality_score=0.60,  # Quality dropped!
                latency=1.0,
            )
            await bandit.update(feedback1, test_features)

            feedback2 = BanditFeedback(
                model_id=arm2_id,
                cost=0.0002,
                quality_score=0.95,  # Quality improved!
                latency=1.2,
            )
            await bandit.update(feedback2, test_features)

        # Now arm2 should have higher alpha (window only sees recent data)
        alpha1_phase2 = bandit.alpha[arm1_id]
        alpha2_phase2 = bandit.alpha[arm2_id]
        assert alpha2_phase2 > alpha1_phase2  # Arm2 now better


class TestUCB1NonStationary:
    """Tests for UCB1 with sliding window."""

    @pytest.mark.asyncio
    async def test_sliding_window_mean_recalculation(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test that UCB1 recalculates mean from sliding window."""
        bandit = UCB1Bandit(arms=test_arms, window_size=3, random_seed=42)

        arm = test_arms[0]
        model_id = arm.model_id

        # Add observations with varying rewards
        qualities = [0.5, 0.6, 0.7, 0.8, 0.9]
        for q in qualities:
            feedback = BanditFeedback(
                model_id=model_id, cost=0.001, quality_score=q, latency=1.0
            )
            await bandit.update(feedback, test_features)

        # Window should only have last 3 observations
        assert len(bandit.reward_history[model_id]) == 3

        # Calculate expected mean from last 3 qualities (0.7, 0.8, 0.9)
        # Use bandit's reward weights for consistency
        last_3_qualities = [0.7, 0.8, 0.9]
        expected_rewards = []
        for q in last_3_qualities:
            fb = BanditFeedback(
                model_id=model_id, cost=0.001, quality_score=q, latency=1.0
            )
            expected_rewards.append(
                fb.calculate_reward(
                    quality_weight=bandit.reward_weights["quality"],
                    cost_weight=bandit.reward_weights["cost"],
                    latency_weight=bandit.reward_weights["latency"],
                )
            )

        expected_mean = sum(expected_rewards) / len(expected_rewards)
        assert abs(bandit.mean_reward[model_id] - expected_mean) < 0.001


class TestEpsilonGreedyNonStationary:
    """Tests for Epsilon-Greedy with sliding window."""

    @pytest.mark.asyncio
    async def test_window_size_configuration(self, test_arms: list[ModelArm]) -> None:
        """Test different window size configurations."""
        # Test with window
        bandit1 = EpsilonGreedyBandit(arms=test_arms, window_size=100)
        assert bandit1.window_size == 100
        assert all(h.maxlen == 100 for h in bandit1.reward_history.values())

        # Test without window (unlimited)
        bandit2 = EpsilonGreedyBandit(arms=test_arms, window_size=0)
        assert bandit2.window_size == 0
        assert all(h.maxlen is None for h in bandit2.reward_history.values())


class TestLinUCBNonStationary:
    """Tests for LinUCB with sliding window."""

    @pytest.mark.asyncio
    async def test_observation_history_storage(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test that LinUCB stores observations (x, r) in history."""
        bandit = LinUCBBandit(arms=test_arms, window_size=10, random_seed=42)

        arm = test_arms[0]
        model_id = arm.model_id

        # Add observation
        feedback = BanditFeedback(
            model_id=model_id, cost=0.001, quality_score=0.9, latency=1.0
        )
        await bandit.update(feedback, test_features)

        # Check observation was stored
        assert len(bandit.observation_history[model_id]) == 1
        obs_x, obs_r = bandit.observation_history[model_id][0]

        # Feature vector should be (386, 1)
        assert obs_x.shape == (386, 1)

        # Reward should match composite reward using bandit's weights
        expected_reward = feedback.calculate_reward(
            quality_weight=bandit.reward_weights["quality"],
            cost_weight=bandit.reward_weights["cost"],
            latency_weight=bandit.reward_weights["latency"],
        )
        assert abs(obs_r - expected_reward) < 0.001

    @pytest.mark.asyncio
    async def test_matrix_recalculation_from_window(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test that A and b matrices are recalculated from window."""
        bandit = LinUCBBandit(
            arms=test_arms, window_size=2, random_seed=42  # Very small window
        )

        arm = test_arms[0]
        model_id = arm.model_id

        # Add 3 observations (window keeps last 2)
        for i in range(3):
            feedback = BanditFeedback(
                model_id=model_id,
                cost=0.001,
                quality_score=0.8 + i * 0.05,  # 0.8, 0.85, 0.9
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Window should only have last 2 observations
        assert len(bandit.observation_history[model_id]) == 2

        # Manually recalculate expected A and b from last 2 observations
        expected_A = np.identity(bandit.feature_dim)
        expected_b = np.zeros((bandit.feature_dim, 1))

        for obs_x, obs_r in bandit.observation_history[model_id]:
            expected_A += obs_x @ obs_x.T
            expected_b += obs_r * obs_x

        # Check matrices match
        assert np.allclose(bandit.A[model_id], expected_A)
        assert np.allclose(bandit.b[model_id], expected_b)

    @pytest.mark.asyncio
    async def test_adapts_to_feature_shift(self, test_arms: list[ModelArm]) -> None:
        """Test LinUCB adapts when context-reward relationship changes."""
        bandit = LinUCBBandit(arms=test_arms, window_size=50, random_seed=42)

        arm = test_arms[0]
        model_id = arm.model_id

        # Phase 1: Simple queries (low complexity) get good quality
        for _ in range(50):
            features_simple = QueryFeatures(
                embedding=[0.1] * 384,
                token_count=10,  # Simple
                complexity_score=0.2,  # Low complexity
                query_text="Simple query",
            )
            feedback = BanditFeedback(
                model_id=model_id,
                cost=0.001,
                quality_score=0.95,  # High quality for simple
                latency=0.5,
            )
            await bandit.update(feedback, features_simple)

        # Store theta after phase 1
        A_inv_phase1 = np.linalg.inv(bandit.A[model_id])
        theta_phase1 = A_inv_phase1 @ bandit.b[model_id]

        # Phase 2: Relationship changes - complex queries now get good quality
        # (This will push out all Phase 1 observations)
        for _ in range(100):
            features_complex = QueryFeatures(
                embedding=[0.9] * 384,
                token_count=500,  # Complex
                complexity_score=0.9,  # High complexity
                query_text="Complex technical query",
            )
            feedback = BanditFeedback(
                model_id=model_id,
                cost=0.001,
                quality_score=0.95,  # High quality for complex now
                latency=2.0,
            )
            await bandit.update(feedback, features_complex)

        # Store theta after phase 2
        A_inv_phase2 = np.linalg.inv(bandit.A[model_id])
        theta_phase2 = A_inv_phase2 @ bandit.b[model_id]

        # Theta should have changed significantly (different feature-reward relationship)
        theta_diff = np.linalg.norm(theta_phase2 - theta_phase1)
        assert theta_diff > 0.1  # Significant change


class TestNonStationaryReset:
    """Test that reset() clears sliding window history."""

    @pytest.mark.asyncio
    async def test_thompson_reset_clears_history(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test Thompson Sampling reset clears history."""
        bandit = ThompsonSamplingBandit(arms=test_arms, window_size=100)

        # Add some observations
        for _ in range(10):
            feedback = BanditFeedback(
                model_id=test_arms[0].model_id,
                cost=0.001,
                quality_score=0.9,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Reset
        bandit.reset()

        # History should be cleared
        assert all(len(h) == 0 for h in bandit.reward_history.values())
        assert all(
            bandit.alpha[arm.model_id] == bandit.prior_alpha for arm in test_arms
        )

    @pytest.mark.asyncio
    async def test_linucb_reset_clears_observations(
        self, test_arms: list[ModelArm], test_features: QueryFeatures
    ) -> None:
        """Test LinUCB reset clears observation history."""
        bandit = LinUCBBandit(arms=test_arms, window_size=50)

        # Add observations
        for _ in range(10):
            feedback = BanditFeedback(
                model_id=test_arms[0].model_id,
                cost=0.001,
                quality_score=0.9,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Reset
        bandit.reset()

        # Observation history should be cleared
        assert all(len(h) == 0 for h in bandit.observation_history.values())

        # Matrices should be back to identity/zero
        for model_id in bandit.arms:
            assert np.allclose(bandit.A[model_id], np.identity(bandit.feature_dim))
            assert np.allclose(bandit.b[model_id], np.zeros((bandit.feature_dim, 1)))
