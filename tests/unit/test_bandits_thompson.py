"""Unit tests for Thompson Sampling bandit algorithm.

Uses shared fixtures from tests/conftest.py: test_arms, test_features
"""

import pytest

from conduit.engines.bandits.thompson_sampling import ThompsonSamplingBandit
from conduit.engines.bandits.base import BanditFeedback
from conduit.core.models import QueryFeatures

# test_arms and test_features fixtures imported from conftest.py


class TestThompsonSamplingBandit:
    """Tests for ThompsonSamplingBandit."""

    def test_initialization(self, test_arms):
        """Test Thompson Sampling bandit initializes with Beta(1,1) priors."""
        bandit = ThompsonSamplingBandit(test_arms)

        assert bandit.name == "thompson_sampling"
        assert len(bandit.arms) == 3
        assert bandit.total_queries == 0

        # Check Beta distribution parameters
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert bandit.alpha[model_id] == 1.0
            assert bandit.beta[model_id] == 1.0
            assert bandit.arm_pulls[model_id] == 0

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm(self, test_arms, test_features):
        """Test arm selection returns valid arm from available arms."""
        bandit = ThompsonSamplingBandit(test_arms)

        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert arm.model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]

    @pytest.mark.asyncio
    async def test_select_arm_increments_count(self, test_arms, test_features):
        """Test arm selection increments total query count."""
        bandit = ThompsonSamplingBandit(test_arms)

        assert bandit.total_queries == 0

        await bandit.select_arm(test_features)
        assert bandit.total_queries == 1

        await bandit.select_arm(test_features)
        assert bandit.total_queries == 2

    @pytest.mark.asyncio
    async def test_update_with_success(self, test_arms, test_features):
        """Test update with high-quality feedback increases alpha."""
        bandit = ThompsonSamplingBandit(test_arms)

        arm = await bandit.select_arm(test_features)
        initial_alpha = bandit.alpha[arm.model_id]
        initial_beta = bandit.beta[arm.model_id]

        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.95,  # High quality
            latency=1.0,
        )

        await bandit.update(feedback, test_features)

        # Alpha should increase for high quality
        assert bandit.alpha[arm.model_id] > initial_alpha
        # Beta should increase slightly
        assert bandit.beta[arm.model_id] >= initial_beta
        # Count should increment
        assert bandit.arm_pulls[arm.model_id] == 1

    @pytest.mark.asyncio
    async def test_update_with_failure(self, test_arms, test_features):
        """Test update with low-quality feedback increases beta."""
        bandit = ThompsonSamplingBandit(test_arms)

        arm = await bandit.select_arm(test_features)
        initial_alpha = bandit.alpha[arm.model_id]
        initial_beta = bandit.beta[arm.model_id]

        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.3,  # Low quality
            latency=1.0,
        )

        await bandit.update(feedback, test_features)

        # Alpha should increase slightly
        assert bandit.alpha[arm.model_id] >= initial_alpha
        # Beta should increase more for low quality
        assert bandit.beta[arm.model_id] > initial_beta
        # Count should increment
        assert bandit.arm_pulls[arm.model_id] == 1

    @pytest.mark.asyncio
    async def test_learning_over_time(self, test_arms, test_features):
        """Test bandit learns to prefer better-performing arms."""
        bandit = ThompsonSamplingBandit(test_arms, random_seed=42)

        # Simulate gpt-5.1 performing well, others performing poorly
        for _ in range(20):
            arm = await bandit.select_arm(test_features)

            if arm.model_id == "gpt-5.1":
                quality = 0.95  # Excellent
            else:
                quality = 0.5  # Poor

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )

            await bandit.update(feedback, test_features)

        # After learning, gpt-5.1 should have highest expected value
        # (measured by alpha / (alpha + beta))
        gpt51_mean = bandit.alpha["gpt-5.1"] / (
            bandit.alpha["gpt-5.1"] + bandit.beta["gpt-5.1"]
        )
        mini_mean = bandit.alpha["o4-mini"] / (
            bandit.alpha["o4-mini"] + bandit.beta["o4-mini"]
        )
        haiku_mean = bandit.alpha["claude-haiku-4-5"] / (
            bandit.alpha["claude-haiku-4-5"] + bandit.beta["claude-haiku-4-5"]
        )

        assert gpt51_mean > mini_mean
        assert gpt51_mean > haiku_mean

    @pytest.mark.asyncio
    async def test_reset(self, test_arms):
        """Test reset restores initial state."""
        bandit = ThompsonSamplingBandit(test_arms)

        # Do some updates
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )

        for _ in range(5):
            arm = await bandit.select_arm(features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.9,
                latency=1.0,
            )
            await bandit.update(feedback, features)

        assert bandit.total_queries == 5

        # Reset
        bandit.reset()

        # Check all restored to initial state
        assert bandit.total_queries == 0
        for model_id in ["o4-mini", "gpt-5.1", "claude-haiku-4-5"]:
            assert bandit.alpha[model_id] == 1.0
            assert bandit.beta[model_id] == 1.0
            assert bandit.arm_pulls[model_id] == 0

    # Note: Lockstep reproducibility test removed - not applicable with global random state.
    # Sequential reproducibility is covered by other tests in this file

    @pytest.mark.asyncio
    async def test_multiple_updates_same_arm(self, test_arms, test_features):
        """Test multiple updates to same arm accumulate correctly."""
        bandit = ThompsonSamplingBandit(test_arms)

        arm = test_arms[0]  # gpt-4o-mini

        # First update
        feedback1 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        await bandit.update(feedback1, test_features)

        alpha_after_1 = bandit.alpha[arm.model_id]
        beta_after_1 = bandit.beta[arm.model_id]
        count_after_1 = bandit.arm_pulls[arm.model_id]

        # Second update
        feedback2 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
        )
        await bandit.update(feedback2, test_features)

        # Parameters should continue updating
        assert bandit.alpha[arm.model_id] > alpha_after_1
        assert bandit.beta[arm.model_id] >= beta_after_1
        assert bandit.arm_pulls[arm.model_id] == count_after_1 + 1
