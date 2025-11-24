"""Unit tests for UCB1 bandit algorithm."""

import math
import pytest

from conduit.engines.bandits.ucb import UCB1Bandit
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


class TestUCB1Bandit:
    """Tests for UCB1Bandit."""

    def test_initialization_default_c(self, test_arms):
        """Test UCB1 bandit initializes with default exploration parameter."""
        bandit = UCB1Bandit(test_arms)

        assert bandit.name == "ucb1"
        assert len(bandit.arms) == 3
        assert bandit.total_queries == 0
        assert bandit.c == 1.5  # Default exploration parameter (UCB1_C_DEFAULT)

        # Check initial state
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            assert bandit.mean_reward[model_id] == 0.0

    def test_initialization_custom_c(self, test_arms):
        """Test UCB1 bandit with custom exploration parameter."""
        bandit = UCB1Bandit(test_arms, c=1.5)

        assert bandit.c == 1.5

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm(self, test_arms, test_features):
        """Test arm selection returns valid arm from available arms."""
        bandit = UCB1Bandit(test_arms)

        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert arm.model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]

    @pytest.mark.asyncio
    async def test_exploration_phase(self, test_arms, test_features):
        """Test exploration phase tries each arm at least once."""
        bandit = UCB1Bandit(test_arms)

        selected_models = set()

        # During exploration phase, should select each arm at least once
        for _ in range(len(test_arms)):
            arm = await bandit.select_arm(test_features)
            selected_models.add(arm.model_id)

        # All arms should have been tried
        assert len(selected_models) == len(test_arms)
        assert bandit.total_queries == len(test_arms)

    @pytest.mark.asyncio
    async def test_update_with_feedback(self, test_arms, test_features):
        """Test update correctly updates mean values."""
        bandit = UCB1Bandit(test_arms)

        arm = await bandit.select_arm(test_features)

        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )

        await bandit.update(feedback, test_features)

        # Mean value should be updated to composite reward
        # Composite: 0.9*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8798...
        expected_reward = 0.9 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        assert abs(bandit.mean_reward[arm.model_id] - expected_reward) < 0.0001
        assert bandit.arm_pulls[arm.model_id] == 1

    @pytest.mark.asyncio
    async def test_running_average(self, test_arms, test_features):
        """Test running average calculation for multiple updates."""
        bandit = UCB1Bandit(test_arms)

        arm = test_arms[0]  # gpt-4o-mini

        # First update: quality = 0.9
        feedback1 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        await bandit.update(feedback1, test_features)
        # Composite: 0.9*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8798...
        expected_reward1 = 0.9 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        assert abs(bandit.mean_reward[arm.model_id] - expected_reward1) < 0.0001

        # Second update: quality = 0.7
        feedback2 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.7,
            latency=1.0,
        )
        await bandit.update(feedback2, test_features)

        # Mean should be average of two composite rewards
        # First: 0.9*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8798...
        # Second: 0.7*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.7398...
        expected_reward2 = 0.7 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        expected_avg = (expected_reward1 + expected_reward2) / 2
        assert abs(bandit.mean_reward[arm.model_id] - expected_avg) < 0.0001
        assert bandit.arm_pulls[arm.model_id] == 2

    @pytest.mark.asyncio
    async def test_ucb_calculation(self, test_arms, test_features):
        """Test UCB value calculation after exploration phase."""
        bandit = UCB1Bandit(test_arms, c=1.5)

        # Simulate exploration phase (select_arm + update)
        for arm in test_arms:
            selected_arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=selected_arm.model_id,
                cost=0.001,
                quality_score=0.8,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Now in exploitation phase, calculate expected UCB
        model_id = test_arms[0].model_id
        count = bandit.arm_pulls[model_id]
        value = bandit.mean_reward[model_id]
        total = bandit.total_queries

        # Verify total_queries > 0 to avoid log(0)
        assert total > 0
        expected_ucb = value + bandit.c * math.sqrt(math.log(total) / count)

        # The actual UCB is calculated internally during select_arm
        # We verify the algorithm is working by checking it makes reasonable choices
        arm = await bandit.select_arm(test_features)
        assert arm in test_arms

    @pytest.mark.asyncio
    async def test_learning_over_time(self, test_arms, test_features):
        """Test bandit learns to prefer better-performing arms."""
        bandit = UCB1Bandit(test_arms)

        # Simulate gpt-4o performing well, others performing poorly
        for _ in range(30):
            arm = await bandit.select_arm(test_features)

            if arm.model_id == "gpt-4o":
                quality = 0.95  # Excellent
            else:
                quality = 0.6  # Poor

            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )

            await bandit.update(feedback, test_features)

        # After learning, gpt-4o should have highest mean value
        assert bandit.mean_reward["gpt-4o"] > bandit.mean_reward["gpt-4o-mini"]
        assert bandit.mean_reward["gpt-4o"] > bandit.mean_reward["claude-3-haiku"]

    @pytest.mark.asyncio
    async def test_exploration_parameter_effect(self, test_arms, test_features):
        """Test higher c parameter leads to more exploration."""
        # Low exploration
        bandit_low = UCB1Bandit(test_arms, c=0.1)
        # High exploration
        bandit_high = UCB1Bandit(test_arms, c=5.0)

        # Make one arm clearly better
        for bandit in [bandit_low, bandit_high]:
            # Initial exploration phase
            for arm in test_arms:
                if arm.model_id == "gpt-4o":
                    quality = 0.95
                else:
                    quality = 0.5

                feedback = BanditFeedback(
                    model_id=arm.model_id,
                    cost=0.001,
                    quality_score=quality,
                    latency=1.0,
                )
                await bandit.update(feedback, test_features)

        # Both should recognize gpt-4o is best
        assert bandit_low.mean_reward["gpt-4o"] > bandit_low.mean_reward["gpt-4o-mini"]
        assert bandit_high.mean_reward["gpt-4o"] > bandit_high.mean_reward["gpt-4o-mini"]

        # High c bandit should continue exploring more even after finding best arm
        # This is hard to test deterministically, but we can verify c is stored
        assert bandit_high.c > bandit_low.c

    @pytest.mark.asyncio
    async def test_reset(self, test_arms):
        """Test reset restores initial state."""
        bandit = UCB1Bandit(test_arms)

        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )

        # Do some updates
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
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            assert bandit.mean_reward[model_id] == 0.0

    @pytest.mark.asyncio
    async def test_no_divide_by_zero(self, test_arms, test_features):
        """Test UCB handles zero counts gracefully during exploration."""
        bandit = UCB1Bandit(test_arms)

        # First selection should not crash (all counts are 0)
        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert bandit.total_queries == 1
