"""Unit tests for Epsilon-Greedy bandit algorithm."""

import pytest

from conduit.engines.bandits.epsilon_greedy import EpsilonGreedyBandit
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


class TestEpsilonGreedyBandit:
    """Tests for EpsilonGreedyBandit."""

    def test_initialization_default_epsilon(self, test_arms):
        """Test Epsilon-Greedy bandit initializes with default epsilon."""
        bandit = EpsilonGreedyBandit(test_arms)

        assert bandit.name == "epsilon_greedy"
        assert len(bandit.arms) == 3
        assert bandit.total_queries == 0
        assert bandit.epsilon == 0.1  # Default exploration rate
        assert bandit.decay == 1.0  # No decay by default

        # Check initial state
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            assert bandit.mean_reward[model_id] == 0.0

    def test_initialization_custom_epsilon(self, test_arms):
        """Test Epsilon-Greedy bandit with custom epsilon."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.2, decay=0.995)

        assert bandit.epsilon == 0.2
        assert bandit.decay == 0.995  # Decay enabled

    def test_initialization_epsilon_validation(self, test_arms):
        """Test epsilon must be between 0 and 1."""
        # Valid epsilon values
        EpsilonGreedyBandit(test_arms, epsilon=0.0)
        EpsilonGreedyBandit(test_arms, epsilon=0.5)
        EpsilonGreedyBandit(test_arms, epsilon=1.0)

        # Invalid epsilon should be caught by Pydantic
        with pytest.raises(Exception):
            EpsilonGreedyBandit(test_arms, epsilon=1.5)
        with pytest.raises(Exception):
            EpsilonGreedyBandit(test_arms, epsilon=-0.1)

    @pytest.mark.asyncio
    async def test_select_arm_returns_valid_arm(self, test_arms, test_features):
        """Test arm selection returns valid arm from available arms."""
        bandit = EpsilonGreedyBandit(test_arms)

        arm = await bandit.select_arm(test_features)

        assert arm in test_arms
        assert arm.model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]

    @pytest.mark.asyncio
    async def test_exploration_vs_exploitation(self, test_arms, test_features):
        """Test bandit explores (random) vs exploits (best arm)."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.5, random_seed=42)

        # Initialize with some feedback to establish a best arm
        for arm in test_arms:
            quality = 0.95 if arm.model_id == "gpt-4o" else 0.5
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # With epsilon=0.5, should see mix of exploitation and exploration
        selections = []
        for _ in range(20):
            arm = await bandit.select_arm(test_features)
            selections.append(arm.model_id)

        # Should see gpt-4o selected (exploitation)
        assert "gpt-4o" in selections
        # Should also see other arms selected (exploration)
        # With epsilon=0.5, roughly half should be random exploration
        unique_selections = set(selections)
        assert len(unique_selections) >= 2  # At least 2 different arms selected

    @pytest.mark.asyncio
    async def test_pure_exploitation(self, test_arms, test_features):
        """Test with epsilon=0, only exploits (selects best arm)."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.0)

        # Initialize with feedback making gpt-4o clearly best
        for arm in test_arms:
            quality = 0.95 if arm.model_id == "gpt-4o" else 0.5
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=quality,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # With epsilon=0, should always select best arm
        for _ in range(10):
            arm = await bandit.select_arm(test_features)
            assert arm.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_pure_exploration(self, test_arms, test_features):
        """Test with epsilon=1, only explores (random selection)."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=1.0, random_seed=42)

        # Initialize with feedback
        for arm in test_arms:
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.8,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # With epsilon=1, should always explore (random selection)
        selections = []
        for _ in range(30):
            arm = await bandit.select_arm(test_features)
            selections.append(arm.model_id)

        # Should see variety of arms due to random exploration
        unique_selections = set(selections)
        assert len(unique_selections) >= 2  # At least 2 different arms

    @pytest.mark.asyncio
    async def test_update_with_feedback(self, test_arms, test_features):
        """Test update correctly updates mean values."""
        bandit = EpsilonGreedyBandit(test_arms)

        arm = test_arms[0]  # gpt-4o-mini

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
        bandit = EpsilonGreedyBandit(test_arms)

        arm = test_arms[0]  # gpt-4o-mini

        # First update: quality = 0.8
        feedback1 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
        )
        await bandit.update(feedback1, test_features)
        # Composite: 0.8*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8098...
        expected_reward1 = 0.8 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        assert abs(bandit.mean_reward[arm.model_id] - expected_reward1) < 0.0001

        # Second update: quality = 1.0
        feedback2 = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=1.0,
            latency=1.0,
        )
        await bandit.update(feedback2, test_features)

        # Mean should be average of two composite rewards
        # First: 0.8*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.8098...
        # Second: 1.0*0.7 + (1/(1+0.001))*0.2 + (1/(1+1.0))*0.1 = 0.9298...
        expected_reward2 = 1.0 * 0.7 + (1 / (1 + 0.001)) * 0.2 + (1 / (1 + 1.0)) * 0.1
        expected_avg = (expected_reward1 + expected_reward2) / 2
        assert abs(bandit.mean_reward[arm.model_id] - expected_avg) < 0.0001
        assert bandit.arm_pulls[arm.model_id] == 2

    @pytest.mark.asyncio
    async def test_epsilon_decay(self, test_arms, test_features):
        """Test epsilon decay reduces exploration over time."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.5, decay=0.99)

        initial_epsilon = bandit.epsilon

        # Simulate many queries
        for _ in range(100):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.8,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Epsilon should have decayed
        assert bandit.epsilon < initial_epsilon
        # But never go below minimum (0.01)
        assert bandit.epsilon >= 0.01

    @pytest.mark.asyncio
    async def test_no_decay_constant_epsilon(self, test_arms, test_features):
        """Test without decay, epsilon remains constant."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.3, decay=1.0)

        initial_epsilon = bandit.epsilon

        # Simulate many queries
        for _ in range(100):
            arm = await bandit.select_arm(test_features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.8,
                latency=1.0,
            )
            await bandit.update(feedback, test_features)

        # Epsilon should remain constant
        assert bandit.epsilon == initial_epsilon

    @pytest.mark.asyncio
    async def test_learning_over_time(self, test_arms, test_features):
        """Test bandit learns to prefer better-performing arms."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.1)  # Low exploration

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
    async def test_reset(self, test_arms):
        """Test reset restores initial state."""
        bandit = EpsilonGreedyBandit(test_arms, epsilon=0.3, decay=0.99)

        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )

        # Do some updates (with decay)
        initial_epsilon = bandit.epsilon
        for _ in range(10):
            arm = await bandit.select_arm(features)
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001,
                quality_score=0.9,
                latency=1.0,
            )
            await bandit.update(feedback, features)

        # Epsilon should have decayed
        assert bandit.epsilon < initial_epsilon

        # Reset
        bandit.reset()

        # Check all restored to initial state
        assert bandit.total_queries == 0
        assert bandit.epsilon == 0.3  # Restored to initial epsilon
        for model_id in ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]:
            assert bandit.arm_pulls[model_id] == 0
            assert bandit.mean_reward[model_id] == 0.0

    @pytest.mark.asyncio
    async def test_random_seed_reproducibility(self, test_arms, test_features):
        """Test random seed produces reproducible results."""
        bandit1 = EpsilonGreedyBandit(test_arms, epsilon=0.5, random_seed=42)
        bandit2 = EpsilonGreedyBandit(test_arms, epsilon=0.5, random_seed=42)

        # Same seed should produce same exploration/exploitation decisions
        for _ in range(10):
            arm1 = await bandit1.select_arm(test_features)
            arm2 = await bandit2.select_arm(test_features)
            assert arm1.model_id == arm2.model_id
