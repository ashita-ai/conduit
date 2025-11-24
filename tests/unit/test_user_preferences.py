"""Unit tests for user preferences feature."""

import pytest
from pydantic import ValidationError

from conduit.core.models import Query, UserPreferences
from conduit.core.defaults import PREFERENCE_WEIGHTS
from conduit.engines.bandits.base import BanditFeedback, ModelArm


class TestUserPreferences:
    """Tests for UserPreferences model."""

    def test_default_preference(self):
        """Test default preference is 'balanced'."""
        prefs = UserPreferences()
        assert prefs.optimize_for == "balanced"

    def test_valid_preferences(self):
        """Test all valid preference options."""
        for option in ["cost", "speed", "quality", "balanced"]:
            prefs = UserPreferences(optimize_for=option)
            assert prefs.optimize_for == option

    def test_invalid_preference(self):
        """Test invalid preference option is rejected."""
        with pytest.raises(ValidationError):
            UserPreferences(optimize_for="invalid")

    def test_preference_weights_defined(self):
        """Test all preference options have weights defined."""
        for option in ["cost", "speed", "quality", "balanced"]:
            assert option in PREFERENCE_WEIGHTS
            weights = PREFERENCE_WEIGHTS[option]
            assert "quality" in weights
            assert "cost" in weights
            assert "latency" in weights
            # Weights should sum to 1.0
            total = weights["quality"] + weights["cost"] + weights["latency"]
            assert abs(total - 1.0) < 0.01

    def test_cost_preference_weights(self):
        """Test cost preference prioritizes cost."""
        weights = PREFERENCE_WEIGHTS["cost"]
        assert weights["cost"] == 0.5
        assert weights["quality"] == 0.4
        assert weights["latency"] == 0.1

    def test_speed_preference_weights(self):
        """Test speed preference prioritizes latency."""
        weights = PREFERENCE_WEIGHTS["speed"]
        assert weights["latency"] == 0.5
        assert weights["quality"] == 0.4
        assert weights["cost"] == 0.1

    def test_quality_preference_weights(self):
        """Test quality preference prioritizes quality."""
        weights = PREFERENCE_WEIGHTS["quality"]
        assert weights["quality"] == 0.8
        assert weights["cost"] == 0.1
        assert weights["latency"] == 0.1

    def test_balanced_preference_weights(self):
        """Test balanced preference has current defaults."""
        weights = PREFERENCE_WEIGHTS["balanced"]
        assert weights["quality"] == 0.7
        assert weights["cost"] == 0.2
        assert weights["latency"] == 0.1


class TestQueryWithPreferences:
    """Tests for Query model with preferences field."""

    def test_query_default_preferences(self):
        """Test Query has default balanced preferences."""
        query = Query(text="Test query")
        assert query.preferences is not None
        assert query.preferences.optimize_for == "balanced"

    def test_query_with_explicit_preferences(self):
        """Test Query accepts explicit preferences."""
        prefs = UserPreferences(optimize_for="cost")
        query = Query(text="Test query", preferences=prefs)
        assert query.preferences.optimize_for == "cost"

    def test_query_preferences_serialization(self):
        """Test Query with preferences serializes correctly."""
        prefs = UserPreferences(optimize_for="quality")
        query = Query(text="Test query", preferences=prefs)
        
        # Test model_dump
        data = query.model_dump()
        assert "preferences" in data
        assert data["preferences"]["optimize_for"] == "quality"

    def test_query_preferences_from_dict(self):
        """Test Query can be created from dict with preferences."""
        data = {
            "text": "Test query",
            "preferences": {"optimize_for": "speed"}
        }
        query = Query(**data)
        assert query.preferences.optimize_for == "speed"


class TestBanditFeedbackWithPreferences:
    """Tests for BanditFeedback reward calculation with preferences."""

    def test_calculate_reward_with_preferences_cost(self):
        """Test reward calculation with cost optimization."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.0001,  # Very low cost
            quality_score=0.8,  # Good quality
            latency=2.0,  # Moderate latency
        )
        prefs = UserPreferences(optimize_for="cost")
        
        reward = feedback.calculate_reward_with_preferences(prefs)
        
        # With cost optimization (0.5 weight), low cost should give high reward
        assert 0.0 <= reward <= 1.0
        # Cost of 0.0001 normalizes to ~0.9999, so with 0.5 weight and
        # quality 0.8 with 0.4 weight, should be around 0.82
        assert reward > 0.75

    def test_calculate_reward_with_preferences_speed(self):
        """Test reward calculation with speed optimization."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.01,  # Higher cost
            quality_score=0.8,  # Good quality
            latency=0.5,  # Very low latency
        )
        prefs = UserPreferences(optimize_for="speed")
        
        reward = feedback.calculate_reward_with_preferences(prefs)
        
        # With speed optimization (0.5 weight), low latency should give high reward
        assert 0.0 <= reward <= 1.0
        # Latency 0.5 normalizes to ~0.67, with 0.5 weight should dominate
        assert reward > 0.5

    def test_calculate_reward_with_preferences_quality(self):
        """Test reward calculation with quality optimization."""
        feedback = BanditFeedback(
            model_id="gpt-4o",
            cost=0.1,  # High cost
            quality_score=0.95,  # Excellent quality
            latency=3.0,  # Slow
        )
        prefs = UserPreferences(optimize_for="quality")
        
        reward = feedback.calculate_reward_with_preferences(prefs)
        
        # With quality optimization (0.8 weight), high quality dominates
        assert 0.0 <= reward <= 1.0
        # Quality 0.95 with 0.8 weight should be around 0.76+
        assert reward > 0.75

    def test_calculate_reward_with_preferences_balanced(self):
        """Test reward calculation with balanced optimization."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.85,
            latency=1.5,
        )
        prefs = UserPreferences(optimize_for="balanced")
        
        reward = feedback.calculate_reward_with_preferences(prefs)
        
        # Should use default weights (0.7, 0.2, 0.1)
        assert 0.0 <= reward <= 1.0

    def test_different_preferences_different_rewards(self):
        """Test same feedback gives different rewards with different preferences."""
        # Scenario: Cheap but slower model
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.0001,  # Very cheap
            quality_score=0.8,  # Good
            latency=3.0,  # Slow
        )
        
        cost_pref = UserPreferences(optimize_for="cost")
        speed_pref = UserPreferences(optimize_for="speed")
        
        cost_reward = feedback.calculate_reward_with_preferences(cost_pref)
        speed_reward = feedback.calculate_reward_with_preferences(speed_pref)
        
        # Cost preference should rate this higher due to low cost
        # Speed preference should rate this lower due to high latency
        assert cost_reward > speed_reward

    def test_preferences_vs_default_weights(self):
        """Test preferences override default weights."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.85,
            latency=1.0,
        )
        
        # Default weights (balanced)
        default_reward = feedback.calculate_reward()
        
        # With balanced preferences (should be same)
        balanced_pref = UserPreferences(optimize_for="balanced")
        pref_reward = feedback.calculate_reward_with_preferences(balanced_pref)
        
        # Should be identical since balanced uses default weights
        assert abs(default_reward - pref_reward) < 0.001


@pytest.mark.asyncio
class TestBanditAlgorithmWithPreferences:
    """Tests for bandit algorithms using preferences."""

    async def test_ucb1_update_with_preferences(self):
        """Test UCB1 update accepts preferences parameter."""
        from conduit.engines.bandits import UCB1Bandit
        from conduit.core.models import QueryFeatures
        
        arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_input_token=0.15,
                cost_per_output_token=0.60,
                expected_quality=0.8,
            ),
            ModelArm(
                model_id="gpt-4o",
                provider="openai",
                model_name="gpt-4o",
                cost_per_input_token=2.50,
                cost_per_output_token=10.00,
                expected_quality=0.95,
            ),
        ]
        
        bandit = UCB1Bandit(arms)
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
        
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        
        # Should accept preferences parameter
        prefs = UserPreferences(optimize_for="cost")
        await bandit.update(feedback, features, preferences=prefs)
        
        # Verify update worked
        assert bandit.arm_pulls["gpt-4o-mini"] == 1

    async def test_linucb_update_with_preferences(self):
        """Test LinUCB update accepts preferences parameter."""
        from conduit.engines.bandits import LinUCBBandit
        from conduit.core.models import QueryFeatures
        
        arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_input_token=0.15,
                cost_per_output_token=0.60,
                expected_quality=0.8,
            ),
        ]
        
        bandit = LinUCBBandit(arms, feature_dim=387)
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
        
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        
        # Should accept preferences parameter
        prefs = UserPreferences(optimize_for="quality")
        await bandit.update(feedback, features, preferences=prefs)
        
        # Verify update worked
        assert bandit.arm_pulls["gpt-4o-mini"] == 1

    async def test_hybrid_router_update_with_preferences(self):
        """Test HybridRouter update accepts preferences parameter."""
        from conduit.engines.hybrid_router import HybridRouter
        from conduit.core.models import QueryFeatures
        
        models = ["gpt-4o-mini", "gpt-4o"]
        router = HybridRouter(models=models, switch_threshold=10)
        
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
        
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.9,
            latency=1.0,
        )
        
        # Should accept preferences parameter (UCB1 phase)
        prefs = UserPreferences(optimize_for="speed")
        await router.update(feedback, features, preferences=prefs)
        
        # Verify update worked
        assert router.ucb1.arm_pulls["gpt-4o-mini"] == 1
