"""Tests for conduit.core.reward_calculation module."""

import pytest

from conduit.core.reward_calculation import (
    apply_user_preferences,
    calculate_composite_reward,
    normalize_cost,
    normalize_latency,
    normalize_quality,
    validate_weights,
)


class TestNormalizeQuality:
    """Tests for normalize_quality function."""

    def test_valid_quality_unchanged(self):
        """Quality in valid range should be unchanged."""
        assert normalize_quality(0.95) == 0.95
        assert normalize_quality(0.5) == 0.5
        assert normalize_quality(0.0) == 0.0
        assert normalize_quality(1.0) == 1.0

    def test_quality_above_one_clamped(self):
        """Quality above 1.0 should be clamped to 1.0."""
        assert normalize_quality(1.2) == 1.0
        assert normalize_quality(2.0) == 1.0

    def test_quality_below_zero_clamped(self):
        """Quality below 0.0 should be clamped to 0.0."""
        assert normalize_quality(-0.1) == 0.0
        assert normalize_quality(-1.0) == 0.0


class TestNormalizeCost:
    """Tests for normalize_cost function."""

    def test_zero_cost_gives_one(self):
        """Zero cost should normalize to 1.0 (best score)."""
        assert normalize_cost(0.0) == 1.0

    def test_one_dollar_gives_half(self):
        """One dollar cost should normalize to 0.5."""
        assert normalize_cost(1.0) == 0.5

    def test_ten_dollars_gives_low_score(self):
        """Ten dollar cost should normalize to ~0.09."""
        result = normalize_cost(10.0)
        assert 0.09 < result < 0.10

    def test_higher_cost_lower_score(self):
        """Higher cost should result in lower normalized score."""
        assert normalize_cost(0.01) > normalize_cost(0.1)
        assert normalize_cost(0.1) > normalize_cost(1.0)
        assert normalize_cost(1.0) > normalize_cost(10.0)

    def test_negative_cost_raises(self):
        """Negative cost should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            normalize_cost(-0.01)


class TestNormalizeLatency:
    """Tests for normalize_latency function."""

    def test_zero_latency_gives_one(self):
        """Zero latency should normalize to 1.0 (best score)."""
        assert normalize_latency(0.0) == 1.0

    def test_one_second_gives_half(self):
        """One second latency should normalize to 0.5."""
        assert normalize_latency(1.0) == 0.5

    def test_ten_seconds_gives_low_score(self):
        """Ten second latency should normalize to ~0.09."""
        result = normalize_latency(10.0)
        assert 0.09 < result < 0.10

    def test_higher_latency_lower_score(self):
        """Higher latency should result in lower normalized score."""
        assert normalize_latency(0.1) > normalize_latency(1.0)
        assert normalize_latency(1.0) > normalize_latency(5.0)
        assert normalize_latency(5.0) > normalize_latency(10.0)

    def test_negative_latency_raises(self):
        """Negative latency should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            normalize_latency(-0.5)


class TestValidateWeights:
    """Tests for validate_weights function."""

    def test_valid_weights_no_error(self):
        """Weights summing to 1.0 should not raise."""
        validate_weights(0.7, 0.2, 0.1)  # No exception
        validate_weights(0.33, 0.33, 0.34)  # Close enough
        validate_weights(1.0, 0.0, 0.0)  # Single weight
        validate_weights(0.0, 1.0, 0.0)  # Different single weight

    def test_weights_within_tolerance(self):
        """Weights within tolerance should not raise."""
        validate_weights(0.7, 0.2, 0.105, tolerance=0.01)  # Sum = 1.005

    def test_weights_not_summing_to_one_raises(self):
        """Weights not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights(0.5, 0.5, 0.5)  # Sum = 1.5

        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_weights(0.3, 0.3, 0.3)  # Sum = 0.9


class TestCalculateCompositeReward:
    """Tests for calculate_composite_reward function."""

    def test_default_weights(self):
        """Test with default weights (70% quality, 20% cost, 10% latency)."""
        # High quality, low cost, moderate latency
        reward = calculate_composite_reward(
            quality=0.95,
            cost=0.001,
            latency=1.5,
        )
        # Expected: 0.7 * 0.95 + 0.2 * (1/1.001) + 0.1 * (1/2.5)
        # = 0.665 + 0.1998 + 0.04 = ~0.905
        assert 0.85 < reward < 0.95

    def test_perfect_scores_near_one(self):
        """Perfect quality, zero cost, zero latency should give ~1.0."""
        reward = calculate_composite_reward(
            quality=1.0,
            cost=0.0,
            latency=0.0,
        )
        assert reward == pytest.approx(1.0)

    def test_worst_scores_near_zero(self):
        """Worst quality, high cost, high latency should give low reward."""
        reward = calculate_composite_reward(
            quality=0.0,
            cost=100.0,  # Very high
            latency=100.0,  # Very high
        )
        assert reward < 0.05

    def test_custom_weights_quality_focus(self):
        """Test with 100% quality weight."""
        reward = calculate_composite_reward(
            quality=0.95,
            cost=100.0,  # Very high, but ignored
            latency=100.0,  # Very high, but ignored
            quality_weight=1.0,
            cost_weight=0.0,
            latency_weight=0.0,
        )
        assert reward == 0.95

    def test_custom_weights_cost_focus(self):
        """Test with 100% cost weight."""
        reward = calculate_composite_reward(
            quality=0.0,  # Ignored
            cost=0.0,  # Free
            latency=100.0,  # Ignored
            quality_weight=0.0,
            cost_weight=1.0,
            latency_weight=0.0,
        )
        assert reward == 1.0

    def test_invalid_weights_raises(self):
        """Invalid weights should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            calculate_composite_reward(
                quality=0.95,
                cost=0.001,
                latency=1.5,
                quality_weight=0.5,
                cost_weight=0.5,
                latency_weight=0.5,  # Sum = 1.5
            )

    def test_negative_cost_raises(self):
        """Negative cost should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_composite_reward(quality=0.9, cost=-0.01, latency=1.0)

    def test_negative_latency_raises(self):
        """Negative latency should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_composite_reward(quality=0.9, cost=0.01, latency=-1.0)


class TestApplyUserPreferences:
    """Tests for apply_user_preferences function."""

    def test_no_preferences_unchanged(self):
        """No preferences should return original weights."""
        q, c, l = apply_user_preferences(0.7, 0.2, 0.1)
        assert abs(q - 0.7) < 0.01
        assert abs(c - 0.2) < 0.01
        assert abs(l - 0.1) < 0.01

    def test_default_multipliers_unchanged(self):
        """Default multipliers (1.0) should return original weights."""
        q, c, l = apply_user_preferences(
            0.7, 0.2, 0.1,
            quality_preference=1.0,
            cost_preference=1.0,
            latency_preference=1.0,
        )
        assert abs(q - 0.7) < 0.01
        assert abs(c - 0.2) < 0.01
        assert abs(l - 0.1) < 0.01

    def test_double_cost_preference(self):
        """Doubling cost preference should increase cost weight."""
        q, c, l = apply_user_preferences(
            0.7, 0.2, 0.1,
            cost_preference=2.0,
        )
        # Original: 0.7, 0.4, 0.1 = 1.2 total
        # Normalized: 0.7/1.2, 0.4/1.2, 0.1/1.2
        assert c > 0.3  # Cost weight increased
        assert q + c + l == pytest.approx(1.0)

    def test_zero_quality_preference(self):
        """Zero quality preference should remove quality weight."""
        q, c, l = apply_user_preferences(
            0.7, 0.2, 0.1,
            quality_preference=0.0,
        )
        # Original: 0.0, 0.2, 0.1 = 0.3 total
        # Normalized: 0, 0.2/0.3, 0.1/0.3
        assert q == 0.0
        assert c == pytest.approx(0.2 / 0.3)
        assert l == pytest.approx(0.1 / 0.3)

    def test_all_zero_returns_equal(self):
        """All zero preferences should return equal weights."""
        q, c, l = apply_user_preferences(
            0.7, 0.2, 0.1,
            quality_preference=0.0,
            cost_preference=0.0,
            latency_preference=0.0,
        )
        assert q == pytest.approx(1.0 / 3)
        assert c == pytest.approx(1.0 / 3)
        assert l == pytest.approx(1.0 / 3)

    def test_weights_always_sum_to_one(self):
        """Output weights should always sum to 1.0."""
        test_cases = [
            (1.5, 0.5, 0.5),  # Various multipliers
            (0.5, 1.5, 0.5),
            (0.5, 0.5, 1.5),
            (2.0, 2.0, 2.0),
            (0.1, 0.1, 2.0),
        ]
        for qp, cp, lp in test_cases:
            q, c, l = apply_user_preferences(
                0.7, 0.2, 0.1,
                quality_preference=qp,
                cost_preference=cp,
                latency_preference=lp,
            )
            assert q + c + l == pytest.approx(1.0)

    def test_multipliers_clamped(self):
        """Multipliers outside [0, 2] should be clamped."""
        # Very high multiplier should be treated as 2.0
        q1, c1, l1 = apply_user_preferences(
            0.7, 0.2, 0.1,
            cost_preference=10.0,  # Will be clamped to 2.0
        )
        q2, c2, l2 = apply_user_preferences(
            0.7, 0.2, 0.1,
            cost_preference=2.0,  # Max allowed
        )
        assert c1 == pytest.approx(c2)

        # Negative multiplier should be treated as 0.0
        q3, c3, l3 = apply_user_preferences(
            0.7, 0.2, 0.1,
            quality_preference=-5.0,  # Will be clamped to 0.0
        )
        assert q3 == 0.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_bandit_feedback_compatibility(self):
        """Test that output matches BanditFeedback.calculate_reward()."""
        from conduit.engines.bandits.base import BanditFeedback

        feedback = BanditFeedback(
            model_id="test",
            cost=0.001,
            quality_score=0.95,
            latency=1.5,
        )

        # Direct function call
        direct_reward = calculate_composite_reward(
            quality=0.95,
            cost=0.001,
            latency=1.5,
        )

        # Through BanditFeedback
        feedback_reward = feedback.calculate_reward()

        assert direct_reward == pytest.approx(feedback_reward)

    def test_user_preferences_flow(self):
        """Test applying user preferences then calculating reward."""
        # User wants to prioritize cost
        q_weight, c_weight, l_weight = apply_user_preferences(
            0.7, 0.2, 0.1,
            cost_preference=2.0,
        )

        reward = calculate_composite_reward(
            quality=0.9,
            cost=0.001,  # Very cheap
            latency=2.0,  # Slow
            quality_weight=q_weight,
            cost_weight=c_weight,
            latency_weight=l_weight,
        )

        # With cost priority, cheap model should score well
        assert reward > 0.8
