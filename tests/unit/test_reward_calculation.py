"""Tests for multi-objective reward calculation (Phase 3)."""

import pytest

from conduit.engines.bandits.base import BanditFeedback


class TestRewardCalculation:
    """Tests for BanditFeedback.calculate_reward() method."""

    def test_reward_with_default_weights(self) -> None:
        """Test reward calculation with default weights (70/20/10)."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.0001,  # Very cheap
            quality_score=0.95,  # High quality
            latency=1.0,  # Reasonable speed
        )

        # Default weights: quality=0.70, cost=0.20, latency=0.10
        reward = feedback.calculate_reward()

        # Components:
        # quality_norm = 0.95 (as-is)
        # cost_norm = 1 / (1 + 0.0001) ≈ 0.9999
        # latency_norm = 1 / (1 + 1.0) = 0.5
        # reward = 0.70 * 0.95 + 0.20 * 0.9999 + 0.10 * 0.5
        #       = 0.665 + 0.19998 + 0.05
        #       ≈ 0.915

        assert 0.90 < reward < 0.92  # Quality dominates (70%)

    def test_reward_expensive_query(self) -> None:
        """Test reward with high cost (should reduce reward)."""
        feedback = BanditFeedback(
            model_id="gpt-4o",
            cost=1.0,  # Expensive
            quality_score=0.95,  # High quality
            latency=1.0,
        )

        reward = feedback.calculate_reward()

        # cost_norm = 1 / (1 + 1.0) = 0.5 (cost hurts reward)
        # reward ≈ 0.70 * 0.95 + 0.20 * 0.5 + 0.10 * 0.5
        #       ≈ 0.665 + 0.10 + 0.05 = 0.815

        assert 0.80 < reward < 0.83

    def test_reward_slow_query(self) -> None:
        """Test reward with high latency (should reduce reward)."""
        feedback = BanditFeedback(
            model_id="claude-opus",
            cost=0.01,
            quality_score=0.98,  # Very high quality
            latency=10.0,  # Slow
        )

        reward = feedback.calculate_reward()

        # latency_norm = 1 / (1 + 10.0) ≈ 0.09 (latency hurts reward)
        # reward ≈ 0.70 * 0.98 + 0.20 * 0.99 + 0.10 * 0.09
        #       ≈ 0.686 + 0.198 + 0.009 ≈ 0.893

        assert 0.88 < reward < 0.90

    def test_reward_poor_quality(self) -> None:
        """Test reward with poor quality (should significantly reduce reward)."""
        feedback = BanditFeedback(
            model_id="weak-model",
            cost=0.0001,  # Very cheap
            quality_score=0.3,  # Poor quality
            latency=0.5,  # Fast
        )

        reward = feedback.calculate_reward()

        # quality_norm = 0.3 (poor quality dominates due to 70% weight)
        # reward ≈ 0.70 * 0.3 + 0.20 * 1.0 + 0.10 * 0.67
        #       ≈ 0.21 + 0.20 + 0.067 ≈ 0.477

        assert 0.45 < reward < 0.50

    def test_reward_perfect_scenario(self) -> None:
        """Test reward with perfect quality, zero cost, zero latency."""
        feedback = BanditFeedback(
            model_id="perfect-model",
            cost=0.0,
            quality_score=1.0,
            latency=0.0,
        )

        reward = feedback.calculate_reward()

        # All components = 1.0
        # reward = 0.70 * 1.0 + 0.20 * 1.0 + 0.10 * 1.0 = 1.0
        assert reward == pytest.approx(1.0)

    def test_reward_worst_scenario(self) -> None:
        """Test reward with worst quality, high cost, high latency."""
        feedback = BanditFeedback(
            model_id="terrible-model",
            cost=100.0,  # Very expensive
            quality_score=0.0,  # Worst quality
            latency=100.0,  # Very slow
        )

        reward = feedback.calculate_reward()

        # quality_norm = 0.0
        # cost_norm = 1 / (1 + 100) ≈ 0.0099
        # latency_norm = 1 / (1 + 100) ≈ 0.0099
        # reward ≈ 0.70 * 0.0 + 0.20 * 0.0099 + 0.10 * 0.0099 ≈ 0.003

        assert 0.0 < reward < 0.01  # Near-zero but positive

    def test_reward_custom_weights(self) -> None:
        """Test reward with custom weight configuration."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.9,
            latency=1.5,
        )

        # Custom weights: cost-optimized (quality=0.5, cost=0.4, latency=0.1)
        reward = feedback.calculate_reward(
            quality_weight=0.5,
            cost_weight=0.4,
            latency_weight=0.1,
        )

        # cost_norm = 1 / (1 + 0.001) ≈ 0.999
        # latency_norm = 1 / (1 + 1.5) = 0.4
        # reward = 0.5 * 0.9 + 0.4 * 0.999 + 0.1 * 0.4
        #       = 0.45 + 0.3996 + 0.04 ≈ 0.890

        assert 0.88 < reward < 0.90

    def test_reward_weights_sum_validation(self) -> None:
        """Test that weights must sum to 1.0."""
        feedback = BanditFeedback(
            model_id="test",
            cost=0.01,
            quality_score=0.9,
            latency=1.0,
        )

        # Weights don't sum to 1.0 (sum = 0.8)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            feedback.calculate_reward(
                quality_weight=0.5,
                cost_weight=0.2,
                latency_weight=0.1,  # sum = 0.8
            )

    def test_reward_asymptotic_cost_normalization(self) -> None:
        """Test asymptotic normalization behavior for cost."""
        # Test cost normalization: 1 / (1 + cost)
        # cost=0 → 1.0, cost=1 → 0.5, cost=10 → 0.09

        test_cases = [
            (0.0, 1.0),  # Zero cost = perfect (1.0)
            (1.0, 0.5),  # Cost=1 → normalized to 0.5
            (10.0, 0.09),  # High cost → low normalized value
        ]

        for cost, expected_norm in test_cases:
            feedback = BanditFeedback(
                model_id="test",
                cost=cost,
                quality_score=0.0,  # Zero quality to isolate cost
                latency=0.0,  # Zero latency to isolate cost
            )

            reward = feedback.calculate_reward(
                quality_weight=0.0,
                cost_weight=1.0,
                latency_weight=0.0,
            )

            assert reward == pytest.approx(expected_norm, abs=0.01)

    def test_reward_asymptotic_latency_normalization(self) -> None:
        """Test asymptotic normalization behavior for latency."""
        # Test latency normalization: 1 / (1 + latency)

        test_cases = [
            (0.0, 1.0),  # Zero latency = perfect
            (1.0, 0.5),  # Latency=1 → 0.5
            (10.0, 0.09),  # High latency → low value
        ]

        for latency, expected_norm in test_cases:
            feedback = BanditFeedback(
                model_id="test",
                cost=0.0,  # Zero cost to isolate latency
                quality_score=0.0,  # Zero quality to isolate latency
                latency=latency,
            )

            reward = feedback.calculate_reward(
                quality_weight=0.0,
                cost_weight=0.0,
                latency_weight=1.0,
            )

            assert reward == pytest.approx(expected_norm, abs=0.01)

    def test_reward_equal_weights(self) -> None:
        """Test reward with equal weights (33/33/34)."""
        feedback = BanditFeedback(
            model_id="test",
            cost=1.0,  # cost_norm = 0.5
            quality_score=0.6,  # quality_norm = 0.6
            latency=2.0,  # latency_norm = 0.33
        )

        # Equal weights
        reward = feedback.calculate_reward(
            quality_weight=0.33,
            cost_weight=0.33,
            latency_weight=0.34,
        )

        # reward = 0.33 * 0.6 + 0.33 * 0.5 + 0.34 * 0.33
        #       = 0.198 + 0.165 + 0.112 ≈ 0.475

        assert 0.47 < reward < 0.48

    def test_reward_quality_only(self) -> None:
        """Test reward with 100% quality weight (ignores cost and latency)."""
        feedback = BanditFeedback(
            model_id="test",
            cost=100.0,  # Expensive (should be ignored)
            quality_score=0.85,
            latency=50.0,  # Slow (should be ignored)
        )

        reward = feedback.calculate_reward(
            quality_weight=1.0,
            cost_weight=0.0,
            latency_weight=0.0,
        )

        # Only quality matters
        assert reward == 0.85

    def test_reward_cost_only(self) -> None:
        """Test reward with 100% cost weight (ignores quality and latency)."""
        feedback = BanditFeedback(
            model_id="test",
            cost=0.5,  # cost_norm = 1/(1+0.5) = 0.67
            quality_score=0.1,  # Poor quality (should be ignored)
            latency=100.0,  # Slow (should be ignored)
        )

        reward = feedback.calculate_reward(
            quality_weight=0.0,
            cost_weight=1.0,
            latency_weight=0.0,
        )

        # Only cost matters: 1 / (1 + 0.5) = 0.666...
        assert reward == pytest.approx(0.6667, abs=0.001)

    def test_reward_range_always_0_to_1(self) -> None:
        """Test that reward is always in [0, 1] range."""
        # Test various extreme scenarios
        test_cases = [
            (0.0, 0.0, 0.0),  # All zeros
            (1.0, 1.0, 1.0),  # All ones
            (100.0, 0.0, 100.0),  # High cost and latency
            (0.0, 1.0, 0.0),  # Perfect quality only
        ]

        for cost, quality, latency in test_cases:
            feedback = BanditFeedback(
                model_id="test",
                cost=cost,
                quality_score=quality,
                latency=latency,
            )

            reward = feedback.calculate_reward()

            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0, 1] range"
