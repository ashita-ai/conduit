"""Unit tests for feedback blending (70/30 explicit/implicit).

Tests the implementation of the documented 70/30 feedback blending formula
that combines explicit user feedback with implicit behavioral signals.

Reference: AGENTS.md states "Integration: 70% explicit + 30% implicit = final reward"
"""

import pytest

from conduit.core.models import ImplicitFeedback
from conduit.feedback.blending import (
    DEFAULT_EXPLICIT_WEIGHT,
    DEFAULT_IMPLICIT_WEIGHT,
    blend_feedback,
    compute_blended_confidence,
    compute_implicit_score,
    load_blending_weights,
)


class TestLoadBlendingWeights:
    """Tests for load_blending_weights configuration loading."""

    def test_default_weights_are_70_30(self):
        """Test default weights match documented 70/30 split."""
        assert DEFAULT_EXPLICIT_WEIGHT == 0.70
        assert DEFAULT_IMPLICIT_WEIGHT == 0.30

    def test_load_blending_weights_returns_tuple(self):
        """Test load_blending_weights returns tuple of floats."""
        explicit, implicit = load_blending_weights()

        assert isinstance(explicit, float)
        assert isinstance(implicit, float)
        assert 0.0 <= explicit <= 1.0
        assert 0.0 <= implicit <= 1.0

    def test_weights_sum_to_one(self):
        """Test loaded weights sum to 1.0."""
        explicit, implicit = load_blending_weights()
        total = explicit + implicit

        assert abs(total - 1.0) < 0.01  # Within tolerance


class TestComputeImplicitScore:
    """Tests for compute_implicit_score conversion."""

    def test_perfect_response_scores_one(self):
        """Test perfect response (no errors, no retries, fast) scores 1.0."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=False,
            error_type=None,
            latency_seconds=1.0,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=False,
        )

        score = compute_implicit_score(implicit)
        assert score == 1.0

    def test_error_severely_penalizes_score(self):
        """Test error occurrence causes severe score penalty."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=True,
            error_type="api_error",
            latency_seconds=5.0,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=False,
        )

        score = compute_implicit_score(implicit)
        # Error = 0.0 (weight 0.5), no retry = 1.0 (weight 0.3), high latency = 1.0 (weight 0.2)
        # Score = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 1.0 = 0.5
        assert score == 0.5

    def test_retry_penalizes_score(self):
        """Test retry detection penalizes score."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=False,
            error_type=None,
            latency_seconds=1.0,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=True,
            retry_delay_seconds=30.0,
            similarity_score=0.92,
        )

        score = compute_implicit_score(implicit)
        # No error = 1.0 (weight 0.5), retry = 0.0 (weight 0.3), high latency = 1.0 (weight 0.2)
        # Score = 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 1.0 = 0.7
        assert score == 0.7

    def test_slow_latency_penalizes_score(self):
        """Test low latency tolerance penalizes score."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=False,
            error_type=None,
            latency_seconds=45.0,  # Very slow
            latency_accepted=True,
            latency_tolerance="low",
            retry_detected=False,
        )

        score = compute_implicit_score(implicit)
        # No error = 1.0 (weight 0.5), no retry = 1.0 (weight 0.3), low latency = 0.3 (weight 0.2)
        # Score = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.3 = 0.86
        assert abs(score - 0.86) < 0.01

    def test_user_timeout_penalizes_latency_further(self):
        """Test latency_accepted=False adds additional penalty."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=False,
            error_type=None,
            latency_seconds=20.0,
            latency_accepted=False,  # User didn't wait
            latency_tolerance="medium",
            retry_detected=False,
        )

        score = compute_implicit_score(implicit)
        # No error = 1.0 (0.5), no retry = 1.0 (0.3), medium latency = 0.6 * 0.5 = 0.3 (0.2)
        # Score = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.3 = 0.86
        expected = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * (0.6 * 0.5)
        assert abs(score - expected) < 0.01

    def test_worst_case_all_signals_bad(self):
        """Test worst case: error + retry + slow latency."""
        implicit = ImplicitFeedback(
            query_id="q1",
            model_id="gpt-4o-mini",
            timestamp=1000.0,
            error_occurred=True,
            error_type="timeout",
            latency_seconds=60.0,
            latency_accepted=False,
            latency_tolerance="low",
            retry_detected=True,
        )

        score = compute_implicit_score(implicit)
        # Error = 0.0, retry = 0.0, low latency not accepted = 0.3 * 0.5 = 0.15
        expected = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 0.15
        assert score < 0.1  # Very low score

    def test_score_bounded_zero_to_one(self):
        """Test score is always in [0, 1] range."""
        # Best case
        best = ImplicitFeedback(
            query_id="q1", model_id="m1", timestamp=0,
            error_occurred=False, latency_seconds=0.1,
            latency_accepted=True, latency_tolerance="high",
            retry_detected=False
        )
        assert 0.0 <= compute_implicit_score(best) <= 1.0

        # Worst case
        worst = ImplicitFeedback(
            query_id="q1", model_id="m1", timestamp=0,
            error_occurred=True, error_type="fatal",
            latency_seconds=100.0, latency_accepted=False,
            latency_tolerance="low", retry_detected=True
        )
        assert 0.0 <= compute_implicit_score(worst) <= 1.0


class TestBlendFeedback:
    """Tests for blend_feedback combining explicit and implicit scores."""

    def test_both_signals_uses_weighted_blend(self):
        """Test 70/30 blend when both signals available."""
        result = blend_feedback(
            explicit_score=0.9,
            implicit_score=0.6,
            explicit_weight=0.7,
            implicit_weight=0.3,
        )

        expected = 0.7 * 0.9 + 0.3 * 0.6
        assert abs(result - expected) < 0.001

    def test_only_explicit_returns_explicit(self):
        """Test only explicit score returns explicit directly."""
        result = blend_feedback(
            explicit_score=0.85,
            implicit_score=None,
        )

        assert result == 0.85

    def test_only_implicit_returns_implicit(self):
        """Test only implicit score returns implicit directly."""
        result = blend_feedback(
            explicit_score=None,
            implicit_score=0.7,
        )

        assert result == 0.7

    def test_neither_returns_conservative_default(self):
        """Test neither signal returns conservative 0.5."""
        result = blend_feedback(
            explicit_score=None,
            implicit_score=None,
        )

        assert result == 0.5

    def test_custom_weights_applied(self):
        """Test custom weights override defaults."""
        # 50/50 split
        result = blend_feedback(
            explicit_score=1.0,
            implicit_score=0.0,
            explicit_weight=0.5,
            implicit_weight=0.5,
        )

        assert result == 0.5

    def test_edge_case_perfect_explicit_poor_implicit(self):
        """Test perfect explicit with poor implicit still good overall."""
        result = blend_feedback(
            explicit_score=1.0,  # User loves it
            implicit_score=0.3,  # But had some issues
        )

        # 0.7 * 1.0 + 0.3 * 0.3 = 0.79
        expected = 0.7 * 1.0 + 0.3 * 0.3
        assert abs(result - expected) < 0.001
        assert result > 0.7  # Still good overall

    def test_edge_case_poor_explicit_perfect_implicit(self):
        """Test poor explicit with perfect implicit."""
        result = blend_feedback(
            explicit_score=0.2,  # User unhappy
            implicit_score=1.0,  # But no technical issues
        )

        # 0.7 * 0.2 + 0.3 * 1.0 = 0.44
        expected = 0.7 * 0.2 + 0.3 * 1.0
        assert abs(result - expected) < 0.001
        assert result < 0.5  # User unhappiness dominates

    def test_result_bounded_zero_to_one(self):
        """Test result is always in [0, 1] range."""
        cases = [
            (1.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ]

        for explicit, implicit in cases:
            result = blend_feedback(explicit, implicit)
            assert 0.0 <= result <= 1.0


class TestComputeBlendedConfidence:
    """Tests for compute_blended_confidence."""

    def test_explicit_confidence_preserved(self):
        """Test explicit confidence is preserved when provided."""
        confidence = compute_blended_confidence(
            explicit_confidence=0.95,
            has_implicit=True,
            implicit_score=0.8,
        )

        assert confidence == 0.95

    def test_only_implicit_returns_soft_confidence(self):
        """Test only implicit signals get soft confidence (0.5)."""
        confidence = compute_blended_confidence(
            explicit_confidence=None,
            has_implicit=True,
            implicit_score=0.7,
        )

        assert confidence == 0.5

    def test_neither_returns_conservative_confidence(self):
        """Test no signals returns very conservative confidence."""
        confidence = compute_blended_confidence(
            explicit_confidence=None,
            has_implicit=False,
        )

        assert confidence == 0.3

    def test_strong_negative_implicit_boosts_confidence(self):
        """Test strong negative implicit signal boosts confidence."""
        confidence = compute_blended_confidence(
            explicit_confidence=0.6,
            has_implicit=True,
            implicit_score=0.1,  # Strong negative signal
        )

        # Base 0.6 + 0.2 boost = 0.8
        assert confidence == 0.8

    def test_weak_implicit_no_boost(self):
        """Test weak implicit signal doesn't boost confidence."""
        confidence = compute_blended_confidence(
            explicit_confidence=0.7,
            has_implicit=True,
            implicit_score=0.6,  # Not strongly negative
        )

        assert confidence == 0.7  # No boost

    def test_confidence_capped_at_one(self):
        """Test confidence is capped at 1.0."""
        confidence = compute_blended_confidence(
            explicit_confidence=0.95,  # Already high
            has_implicit=True,
            implicit_score=0.1,  # Would boost by 0.2
        )

        assert confidence == 1.0  # Capped


class TestIntegrationScenarios:
    """Integration tests for realistic feedback scenarios."""

    def test_happy_path_user_thumbs_up_fast_response(self):
        """Test happy path: user gives thumbs up, response was fast."""
        # Implicit: fast, no errors, no retries
        implicit = ImplicitFeedback(
            query_id="q1", model_id="gpt-4o-mini", timestamp=1000.0,
            error_occurred=False, latency_seconds=0.5,
            latency_accepted=True, latency_tolerance="high",
            retry_detected=False
        )

        implicit_score = compute_implicit_score(implicit)
        assert implicit_score == 1.0

        # Explicit: thumbs up = 1.0
        final_score = blend_feedback(
            explicit_score=1.0,
            implicit_score=implicit_score,
        )

        assert final_score == 1.0  # Perfect score

    def test_user_happy_but_slow_response(self):
        """Test user happy despite slow response."""
        implicit = ImplicitFeedback(
            query_id="q1", model_id="gpt-4o-mini", timestamp=1000.0,
            error_occurred=False, latency_seconds=25.0,  # Medium slow
            latency_accepted=True, latency_tolerance="medium",
            retry_detected=False
        )

        implicit_score = compute_implicit_score(implicit)
        # No error = 1.0 (0.5), no retry = 1.0 (0.3), medium = 0.6 (0.2)
        expected_implicit = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.6
        assert abs(implicit_score - expected_implicit) < 0.01

        # User still gave thumbs up
        final_score = blend_feedback(
            explicit_score=1.0,
            implicit_score=implicit_score,
        )

        # 0.7 * 1.0 + 0.3 * 0.92 = 0.976
        assert final_score > 0.9  # Still very good

    def test_user_unhappy_error_occurred(self):
        """Test user unhappy and error occurred."""
        implicit = ImplicitFeedback(
            query_id="q1", model_id="gpt-4o-mini", timestamp=1000.0,
            error_occurred=True, error_type="model_refusal",
            latency_seconds=5.0, latency_accepted=True, latency_tolerance="high",
            retry_detected=True  # User retried
        )

        implicit_score = compute_implicit_score(implicit)
        # Error = 0.0 (0.5), retry = 0.0 (0.3), high latency = 1.0 (0.2)
        expected_implicit = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0
        assert abs(implicit_score - expected_implicit) < 0.01

        # User gave thumbs down
        final_score = blend_feedback(
            explicit_score=0.0,
            implicit_score=implicit_score,
        )

        # Both signals bad -> very low score
        assert final_score < 0.1

    def test_no_explicit_feedback_implicit_only(self):
        """Test learning continues with only implicit feedback."""
        # User didn't provide feedback, but no errors occurred
        implicit = ImplicitFeedback(
            query_id="q1", model_id="gpt-4o-mini", timestamp=1000.0,
            error_occurred=False, latency_seconds=2.0,
            latency_accepted=True, latency_tolerance="high",
            retry_detected=False
        )

        implicit_score = compute_implicit_score(implicit)
        assert implicit_score == 1.0

        # No explicit feedback
        final_score = blend_feedback(
            explicit_score=None,
            implicit_score=implicit_score,
        )

        # Falls back to implicit score
        assert final_score == 1.0

        confidence = compute_blended_confidence(
            explicit_confidence=None,
            has_implicit=True,
            implicit_score=implicit_score,
        )

        # Soft confidence for implicit-only
        assert confidence == 0.5
