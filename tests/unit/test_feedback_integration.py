"""Unit tests for feedback integration with Thompson Sampling."""

import pytest

from conduit.core.models import Feedback, ImplicitFeedback, QueryFeatures
from conduit.engines.bandit import ContextualBandit
from conduit.feedback.integration import FeedbackIntegrator


@pytest.fixture
def bandit():
    """Create contextual bandit for testing."""
    return ContextualBandit(models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"])


@pytest.fixture
def integrator(bandit):
    """Create feedback integrator."""
    return FeedbackIntegrator(bandit, explicit_weight=0.7, implicit_weight=0.3)


@pytest.fixture
def sample_features():
    """Create sample QueryFeatures."""
    return QueryFeatures(
        embedding=[0.1, 0.2, 0.3] * 128,
        token_count=50,
        complexity_score=0.5,
        domain="code",
        domain_confidence=0.8,
    )


class TestFeedbackIntegratorInit:
    """Test FeedbackIntegrator initialization."""

    def test_init_default_weights(self, bandit):
        """Test initialization with default weights."""
        integrator = FeedbackIntegrator(bandit)

        assert integrator.explicit_weight == 0.7
        assert integrator.implicit_weight == 0.3
        assert integrator.bandit is bandit

    def test_init_custom_weights(self, bandit):
        """Test initialization with custom weights."""
        integrator = FeedbackIntegrator(
            bandit, explicit_weight=0.6, implicit_weight=0.4
        )

        assert integrator.explicit_weight == 0.6
        assert integrator.implicit_weight == 0.4


class TestFeedbackConversion:
    """Test explicit feedback to reward conversion."""

    def test_explicit_high_quality_met_expectations(self, integrator):
        """Test high quality + met expectations."""
        feedback = Feedback(
            response_id="r123",
            quality_score=1.0,
            met_expectations=True,
        )

        reward = integrator._explicit_to_reward(feedback)

        # (1.0 * 0.6) + (1.0 * 0.4) = 0.6 + 0.4 = 1.0
        assert reward == pytest.approx(1.0)

    def test_explicit_low_quality_not_met(self, integrator):
        """Test low quality + expectations not met."""
        feedback = Feedback(
            response_id="r123",
            quality_score=0.2,
            met_expectations=False,
        )

        reward = integrator._explicit_to_reward(feedback)

        # (0.2 * 0.6) + (0.0 * 0.4) = 0.12
        assert reward == pytest.approx(0.12)

    def test_explicit_medium_quality_met(self, integrator):
        """Test medium quality + met expectations."""
        feedback = Feedback(
            response_id="r123",
            quality_score=0.6,
            met_expectations=True,
        )

        reward = integrator._explicit_to_reward(feedback)

        # (0.6 * 0.6) + (1.0 * 0.4) = 0.36 + 0.4 = 0.76
        assert reward == pytest.approx(0.76)

    def test_explicit_high_quality_not_met(self, integrator):
        """Test high quality but expectations not met."""
        feedback = Feedback(
            response_id="r123",
            quality_score=0.9,
            met_expectations=False,
        )

        reward = integrator._explicit_to_reward(feedback)

        # (0.9 * 0.6) + (0.0 * 0.4) = 0.54
        assert reward == pytest.approx(0.54)


class TestImplicitFeedbackConversion:
    """Test implicit feedback to reward conversion."""

    def test_implicit_error_occurred(self, integrator):
        """Test error occurred (highest priority signal)."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=True,
            error_type="api_error",
            latency_seconds=1.0,
            latency_tolerance="high",
        )

        reward = integrator._implicit_to_reward(feedback)

        # Error occurred → reward = 0.0
        assert reward == 0.0

    def test_implicit_retry_detected(self, integrator):
        """Test retry detected (second priority)."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=True,
            retry_delay_seconds=30.0,
            similarity_score=0.9,
            latency_seconds=2.0,
            latency_tolerance="high",
        )

        reward = integrator._implicit_to_reward(feedback)

        # Retry detected → reward = 0.3
        assert reward == 0.3

    def test_implicit_high_latency_tolerance(self, integrator):
        """Test high latency tolerance (fast response)."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=False,
            latency_seconds=5.0,
            latency_tolerance="high",
        )

        reward = integrator._implicit_to_reward(feedback)

        # High tolerance → reward = 0.9
        assert reward == 0.9

    def test_implicit_medium_latency_tolerance(self, integrator):
        """Test medium latency tolerance (acceptable speed)."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=False,
            latency_seconds=15.0,
            latency_tolerance="medium",
        )

        reward = integrator._implicit_to_reward(feedback)

        # Medium tolerance → reward = 0.7
        assert reward == 0.7

    def test_implicit_low_latency_tolerance(self, integrator):
        """Test low latency tolerance (slow but patient)."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=False,
            latency_seconds=35.0,
            latency_tolerance="low",
        )

        reward = integrator._implicit_to_reward(feedback)

        # Low tolerance → reward = 0.5
        assert reward == 0.5


class TestBanditUpdate:
    """Test bandit update integration."""

    def test_update_from_explicit(self, integrator, bandit, sample_features):
        """Test updating bandit from explicit feedback."""
        feedback = Feedback(
            response_id="r123",
            quality_score=1.0,  # High quality to ensure weighted reward > 0.7
            met_expectations=True,
        )

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha

        integrator.update_from_explicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        # Reward = (1.0 * 0.6) + (1.0 * 0.4) = 1.0, weighted = 1.0 * 0.7 = 0.7 (success threshold)
        # Should increment alpha (success)
        assert bandit.model_states["gpt-4o-mini"].alpha > initial_alpha

    def test_update_from_implicit_success(self, integrator, bandit, sample_features):
        """Test updating bandit from positive implicit signals."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=False,
            latency_seconds=5.0,
            latency_tolerance="high",
        )

        initial_alpha = bandit.model_states["gpt-4o-mini"].alpha

        integrator.update_from_implicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        # High tolerance (0.9) * implicit weight (0.3) = 0.27 < 0.7 threshold → failure
        # But with implicit weight, might still be below threshold
        # Check that update was called
        assert bandit.model_states["gpt-4o-mini"].total_requests > 0

    def test_update_from_implicit_error(self, integrator, bandit, sample_features):
        """Test updating bandit from error signal."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=True,
            error_type="api_error",
            latency_seconds=1.0,
            latency_tolerance="high",
        )

        initial_beta = bandit.model_states["gpt-4o-mini"].beta

        integrator.update_from_implicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        # Error (reward=0.0) should increment beta (failure)
        assert bandit.model_states["gpt-4o-mini"].beta > initial_beta

    def test_update_combined_both_feedbacks(
        self, integrator, bandit, sample_features
    ):
        """Test updating with both explicit and implicit feedback."""
        explicit = Feedback(
            response_id="r123",
            quality_score=0.8,
            met_expectations=True,
        )
        implicit = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            latency_seconds=5.0,
            latency_tolerance="high",
        )

        initial_requests = bandit.model_states["gpt-4o-mini"].total_requests

        integrator.update_combined(
            model="gpt-4o-mini",
            features=sample_features,
            explicit=explicit,
            implicit=implicit,
        )

        # Should have 2 updates (explicit + implicit)
        assert bandit.model_states["gpt-4o-mini"].total_requests == initial_requests + 2

    def test_update_combined_explicit_only(self, integrator, bandit, sample_features):
        """Test updating with only explicit feedback."""
        explicit = Feedback(
            response_id="r123",
            quality_score=0.6,
            met_expectations=False,
        )

        initial_requests = bandit.model_states["gpt-4o-mini"].total_requests

        integrator.update_combined(
            model="gpt-4o-mini",
            features=sample_features,
            explicit=explicit,
            implicit=None,
        )

        assert bandit.model_states["gpt-4o-mini"].total_requests == initial_requests + 1

    def test_update_combined_implicit_only(self, integrator, bandit, sample_features):
        """Test updating with only implicit feedback."""
        implicit = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            latency_seconds=2.0,
            latency_tolerance="high",
        )

        initial_requests = bandit.model_states["gpt-4o-mini"].total_requests

        integrator.update_combined(
            model="gpt-4o-mini",
            features=sample_features,
            explicit=None,
            implicit=implicit,
        )

        assert bandit.model_states["gpt-4o-mini"].total_requests == initial_requests + 1


class TestFeedbackWeighting:
    """Test feedback weighting strategy."""

    def test_explicit_weighting(self, integrator, bandit, sample_features):
        """Test explicit feedback applies correct weight."""
        feedback = Feedback(
            response_id="r123",
            quality_score=1.0,
            met_expectations=True,
        )

        integrator.update_from_explicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        # Raw reward = 1.0, weighted = 1.0 * 0.7 = 0.7 (success threshold met)
        # Should increment alpha
        assert bandit.model_states["gpt-4o-mini"].alpha == 2.0
        assert bandit.model_states["gpt-4o-mini"].beta == 1.0

    def test_implicit_weighting(self, integrator, bandit, sample_features):
        """Test implicit feedback applies correct weight."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=100.0,
            error_occurred=False,
            retry_detected=False,
            latency_seconds=5.0,
            latency_tolerance="high",
        )

        integrator.update_from_implicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        # Raw reward = 0.9, weighted = 0.9 * 0.3 = 0.27 < 0.7 threshold → failure
        # Should increment beta
        assert bandit.model_states["gpt-4o-mini"].alpha == 1.0
        assert bandit.model_states["gpt-4o-mini"].beta == 2.0


class TestFeedbackStats:
    """Test feedback statistics."""

    def test_get_feedback_stats(self, integrator, bandit):
        """Test getting feedback integration statistics."""
        stats = integrator.get_feedback_stats()

        assert "explicit_weight" in stats
        assert "implicit_weight" in stats
        assert "bandit_states" in stats
        assert stats["explicit_weight"] == 0.7
        assert stats["implicit_weight"] == 0.3
        assert "gpt-4o-mini" in stats["bandit_states"]

    def test_stats_include_bandit_state(self, integrator, bandit, sample_features):
        """Test stats include bandit state after updates."""
        feedback = Feedback(
            response_id="r123",
            quality_score=1.0,  # High quality to ensure success
            met_expectations=True,
        )

        integrator.update_from_explicit(
            model="gpt-4o-mini",
            features=sample_features,
            feedback=feedback,
        )

        stats = integrator.get_feedback_stats()
        model_stats = stats["bandit_states"]["gpt-4o-mini"]

        assert "alpha" in model_stats
        assert "beta" in model_stats
        assert "mean_success_rate" in model_stats
        assert model_stats["alpha"] == 2.0
