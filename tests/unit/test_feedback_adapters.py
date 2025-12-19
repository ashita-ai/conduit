"""Tests for feedback adapters.

Tests the pluggable adapter system for converting user feedback signals
to bandit rewards.
"""

import pytest
from datetime import datetime, timezone

from conduit.feedback.adapters import (
    CompletionTimeAdapter,
    FeedbackAdapter,
    QualityScoreAdapter,
    RatingAdapter,
    RegenerationAdapter,
    TaskSuccessAdapter,
    ThumbsAdapter,
)
from conduit.feedback.models import FeedbackEvent, RewardMapping


class TestThumbsAdapter:
    """Tests for ThumbsAdapter."""

    def test_thumbs_up(self):
        """Thumbs up should return reward 1.0."""
        adapter = ThumbsAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="thumbs",
            payload={"value": "up"},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0
        assert result.confidence == 1.0

    def test_thumbs_down(self):
        """Thumbs down should return reward 0.0."""
        adapter = ThumbsAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="thumbs",
            payload={"value": "down"},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0
        assert result.confidence == 1.0

    def test_thumbs_unknown(self):
        """Unknown value should return neutral with low confidence."""
        adapter = ThumbsAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="thumbs",
            payload={"value": "maybe"},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5
        assert result.confidence == 0.5

    def test_thumbs_missing_value(self):
        """Missing value should return neutral."""
        adapter = ThumbsAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="thumbs",
            payload={},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5
        assert result.confidence == 0.5

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = ThumbsAdapter()
        assert adapter.signal_type == "thumbs"


class TestRatingAdapter:
    """Tests for RatingAdapter."""

    def test_max_rating_uncalibrated(self):
        """Max rating (5/5) with uncalibrated mode should return reward 1.0."""
        adapter = RatingAdapter(calibrated=False)  # Disable calibration
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 5},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0
        assert result.confidence == 1.0

    def test_min_rating_uncalibrated(self):
        """Min rating (1/5) with uncalibrated mode should return reward 0.0."""
        adapter = RatingAdapter(calibrated=False)
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 1},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0
        assert result.confidence == 1.0

    def test_mid_rating(self):
        """Mid rating (3/5) should return reward 0.5 (same for calibrated/uncalibrated)."""
        adapter = RatingAdapter()  # Calibrated by default
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 3},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5  # Midpoint is same for sigmoid
        assert result.confidence == 1.0

    def test_custom_scale_uncalibrated(self):
        """Custom 0-10 scale with uncalibrated mode should normalize correctly."""
        adapter = RatingAdapter(min_rating=0, max_rating=10, calibrated=False)
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 7},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.7
        assert result.confidence == 1.0

    def test_rating_out_of_range_uncalibrated(self):
        """Rating outside range should be clamped."""
        adapter = RatingAdapter(calibrated=False)  # 1-5 scale
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 10},  # Above max
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0  # Clamped to max

    def test_missing_rating(self):
        """Missing rating should use middle of range."""
        adapter = RatingAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5  # Middle of 1-5 range

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = RatingAdapter()
        assert adapter.signal_type == "rating"

    def test_calibrated_high_rating(self):
        """High rating with calibration should return reward close to 1.0."""
        adapter = RatingAdapter(calibrated=True)  # Default
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 5},
        )
        result = adapter.to_reward(event)

        # Calibrated 5/5 should be very close to 1.0 (sigmoid approaches but doesn't reach)
        assert result.reward > 0.95
        assert result.confidence == 1.0

    def test_calibrated_low_rating(self):
        """Low rating with calibration should return reward close to 0.0."""
        adapter = RatingAdapter(calibrated=True)
        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 1},
        )
        result = adapter.to_reward(event)

        # Calibrated 1/5 should be very close to 0.0
        assert result.reward < 0.05
        assert result.confidence == 1.0

    def test_calibrated_high_rating_stronger(self):
        """4/5 rating with calibration should be higher than linear."""
        adapter_calibrated = RatingAdapter(calibrated=True)
        adapter_linear = RatingAdapter(calibrated=False)

        event = FeedbackEvent(
            query_id="q1",
            signal_type="rating",
            payload={"rating": 4},  # 0.75 linear
        )

        calibrated_result = adapter_calibrated.to_reward(event)
        linear_result = adapter_linear.to_reward(event)

        # Calibrated should push 4/5 higher than 0.75
        assert calibrated_result.reward > linear_result.reward
        assert linear_result.reward == 0.75


class TestTaskSuccessAdapter:
    """Tests for TaskSuccessAdapter."""

    def test_success(self):
        """Success should return reward 1.0."""
        adapter = TaskSuccessAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="task_success",
            payload={"success": True},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0
        assert result.confidence == 1.0

    def test_failure(self):
        """Failure should return reward 0.0."""
        adapter = TaskSuccessAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="task_success",
            payload={"success": False},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0
        assert result.confidence == 1.0

    def test_missing_success(self):
        """Missing success defaults to False."""
        adapter = TaskSuccessAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="task_success",
            payload={},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = TaskSuccessAdapter()
        assert adapter.signal_type == "task_success"


class TestQualityScoreAdapter:
    """Tests for QualityScoreAdapter."""

    def test_high_score(self):
        """High quality score should pass through."""
        adapter = QualityScoreAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="quality_score",
            payload={"score": 0.95},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.95
        assert result.confidence == 1.0

    def test_low_score(self):
        """Low quality score should pass through."""
        adapter = QualityScoreAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="quality_score",
            payload={"score": 0.15},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.15
        assert result.confidence == 1.0

    def test_score_clamped_high(self):
        """Score above 1.0 should be clamped."""
        adapter = QualityScoreAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="quality_score",
            payload={"score": 1.5},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0

    def test_score_clamped_low(self):
        """Score below 0.0 should be clamped."""
        adapter = QualityScoreAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="quality_score",
            payload={"score": -0.5},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0

    def test_missing_score(self):
        """Missing score defaults to 0.5."""
        adapter = QualityScoreAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="quality_score",
            payload={},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = QualityScoreAdapter()
        assert adapter.signal_type == "quality_score"


class TestCompletionTimeAdapter:
    """Tests for CompletionTimeAdapter."""

    def test_instant_completion(self):
        """Instant completion should return reward ~1.0."""
        adapter = CompletionTimeAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="completion_time",
            payload={"seconds": 0},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0
        assert result.confidence == 0.7  # Time is proxy metric

    def test_max_time_completion(self):
        """Max time completion should return reward 0.0."""
        adapter = CompletionTimeAdapter(max_seconds=300)
        event = FeedbackEvent(
            query_id="q1",
            signal_type="completion_time",
            payload={"seconds": 300},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0

    def test_half_time_completion(self):
        """Half max time should return reward ~0.5."""
        adapter = CompletionTimeAdapter(max_seconds=100)
        event = FeedbackEvent(
            query_id="q1",
            signal_type="completion_time",
            payload={"seconds": 50},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.5

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = CompletionTimeAdapter()
        assert adapter.signal_type == "completion_time"


class TestFeedbackEvent:
    """Tests for FeedbackEvent model."""

    def test_default_timestamp(self):
        """Event should have default timestamp."""
        event = FeedbackEvent(
            query_id="q1",
            signal_type="thumbs",
        )
        assert event.timestamp is not None

    def test_custom_payload(self):
        """Event should preserve custom payload."""
        payload = {"custom_field": 123, "nested": {"key": "value"}}
        event = FeedbackEvent(
            query_id="q1",
            signal_type="custom",
            payload=payload,
        )
        assert event.payload == payload


class TestRegenerationAdapter:
    """Tests for RegenerationAdapter."""

    def test_regenerated(self):
        """Regeneration should return reward 0.0 with 0.8 confidence."""
        adapter = RegenerationAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="regeneration",
            payload={"regenerated": True},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.0
        assert result.confidence == 0.8  # Implicit signal

    def test_not_regenerated(self):
        """No regeneration should return reward 1.0."""
        adapter = RegenerationAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="regeneration",
            payload={"regenerated": False},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0
        assert result.confidence == 0.8

    def test_accepted_after_regeneration(self):
        """Acceptance after regeneration should return partial reward."""
        adapter = RegenerationAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="regeneration",
            payload={"regenerated": True, "accepted_after_regeneration": True},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.3  # Partial acceptance
        assert result.confidence == 0.8

    def test_custom_rewards(self):
        """Custom reward values should be used."""
        adapter = RegenerationAdapter(
            regeneration_reward=0.1,
            acceptance_reward=0.9,
            partial_acceptance_reward=0.5,
            confidence=0.7,
        )
        event = FeedbackEvent(
            query_id="q1",
            signal_type="regeneration",
            payload={"regenerated": True},
        )
        result = adapter.to_reward(event)

        assert result.reward == 0.1
        assert result.confidence == 0.7

    def test_signal_type(self):
        """Adapter should report correct signal type."""
        adapter = RegenerationAdapter()
        assert adapter.signal_type == "regeneration"

    def test_missing_regenerated_flag(self):
        """Missing regenerated flag defaults to False (accepted)."""
        adapter = RegenerationAdapter()
        event = FeedbackEvent(
            query_id="q1",
            signal_type="regeneration",
            payload={},
        )
        result = adapter.to_reward(event)

        assert result.reward == 1.0  # Defaults to acceptance


class TestRewardMapping:
    """Tests for RewardMapping model."""

    def test_default_confidence(self):
        """Default confidence should be 1.0."""
        mapping = RewardMapping(reward=0.5)
        assert mapping.confidence == 1.0

    def test_reward_bounds(self):
        """Reward should be bounded 0.0-1.0."""
        # Valid values
        RewardMapping(reward=0.0)
        RewardMapping(reward=1.0)
        RewardMapping(reward=0.5)

        # Invalid values should raise
        with pytest.raises(ValueError):
            RewardMapping(reward=-0.1)

        with pytest.raises(ValueError):
            RewardMapping(reward=1.1)
