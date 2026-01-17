"""Unit tests for feedback signal detection."""

import pytest

from conduit.feedback.signals import (
    ErrorSignal,
    LatencySignal,
    RetrySignal,
    SignalDetector,
)


class TestErrorSignal:
    """Test ErrorSignal model."""

    def test_error_signal_default(self):
        """Test error signal with default values."""
        signal = ErrorSignal()
        assert signal.occurred is False
        assert signal.error_type is None
        assert signal.model_id is None

    def test_error_signal_occurred(self):
        """Test error signal with error occurred."""
        signal = ErrorSignal(
            occurred=True,
            error_type="api_error",
            model_id="gpt-4o-mini",
        )
        assert signal.occurred is True
        assert signal.error_type == "api_error"
        assert signal.model_id == "gpt-4o-mini"


class TestLatencySignal:
    """Test LatencySignal model."""

    def test_latency_signal_defaults(self):
        """Test latency signal with defaults."""
        signal = LatencySignal(actual_latency_seconds=5.0)
        assert signal.accepted is True
        assert signal.actual_latency_seconds == 5.0
        assert signal.tolerance_level == "high"

    def test_categorize_tolerance_high(self):
        """Test high tolerance categorization (< 10s)."""
        signal = LatencySignal(actual_latency_seconds=5.0)
        signal = signal.categorize_tolerance()
        assert signal.tolerance_level == "high"

    def test_categorize_tolerance_medium(self):
        """Test medium tolerance categorization (10-30s)."""
        signal = LatencySignal(actual_latency_seconds=15.0)
        signal = signal.categorize_tolerance()
        assert signal.tolerance_level == "medium"

    def test_categorize_tolerance_low(self):
        """Test low tolerance categorization (> 30s)."""
        signal = LatencySignal(actual_latency_seconds=35.0)
        signal = signal.categorize_tolerance()
        assert signal.tolerance_level == "low"

    def test_categorize_tolerance_boundary_medium(self):
        """Test boundary at 10s (medium)."""
        signal = LatencySignal(actual_latency_seconds=10.1)
        signal = signal.categorize_tolerance()
        assert signal.tolerance_level == "medium"

    def test_categorize_tolerance_boundary_low(self):
        """Test boundary at 30s (low)."""
        signal = LatencySignal(actual_latency_seconds=30.1)
        signal = signal.categorize_tolerance()
        assert signal.tolerance_level == "low"


class TestRetrySignal:
    """Test RetrySignal model."""

    def test_retry_signal_default(self):
        """Test retry signal with default values."""
        signal = RetrySignal()
        assert signal.detected is False
        assert signal.delay_seconds is None
        assert signal.similarity_score is None
        assert signal.original_query_id is None

    def test_retry_signal_detected(self):
        """Test retry signal with retry detected."""
        signal = RetrySignal(
            detected=True,
            delay_seconds=30.0,
            similarity_score=0.92,
            original_query_id="q123",
        )
        assert signal.detected is True
        assert signal.delay_seconds == 30.0
        assert signal.similarity_score == 0.92
        assert signal.original_query_id == "q123"


class TestSignalDetectorError:
    """Test SignalDetector.detect_error method."""

    def test_detect_error_execution_failed(self):
        """Test error detection when execution status is error."""
        signal = SignalDetector.detect_error(
            response_text="Some text",
            execution_status="error",
            execution_error="Connection timeout",
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is True
        assert signal.error_type == "Connection timeout"
        assert signal.model_id == "gpt-4o-mini"

    def test_detect_error_empty_response(self):
        """Test error detection for empty response."""
        signal = SignalDetector.detect_error(
            response_text="",
            execution_status="success",
            execution_error=None,
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is True
        assert signal.error_type == "empty_response"
        assert signal.model_id == "gpt-4o-mini"

    def test_detect_error_short_response(self):
        """Test error detection for very short response (< 10 chars)."""
        signal = SignalDetector.detect_error(
            response_text="Error",
            execution_status="success",
            execution_error=None,
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is True
        assert signal.error_type == "empty_response"

    def test_detect_error_none_response(self):
        """Test error detection for None response."""
        signal = SignalDetector.detect_error(
            response_text=None,
            execution_status="success",
            execution_error=None,
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is True
        assert signal.error_type == "empty_response"

    def test_detect_error_model_refusal_patterns(self):
        """Test error detection for model refusal patterns."""
        refusal_texts = [
            "I apologize, but I cannot help with that",
            "I'm unable to process this request",
            "Error: Invalid input",
            "Exception: NullPointerException",
            "Failed to generate response",
        ]

        for text in refusal_texts:
            signal = SignalDetector.detect_error(
                response_text=text,
                execution_status="success",
                execution_error=None,
                model_id="gpt-4o-mini",
            )

            assert signal.occurred is True, f"Failed to detect error in: {text}"
            assert signal.error_type == "model_refusal_or_error"

    def test_detect_error_valid_response(self):
        """Test no error for valid response."""
        signal = SignalDetector.detect_error(
            response_text="This is a valid response with enough content.",
            execution_status="success",
            execution_error=None,
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is False
        assert signal.model_id == "gpt-4o-mini"

    def test_detect_error_case_insensitive(self):
        """Test error pattern matching is case-insensitive."""
        signal = SignalDetector.detect_error(
            response_text="i apologize, but i cannot help",
            execution_status="success",
            execution_error=None,
            model_id="gpt-4o-mini",
        )

        assert signal.occurred is True


class TestSignalDetectorLatency:
    """Test SignalDetector.detect_latency method."""

    def test_detect_latency_fast(self):
        """Test latency detection for fast response (< 10s)."""
        signal = SignalDetector.detect_latency(
            request_start_time=100.0,
            response_complete_time=105.0,
        )

        assert signal.accepted is True
        assert signal.actual_latency_seconds == 5.0
        assert signal.tolerance_level == "high"

    def test_detect_latency_medium(self):
        """Test latency detection for medium response (10-30s)."""
        signal = SignalDetector.detect_latency(
            request_start_time=100.0,
            response_complete_time=115.0,
        )

        assert signal.accepted is True
        assert signal.actual_latency_seconds == 15.0
        assert signal.tolerance_level == "medium"

    def test_detect_latency_slow(self):
        """Test latency detection for slow response (> 30s)."""
        signal = SignalDetector.detect_latency(
            request_start_time=100.0,
            response_complete_time=140.0,
        )

        assert signal.accepted is True
        assert signal.actual_latency_seconds == 40.0
        assert signal.tolerance_level == "low"

    def test_detect_latency_zero(self):
        """Test latency detection for instant response."""
        signal = SignalDetector.detect_latency(
            request_start_time=100.0,
            response_complete_time=100.0,
        )

        assert signal.actual_latency_seconds == 0.0
        assert signal.tolerance_level == "high"


class TestSignalDetectorRetry:
    """Test SignalDetector.detect_retry_from_similarity method."""

    def test_detect_retry_high_similarity_short_time(self):
        """Test retry detection with high similarity and short delay."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.92,
            time_delta_seconds=30.0,
            previous_query_id="q123",
            similarity_threshold=0.85,
            time_window_seconds=300.0,
        )

        assert signal.detected is True
        assert signal.delay_seconds == 30.0
        assert signal.similarity_score == 0.92
        assert signal.original_query_id == "q123"

    def test_detect_retry_exact_match(self):
        """Test retry detection with exact match (similarity 1.0)."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=1.0,
            time_delta_seconds=60.0,
            previous_query_id="q123",
        )

        assert signal.detected is True
        assert signal.similarity_score == 1.0

    def test_detect_retry_low_similarity(self):
        """Test no retry with low similarity."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.5,
            time_delta_seconds=30.0,
            previous_query_id="q123",
        )

        assert signal.detected is False

    def test_detect_retry_boundary_similarity(self):
        """Test retry detection at similarity threshold boundary."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.85,
            time_delta_seconds=30.0,
            previous_query_id="q123",
            similarity_threshold=0.85,
        )

        assert signal.detected is True

    def test_detect_retry_time_window_exceeded(self):
        """Test no retry when time window exceeded."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.95,
            time_delta_seconds=400.0,  # > 300 seconds
            previous_query_id="q123",
            time_window_seconds=300.0,
        )

        assert signal.detected is False

    def test_detect_retry_boundary_time_window(self):
        """Test retry detection at time window boundary."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.9,
            time_delta_seconds=300.0,
            previous_query_id="q123",
            time_window_seconds=300.0,
        )

        # At exactly the boundary, still within window (<=)
        assert signal.detected is True

    def test_detect_retry_custom_thresholds(self):
        """Test retry detection with custom thresholds."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.75,
            time_delta_seconds=120.0,
            previous_query_id="q123",
            similarity_threshold=0.7,
            time_window_seconds=600.0,
        )

        assert signal.detected is True
