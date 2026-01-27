"""Edge case tests for signal detection logic in feedback/signals.py."""

from unittest.mock import MagicMock, patch

import pytest

from conduit.feedback.signals import (
    ErrorSignal,
    LatencySignal,
    RetrySignal,
    SignalDetector,
)


class TestSignalDetectorEdgeCases:
    """Tests for boundary conditions and edge cases in SignalDetector."""

    # --- detect_error tests ---

    def test_detect_error_none_inputs(self) -> None:
        """Test detect_error with None inputs where allowable."""
        # None response_text, but success status -> Should be treated as empty response error
        signal = SignalDetector.detect_error(
            response_text=None,
            execution_status="success",
            execution_error=None,
            model_id="test-model",
        )
        assert signal.occurred is True
        assert signal.error_type == "empty_response"
        assert signal.model_id == "test-model"

    def test_detect_error_none_execution_error(self) -> None:
        """Test detect_error with error status but None error message."""
        signal = SignalDetector.detect_error(
            response_text="some text",
            execution_status="error",
            execution_error=None,
            model_id="test-model",
        )
        assert signal.occurred is True
        assert signal.error_type == "unknown_error"

    def test_detect_error_empty_strings(self) -> None:
        """Test detect_error with empty string inputs."""
        # Empty response text -> empty_response error
        signal = SignalDetector.detect_error(
            response_text="",
            execution_status="success",
            execution_error="",
            model_id="test-model",
        )
        assert signal.occurred is True
        assert signal.error_type == "empty_response"

    @pytest.mark.parametrize("short_text", ["hi", "   short   ", "123456789"])
    def test_detect_error_short_response(self, short_text: str) -> None:
        """Test detect_error with response text shorter than 10 chars."""
        signal = SignalDetector.detect_error(
            response_text=short_text,
            execution_status="success",
            execution_error=None,
            model_id="test-model",
        )
        assert signal.occurred is True
        assert signal.error_type == "empty_response"

    def test_detect_error_exact_boundary(self) -> None:
        """Test detect_error at exact 10-char boundary (len < 10 triggers error)."""
        signal = SignalDetector.detect_error(
            response_text="0123456789",
            execution_status="success",
            execution_error=None,
            model_id="test-model",
        )
        assert signal.occurred is False

    @pytest.mark.parametrize("error_phrase", [
        "Error: Something went wrong",
        "Exception: NullPointer",
        "I apologize, but I cannot do that",
        "I'm unable to assist",
        "Failed to generate",
    ])
    def test_detect_error_content_patterns(self, error_phrase: str) -> None:
        """Test detect_error with various error patterns in text."""
        signal = SignalDetector.detect_error(
            response_text=f"Some prefix {error_phrase} some suffix",
            execution_status="success",
            execution_error=None,
            model_id="test-model",
        )
        assert signal.occurred is True
        assert signal.error_type == "model_refusal_or_error"

    # --- detect_latency tests ---

    @pytest.mark.parametrize("start,end,expected_latency", [
        (100.0, 100.0, 0.0),      # Zero latency
        (100.0, 100.001, 0.001),  # Micro latency
        (100.0, 160.0, 60.0),     # High latency
    ])
    @patch("conduit.feedback.signals.load_feedback_config")
    def test_detect_latency_calculations(
        self, mock_load_config: MagicMock, start: float, end: float, expected_latency: float
    ) -> None:
        """Test latency calculation accuracy."""
        mock_load_config.return_value = {
            "latency_detection": {
                "high_tolerance_max": 10.0,
                "medium_tolerance_max": 30.0,
            }
        }
        signal = SignalDetector.detect_latency(start, end)
        assert signal.actual_latency_seconds == pytest.approx(expected_latency)
        assert signal.accepted is True

    @patch("conduit.feedback.signals.load_feedback_config")
    def test_categorize_tolerance_boundaries(self, mock_load_config: MagicMock) -> None:
        """Test latency categorization at exact threshold boundaries."""
        # Mock config to have predictable thresholds
        mock_load_config.return_value = {
            "latency_detection": {
                "high_tolerance_max": 10.0,   # 0-10: high
                "medium_tolerance_max": 30.0  # 10-30: medium, >30: low
            }
        }

        # Case 1: Exactly at high/medium boundary (10.0) -> Should be high
        # Logic: if latency > 10.0 (medium) else (high)
        s1 = LatencySignal(actual_latency_seconds=10.0, accepted=True).categorize_tolerance()
        assert s1.tolerance_level == "high"

        # Case 2: Just above boundary (10.0001) -> Should be medium
        s2 = LatencySignal(actual_latency_seconds=10.0001, accepted=True).categorize_tolerance()
        assert s2.tolerance_level == "medium"

        # Case 3: Exactly at medium/low boundary (30.0) -> Should be medium
        s3 = LatencySignal(actual_latency_seconds=30.0, accepted=True).categorize_tolerance()
        assert s3.tolerance_level == "medium"

        # Case 4: Just above boundary (30.0001) -> Should be low
        s4 = LatencySignal(actual_latency_seconds=30.0001, accepted=True).categorize_tolerance()
        assert s4.tolerance_level == "low"

        mock_load_config.assert_called()

    # --- detect_retry tests ---

    @pytest.mark.parametrize("score,threshold,expected_detected", [
        (0.85, 0.85, True),   # Exact match
        (0.849, 0.85, False), # Just below
        (0.851, 0.85, True),  # Just above
        (1.0, 0.85, True),    # Max score
        (0.0, 0.85, False),   # Min score
    ])
    def test_detect_retry_similarity_boundaries(self, score: float, threshold: float, expected_detected: bool) -> None:
        """Test detect_retry_from_similarity at score boundaries."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=score,
            time_delta_seconds=10.0,
            previous_query_id="prev-1",
            similarity_threshold=threshold,
        )
        assert signal.detected is expected_detected

    @pytest.mark.parametrize("delta,window,expected_detected", [
        (300.0, 300.0, True),   # Exact window edge
        (300.1, 300.0, False),  # Just outside window
        (0.0, 300.0, True),     # Immediate retry
    ])
    def test_detect_retry_time_window_boundaries(self, delta: float, window: float, expected_detected: bool) -> None:
        """Test detect_retry_from_similarity at time window boundaries."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=0.9,  # High similarity
            time_delta_seconds=delta,
            previous_query_id="prev-1",
            time_window_seconds=window,
        )
        assert signal.detected is expected_detected

    def test_detect_retry_outside_window_returns_empty(self) -> None:
        """Test that retry outside window returns empty signal."""
        signal = SignalDetector.detect_retry_from_similarity(
            similarity_score=1.0,
            time_delta_seconds=301.0,
            previous_query_id="prev-1",
            time_window_seconds=300.0,
        )
        assert signal.detected is False
        assert signal.delay_seconds is None
        assert signal.similarity_score is None
        assert signal.original_query_id is None
