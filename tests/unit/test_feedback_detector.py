"""Unit tests for Implicit Feedback Detector."""

import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.models import ImplicitFeedback, QueryFeatures
from conduit.feedback.detector import ImplicitFeedbackDetector
from conduit.feedback.history import QueryHistoryEntry, QueryHistoryTracker
from conduit.feedback.signals import ErrorSignal, LatencySignal, RetrySignal


@pytest.fixture
def mock_history_tracker():
    """Create mock query history tracker."""
    tracker = AsyncMock()
    tracker.add_query = AsyncMock()
    tracker.find_similar_query = AsyncMock(return_value=None)
    return tracker


@pytest.fixture
def detector(mock_history_tracker):
    """Create detector with mock history tracker."""
    return ImplicitFeedbackDetector(
        history_tracker=mock_history_tracker,
        retry_similarity_threshold=0.85,
        retry_time_window_seconds=300.0)


@pytest.fixture
def sample_features():
    """Create sample query features with normalized embedding."""
    # Create normalized embedding (unit length)
    raw_embedding = [0.1] * 384
    magnitude = math.sqrt(sum(x**2 for x in raw_embedding))
    normalized_embedding = [x / magnitude for x in raw_embedding]

    return QueryFeatures(
        embedding=normalized_embedding,
        token_count=50,
        complexity_score=0.5
    )


class TestImplicitFeedbackDetector:
    """Tests for ImplicitFeedbackDetector."""

    def test_initialization(self, mock_history_tracker):
        """Test detector initialization."""
        detector = ImplicitFeedbackDetector(
            history_tracker=mock_history_tracker,
            retry_similarity_threshold=0.85,
            retry_time_window_seconds=300.0)

        assert detector.history == mock_history_tracker
        assert detector.retry_similarity_threshold == 0.85
        assert detector.retry_time_window == 300.0

    @pytest.mark.asyncio
    async def test_detect_success_no_signals(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with successful response and no implicit signals."""
        start_time = time.time()
        end_time = start_time + 1.5

        feedback = await detector.detect(
            query="What is Python?",
            query_id="q123",
            features=sample_features,
            response_text="Python is a programming language...",
            model_id="gpt-4o-mini",
            execution_status="success",
            execution_error=None,
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        # Verify feedback structure
        assert isinstance(feedback, ImplicitFeedback)
        assert feedback.query_id == "q123"
        assert feedback.model_id == "gpt-4o-mini"
        assert feedback.error_occurred is False
        assert feedback.retry_detected is False
        assert feedback.latency_accepted is True

        # Verify history tracker was called
        mock_history_tracker.add_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_error_signal(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with error signal."""
        start_time = time.time()
        end_time = start_time + 1.0

        feedback = await detector.detect(
            query="What is Python?",
            query_id="q456",
            features=sample_features,
            response_text=None,  # No response on error
            model_id="gpt-4o-mini",
            execution_status="error",
            execution_error="API timeout",
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        assert feedback.error_occurred is True
        assert feedback.error_type is not None

    @pytest.mark.asyncio
    async def test_detect_high_latency(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with high latency signal."""
        start_time = time.time()
        end_time = start_time + 35.0  # > 30s = low tolerance

        feedback = await detector.detect(
            query="Complex query",
            query_id="q789",
            features=sample_features,
            response_text="Response",
            model_id="gpt-4o",
            execution_status="success",
            execution_error=None,
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        assert feedback.latency_seconds == 35.0
        assert feedback.latency_tolerance == "low"

    @pytest.mark.asyncio
    async def test_detect_retry_signal(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with retry signal."""
        # Create normalized embedding (unit length)
        raw_embedding = [0.1] * 384
        magnitude = math.sqrt(sum(x**2 for x in raw_embedding))
        normalized_embedding = [x / magnitude for x in raw_embedding]

        # Setup: Previous query in history
        previous_time = time.time() - 60.0  # 1 minute ago
        previous_query = QueryHistoryEntry(
            query_id="q_prev",
            query_text="What is Python?",
            embedding=normalized_embedding,  # Normalized embedding for cosine similarity
            timestamp=previous_time,
            user_id="user_abc",
            model_used="gpt-4o-mini")
        mock_history_tracker.find_similar_query = AsyncMock(
            return_value=previous_query
        )

        start_time = time.time()
        end_time = start_time + 1.0

        feedback = await detector.detect(
            query="What is Python?",  # Same query = retry
            query_id="q_retry",
            features=sample_features,
            response_text="Response",
            model_id="gpt-4o",
            execution_status="success",
            execution_error=None,
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        assert feedback.retry_detected is True
        assert feedback.retry_delay_seconds is not None
        assert feedback.similarity_score is not None
        assert feedback.original_query_id == "q_prev"

    @pytest.mark.asyncio
    async def test_detect_retry_no_match(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with no retry match."""
        # Setup: No similar query found
        mock_history_tracker.find_similar_query = AsyncMock(return_value=None)

        start_time = time.time()
        end_time = start_time + 1.0

        feedback = await detector.detect(
            query="Unique query",
            query_id="q_unique",
            features=sample_features,
            response_text="Response",
            model_id="gpt-4o-mini",
            execution_status="success",
            execution_error=None,
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        assert feedback.retry_detected is False
        assert feedback.retry_delay_seconds is None
        assert feedback.similarity_score is None

    @pytest.mark.asyncio
    async def test_detect_combined_signals(
        self, detector, mock_history_tracker, sample_features
    ):
        """Test detect with multiple signals (error + latency)."""
        start_time = time.time()
        end_time = start_time + 35.0  # High latency

        feedback = await detector.detect(
            query="Complex query",
            query_id="q_multi",
            features=sample_features,
            response_text=None,
            model_id="gpt-4o-mini",
            execution_status="error",
            execution_error="Rate limit exceeded",
            request_start_time=start_time,
            response_complete_time=end_time,
            user_id="user_abc")

        # Both error and latency signals present
        assert feedback.error_occurred is True
        assert feedback.latency_seconds == 35.0
        assert feedback.latency_tolerance == "low"

    def test_cosine_similarity(self, detector):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = detector._cosine_similarity(vec1, vec2)
        assert similarity == 1.0

        vec3 = [0.6, 0.8, 0.0]
        vec4 = [0.8, 0.6, 0.0]
        similarity2 = detector._cosine_similarity(vec3, vec4)
        assert 0.9 < similarity2 < 1.0  # High but not perfect

    def test_log_signals_error(self, detector):
        """Test _log_signals with error signal."""
        feedback = ImplicitFeedback(
            query_id="q123",
            model_id="gpt-4o-mini",
            timestamp=time.time(),
            error_occurred=True,
            error_type="api_error",
            latency_seconds=1.0,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=False)

        # Should log without errors
        detector._log_signals(feedback)

    def test_log_signals_retry(self, detector):
        """Test _log_signals with retry signal."""
        feedback = ImplicitFeedback(
            query_id="q456",
            model_id="gpt-4o",
            timestamp=time.time(),
            error_occurred=False,
            latency_seconds=2.0,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=True,
            retry_delay_seconds=60.0,
            similarity_score=0.95,
            original_query_id="q_prev")

        # Should log without errors
        detector._log_signals(feedback)

    def test_log_signals_high_latency(self, detector):
        """Test _log_signals with high latency."""
        feedback = ImplicitFeedback(
            query_id="q789",
            model_id="claude-3-5-sonnet",
            timestamp=time.time(),
            error_occurred=False,
            latency_seconds=35.0,
            latency_accepted=True,
            latency_tolerance="low",
            retry_detected=False)

        # Should log without errors
        detector._log_signals(feedback)

    def test_log_signals_no_signals(self, detector):
        """Test _log_signals with no significant signals."""
        feedback = ImplicitFeedback(
            query_id="q999",
            model_id="gpt-4o-mini",
            timestamp=time.time(),
            error_occurred=False,
            latency_seconds=1.5,
            latency_accepted=True,
            latency_tolerance="high",
            retry_detected=False)

        # Should not log anything
        detector._log_signals(feedback)

    def test_initialization_custom_thresholds(self, mock_history_tracker):
        """Test detector with custom retry thresholds."""
        detector = ImplicitFeedbackDetector(
            history_tracker=mock_history_tracker,
            retry_similarity_threshold=0.90,
            retry_time_window_seconds=600.0)

        assert detector.retry_similarity_threshold == 0.90
        assert detector.retry_time_window == 600.0
