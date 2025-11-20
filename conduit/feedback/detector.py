"""Implicit feedback detector for behavioral signal analysis.

Orchestrates detection of implicit behavioral signals:
- Error detection from model responses
- Latency tolerance from response times
- Retry behavior from query similarity

Integrates with QueryHistoryTracker for retry detection and
produces ImplicitFeedback objects for Thompson Sampling updates.
"""

import logging
import time

from conduit.core.models import ImplicitFeedback, QueryFeatures
from conduit.feedback.history import QueryHistoryTracker
from conduit.feedback.signals import RetrySignal, SignalDetector

logger = logging.getLogger(__name__)


class ImplicitFeedbackDetector:
    """Orchestrates implicit behavioral signal detection.

    Combines error, latency, and retry signal detection into
    a unified ImplicitFeedback object for ML model updates.

    Architecture:
        - Error detection: Analyze response content and execution status
        - Latency detection: Track response time and user patience
        - Retry detection: Compare with recent query history via embeddings

    Example:
        >>> detector = ImplicitFeedbackDetector(history_tracker)
        >>> feedback = await detector.detect(
        ...     query="What is Python?",
        ...     query_id="q123",
        ...     features=query_features,
        ...     response_text="Python is...",
        ...     model_id="gpt-4o-mini",
        ...     execution_status="success",
        ...     request_start_time=start_time,
        ...     response_complete_time=end_time,
        ...     user_id="user_abc"
        ... )
    """

    def __init__(
        self,
        history_tracker: QueryHistoryTracker,
        retry_similarity_threshold: float = 0.85,
        retry_time_window_seconds: float = 300.0,
    ):
        """Initialize implicit feedback detector.

        Args:
            history_tracker: Query history tracker for retry detection
            retry_similarity_threshold: Minimum similarity for retry (0.85 default)
            retry_time_window_seconds: Time window for retry detection (5 min default)
        """
        self.history = history_tracker
        self.retry_similarity_threshold = retry_similarity_threshold
        self.retry_time_window = retry_time_window_seconds

    async def detect(
        self,
        query: str,
        query_id: str,
        features: QueryFeatures,
        response_text: str | None,
        model_id: str,
        execution_status: str,
        execution_error: str | None,
        request_start_time: float,
        response_complete_time: float,
        user_id: str,
    ) -> ImplicitFeedback:
        """Detect all implicit behavioral signals.

        Args:
            query: Original query text
            query_id: Unique query identifier
            features: Query features (includes embedding)
            response_text: Model response text (None if execution failed)
            model_id: Model that generated response
            execution_status: Execution status ("success" or "error")
            execution_error: Error message if execution failed
            request_start_time: Request timestamp (epoch seconds)
            response_complete_time: Response timestamp (epoch seconds)
            user_id: User identifier (API key or client ID)

        Returns:
            ImplicitFeedback with all detected signals

        Side Effects:
            - Adds current query to history tracker
            - Logs signal detections for monitoring
        """
        # 1. Error Signal Detection
        error_signal = SignalDetector.detect_error(
            response_text=response_text,
            execution_status=execution_status,
            execution_error=execution_error,
            model_id=model_id,
        )

        # 2. Latency Signal Detection
        latency_signal = SignalDetector.detect_latency(
            request_start_time=request_start_time,
            response_complete_time=response_complete_time,
        )

        # 3. Retry Signal Detection
        retry_signal = await self._detect_retry(
            current_embedding=features.embedding,
            user_id=user_id,
        )

        # 4. Store current query in history for future retry detection
        await self.history.add_query(
            query_id=query_id,
            query_text=query,
            features=features,
            user_id=user_id,
            model_used=model_id,
        )

        # 5. Build ImplicitFeedback object
        feedback = ImplicitFeedback(
            query_id=query_id,
            model_id=model_id,
            timestamp=response_complete_time,
            # Error signals
            error_occurred=error_signal.occurred,
            error_type=error_signal.error_type,
            # Latency signals
            latency_seconds=latency_signal.actual_latency_seconds,
            latency_accepted=latency_signal.accepted,
            latency_tolerance=latency_signal.tolerance_level,
            # Retry signals
            retry_detected=retry_signal.detected,
            retry_delay_seconds=retry_signal.delay_seconds,
            similarity_score=retry_signal.similarity_score,
            original_query_id=retry_signal.original_query_id,
        )

        # Log significant signals for monitoring
        self._log_signals(feedback)

        return feedback

    async def _detect_retry(
        self,
        current_embedding: list[float],
        user_id: str,
    ) -> RetrySignal:
        """Detect retry behavior via semantic similarity.

        Args:
            current_embedding: Embedding of current query
            user_id: User identifier

        Returns:
            RetrySignal with detection results

        Algorithm:
            1. Find most similar recent query within time window
            2. If similarity >= threshold: retry detected
            3. Return RetrySignal with delay and similarity metrics
        """
        similar_query = await self.history.find_similar_query(
            current_embedding=current_embedding,
            user_id=user_id,
            similarity_threshold=self.retry_similarity_threshold,
            time_window_seconds=self.retry_time_window,
        )

        if similar_query is None:
            return RetrySignal(
                detected=False,
                delay_seconds=None,
                similarity_score=None,
                original_query_id=None,
            )

        # Calculate time delta
        current_time = time.time()
        time_delta = current_time - similar_query.timestamp

        return RetrySignal(
            detected=True,
            delay_seconds=time_delta,
            similarity_score=self._cosine_similarity(
                current_embedding, similar_query.embedding
            ),
            original_query_id=similar_query.query_id,
        )

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity (dot product for normalized vectors)."""
        return float(sum(a * b for a, b in zip(vec1, vec2)))

    def _log_signals(self, feedback: ImplicitFeedback) -> None:
        """Log detected signals for monitoring and debugging.

        Args:
            feedback: ImplicitFeedback with detected signals
        """
        signals = []

        if feedback.error_occurred:
            signals.append(f"ERROR({feedback.error_type})")

        if feedback.retry_detected:
            signals.append(
                f"RETRY(delay={feedback.retry_delay_seconds:.1f}s, "
                f"similarity={feedback.similarity_score:.2f})"
            )

        if feedback.latency_seconds > 10:
            signals.append(
                f"LATENCY({feedback.latency_seconds:.1f}s, "
                f"tolerance={feedback.latency_tolerance})"
            )

        if signals:
            logger.info(
                f"Implicit signals detected for {feedback.model_id} "
                f"(query={feedback.query_id}): {', '.join(signals)}"
            )
