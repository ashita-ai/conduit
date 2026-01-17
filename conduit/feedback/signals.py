"""Signal detection classes for implicit feedback.

Individual signal detectors that analyze specific behavioral patterns:
- RetrySignal: Semantic similarity-based retry detection
- LatencySignal: Response time and user patience analysis
- ErrorSignal: Model failure and quality issue detection
"""

from pydantic import BaseModel, Field

from conduit.core.config import load_feedback_config


class RetrySignal(BaseModel):
    """Immutable retry behavior detection signal.

    Detects when a user re-submits a semantically similar query,
    indicating dissatisfaction with the previous response.

    Attributes:
        detected: Whether a retry was detected
        delay_seconds: Time between original and retry query
        similarity_score: Cosine similarity to previous query (0.0-1.0)
        original_query_id: ID of the query being retried
    """

    model_config = {"frozen": True}

    detected: bool = Field(default=False, description="Retry detected")
    delay_seconds: float | None = Field(
        None, description="Time since original query", ge=0.0
    )
    similarity_score: float | None = Field(
        None, description="Similarity to previous query (0-1)", ge=0.0, le=1.0
    )
    original_query_id: str | None = Field(None, description="Original query ID")


class LatencySignal(BaseModel):
    """Immutable latency acceptance and tolerance signal.

    Tracks response time and whether the user demonstrated patience
    or exhibited signs of dissatisfaction with slow responses.

    Attributes:
        accepted: User waited for response (didn't timeout)
        actual_latency_seconds: Measured response time
        tolerance_level: Categorized patience (high/medium/low)
    """

    model_config = {"frozen": True}

    accepted: bool = Field(default=True, description="User waited for response")
    actual_latency_seconds: float = Field(
        ..., description="Response time in seconds", ge=0.0
    )
    tolerance_level: str = Field(
        default="high",
        description="Latency tolerance (high/medium/low)",
    )

    def categorize_tolerance(self) -> "LatencySignal":
        """Return a new LatencySignal with categorized tolerance."""
        config = load_feedback_config()
        thresholds = config["latency_detection"]

        if self.actual_latency_seconds > thresholds["medium_tolerance_max"]:
            level = "low"  # Very slow, user was patient
        elif self.actual_latency_seconds > thresholds["high_tolerance_max"]:
            level = "medium"  # Somewhat slow
        else:
            level = "high"  # Fast, no patience needed

        return self.model_copy(update={"tolerance_level": level})


class ErrorSignal(BaseModel):
    """Immutable error occurrence and type detection signal.

    Captures model failures, API errors, and quality issues
    like empty responses or error patterns in output.

    Attributes:
        occurred: Whether an error was detected
        error_type: Classification of error (api_error, timeout, empty_response, etc.)
        model_id: Which model produced the error
    """

    model_config = {"frozen": True}

    occurred: bool = Field(default=False, description="Error detected")
    error_type: str | None = Field(None, description="Error classification")
    model_id: str | None = Field(None, description="Model that errored")


class SignalDetector:
    """Individual signal detection methods.

    Stateless utility class providing detection algorithms for
    each type of implicit behavioral signal.
    """

    @staticmethod
    def detect_error(
        response_text: str | None,
        execution_status: str,
        execution_error: str | None,
        model_id: str,
    ) -> ErrorSignal:
        """Detect error signals from response.

        Args:
            response_text: Model response text
            execution_status: Execution status (success/error)
            execution_error: Error message if execution failed
            model_id: Model identifier

        Returns:
            ErrorSignal with detection results

        Detection Logic:
            1. Hard errors: API failures, timeouts, exceptions
            2. Soft errors: Empty/very short responses
            3. Content errors: Error patterns in response text
        """
        # Hard errors from execution
        if execution_status == "error":
            return ErrorSignal(
                occurred=True,
                error_type=execution_error or "unknown_error",
                model_id=model_id,
            )

        # Soft errors: empty or very short response
        if not response_text or len(response_text.strip()) < 10:
            return ErrorSignal(
                occurred=True,
                error_type="empty_response",
                model_id=model_id,
            )

        # Content errors: error patterns in response
        error_patterns = [
            "I apologize, but I",
            "I cannot",
            "I'm unable to",
            "Error:",
            "Exception:",
            "Failed to",
        ]

        response_lower = response_text.lower()
        for pattern in error_patterns:
            if pattern.lower() in response_lower:
                return ErrorSignal(
                    occurred=True,
                    error_type="model_refusal_or_error",
                    model_id=model_id,
                )

        return ErrorSignal(occurred=False, error_type=None, model_id=model_id)

    @staticmethod
    def detect_latency(
        request_start_time: float,
        response_complete_time: float,
    ) -> LatencySignal:
        """Detect latency signals from timing.

        Args:
            request_start_time: Request timestamp (epoch seconds)
            response_complete_time: Response timestamp (epoch seconds)

        Returns:
            LatencySignal with timing analysis

        Logic:
            - If we returned a response, user waited (accepted=True)
            - Categorize tolerance based on actual latency:
              - <10s: high tolerance (fast response)
              - 10-30s: medium tolerance (somewhat slow)
              - >30s: low tolerance (very slow, user was patient)
        """
        latency = response_complete_time - request_start_time

        signal = LatencySignal(
            accepted=True,  # They received response, so they waited
            actual_latency_seconds=latency,
        )

        return signal.categorize_tolerance()

    @staticmethod
    def detect_retry_from_similarity(
        similarity_score: float,
        time_delta_seconds: float,
        previous_query_id: str,
        similarity_threshold: float = 0.85,
        time_window_seconds: float = 300.0,  # 5 minutes
    ) -> RetrySignal:
        """Detect retry based on similarity and timing.

        Args:
            similarity_score: Cosine similarity to previous query (0-1)
            time_delta_seconds: Time since previous query
            previous_query_id: ID of potentially retried query
            similarity_threshold: Minimum similarity for retry detection
            time_window_seconds: Maximum time for retry consideration

        Returns:
            RetrySignal with detection results

        Logic:
            - High similarity (>0.85) + Short time (<5min) = Retry
            - Exact match (1.0) = Definite retry
            - Low similarity or long delay = Not a retry
        """
        # Check if within time window
        if time_delta_seconds > time_window_seconds:
            return RetrySignal(
                detected=False,
                delay_seconds=None,
                similarity_score=None,
                original_query_id=None,
            )

        # Check similarity threshold
        if similarity_score >= similarity_threshold:
            return RetrySignal(
                detected=True,
                delay_seconds=time_delta_seconds,
                similarity_score=similarity_score,
                original_query_id=previous_query_id,
            )

        return RetrySignal(
            detected=False,
            delay_seconds=None,
            similarity_score=None,
            original_query_id=None,
        )
