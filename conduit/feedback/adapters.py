"""Feedback adapters for converting user signals to bandit rewards.

This module provides pluggable adapters for different feedback signal types:
- ThumbsAdapter: Thumbs up/down binary feedback
- RatingAdapter: Numeric ratings (1-5 stars, 0-10 scale, etc.)
- TaskSuccessAdapter: Task completion success/failure
- QualityScoreAdapter: Direct quality score (0.0-1.0) from evaluators/humans
- CompletionTimeAdapter: Task completion time as quality proxy
- RegenerationAdapter: User regenerated response (implicit negative signal)

Each adapter converts a FeedbackEvent to a RewardMapping that can be
used to update bandit weights via Router.update().

Design Note - RewardMapping.confidence:
    The confidence field controls how strongly the feedback affects bandit updates:
    - 1.0: Full weight update (explicit user feedback)
    - 0.5-0.8: Partial weight (implicit signals like regeneration)
    - 0.0: No update (use for analytics only)

    When FeedbackCollector records feedback, it scales the update by confidence.
    This enables softer learning from uncertain signals.

Usage:
    >>> from conduit.feedback.adapters import ThumbsAdapter, QualityScoreAdapter
    >>>
    >>> # Built-in adapters
    >>> thumbs = ThumbsAdapter()
    >>> event = FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"})
    >>> reward = thumbs.to_reward(event)
    >>> print(reward.reward)  # 1.0
    >>>
    >>> # Direct quality score from human reviewer or Arbiter
    >>> score = QualityScoreAdapter()
    >>> event = FeedbackEvent(query_id="q1", signal_type="quality_score", payload={"score": 0.87})
    >>> reward = score.to_reward(event)
    >>> print(reward.reward)  # 0.87
"""

from abc import ABC, abstractmethod

from conduit.feedback.models import FeedbackEvent, RewardMapping


class FeedbackAdapter(ABC):
    """Base class for converting user feedback signals to bandit rewards.

    Subclasses implement the `to_reward` method to convert their
    specific signal type to a normalized reward value.

    Attributes:
        signal_type: The signal type this adapter handles (e.g., "thumbs", "rating")
    """

    @property
    @abstractmethod
    def signal_type(self) -> str:
        """Signal type this adapter handles (e.g., 'thumbs', 'rating')."""
        pass

    @abstractmethod
    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert feedback event to bandit reward.

        Args:
            event: FeedbackEvent with signal-specific payload

        Returns:
            RewardMapping with reward (0.0-1.0) and confidence
        """
        pass


class ThumbsAdapter(FeedbackAdapter):
    """Thumbs up/down binary feedback adapter.

    Converts thumbs feedback to binary rewards:
    - "up" -> reward=1.0 (positive)
    - "down" -> reward=0.0 (negative)
    - unknown -> reward=0.5 (neutral, low confidence)

    Payload:
        {"value": "up" | "down"}
    """

    @property
    def signal_type(self) -> str:
        return "thumbs"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert thumbs up/down to reward."""
        value = event.payload.get("value", "").lower()

        if value == "up":
            return RewardMapping(reward=1.0, confidence=1.0)
        elif value == "down":
            return RewardMapping(reward=0.0, confidence=1.0)
        else:
            return RewardMapping(reward=0.5, confidence=0.5)


class RatingAdapter(FeedbackAdapter):
    """Numeric rating feedback adapter (1-5 stars, 0-10 scale, etc.).

    Normalizes rating values to 0.0-1.0 reward range with optional
    calibration curve to handle bimodal rating distributions.

    Rating Calibration:
        User ratings often follow bimodal distributions (many 1s and 5s, few 3s).
        Linear normalization assumes uniform distribution, which underweights
        extreme ratings. Sigmoid calibration maps ratings through a curve
        that better reflects user intent:
        - Low ratings (1-2): More strongly negative
        - Middle ratings (3): Near neutral (0.5)
        - High ratings (4-5): More strongly positive

    Payload:
        {"rating": float}
    """

    def __init__(
        self,
        min_rating: float = 1.0,
        max_rating: float = 5.0,
        calibrated: bool = True,
        calibration_steepness: float = 2.0,
    ):
        """Initialize rating adapter.

        Args:
            min_rating: Minimum rating value (default 1.0)
            max_rating: Maximum rating value (default 5.0)
            calibrated: Use sigmoid calibration curve (default True)
            calibration_steepness: Steepness of sigmoid curve (default 2.0).
                Higher values create sharper transitions at extremes.
        """
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.calibrated = calibrated
        self.calibration_steepness = calibration_steepness

    @property
    def signal_type(self) -> str:
        return "rating"

    def _calibrate(self, normalized: float) -> float:
        """Apply sigmoid calibration to normalized rating.

        Maps [0, 1] -> [0, 1] with steeper transitions at extremes.
        Uses: 1 / (1 + exp(-k * (x - 0.5) * 4)) where k = steepness

        Args:
            normalized: Linear normalized rating [0, 1]

        Returns:
            Calibrated reward [0, 1]
        """
        import math

        # Shift to center at 0, scale, then apply sigmoid
        x = (normalized - 0.5) * 4 * self.calibration_steepness
        try:
            calibrated = 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            # Handle extreme values
            calibrated = 0.0 if x < 0 else 1.0

        return calibrated

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert numeric rating to normalized (optionally calibrated) reward."""
        default_rating = (self.min_rating + self.max_rating) / 2
        rating = event.payload.get("rating", default_rating)

        if self.max_rating == self.min_rating:
            normalized = 0.5
        else:
            normalized = (rating - self.min_rating) / (
                self.max_rating - self.min_rating
            )

        normalized = max(0.0, min(1.0, normalized))

        if self.calibrated:
            reward = self._calibrate(normalized)
        else:
            reward = normalized

        return RewardMapping(reward=reward, confidence=1.0)


class TaskSuccessAdapter(FeedbackAdapter):
    """Task completion success/failure adapter.

    Converts task completion signals to binary rewards:
    - success=True -> reward=1.0
    - success=False -> reward=0.0

    Use this for objective success criteria (code ran, API call succeeded,
    test passed, etc.).

    Payload:
        {"success": bool}
    """

    @property
    def signal_type(self) -> str:
        return "task_success"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert task success to reward."""
        success = event.payload.get("success", False)
        return RewardMapping(reward=1.0 if success else 0.0, confidence=1.0)


class QualityScoreAdapter(FeedbackAdapter):
    """Direct quality score adapter for human reviewers or automated evaluators.

    Accepts pre-computed quality scores (0.0-1.0), passing them through
    as rewards without transformation. Use this when:
    - Human reviewers score responses
    - External evaluators (like Arbiter/GPT-4) provide quality assessments
    - You have custom scoring logic outside the adapter system

    Payload:
        {"score": float}  # 0.0-1.0

    Example:
        >>> # From human reviewer
        >>> event = FeedbackEvent(
        ...     query_id="q1",
        ...     signal_type="quality_score",
        ...     payload={"score": 0.85}
        ... )
        >>>
        >>> # From Arbiter evaluation
        >>> arbiter_score = await arbiter.evaluate(query, response)
        >>> event = FeedbackEvent(
        ...     query_id=decision.query_id,
        ...     signal_type="quality_score",
        ...     payload={"score": arbiter_score}
        ... )
    """

    @property
    def signal_type(self) -> str:
        return "quality_score"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Pass through quality score as reward.

        Args:
            event: FeedbackEvent with payload {"score": float}

        Returns:
            RewardMapping with score clamped to 0.0-1.0
        """
        score = event.payload.get("score", 0.5)
        clamped = max(0.0, min(1.0, float(score)))
        return RewardMapping(reward=clamped, confidence=1.0)


class CompletionTimeAdapter(FeedbackAdapter):
    """User session completion time adapter.

    Converts task completion time to reward. Faster completion indicates
    the model's response was helpful and actionable.

    Payload:
        {"seconds": float}

    Note:
        This is a proxy metric - faster isn't always better (user might
        give up). Use with lower confidence.
    """

    def __init__(self, target_seconds: float = 60.0, max_seconds: float = 300.0):
        """Initialize completion time adapter.

        Args:
            target_seconds: Target completion time (reward=0.5 at this time)
            max_seconds: Maximum time before reward=0.0
        """
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds

    @property
    def signal_type(self) -> str:
        return "completion_time"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert completion time to reward.

        Reward is 1.0 at 0 seconds, decreasing linearly to 0.0 at max_seconds.
        """
        seconds = event.payload.get("seconds", self.target_seconds)

        if seconds <= 0:
            reward = 1.0
        elif seconds >= self.max_seconds:
            reward = 0.0
        else:
            reward = 1.0 - (seconds / self.max_seconds)

        return RewardMapping(
            reward=max(0.0, min(1.0, reward)),
            confidence=0.7,  # Time is a proxy metric
        )


class RegenerationAdapter(FeedbackAdapter):
    """User regenerated response adapter (implicit negative signal).

    When a user clicks "regenerate" or "try again", it indicates
    dissatisfaction with the current response. This is a strong
    implicit signal that the model underperformed.

    Signal Interpretation:
        - regenerated=True: User was unsatisfied (reward=0.0)
        - regenerated=False: User accepted response (reward=1.0)
        - accepted_after_regeneration: Partial success (reward=0.3)

    Confidence:
        Uses 0.8 confidence because regeneration is implicit (user didn't
        explicitly rate the response). This causes softer bandit updates
        than explicit thumbs down.

    Payload:
        {"regenerated": bool, "accepted_after_regeneration": bool (optional)}

    Example:
        >>> adapter = RegenerationAdapter()
        >>> # User regenerated (dissatisfied)
        >>> event = FeedbackEvent(
        ...     query_id="q1",
        ...     signal_type="regeneration",
        ...     payload={"regenerated": True}
        ... )
        >>> reward = adapter.to_reward(event)
        >>> print(reward.reward, reward.confidence)  # 0.0, 0.8
        >>>
        >>> # User accepted regenerated response
        >>> event = FeedbackEvent(
        ...     query_id="q2",
        ...     signal_type="regeneration",
        ...     payload={"regenerated": True, "accepted_after_regeneration": True}
        ... )
        >>> reward = adapter.to_reward(event)
        >>> print(reward.reward, reward.confidence)  # 0.3, 0.8
    """

    def __init__(
        self,
        regeneration_reward: float = 0.0,
        acceptance_reward: float = 1.0,
        partial_acceptance_reward: float = 0.3,
        confidence: float = 0.8,
    ):
        """Initialize regeneration adapter.

        Args:
            regeneration_reward: Reward when user regenerated (default 0.0)
            acceptance_reward: Reward when user accepted without regenerating (default 1.0)
            partial_acceptance_reward: Reward when user accepted after regeneration (default 0.3)
            confidence: Confidence level for implicit signal (default 0.8)
        """
        self.regeneration_reward = regeneration_reward
        self.acceptance_reward = acceptance_reward
        self.partial_acceptance_reward = partial_acceptance_reward
        self.confidence = confidence

    @property
    def signal_type(self) -> str:
        return "regeneration"

    def to_reward(self, event: FeedbackEvent) -> RewardMapping:
        """Convert regeneration signal to reward.

        Returns:
            RewardMapping with appropriate reward based on regeneration state
        """
        regenerated = event.payload.get("regenerated", False)
        accepted_after = event.payload.get("accepted_after_regeneration", False)

        if not regenerated:
            # User accepted without regenerating
            return RewardMapping(
                reward=self.acceptance_reward, confidence=self.confidence
            )
        elif accepted_after:
            # User regenerated but eventually accepted
            return RewardMapping(
                reward=self.partial_acceptance_reward, confidence=self.confidence
            )
        else:
            # User regenerated (implicit negative)
            return RewardMapping(
                reward=self.regeneration_reward, confidence=self.confidence
            )
