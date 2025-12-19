"""Base classes and interfaces for bandit algorithms.

All bandit algorithms in Conduit use QueryFeatures from conduit.core.models
for context, ensuring consistency across the routing system.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field, computed_field

from conduit.core.config import load_feature_dimensions
from conduit.core.models import QueryFeatures
from conduit.core.reward_calculation import calculate_composite_reward

if TYPE_CHECKING:
    from conduit.core.state_store import BanditState


class ModelArm(BaseModel):
    """Represents a model (arm) in the multi-armed bandit.

    Attributes:
        model_id: Unique identifier (e.g., "openai:gpt-4o-mini")
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model_name: Model name within provider
        cost_per_input_token: Cost in USD per 1K input tokens
        cost_per_output_token: Cost in USD per 1K output tokens
        expected_quality: Prior estimate of quality (0-1 scale)
        metadata: Additional model characteristics
    """

    model_id: str
    provider: str
    model_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    expected_quality: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_name(self) -> str:
        """Get full model name for PydanticAI."""
        return f"{self.provider}:{self.model_name}"


class BanditFeedback(BaseModel):
    """Feedback from executing a model selection.

    Attributes:
        model_id: Which model was selected
        cost: Actual cost incurred (USD)
        quality_score: Quality score from evaluation (0-1 scale)
        latency: Response latency in seconds
        success: Whether execution succeeded
        confidence: How confident we are in this feedback (0-1 scale).
            Used for weighted bandit updates. 1.0 = full weight,
            0.5 = half weight (softer update). Default: 1.0
        metadata: Additional feedback data (token counts, etc.)

    Multi-Objective Reward Function (Phase 3):
        Composite reward combines quality, cost, and latency using configurable weights.
        Default weights: 70% quality, 20% cost, 10% latency
        All metrics normalized to [0, 1] range where higher is better.

    Confidence-Weighted Updates:
        When confidence < 1.0, bandit algorithms apply softer updates:
        - Thompson Sampling: Partial observation (alpha += confidence * reward)
        - LinUCB: Weighted matrix updates (A += confidence * x @ x.T)
        Use lower confidence for implicit signals (regeneration, time-based).
    """

    model_id: str
    cost: float = Field(..., ge=0.0)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    latency: float = Field(..., ge=0.0)
    success: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def calculate_reward(
        self,
        quality_weight: float | None = None,
        cost_weight: float | None = None,
        latency_weight: float | None = None,
    ) -> float:
        """Calculate composite reward from quality, cost, and latency.

        Delegates to conduit.core.reward_calculation.calculate_composite_reward()
        for the actual calculation. See that module for implementation details.

        Args:
            quality_weight: Weight for quality component (default: 0.70)
            cost_weight: Weight for cost component (default: 0.20)
            latency_weight: Weight for latency component (default: 0.10)

        Returns:
            Composite reward in [0, 1] range

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> reward = feedback.calculate_reward()
            >>> print(f"{reward:.3f}")  # ~0.90 (high quality dominates)
            0.903
        """
        return calculate_composite_reward(
            quality=self.quality_score,
            cost=self.cost,
            latency=self.latency,
            quality_weight=quality_weight,
            cost_weight=cost_weight,
            latency_weight=latency_weight,
        )


class BanditAlgorithm(ABC):
    """Abstract base class for multi-armed bandit algorithms.

    All bandit algorithms must implement:
    1. select_arm: Choose which model to use for a query
    2. update: Update internal state with feedback
    3. reset: Reset algorithm state
    4. to_state: Serialize state for persistence
    5. from_state: Restore state from persistence

    Contract Guarantees:
        - Thread Safety: All implementations MUST be thread-safe for async operations
        - State Invariants: After update(), total_queries must increment by 1
        - Error Handling: All methods must handle invalid inputs gracefully
        - Persistence: to_state() must capture ALL state needed for from_state()
        - Idempotency: reset() followed by from_state(initial_state) restores initial state

    State Management:
        - All implementations maintain total_queries counter
        - State serialization must be lossless (from_state(to_state()) = identity)
        - from_state() validates compatibility with algorithm configuration
        - reset() clears all learned state but preserves arms configuration

    Error Handling:
        - select_arm() never fails (may use random selection as fallback)
        - update() validates feedback before state modification
        - from_state() raises ValueError for incompatible state
        - Async methods handle cancellation gracefully

    Performance Design Goals (not enforced):
        - select_arm() should complete quickly (typically <100ms)
        - update() should complete quickly (typically <50ms)
        - Actual performance depends on deployment environment and load

    Uses Conduit's QueryFeatures for context instead of custom data structures.
    """

    def __init__(self, name: str, arms: list[ModelArm]) -> None:
        """Initialize bandit algorithm.

        Args:
            name: Algorithm name for identification
            arms: List of available model arms
        """
        self.name = name
        self.arms = {arm.model_id: arm for arm in arms}
        self.arm_list = arms
        self.n_arms = len(arms)
        self.total_queries = 0

    @abstractmethod
    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select which model arm to pull for this query.

        Contract Guarantees:
            - MUST always return a valid arm from self.arms
            - MUST NOT raise exceptions (use fallback selection if needed)
            - MUST be thread-safe for concurrent calls
            - Result is deterministic for same features and state

        Performance:
            - Designed to be fast (typically <100ms in normal conditions)
            - Actual performance depends on deployment environment

        State Modifications:
            - MAY increment internal counters (implementation-specific)
            - MUST NOT modify features parameter

        Args:
            features: Query features from QueryAnalyzer

        Returns:
            Selected model arm (always one of self.arms values)

        Example:
            >>> features = QueryFeatures(
            ...     embedding=[0.1] * 384,
            ...     token_count=10,
            ...     complexity_score=0.5,
            ...     domain="general",            ... )
            >>> arm = await algorithm.select_arm(features)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
            >>> assert arm in algorithm.arms.values()  # Guaranteed
        """
        pass

    @abstractmethod
    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update algorithm state with feedback from arm pull.

        Contract Guarantees:
            - MUST increment self.total_queries by 1
            - MUST validate feedback.model_id is in self.arms
            - MUST be thread-safe for concurrent updates
            - MUST handle invalid feedback gracefully (log warning, skip update)

        Performance:
            - Designed to be fast (typically <50ms in normal conditions)
            - Actual performance depends on deployment environment

        State Modifications:
            - Updates algorithm-specific state (matrices, counters, distributions)
            - Increments total_queries counter
            - MAY update per-arm statistics

        Validation:
            - feedback.model_id must exist in self.arms (raises ValueError if not)
            - feedback.quality_score must be in [0, 1] (Pydantic validation)
            - feedback.cost must be >= 0 (Pydantic validation)
            - feedback.latency must be >= 0 (Pydantic validation)

        Args:
            feedback: Feedback from model execution
            features: Original query features

        Raises:
            ValueError: If feedback.model_id not in self.arms

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> queries_before = algorithm.total_queries
            >>> await algorithm.update(feedback, features)
            >>> assert algorithm.total_queries == queries_before + 1  # Guaranteed
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset algorithm to initial state.

        Contract Guarantees:
            - MUST set total_queries to 0
            - MUST reset all learned parameters to initial values
            - MUST preserve arms configuration (self.arms unchanged)
            - MUST be idempotent (reset twice = reset once)
            - Result state equals newly constructed algorithm state

        State Modifications:
            - Clears all learned parameters and history
            - Resets total_queries to 0
            - Reinitializes algorithm-specific state (matrices, distributions, etc.)
            - Does NOT modify self.arms or self.arm_list

        Use Cases:
            - Running multiple independent experiments
            - Testing algorithm behavior from clean state
            - Resetting after configuration changes

        Example:
            >>> algorithm.total_queries = 1000
            >>> algorithm.reset()
            >>> assert algorithm.total_queries == 0  # Guaranteed
            >>> assert len(algorithm.arms) == initial_arm_count  # Arms preserved
        """
        pass

    @abstractmethod
    def to_state(self) -> "BanditState":
        """Serialize algorithm state for persistence.

        Contract Guarantees:
            - MUST capture ALL state needed for complete restoration
            - MUST be lossless (from_state(to_state()) restores exact state)
            - Result MUST be JSON-serializable via BanditState.model_dump()
            - MUST include algorithm identifier for validation

        Performance:
            - Designed to be fast (typically <1s for typical state sizes)
            - Actual performance depends on state size and deployment environment

        State Captured:
            - Algorithm name and configuration
            - Total queries counter
            - All learned parameters (matrices, vectors, distributions)
            - Per-arm statistics and history
            - Algorithm-specific metadata

        Serialization:
            Converts all internal state (matrices, vectors, counters) to a
            BanditState object that can be serialized to JSON for database storage.

        Returns:
            BanditState object containing all algorithm state

        Example:
            >>> state = algorithm.to_state()
            >>> state.algorithm
            "linucb"
            >>> state.total_queries
            1000
            >>> len(state.A_matrices)  # LinUCB-specific
            5
            >>> json_str = state.model_dump_json()  # JSON-serializable
        """
        pass

    @abstractmethod
    def from_state(self, state: "BanditState") -> None:
        """Restore algorithm state from persisted data.

        Contract Guarantees:
            - MUST restore ALL state captured by to_state()
            - MUST validate state compatibility before applying
            - MUST be inverse of to_state() (lossless restoration)
            - MUST raise ValueError for incompatible state
            - State after restoration MUST equal state before serialization

        Validation Required:
            - Algorithm name matches (e.g., "linucb" state for LinUCB)
            - Feature dimensions match configuration
            - Number of arms matches current configuration
            - State structure matches expected format

        State Restoration:
            Loads internal state from a BanditState object, typically loaded
            from database storage after a server restart.

        Args:
            state: BanditState object with serialized state

        Raises:
            ValueError: If state is incompatible with algorithm configuration
                       (wrong algorithm, dimension mismatch, arm mismatch)

        Example:
            >>> state = await store.load_bandit_state("router-1", "linucb")
            >>> algorithm.from_state(state)
            >>> assert algorithm.total_queries == state.total_queries  # Restored
            >>> assert algorithm.name == state.algorithm  # Validated
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get algorithm statistics and state.

        Returns:
            Dictionary with algorithm-specific statistics

        Example:
            >>> stats = algorithm.get_stats()
            >>> print(stats["total_queries"])
            1000
        """
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "n_arms": self.n_arms,
        }

    def _extract_features(self, features: QueryFeatures) -> np.ndarray:
        """Extract feature vector from QueryFeatures.

        Combines embedding vector with metadata features:
        - embedding (384 dims)
        - token_count (1 dim, normalized by config token_count_normalization)
        - complexity_score (1 dim)
                Args:
            features: Query features object

        Returns:
            Feature vector as (d×1) numpy array

        Example:
            >>> features = QueryFeatures(
            ...     embedding=[0.1] * 384,
            ...     token_count=50,
            ...     complexity_score=0.5,
            ...     domain="general",            ... )
            >>> x = bandit._extract_features(features)
            >>> x.shape
            (386, 1)
        """
        # Load feature config for normalization constant
        feature_config = load_feature_dimensions()
        token_normalization = feature_config["token_count_normalization"]

        # Combine embedding with metadata
        feature_vector = np.array(
            features.embedding
            + [
                features.token_count / token_normalization,
                features.complexity_score,
            ]
        )

        # Reshape to column vector (d×1)
        return feature_vector.reshape(-1, 1)
