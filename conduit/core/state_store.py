"""State persistence interface for bandit algorithms.

This module provides abstract interfaces and data structures for persisting
bandit algorithm state across server restarts. Supports serialization of
numpy arrays and complex state structures to JSON for PostgreSQL storage.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class RouterPhase(str, Enum):
    """Phase of the HybridRouter lifecycle."""

    UCB1 = "ucb1"
    LINUCB = "linucb"


class BanditState(BaseModel):
    """Serializable state for a bandit algorithm.

    Supports numpy array serialization via custom encoders.
    All numpy arrays are stored as nested lists for JSON compatibility.
    """

    algorithm: str = Field(description="Algorithm type (ucb1, linucb, etc.)")
    arm_ids: list[str] = Field(description="List of arm IDs in order")

    # Common state across algorithms
    arm_pulls: dict[str, int] = Field(default_factory=dict)
    arm_successes: dict[str, int] = Field(default_factory=dict)
    total_queries: int = 0

    # UCB1-specific state
    mean_reward: dict[str, float] = Field(default_factory=dict)
    sum_reward: dict[str, float] = Field(default_factory=dict)
    explored_arms: list[str] = Field(default_factory=list)
    reward_history: list[dict[str, Any]] = Field(default_factory=list)

    # LinUCB-specific state (numpy arrays as nested lists)
    A_matrices: dict[str, list[list[float]]] = Field(
        default_factory=dict, description="A matrices per arm (d x d)"
    )
    b_vectors: dict[str, list[float]] = Field(
        default_factory=dict, description="b vectors per arm (d x 1 flattened)"
    )
    observation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Thompson Sampling state
    alpha_params: dict[str, float] = Field(
        default_factory=dict, description="Beta distribution alpha per arm"
    )
    beta_params: dict[str, float] = Field(
        default_factory=dict, description="Beta distribution beta per arm"
    )

    # Contextual Thompson Sampling state
    mu_vectors: dict[str, list[float]] = Field(
        default_factory=dict, description="Mean vectors per arm"
    )
    sigma_matrices: dict[str, list[list[float]]] = Field(
        default_factory=dict, description="Covariance matrices per arm"
    )

    # Epsilon-Greedy state
    epsilon: float | None = None

    # Oracle baseline state
    oracle_rewards: dict[str, float] = Field(
        default_factory=dict, description="Oracle rewards for query+arm pairs"
    )

    # Dueling Bandit state
    preference_weights: dict[str, list[float]] = Field(
        default_factory=dict, description="Preference weights per arm"
    )
    preference_counts: dict[str, int] = Field(
        default_factory=dict, description="Comparison counts per pair"
    )
    exploration_weight: float | None = None
    learning_rate: float | None = None

    # Algorithm hyperparameters (for restoration)
    alpha: float | None = Field(default=None, description="LinUCB exploration param")
    feature_dim: int | None = Field(default=None, description="Feature dimension")
    window_size: int | None = Field(
        default=None, description="Sliding window for non-stationarity"
    )

    # Embedding configuration (Phase 1 & 2: dimension safety)
    embedding_provider: str | None = Field(
        default=None,
        description="Embedding provider used (openai, cohere, fastembed, etc.)",
    )
    embedding_dimensions: int | None = Field(
        default=None, description="Raw embedding dimensions (384, 1024, 1536, etc.)"
    )
    pca_enabled: bool | None = Field(
        default=None, description="Whether PCA dimensionality reduction was used"
    )
    pca_dimensions: int | None = Field(
        default=None, description="PCA output dimensions if PCA enabled"
    )

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    model_config = {"arbitrary_types_allowed": True}


class HybridRouterState(BaseModel):
    """Serializable state for HybridRouter with configurable algorithms.

    Supports 4 algorithm combinations:
    - UCB1 → LinUCB (default, fast cold start)
    - Thompson Sampling → LinUCB (higher quality cold start)
    - UCB1 → Contextual Thompson Sampling
    - Thompson Sampling → Contextual Thompson Sampling (full Bayesian)

    Maintains backward compatibility with old ucb1_state/linucb_state fields.
    """

    query_count: int = 0
    current_phase: RouterPhase = RouterPhase.UCB1
    transition_threshold: int | float | None = Field(
        default=2000,
        description="Query threshold for phase transition (can be infinity)",
    )

    # Algorithm identifiers (new fields)
    phase1_algorithm: str | None = None  # "ucb1" or "thompson_sampling"
    phase2_algorithm: str | None = None  # "linucb" or "contextual_thompson_sampling"

    # Embedded bandit states (new fields)
    phase1_state: BanditState | None = None
    phase2_state: BanditState | None = None

    # Backward compatibility (deprecated, prefer phase1_state/phase2_state)
    ucb1_state: BanditState | None = None
    linucb_state: BanditState | None = None

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 2  # Increment version for new fields


class StateStore(ABC):
    """Abstract interface for persisting bandit state.

    INTERFACE CONTRACT:

    Guarantees:
        - Atomic Updates: save_*() operations are atomic (all-or-nothing)
        - Consistency: load_*() returns exactly what was last saved
        - Isolation: Operations on different router_ids are independent
        - Durability: Saved state survives process restarts

    Error Handling:
        - save_*(): Raises StateStoreError on failure, never silently fails
        - load_*(): Returns None for missing state, raises StateStoreError for errors
        - delete_state(): Idempotent (safe to call on non-existent state)

    Thread Safety:
        - All methods MUST be safe for concurrent async calls
        - Same router_id operations are serialized internally
        - Different router_ids can be accessed concurrently

    State Invariants:
        - router_id is immutable once created
        - version field in state objects must be preserved
        - updated_at is set by save operations

    Performance Design Goals:
        - save_*(): Typically completes in <50ms for normal state sizes
        - load_*(): Typically completes in <20ms for normal state sizes
        - Actual performance depends on implementation and infrastructure

    Implementations:
        - PostgresStateStore: Production-grade with connection pooling
        - InMemoryStateStore: Testing only (not persistent)

    Example:
        >>> store = PostgresStateStore(pool)
        >>> # Save state
        >>> await store.save_hybrid_router_state("router-1", state)
        >>> # Load state (returns same data)
        >>> loaded = await store.load_hybrid_router_state("router-1")
        >>> assert loaded.query_count == state.query_count
        >>> # Missing state returns None
        >>> missing = await store.load_hybrid_router_state("nonexistent")
        >>> assert missing is None
    """

    @abstractmethod
    async def save_bandit_state(
        self, router_id: str, bandit_id: str, state: BanditState
    ) -> None:
        """Save bandit algorithm state atomically.

        Contract Guarantees:
            - MUST be atomic (either fully saved or not at all)
            - MUST update state.updated_at timestamp
            - MUST be idempotent (safe to retry on failure)
            - MUST be thread-safe for concurrent calls

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit (e.g., "ucb1", "linucb")
            state: Bandit state to persist

        Raises:
            StateStoreError: If save fails (network, storage, serialization)

        Example:
            >>> await store.save_bandit_state("router-1", "linucb", state)
            >>> # Idempotent: safe to call again
            >>> await store.save_bandit_state("router-1", "linucb", state)
        """
        pass

    @abstractmethod
    async def load_bandit_state(
        self, router_id: str, bandit_id: str
    ) -> BanditState | None:
        """Load bandit algorithm state.

        Contract Guarantees:
            - MUST return None for non-existent state (not an error)
            - MUST return exactly what was last saved
            - MUST be thread-safe for concurrent calls
            - MUST NOT modify the stored state

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit

        Returns:
            BanditState if found, None if state doesn't exist

        Raises:
            StateStoreError: If load fails (network, storage, deserialization)
                            NOT raised for missing state (returns None instead)

        Example:
            >>> state = await store.load_bandit_state("router-1", "linucb")
            >>> if state is None:
            ...     print("No saved state, starting fresh")
            ... else:
            ...     bandit.from_state(state)
        """
        pass

    @abstractmethod
    async def save_hybrid_router_state(
        self, router_id: str, state: HybridRouterState
    ) -> None:
        """Save HybridRouter state including embedded bandit states.

        Contract Guarantees:
            - MUST be atomic (entire state saved or none)
            - MUST include embedded phase1_state and phase2_state
            - MUST update state.updated_at timestamp
            - MUST be thread-safe for concurrent calls

        Args:
            router_id: Unique identifier for the router instance
            state: HybridRouter state to persist (includes bandit states)

        Raises:
            StateStoreError: If save fails

        Example:
            >>> state = HybridRouterState(
            ...     query_count=1000,
            ...     current_phase=RouterPhase.LINUCB,
            ...     phase1_state=ucb1.to_state(),
            ...     phase2_state=linucb.to_state(),
            ... )
            >>> await store.save_hybrid_router_state("router-1", state)
        """
        pass

    @abstractmethod
    async def load_hybrid_router_state(
        self, router_id: str
    ) -> HybridRouterState | None:
        """Load HybridRouter state.

        Contract Guarantees:
            - MUST return None for non-existent state
            - MUST return exactly what was last saved
            - MUST include embedded bandit states if they were saved

        Args:
            router_id: Unique identifier for the router instance

        Returns:
            HybridRouterState if found, None if state doesn't exist

        Raises:
            StateStoreError: If load fails (not for missing state)

        Example:
            >>> state = await store.load_hybrid_router_state("router-1")
            >>> if state:
            ...     router.from_state(state)
        """
        pass

    @abstractmethod
    async def delete_state(self, router_id: str) -> None:
        """Delete all state for a router instance.

        Contract Guarantees:
            - MUST be idempotent (safe to call on non-existent state)
            - MUST delete bandit states AND hybrid router state
            - MUST NOT raise error if state doesn't exist

        Args:
            router_id: Unique identifier for the router instance

        Example:
            >>> await store.delete_state("router-1")
            >>> # Safe to call again
            >>> await store.delete_state("router-1")
        """
        pass

    @abstractmethod
    async def list_router_ids(self) -> list[str]:
        """List all router IDs with persisted state.

        Contract Guarantees:
            - MUST return empty list if no state exists
            - MUST include all router IDs that have any saved state

        Returns:
            List of router IDs (empty if none exist)

        Example:
            >>> ids = await store.list_router_ids()
            >>> print(f"Found {len(ids)} routers with saved state")
        """
        pass


class StateStoreError(Exception):
    """Error during state persistence operations."""

    pass


# Utility functions for numpy serialization


def numpy_to_list(arr: np.ndarray) -> list[Any]:
    """Convert numpy array to nested list for JSON serialization.

    Args:
        arr: Numpy array of any shape

    Returns:
        Nested list representation
    """
    result: list[Any] = arr.tolist()
    return result


def list_to_numpy(data: list[Any], dtype: type = float) -> np.ndarray:
    """Convert nested list back to numpy array.

    Args:
        data: Nested list from JSON
        dtype: Numpy dtype for the array

    Returns:
        Numpy array
    """
    return np.array(data, dtype=dtype)


def serialize_bandit_matrices(
    A: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> tuple[dict[str, list[list[float]]], dict[str, list[float]]]:
    """Serialize LinUCB A matrices and b vectors.

    Args:
        A: Dictionary of arm_id -> A matrix (d x d)
        b: Dictionary of arm_id -> b vector (d x 1)

    Returns:
        Tuple of (A_matrices as nested lists, b_vectors as lists)
    """
    A_serialized = {arm_id: numpy_to_list(mat) for arm_id, mat in A.items()}
    b_serialized = {arm_id: numpy_to_list(vec.flatten()) for arm_id, vec in b.items()}
    return A_serialized, b_serialized


def deserialize_bandit_matrices(
    A_data: dict[str, list[list[float]]], b_data: dict[str, list[float]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Deserialize LinUCB A matrices and b vectors.

    Args:
        A_data: Dictionary of arm_id -> A matrix as nested lists
        b_data: Dictionary of arm_id -> b vector as flat list

    Returns:
        Tuple of (A matrices as numpy arrays, b vectors as column vectors)
    """
    A = {arm_id: list_to_numpy(mat) for arm_id, mat in A_data.items()}
    b = {arm_id: list_to_numpy(vec).reshape(-1, 1) for arm_id, vec in b_data.items()}
    return A, b
