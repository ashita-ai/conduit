"""State persistence interface for bandit algorithms.

This module provides abstract interfaces and data structures for persisting
bandit algorithm state across server restarts. Supports serialization of
numpy arrays and complex state structures to JSON for PostgreSQL storage.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
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

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1

    model_config = {"arbitrary_types_allowed": True}


class HybridRouterState(BaseModel):
    """Serializable state for HybridRouter."""

    query_count: int = 0
    current_phase: RouterPhase = RouterPhase.UCB1
    transition_threshold: int = 2000

    # Embedded bandit states
    ucb1_state: BanditState | None = None
    linucb_state: BanditState | None = None

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = 1


class StateStore(ABC):
    """Abstract interface for persisting bandit state.

    Implementations must handle:
    - Saving and loading BanditState
    - Saving and loading HybridRouterState
    - Atomic updates (no partial state corruption)
    - Version compatibility checks
    """

    @abstractmethod
    async def save_bandit_state(
        self, router_id: str, bandit_id: str, state: BanditState
    ) -> None:
        """Save bandit algorithm state.

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit (e.g., "ucb1", "linucb")
            state: Bandit state to persist

        Raises:
            StateStoreError: If save fails
        """
        pass

    @abstractmethod
    async def load_bandit_state(
        self, router_id: str, bandit_id: str
    ) -> BanditState | None:
        """Load bandit algorithm state.

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit

        Returns:
            BanditState if found, None otherwise

        Raises:
            StateStoreError: If load fails (not for missing state)
        """
        pass

    @abstractmethod
    async def save_hybrid_router_state(
        self, router_id: str, state: HybridRouterState
    ) -> None:
        """Save HybridRouter state including embedded bandit states.

        Args:
            router_id: Unique identifier for the router instance
            state: HybridRouter state to persist

        Raises:
            StateStoreError: If save fails
        """
        pass

    @abstractmethod
    async def load_hybrid_router_state(
        self, router_id: str
    ) -> HybridRouterState | None:
        """Load HybridRouter state.

        Args:
            router_id: Unique identifier for the router instance

        Returns:
            HybridRouterState if found, None otherwise

        Raises:
            StateStoreError: If load fails (not for missing state)
        """
        pass

    @abstractmethod
    async def delete_state(self, router_id: str) -> None:
        """Delete all state for a router instance.

        Args:
            router_id: Unique identifier for the router instance
        """
        pass

    @abstractmethod
    async def list_router_ids(self) -> list[str]:
        """List all router IDs with persisted state.

        Returns:
            List of router IDs
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
    return arr.tolist()


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
