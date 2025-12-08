"""State Persistence Example: Save and restore bandit learning across restarts.

This example demonstrates how to persist bandit state to survive server restarts.
Uses an in-memory state store for demo; production would use PostgresStateStore.

Key concepts:
- BanditState: Serializable state for all bandit algorithms
- HybridRouterState: State for the hybrid UCB1 -> LinUCB router
- StateStore: Abstract interface for persistence backends
- to_state() / from_state(): Serialize/deserialize algorithms

Production usage:
    from conduit.core.postgres_state_store import PostgresStateStore
    store = PostgresStateStore(pool)  # asyncpg connection pool
    await router.save_state(store, "production-router")
"""

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

from conduit.core.models import Query
from conduit.core.state_store import (
    BanditState,
    HybridRouterState,
    StateStore,
)
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.embeddings.base import EmbeddingProvider
from conduit.engines.hybrid_router import HybridRouter


# =============================================================================
# Mock Embedding Provider (for demonstration without external API)
# =============================================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for demos that generates deterministic embeddings.

    Uses text hash to create consistent embeddings without external API calls.
    Production should use HuggingFace, OpenAI, or Cohere providers.
    """

    async def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        # Create consistent hash-based embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to 384-dim embedding (BGE-small dimension)
        embedding = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= 384:
                break
            # Convert hex pair to float in [-1, 1]
            val = (int(text_hash[i : i + 2], 16) / 255.0) * 2 - 1
            embedding.append(val)

        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)

        return embedding[:384]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        return [await self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return 384 (same as BGE-small)."""
        return 384

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "mock"


# =============================================================================
# In-Memory State Store (for demonstration)
# =============================================================================


class InMemoryStateStore(StateStore):
    """In-memory implementation of StateStore for testing and demos.

    Production deployments should use PostgresStateStore for durability.
    """

    def __init__(self) -> None:
        """Initialize empty in-memory store."""
        self._bandit_states: dict[str, dict[str, BanditState]] = {}
        self._hybrid_states: dict[str, HybridRouterState] = {}

    async def save_bandit_state(
        self, router_id: str, bandit_id: str, state: BanditState
    ) -> None:
        """Save bandit state to memory."""
        if router_id not in self._bandit_states:
            self._bandit_states[router_id] = {}
        state.updated_at = datetime.now(UTC)
        self._bandit_states[router_id][bandit_id] = state

    async def load_bandit_state(
        self, router_id: str, bandit_id: str
    ) -> BanditState | None:
        """Load bandit state from memory."""
        return self._bandit_states.get(router_id, {}).get(bandit_id)

    async def save_hybrid_router_state(
        self, router_id: str, state: HybridRouterState
    ) -> None:
        """Save hybrid router state to memory."""
        state.updated_at = datetime.now(UTC)
        self._hybrid_states[router_id] = state

    async def load_hybrid_router_state(
        self, router_id: str
    ) -> HybridRouterState | None:
        """Load hybrid router state from memory."""
        return self._hybrid_states.get(router_id)

    async def delete_state(self, router_id: str) -> None:
        """Delete all state for a router."""
        self._bandit_states.pop(router_id, None)
        self._hybrid_states.pop(router_id, None)

    async def list_router_ids(self) -> list[str]:
        """List all router IDs with saved state."""
        ids = set(self._bandit_states.keys()) | set(self._hybrid_states.keys())
        return sorted(ids)


# =============================================================================
# Simulation helpers
# =============================================================================


def simulate_feedback(model_id: str, query_num: int) -> BanditFeedback:
    """Simulate feedback from a model response.

    Creates realistic-ish feedback based on model characteristics.
    """
    import random

    # Different models have different characteristics
    model_profiles = {
        "gpt-4o-mini": {"quality": 0.88, "cost": 0.0001, "latency": 0.5},
        "gpt-4o": {"quality": 0.95, "cost": 0.005, "latency": 1.2},
        "claude-3-5-sonnet": {"quality": 0.94, "cost": 0.003, "latency": 0.9},
    }

    profile = model_profiles.get(
        model_id, {"quality": 0.85, "cost": 0.001, "latency": 0.7}
    )

    # Add some noise
    quality = min(1.0, max(0.0, profile["quality"] + random.gauss(0, 0.05)))
    cost = profile["cost"] * random.uniform(0.9, 1.1)
    latency = profile["latency"] * random.uniform(0.8, 1.2)

    return BanditFeedback(
        model_id=model_id,
        cost=cost,
        quality_score=quality,
        latency=latency,
    )


async def train_router(router: HybridRouter, num_queries: int) -> dict[str, Any]:
    """Train router with simulated queries and feedback."""
    sample_queries = [
        "Explain quantum computing",
        "Write a Python function to sort a list",
        "What is the capital of France?",
        "Summarize this research paper",
        "Debug this async code",
        "Compare REST vs GraphQL",
        "How do transformers work in NLP?",
        "Write a haiku about programming",
    ]

    model_selections: dict[str, int] = {}

    for i in range(num_queries):
        query_text = sample_queries[i % len(sample_queries)]
        query = Query(text=query_text)

        # Route query
        decision = await router.route(query)
        model_id = decision.selected_model

        # Track selections
        model_selections[model_id] = model_selections.get(model_id, 0) + 1

        # Simulate feedback
        feedback = simulate_feedback(model_id, i)
        await router.update(feedback, decision.features)

    return {
        "queries_processed": num_queries,
        "model_selections": model_selections,
        "current_phase": router.current_phase,
        "query_count": router.query_count,
    }


# =============================================================================
# Main demonstration
# =============================================================================


async def main() -> None:
    """Demonstrate state persistence across simulated restarts."""
    print("=" * 60)
    print("State Persistence Demo")
    print("=" * 60)

    # Initialize state store
    store = InMemoryStateStore()
    router_id = "demo-router"

    models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]

    # -------------------------------------------------------------------------
    # Phase 1: Initial training
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Initial Training ---")

    # Create mock analyzer (for demo - production uses real embeddings)
    mock_provider = MockEmbeddingProvider()
    mock_analyzer = QueryAnalyzer(embedding_provider=mock_provider)

    router1 = HybridRouter(
        models=models,
        switch_threshold=100,  # Switch from UCB1 to LinUCB after 100 queries
        analyzer=mock_analyzer,  # Use mock for demo
    )

    # Train for 50 queries (still in UCB1 phase)
    results1 = await train_router(router1, num_queries=50)
    print(f"Trained: {results1['queries_processed']} queries")
    print(f"Phase: {results1['current_phase']}")
    print(f"Model selections: {results1['model_selections']}")

    # Save state
    await router1.save_state(store, router_id)
    print("\nState saved to store")

    # Show saved state info
    state = router1.to_state()
    print(f"  - Query count: {state.query_count}")
    print(f"  - Phase: {state.current_phase.value}")
    print(f"  - UCB1 state present: {state.ucb1_state is not None}")
    print(f"  - LinUCB state present: {state.linucb_state is not None}")

    # -------------------------------------------------------------------------
    # Simulate server restart by creating new router
    # -------------------------------------------------------------------------
    print("\n--- Simulating Server Restart ---")
    del router1  # "Restart"

    # -------------------------------------------------------------------------
    # Phase 2: Resume from saved state
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Resume from Saved State ---")

    # Create new router (simulating new process after restart)
    mock_provider2 = MockEmbeddingProvider()
    mock_analyzer2 = QueryAnalyzer(embedding_provider=mock_provider2)

    router2 = HybridRouter(
        models=models,
        switch_threshold=100,
        analyzer=mock_analyzer2,
    )

    # Load saved state
    loaded = await router2.load_state(store, router_id)
    print(f"State loaded: {loaded}")
    print(f"Resumed query count: {router2.query_count}")
    print(f"Resumed phase: {router2.current_phase}")

    # Continue training (will transition to LinUCB during this phase)
    results2 = await train_router(router2, num_queries=100)
    print(f"\nTrained additional: {results2['queries_processed']} queries")
    print(f"Total queries: {router2.query_count}")
    print(f"Phase: {results2['current_phase']}")
    print(f"Model selections: {results2['model_selections']}")

    # Save updated state
    await router2.save_state(store, router_id)
    print("\nUpdated state saved")

    # -------------------------------------------------------------------------
    # Phase 3: Final verification
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Final Verification ---")

    mock_provider3 = MockEmbeddingProvider()
    mock_analyzer3 = QueryAnalyzer(embedding_provider=mock_provider3)

    router3 = HybridRouter(
        models=models,
        switch_threshold=100,
        analyzer=mock_analyzer3,
    )
    await router3.load_state(store, router_id)

    print(f"Final query count: {router3.query_count}")
    print(f"Final phase: {router3.current_phase}")

    # Show bandit statistics
    stats = router3.get_stats()
    print(f"\nBandit statistics:")
    print(f"  - Total queries: {stats['total_queries']}")
    print(f"  - Arm pulls: {stats.get('arm_pulls', {})}")

    # -------------------------------------------------------------------------
    # Demonstrate state serialization format
    # -------------------------------------------------------------------------
    print("\n--- State Serialization Format ---")

    state = router3.to_state()

    # Show what gets serialized (truncated for readability)
    state_dict = state.model_dump()

    # Show top-level structure
    print("HybridRouterState fields:")
    for key in ["query_count", "current_phase", "transition_threshold"]:
        print(f"  - {key}: {state_dict.get(key)}")

    if state_dict.get("ucb1_state"):
        ucb1 = state_dict["ucb1_state"]
        print("\nUCB1 BanditState fields:")
        print(f"  - algorithm: {ucb1.get('algorithm')}")
        print(f"  - arm_ids: {ucb1.get('arm_ids')}")
        print(f"  - total_queries: {ucb1.get('total_queries')}")
        print(f"  - arm_pulls: {ucb1.get('arm_pulls')}")

    if state_dict.get("linucb_state"):
        linucb = state_dict["linucb_state"]
        print("\nLinUCB BanditState fields:")
        print(f"  - algorithm: {linucb.get('algorithm')}")
        print(f"  - feature_dim: {linucb.get('feature_dim')}")
        print(f"  - alpha: {linucb.get('alpha')}")
        print(f"  - A_matrices count: {len(linucb.get('A_matrices', {}))}")
        print(f"  - b_vectors count: {len(linucb.get('b_vectors', {}))}")

    print("\n" + "=" * 60)
    print("Demo complete! State persisted and restored successfully.")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Production usage note
    # -------------------------------------------------------------------------
    print(
        """
Production Usage:
-----------------
Replace InMemoryStateStore with PostgresStateStore:

    import asyncpg
    from conduit.core.postgres_state_store import PostgresStateStore

    pool = await asyncpg.create_pool(DATABASE_URL)
    store = PostgresStateStore(pool)

    router = HybridRouter(models=["gpt-4o-mini", "gpt-4o"])

    # On startup: Load existing state
    if await router.load_state(store, "production-router"):
        print("Resumed from saved state")
    else:
        print("Starting fresh")

    # Periodically or on shutdown: Save state
    await router.save_state(store, "production-router")
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
