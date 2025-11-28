"""Integration tests for automatic state persistence.

Tests that Router automatically saves and loads state without manual intervention.

These tests require:
- DATABASE_URL environment variable (postgresql://...)
- Database schema migrated (run alembic upgrade head)
"""

import os
from uuid import uuid4

import pytest

from conduit.core.database import Database
from conduit.core.models import Query
from conduit.core.postgres_state_store import PostgresStateStore
from conduit.engines.router import Router


# Skip all tests if DATABASE_URL not available
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not configured",
)


@pytest.fixture
async def db():
    """Create connected database instance."""
    database = Database()
    await database.connect()
    yield database
    await database.disconnect()


@pytest.fixture
async def state_store(db):
    """Create state store from database."""
    return PostgresStateStore(db.pool)


@pytest.mark.asyncio
async def test_auto_load_on_first_route(state_store):
    """Test that router auto-loads saved state on first route() call."""
    router_id = f"test-auto-load-{uuid4()}"

    # Create first router and route queries
    router1 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
        checkpoint_interval=10,
    )

    # Route 5 queries
    for i in range(5):
        query = Query(text=f"Query {i}")
        decision = await router1.route(query)

        # Provide feedback to update weights
        await router1.update(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.8,
            latency=1.0,
            features=decision.features,
        )

    initial_query_count = router1.hybrid_router.query_count
    assert initial_query_count == 5

    # Save final state
    await router1.close()

    # Create second router - should auto-load on first route()
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    # Before first route, state not loaded yet
    assert not router2._state_loaded

    # First route should trigger auto-load
    query = Query(text="First query after restart")
    await router2.route(query)

    # State should be loaded now
    assert router2._state_loaded
    assert router2.hybrid_router.query_count == 6  # 5 + 1

    await router2.close()


@pytest.mark.asyncio
async def test_save_after_every_update(state_store):
    """Test that state is saved after every update() call."""
    router_id = f"test-save-update-{uuid4()}"

    router = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    # Route and update
    query = Query(text="Test query")
    decision = await router.route(query)

    await router.update(
        model_id=decision.selected_model,
        cost=0.001,
        quality_score=0.9,
        latency=1.0,
        features=decision.features,
    )

    # State should be saved immediately after update
    # Verify by loading in new router
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    await router2._load_initial_state()

    assert router2.hybrid_router.query_count == 1
    assert router2._state_loaded

    await router.close()
    await router2.close()


@pytest.mark.asyncio
async def test_periodic_checkpoint(state_store):
    """Test that state is saved periodically based on checkpoint_interval."""
    router_id = f"test-checkpoint-{uuid4()}"

    router = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
        checkpoint_interval=3,  # Save every 3 queries
    )

    # Route 3 queries (should trigger checkpoint on 3rd)
    for i in range(3):
        query = Query(text=f"Query {i}")
        await router.route(query)

    # Query count should be 3 (checkpoint should have triggered)
    assert router.hybrid_router.query_count == 3

    # Verify checkpoint was saved
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    await router2._load_initial_state()
    assert router2.hybrid_router.query_count == 3

    await router.close()
    await router2.close()


@pytest.mark.asyncio
async def test_save_on_close(state_store):
    """Test that final state is saved on close()."""
    router_id = f"test-close-{uuid4()}"

    router = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
        checkpoint_interval=100,  # High interval so only close() saves
    )

    # Route queries but don't hit checkpoint
    for i in range(5):
        query = Query(text=f"Query {i}")
        await router.route(query)

    assert router.hybrid_router.query_count == 5

    # Close should save final state
    await router.close()

    # Verify final state was saved
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    await router2._load_initial_state()
    assert router2.hybrid_router.query_count == 5

    await router2.close()


@pytest.mark.asyncio
async def test_no_persistence_when_disabled(state_store):
    """Test that persistence doesn't occur when auto_persist=False."""
    router_id = f"test-no-persist-{uuid4()}"

    router = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=False,  # Disabled
    )

    # Route queries
    for i in range(5):
        query = Query(text=f"Query {i}")
        await router.route(query)

    assert router.hybrid_router.query_count == 5

    await router.close()

    # New router should NOT load state (nothing was saved)
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=False,
    )

    await router2._load_initial_state()
    assert router2.hybrid_router.query_count == 0  # Fresh start

    await router2.close()


@pytest.mark.asyncio
async def test_crash_recovery_simulation(state_store):
    """Simulate crash recovery by not calling close()."""
    router_id = f"test-crash-{uuid4()}"

    # Simulate normal operation
    router1 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    for i in range(10):
        query = Query(text=f"Query {i}")
        decision = await router1.route(query)

        # Update after each query (triggers auto-save)
        await router1.update(
            model_id=decision.selected_model,
            cost=0.001,
            quality_score=0.75,
            latency=1.0,
            features=decision.features,
        )

    assert router1.hybrid_router.query_count == 10

    # Simulate crash (no close() call)
    del router1

    # Simulate restart - should recover from last save
    router2 = Router(
        models=["gpt-4o-mini", "gpt-4o"],
        state_store=state_store,
        router_id=router_id,
        auto_persist=True,
    )

    # First route triggers auto-load
    query = Query(text="Query after crash")
    await router2.route(query)

    # Should have recovered all 10 queries from before crash
    assert router2.hybrid_router.query_count == 11  # 10 + 1

    await router2.close()
