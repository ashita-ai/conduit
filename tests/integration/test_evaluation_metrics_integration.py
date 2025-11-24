"""Integration tests for evaluation metrics database queries.

These tests require:
- DATABASE_URL environment variable (postgresql://...)
- Database schema migrated (run alembic upgrade head)
- evaluation_metrics table exists
"""

import os
import pytest
from uuid import uuid4

from conduit.core.database import Database


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


@pytest.mark.asyncio
async def test_fetch_latest_evaluation_metrics_empty_table(db):
    """Test fetching metrics when table is empty."""
    metrics = await db.fetch_latest_evaluation_metrics(time_window="last_hour")

    # Should return dict with None values
    assert isinstance(metrics, dict)
    assert metrics["regret_oracle"] is None
    assert metrics["regret_random"] is None
    assert metrics["quality_trend"] is None
    assert metrics["cost_efficiency"] is None
    assert metrics["convergence_rate"] is None


@pytest.mark.asyncio
async def test_fetch_latest_evaluation_metrics_with_data(db):
    """Test fetching metrics when data exists."""
    # Insert test data into evaluation_metrics table
    async with db.pool.acquire() as conn:
        # Clean up any existing test data first
        await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test\": true}'")

        # Insert regret vs oracle
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-regret-oracle-{uuid4()}",
            "regret_vs_oracle",
            0.05,
            "last_hour",
            '{"test": true}',
        )

        # Insert regret vs random
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-regret-random-{uuid4()}",
            "regret_vs_random",
            0.3,
            "last_hour",
            '{"test": true}',
        )

        # Insert cost efficiency
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-cost-efficiency-{uuid4()}",
            "cost_efficiency",
            42.5,
            "last_hour",
            '{"test": true}',
        )

    try:
        # Fetch metrics
        metrics = await db.fetch_latest_evaluation_metrics(time_window="last_hour")

        # Verify results
        assert metrics["regret_oracle"] == 0.05
        assert metrics["regret_random"] == 0.3
        assert metrics["cost_efficiency"] == 42.5
        # These weren't inserted, so should be None
        assert metrics["quality_trend"] is None
        assert metrics["convergence_rate"] is None

    finally:
        # Clean up test data
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test\": true}'")


@pytest.mark.asyncio
async def test_fetch_latest_evaluation_metrics_multiple_timestamps(db):
    """Test that only latest value is returned when multiple exist."""
    import asyncio

    async with db.pool.acquire() as conn:
        # Clean up first
        await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test_multi\": true}'")

        # Insert older value
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW() - INTERVAL '1 hour')
            """,
            f"test-old-{uuid4()}",
            "regret_vs_oracle",
            0.10,  # Old value
            "last_hour",
            '{"test_multi": true}',
        )

        # Small delay to ensure different timestamps
        await asyncio.sleep(0.1)

        # Insert newer value
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-new-{uuid4()}",
            "regret_vs_oracle",
            0.05,  # New value (should be returned)
            "last_hour",
            '{"test_multi": true}',
        )

    try:
        metrics = await db.fetch_latest_evaluation_metrics(time_window="last_hour")

        # Should return the newer value (0.05), not older (0.10)
        assert metrics["regret_oracle"] == 0.05

    finally:
        # Clean up
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test_multi\": true}'")


@pytest.mark.asyncio
async def test_fetch_latest_evaluation_metrics_different_time_windows(db):
    """Test that time_window parameter filters correctly."""
    async with db.pool.acquire() as conn:
        # Clean up first
        await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test_window\": true}'")

        # Insert data for last_hour
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-hour-{uuid4()}",
            "regret_vs_oracle",
            0.05,
            "last_hour",
            '{"test_window": true}',
        )

        # Insert data for last_day
        await conn.execute(
            """
            INSERT INTO evaluation_metrics (id, metric_name, metric_value, time_window, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            f"test-day-{uuid4()}",
            "regret_vs_oracle",
            0.08,
            "last_day",
            '{"test_window": true}',
        )

    try:
        # Fetch last_hour metrics
        hour_metrics = await db.fetch_latest_evaluation_metrics(time_window="last_hour")
        assert hour_metrics["regret_oracle"] == 0.05

        # Fetch last_day metrics
        day_metrics = await db.fetch_latest_evaluation_metrics(time_window="last_day")
        assert day_metrics["regret_oracle"] == 0.08

    finally:
        # Clean up
        async with db.pool.acquire() as conn:
            await conn.execute("DELETE FROM evaluation_metrics WHERE metadata @> '{\"test_window\": true}'")
