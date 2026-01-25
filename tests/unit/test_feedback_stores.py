"""Unit tests for feedback stores.

Tests the RedisFeedbackStore and PostgresFeedbackStore implementations
using mocked backends.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.feedback.models import PendingQuery
from conduit.feedback.stores import (
    InMemoryFeedbackStore,
    PostgresFeedbackStore,
    RedisFeedbackStore,
)


@pytest.fixture
def sample_pending_query():
    """Create a sample pending query for tests."""
    return PendingQuery(
        query_id="query-123",
        model_id="gpt-4o-mini",
        features={"embedding": [0.1] * 10, "token_count": 50},
        session_id="session-456",
        cost=0.001,
        latency=0.5,
        ttl_seconds=3600,
        created_at=datetime.now(timezone.utc),
    )


class TestRedisFeedbackStore:
    """Tests for RedisFeedbackStore."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        redis.setex = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.getdel = AsyncMock(return_value=None)
        redis.delete = AsyncMock()
        redis.ttl = AsyncMock(return_value=3600)
        redis.set = AsyncMock()
        redis.scan = AsyncMock(return_value=(0, []))
        return redis

    @pytest.fixture
    def store(self, mock_redis):
        """Create a RedisFeedbackStore with mocked Redis."""
        return RedisFeedbackStore(
            redis=mock_redis, key_prefix="test:feedback", default_ttl=3600
        )

    @pytest.mark.asyncio
    async def test_save_pending(self, store, mock_redis, sample_pending_query):
        """Test saving a pending query to Redis."""
        await store.save_pending(sample_pending_query)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "test:feedback:query-123"
        assert call_args[0][1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_save_pending_uses_query_ttl(
        self, store, mock_redis, sample_pending_query
    ):
        """Test that save_pending uses query's TTL if provided."""
        sample_pending_query.ttl_seconds = 1800

        await store.save_pending(sample_pending_query)

        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 1800  # Query's TTL

    @pytest.mark.asyncio
    async def test_get_pending_found(self, store, mock_redis, sample_pending_query):
        """Test retrieving an existing pending query."""
        mock_redis.get.return_value = sample_pending_query.model_dump_json().encode()

        result = await store.get_pending("query-123")

        mock_redis.get.assert_called_once_with("test:feedback:query-123")
        assert result is not None
        assert result.query_id == "query-123"
        assert result.model_id == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_pending_not_found(self, store, mock_redis):
        """Test retrieving a non-existent pending query."""
        mock_redis.get.return_value = None

        result = await store.get_pending("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_invalid_json(self, store, mock_redis):
        """Test handling of invalid JSON in Redis."""
        mock_redis.get.return_value = b"invalid json"

        result = await store.get_pending("query-123")

        assert result is None  # Should return None, not raise

    @pytest.mark.asyncio
    async def test_get_and_delete_pending_uses_getdel(
        self, store, mock_redis, sample_pending_query
    ):
        """Test atomic get and delete using GETDEL command."""
        mock_redis.getdel.return_value = sample_pending_query.model_dump_json().encode()

        result = await store.get_and_delete_pending("query-123")

        mock_redis.getdel.assert_called_once_with("test:feedback:query-123")
        assert result is not None
        assert result.query_id == "query-123"

    @pytest.mark.asyncio
    async def test_get_and_delete_pending_fallback(
        self, store, mock_redis, sample_pending_query
    ):
        """Test fallback when GETDEL is not available."""
        # Simulate older Redis without getdel
        mock_redis.getdel.side_effect = AttributeError("no getdel")
        mock_redis.get.return_value = sample_pending_query.model_dump_json().encode()

        result = await store.get_and_delete_pending("query-123")

        mock_redis.get.assert_called_once()
        mock_redis.delete.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_and_delete_pending_not_found(self, store, mock_redis):
        """Test atomic get and delete when query doesn't exist."""
        mock_redis.getdel.return_value = None

        result = await store.get_and_delete_pending("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_pending_success(
        self, store, mock_redis, sample_pending_query
    ):
        """Test updating cost and latency for a pending query."""
        mock_redis.get.return_value = sample_pending_query.model_dump_json().encode()
        mock_redis.ttl.return_value = 1800  # Remaining TTL

        result = await store.update_pending("query-123", cost=0.002, latency=0.8)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_pending_not_found(self, store, mock_redis):
        """Test updating a non-existent pending query."""
        mock_redis.get.return_value = None

        result = await store.update_pending("nonexistent", cost=0.001)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_pending_expired_ttl(
        self, store, mock_redis, sample_pending_query
    ):
        """Test updating when TTL has expired (but key still exists)."""
        mock_redis.get.return_value = sample_pending_query.model_dump_json().encode()
        mock_redis.ttl.return_value = -1  # Expired or no TTL

        result = await store.update_pending("query-123", cost=0.002)

        assert result is True
        # Should use set() instead of setex() when no TTL
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_pending(self, store, mock_redis):
        """Test deleting a pending query."""
        await store.delete_pending("query-123")

        mock_redis.delete.assert_called_once_with("test:feedback:query-123")

    @pytest.mark.asyncio
    async def test_key_generation(self, store):
        """Test Redis key generation."""
        assert store._key("test-id") == "test:feedback:test-id"


class TestPostgresFeedbackStore:
    """Tests for PostgresFeedbackStore."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock connection with async methods."""
        conn = MagicMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.fixture
    def mock_pool(self, mock_connection):
        """Create a mock asyncpg connection pool.

        asyncpg pool.acquire() returns a context manager that yields the connection.
        We need to mock this properly for async with statements.
        """
        # Create async context manager that returns the connection
        async_ctx_manager = AsyncMock()
        async_ctx_manager.__aenter__ = AsyncMock(return_value=mock_connection)
        async_ctx_manager.__aexit__ = AsyncMock(return_value=None)

        pool = MagicMock()
        pool.acquire = MagicMock(return_value=async_ctx_manager)
        return pool

    @pytest.fixture
    def store(self, mock_pool):
        """Create a PostgresFeedbackStore with mocked pool."""
        return PostgresFeedbackStore(pool=mock_pool, table_name="test_pending_queries")

    @pytest.mark.asyncio
    async def test_save_pending(
        self, store, mock_pool, mock_connection, sample_pending_query
    ):
        """Test saving a pending query to PostgreSQL."""
        await store.save_pending(sample_pending_query)

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        # Verify the SQL contains INSERT
        assert "INSERT INTO" in call_args[0][0]
        assert "test_pending_queries" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_pending_found(
        self, store, mock_pool, mock_connection, sample_pending_query
    ):
        """Test retrieving an existing pending query."""
        mock_connection.fetchrow.return_value = {
            "query_id": "query-123",
            "model_id": "gpt-4o-mini",
            "features": {"embedding": [0.1] * 10, "token_count": 50},
            "session_id": "session-456",
            "cost": 0.001,
            "latency": 0.5,
            "ttl_seconds": 3600,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        result = await store.get_pending("query-123")

        mock_connection.fetchrow.assert_called_once()
        assert result is not None
        assert result.query_id == "query-123"

    @pytest.mark.asyncio
    async def test_get_pending_not_found(self, store, mock_pool, mock_connection):
        """Test retrieving a non-existent pending query."""
        mock_connection.fetchrow.return_value = None

        result = await store.get_pending("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_and_delete_pending_atomic(
        self, store, mock_pool, mock_connection, sample_pending_query
    ):
        """Test atomic get and delete using DELETE RETURNING."""
        mock_connection.fetchrow.return_value = {
            "query_id": "query-123",
            "model_id": "gpt-4o-mini",
            "features": {"embedding": [0.1] * 10, "token_count": 50},
            "session_id": "session-456",
            "cost": 0.001,
            "latency": 0.5,
            "ttl_seconds": 3600,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        result = await store.get_and_delete_pending("query-123")

        call_args = mock_connection.fetchrow.call_args
        # Verify DELETE ... RETURNING is used
        assert "DELETE" in call_args[0][0]
        assert "RETURNING" in call_args[0][0]
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_pending_success(
        self, store, mock_pool, mock_connection, sample_pending_query
    ):
        """Test updating cost and latency."""
        mock_connection.execute.return_value = "UPDATE 1"

        result = await store.update_pending("query-123", cost=0.002, latency=0.8)

        assert result is True
        mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_pending_not_found(self, store, mock_pool, mock_connection):
        """Test updating a non-existent query."""
        mock_connection.execute.return_value = "UPDATE 0"

        result = await store.update_pending("nonexistent", cost=0.001)

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_pending(self, store, mock_pool, mock_connection):
        """Test deleting a pending query."""
        await store.delete_pending("query-123")

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert "DELETE FROM" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, store, mock_pool, mock_connection):
        """Test cleaning up expired queries."""
        mock_connection.execute.return_value = "DELETE 5"

        deleted = await store.cleanup_expired()

        call_args = mock_connection.execute.call_args
        assert "DELETE FROM" in call_args[0][0]
        assert "expires_at" in call_args[0][0]
        assert deleted == 5

    @pytest.mark.asyncio
    async def test_get_session_queries(
        self, store, mock_pool, mock_connection, sample_pending_query
    ):
        """Test retrieving queries by session ID."""
        mock_connection.fetch.return_value = [
            {
                "query_id": "query-1",
                "model_id": "gpt-4o-mini",
                "features": {},
                "session_id": "session-456",
                "cost": 0.001,
                "latency": 0.5,
                "ttl_seconds": 3600,
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
            },
            {
                "query_id": "query-2",
                "model_id": "gpt-4o",
                "features": {},
                "session_id": "session-456",
                "cost": 0.01,
                "latency": 1.0,
                "ttl_seconds": 3600,
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
            },
        ]

        results = await store.get_session_queries("session-456")

        assert len(results) == 2
        assert results[0].query_id == "query-1"
        assert results[1].query_id == "query-2"


class TestInMemoryFeedbackStoreEdgeCases:
    """Additional edge case tests for InMemoryFeedbackStore."""

    @pytest.mark.asyncio
    async def test_concurrent_get_and_delete(self, sample_pending_query):
        """Test concurrent get_and_delete operations."""
        store = InMemoryFeedbackStore()
        await store.save_pending(sample_pending_query)

        # Simulate concurrent access
        import asyncio

        results = await asyncio.gather(
            store.get_and_delete_pending("query-123"),
            store.get_and_delete_pending("query-123"),
        )

        # Only one should succeed
        successful = [r for r in results if r is not None]
        assert len(successful) == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_queries(self):
        """Test that cleanup_expired removes old queries."""
        store = InMemoryFeedbackStore()

        # Create an expired query
        expired_query = PendingQuery(
            query_id="expired-123",
            model_id="gpt-4o-mini",
            features={},
            ttl_seconds=1,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.save_pending(expired_query)

        # Create a fresh query
        fresh_query = PendingQuery(
            query_id="fresh-456",
            model_id="gpt-4o-mini",
            features={},
            ttl_seconds=3600,
            created_at=datetime.now(timezone.utc),
        )
        await store.save_pending(fresh_query)

        deleted = await store.cleanup_expired()

        # Expired query should be deleted
        assert deleted == 1
        assert await store.get_pending("expired-123") is None
        assert await store.get_pending("fresh-456") is not None

    @pytest.mark.asyncio
    async def test_update_pending_partial(self, sample_pending_query):
        """Test updating only cost or only latency."""
        store = InMemoryFeedbackStore()
        await store.save_pending(sample_pending_query)

        # Update only cost
        result = await store.update_pending("query-123", cost=0.005)
        assert result is True

        pending = await store.get_pending("query-123")
        assert pending is not None
        assert pending.cost == 0.005
        assert pending.latency == 0.5  # Original value

        # Update only latency
        result = await store.update_pending("query-123", latency=1.5)
        assert result is True

        pending = await store.get_pending("query-123")
        assert pending is not None
        assert pending.latency == 1.5
        assert pending.cost == 0.005  # Previous update preserved

    @pytest.mark.asyncio
    async def test_get_session_queries_empty(self):
        """Test get_session_queries with no matching queries."""
        store = InMemoryFeedbackStore()

        results = await store.get_session_queries("nonexistent-session")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_and_delete_session(self, sample_pending_query):
        """Test atomically getting and deleting all queries for a session."""
        store = InMemoryFeedbackStore()

        # Add multiple queries for same session
        await store.save_pending(sample_pending_query)
        query2 = sample_pending_query.model_copy(update={"query_id": "query-456"})
        await store.save_pending(query2)

        # Add query for different session
        other_session_query = sample_pending_query.model_copy(
            update={"query_id": "query-789", "session_id": "other-session"}
        )
        await store.save_pending(other_session_query)

        results = await store.get_and_delete_session("session-456")

        assert len(results) == 2
        assert await store.get_pending("query-123") is None
        assert await store.get_pending("query-456") is None
        assert await store.get_pending("query-789") is not None
