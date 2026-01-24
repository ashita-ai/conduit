"""Unit tests for Database class with mocked asyncpg.

Tests cover:
- Connection pool creation and management
- Error handling for database operations
- Query, routing, response, and feedback saving
- Model state and pricing operations
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from conduit.core.database import Database
from conduit.core.exceptions import DatabaseError
from conduit.core.models import (
    Feedback,
    ModelState,
    Query,
    QueryConstraints,
    QueryFeatures,
    Response,
    RoutingDecision,
)
from conduit.core.pricing import ModelPricing


class TestDatabaseInit:
    """Tests for Database initialization."""

    def test_init_with_url(self):
        """Test initialization with explicit URL."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        assert db.database_url == "postgresql://user:pass@localhost/test"
        assert db.pool is None

    def test_init_with_env_var(self):
        """Test initialization using environment variable."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://env@host/db"}):
            db = Database()
            assert db.database_url == "postgresql://env@host/db"

    def test_init_without_url_raises_error(self):
        """Test initialization without URL raises ValueError."""
        with patch.dict("os.environ", {"DATABASE_URL": ""}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Database()
            assert "Database URL must be provided" in str(exc_info.value)

    def test_init_with_custom_pool_sizes(self):
        """Test initialization with custom pool sizes."""
        db = Database(
            database_url="postgresql://user:pass@localhost/test",
            min_size=10,
            max_size=50,
        )
        assert db.min_size == 10
        assert db.max_size == 50


class TestDatabaseConnect:
    """Tests for database connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """Test that connect creates a connection pool."""
        mock_pool = MagicMock()

        with patch(
            "conduit.core.database.asyncpg.create_pool", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_pool
            db = Database(database_url="postgresql://user:pass@localhost/test")
            await db.connect()

            assert db.pool == mock_pool
            mock_create.assert_called_once()
            # Verify pool configuration
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["statement_cache_size"] == 0

    @pytest.mark.asyncio
    async def test_connect_raises_database_error_on_failure(self):
        """Test that connection failure raises DatabaseError."""
        with patch(
            "conduit.core.database.asyncpg.create_pool", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("Connection refused")
            db = Database(database_url="postgresql://user:pass@localhost/test")

            with pytest.raises(DatabaseError) as exc_info:
                await db.connect()

            assert "Failed to connect to database" in str(exc_info.value)


class TestDatabaseDisconnect:
    """Tests for database disconnection."""

    @pytest.mark.asyncio
    async def test_disconnect_closes_pool(self):
        """Test that disconnect closes the pool gracefully."""
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = mock_pool

        await db.disconnect()

        mock_pool.close.assert_called_once()
        assert db.pool is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_timeout(self):
        """Test that disconnect handles timeout gracefully."""
        import asyncio

        mock_pool = MagicMock()
        mock_pool.close = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_pool.terminate = MagicMock()

        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = mock_pool

        # Should not raise
        await db.disconnect()

        mock_pool.terminate.assert_called_once()
        assert db.pool is None

    @pytest.mark.asyncio
    async def test_disconnect_with_no_pool(self):
        """Test disconnect when pool is None."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        # Should not raise
        await db.disconnect()
        assert db.pool is None


class TestSaveQuery:
    """Tests for save_query method."""

    @pytest.fixture
    def mock_db(self):
        """Create database with mocked pool."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db.pool = mock_pool
        return db, mock_conn

    @pytest.fixture
    def sample_query(self):
        """Create sample query."""
        return Query(
            id=f"test-{uuid4()}",
            text="What is 2+2?",
            user_id="test-user",
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_save_query_success(self, mock_db, sample_query):
        """Test successful query save."""
        db, mock_conn = mock_db

        query_id = await db.save_query(sample_query)

        assert query_id == sample_query.id
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_query_with_constraints(self, mock_db):
        """Test saving query with constraints and context."""
        db, mock_conn = mock_db

        query = Query(
            id=f"test-{uuid4()}",
            text="Complex query",
            user_id="test-user",
            constraints=QueryConstraints(max_cost=0.01, max_latency=2.0),
            context={"source": "test"},
            created_at=datetime.now(timezone.utc),
        )

        query_id = await db.save_query(query)

        assert query_id == query.id
        # Verify constraints and context are serialized
        call_args = mock_conn.execute.call_args[0]
        assert query.id in call_args

    @pytest.mark.asyncio
    async def test_save_query_without_connection_raises_error(self, sample_query):
        """Test save_query raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_query(sample_query)

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_query_handles_db_error(self, mock_db, sample_query):
        """Test save_query wraps database errors."""
        db, mock_conn = mock_db
        mock_conn.execute.side_effect = Exception("Duplicate key")

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_query(sample_query)

        assert "Failed to save query" in str(exc_info.value)


class TestSaveCompleteInteraction:
    """Tests for save_complete_interaction method."""

    @pytest.fixture
    def mock_db(self):
        """Create database with mocked pool and transaction."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        mock_conn = AsyncMock()

        # Mock transaction() to return an async context manager
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_tx)

        # Mock acquire() to return an async context manager that yields mock_conn
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_acquire)
        db.pool = mock_pool
        return db, mock_conn

    @pytest.fixture
    def sample_response(self):
        """Create sample response."""
        return Response(
            id=f"resp-{uuid4()}",
            query_id=f"query-{uuid4()}",
            model="gpt-4o-mini",
            text="4",
            cost=0.001,
            latency=0.5,
            tokens=20,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_routing(self, sample_response):
        """Create sample routing decision."""
        return RoutingDecision(
            id=f"route-{uuid4()}",
            query_id=sample_response.query_id,
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=QueryFeatures(
                embedding=[0.1] * 384, token_count=10, complexity_score=0.2
            ),
            reasoning="Test",
        )

    @pytest.fixture
    def sample_feedback(self, sample_response):
        """Create sample feedback."""
        return Feedback(
            response_id=sample_response.id,
            quality_score=0.9,
            met_expectations=True,
        )

    @pytest.mark.asyncio
    async def test_save_with_routing_and_response(
        self, mock_db, sample_routing, sample_response
    ):
        """Test saving routing decision and response."""
        db, mock_conn = mock_db

        await db.save_complete_interaction(
            routing=sample_routing, response=sample_response
        )

        # Should have called execute twice (routing + response)
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_save_response_only(self, mock_db, sample_response):
        """Test saving response without routing decision."""
        db, mock_conn = mock_db

        await db.save_complete_interaction(routing=None, response=sample_response)

        # Should have called execute once (response only)
        assert mock_conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_save_with_feedback(
        self, mock_db, sample_routing, sample_response, sample_feedback
    ):
        """Test saving complete interaction with feedback."""
        db, mock_conn = mock_db

        await db.save_complete_interaction(
            routing=sample_routing, response=sample_response, feedback=sample_feedback
        )

        # Should have called execute 3 times (routing + response + feedback)
        assert mock_conn.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_save_without_connection_raises_error(self, sample_response):
        """Test save raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_complete_interaction(routing=None, response=sample_response)

        assert "Database not connected" in str(exc_info.value)


class TestModelStateOperations:
    """Tests for model state operations."""

    @pytest.fixture
    def mock_db(self):
        """Create database with mocked pool."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db.pool = mock_pool
        return db, mock_conn

    @pytest.mark.asyncio
    async def test_update_model_state_success(self, mock_db):
        """Test successful model state update."""
        db, mock_conn = mock_db
        state = ModelState(model_id="gpt-4o-mini", alpha=5.0, beta=2.0)

        await db.update_model_state(state)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_model_state_without_connection(self):
        """Test update raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None
        state = ModelState(model_id="gpt-4o-mini", alpha=5.0, beta=2.0)

        with pytest.raises(DatabaseError) as exc_info:
            await db.update_model_state(state)

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_model_states_success(self, mock_db):
        """Test successful retrieval of model states."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = [
            {
                "model_id": "gpt-4o-mini",
                "alpha": 5.0,
                "beta": 2.0,
                "total_requests": 100,
                "total_cost": 1.5,
                "avg_quality": 0.85,
                "updated_at": datetime.now(timezone.utc),
            }
        ]

        states = await db.get_model_states()

        assert "gpt-4o-mini" in states
        assert states["gpt-4o-mini"].alpha == 5.0
        assert states["gpt-4o-mini"].beta == 2.0

    @pytest.mark.asyncio
    async def test_get_model_states_empty(self, mock_db):
        """Test retrieval when no states exist."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = []

        states = await db.get_model_states()

        assert states == {}

    @pytest.mark.asyncio
    async def test_get_model_states_without_connection(self):
        """Test get states raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_model_states()

        assert "Database not connected" in str(exc_info.value)


class TestModelPricingOperations:
    """Tests for model pricing operations."""

    @pytest.fixture
    def mock_db(self):
        """Create database with mocked pool."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db.pool = mock_pool
        return db, mock_conn

    @pytest.mark.asyncio
    async def test_get_latest_pricing_success(self, mock_db):
        """Test successful retrieval of latest model pricing."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = [
            {
                "model_id": "gpt-4o-mini",
                "input_cost_per_million": 0.15,
                "output_cost_per_million": 0.60,
                "cached_input_cost_per_million": 0.075,
                "source": "openai",
                "snapshot_at": datetime.now(timezone.utc),
            }
        ]

        prices = await db.get_latest_pricing()

        assert "gpt-4o-mini" in prices
        assert prices["gpt-4o-mini"].input_cost_per_million == 0.15
        assert prices["gpt-4o-mini"].output_cost_per_million == 0.60

    @pytest.mark.asyncio
    async def test_get_latest_pricing_without_cached_cost(self, mock_db):
        """Test pricing without cached input cost."""
        db, mock_conn = mock_db
        mock_conn.fetch.return_value = [
            {
                "model_id": "gpt-4o",
                "input_cost_per_million": 2.50,
                "output_cost_per_million": 10.00,
                "cached_input_cost_per_million": None,
                "source": None,
                "snapshot_at": None,
            }
        ]

        prices = await db.get_latest_pricing()

        assert "gpt-4o" in prices
        assert prices["gpt-4o"].cached_input_cost_per_million is None

    @pytest.mark.asyncio
    async def test_get_latest_pricing_without_connection(self):
        """Test get pricing raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_latest_pricing()

        assert "Database not connected" in str(exc_info.value)


class TestResponseRetrieval:
    """Tests for response retrieval."""

    @pytest.fixture
    def mock_db(self):
        """Create database with mocked pool."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        db.pool = mock_pool
        return db, mock_conn

    @pytest.mark.asyncio
    async def test_get_response_by_id_found(self, mock_db):
        """Test successful response retrieval."""
        db, mock_conn = mock_db
        response_id = f"resp-{uuid4()}"
        mock_conn.fetchrow.return_value = {
            "id": response_id,
            "query_id": "query-123",
            "model": "gpt-4o-mini",
            "text": "4",
            "cost": 0.001,
            "latency": 0.5,
            "tokens": 20,
            "created_at": datetime.now(timezone.utc),
        }

        response = await db.get_response_by_id(response_id)

        assert response is not None
        assert response.id == response_id
        assert response.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_response_by_id_not_found(self, mock_db):
        """Test response retrieval when not found."""
        db, mock_conn = mock_db
        mock_conn.fetchrow.return_value = None

        response = await db.get_response_by_id("nonexistent")

        assert response is None

    @pytest.mark.asyncio
    async def test_get_response_without_connection(self):
        """Test get response raises error when not connected."""
        db = Database(database_url="postgresql://user:pass@localhost/test")
        db.pool = None

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_response_by_id("any-id")

        assert "Database not connected" in str(exc_info.value)
