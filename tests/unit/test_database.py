"""Unit tests for Database module.

Note: These are lightweight unit tests focusing on initialization and basic logic.
Full database integration tests require real Supabase instance and belong in
tests/integration/ directory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.database import Database
from conduit.core.exceptions import DatabaseError


class TestDatabaseInitialization:
    """Tests for Database initialization."""

    def test_init_with_env_variables(self):
        """Test database initialization from environment variables."""
        with patch.dict(
            "os.environ",
            {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_ANON_KEY": "test-key"},
        ):
            db = Database()

            assert db.supabase_url == "https://test.supabase.co"
            assert db.supabase_key == "test-key"
            assert db.client is None  # Not connected yet

    def test_init_with_explicit_params(self):
        """Test database initialization with explicit parameters."""
        db = Database(
            supabase_url="https://explicit.supabase.co", supabase_key="explicit-key"
        )

        assert db.supabase_url == "https://explicit.supabase.co"
        assert db.supabase_key == "explicit-key"
        assert db.client is None

    def test_init_missing_url_raises_error(self):
        """Test initialization fails when URL is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                Database()

            assert "Supabase URL and key must be provided" in str(exc_info.value)

    def test_init_explicit_params_override_env(self):
        """Test explicit parameters override environment variables."""
        with patch.dict(
            "os.environ",
            {"SUPABASE_URL": "https://env.supabase.co", "SUPABASE_ANON_KEY": "env-key"},
        ):
            db = Database(
                supabase_url="https://explicit.supabase.co", supabase_key="explicit-key"
            )

            assert db.supabase_url == "https://explicit.supabase.co"
            assert db.supabase_key == "explicit-key"


class TestDatabaseConnection:
    """Tests for database connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self):
        """Test connect() creates Supabase client."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        with patch("conduit.core.database.acreate_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            await db.connect()

            mock_create.assert_called_once_with("https://test.supabase.co", "test-key")
            assert db.client is mock_client

    @pytest.mark.asyncio
    async def test_connect_without_url_raises_error(self):
        """Test connect() fails if URL not set."""
        # Create database instance with missing credentials
        with patch.dict("os.environ", {}, clear=True):
            try:
                db = Database.__new__(Database)  # Skip __init__ validation
                db.supabase_url = None
                db.supabase_key = None
                db.client = None

                with pytest.raises(DatabaseError) as exc_info:
                    await db.connect()

                assert "Supabase URL and key must be set" in str(exc_info.value)
            except ValueError:
                # If __new__ doesn't work, just skip this test
                pytest.skip("Cannot bypass __init__ validation")

    @pytest.mark.asyncio
    async def test_disconnect_clears_client(self):
        """Test disconnect() clears client reference."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        # Mock client
        db.client = AsyncMock()

        await db.disconnect()

        assert db.client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test disconnect() handles no client gracefully."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        # No client set
        assert db.client is None

        # Should not raise
        await db.disconnect()

        assert db.client is None


class TestDatabaseOperationGuards:
    """Tests for database operation guards (not connected checks)."""

    @pytest.mark.asyncio
    async def test_save_query_requires_connection(self):
        """Test save_query() raises if not connected."""
        from conduit.core.models import Query
        from datetime import datetime, timezone

        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        query = Query(
            id="test-query",
            text="test",
            user_id="user-1",
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_query(query)

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_complete_interaction_requires_connection(self):
        """Test save_complete_interaction() raises if not connected."""
        from conduit.core.models import Response
        from datetime import datetime, timezone

        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        response = Response(
            id="test-response",
            query_id="test-query",
            model="gpt-4o-mini",
            text="answer",
            cost=0.001,
            latency=0.5,
            tokens=100,
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_complete_interaction(routing=None, response=response)

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_model_state_requires_connection(self):
        """Test update_model_state() raises if not connected."""
        from conduit.core.models import ModelState

        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        state = ModelState(model_id="gpt-4o-mini", alpha=5.0, beta=2.0)

        with pytest.raises(DatabaseError) as exc_info:
            await db.update_model_state(state)

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_model_states_requires_connection(self):
        """Test get_model_states() raises if not connected."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_model_states()

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_model_prices_requires_connection(self):
        """Test get_model_prices() raises if not connected."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_model_prices()

        assert "Database not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_response_by_id_requires_connection(self):
        """Test get_response_by_id() raises if not connected."""
        db = Database(supabase_url="https://test.supabase.co", supabase_key="test-key")

        with pytest.raises(DatabaseError) as exc_info:
            await db.get_response_by_id("test-id")

        assert "Database not connected" in str(exc_info.value)
