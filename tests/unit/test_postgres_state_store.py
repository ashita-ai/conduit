"""Tests for PostgreSQL state store module.

Tests cover:
- State persistence (save/load) for bandit and hybrid router state
- Optimistic locking with version conflicts
- Table creation and schema management
- Error handling and recovery
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.postgres_state_store import (
    BASE_DELAY_MS,
    MAX_DELAY_MS,
    PostgresStateStore,
    StateVersionConflictError,
)
from conduit.core.state_store import BanditState, HybridRouterState, StateStoreError


@pytest.fixture
def mock_pool():
    """Create a mock database pool."""
    return MagicMock()


@pytest.fixture
def mock_conn():
    """Create a mock database connection."""
    return AsyncMock()


@pytest.fixture
def sample_bandit_state():
    """Create a sample BanditState for testing."""
    return BanditState(
        algorithm="ucb1",
        arm_ids=["model-a", "model-b"],
        arm_pulls={"model-a": 5, "model-b": 3},
        total_queries=8,
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_hybrid_state():
    """Create a sample HybridRouterState for testing."""
    ucb1_state = BanditState(
        algorithm="ucb1",
        arm_ids=["model-a"],
        arm_pulls={"model-a": 1},
        total_queries=1,
        updated_at=datetime.now(UTC),
    )
    linucb_state = BanditState(
        algorithm="linucb",
        arm_ids=["model-a"],
        arm_pulls={"model-a": 1},
        total_queries=1,
        updated_at=datetime.now(UTC),
    )
    return HybridRouterState(
        query_count=1,
        current_phase="ucb1",  # RouterPhase enum value
        ucb1_state=ucb1_state,
        linucb_state=linucb_state,
        updated_at=datetime.now(UTC),
    )


class TestPostgresStateStoreInit:
    """Tests for PostgresStateStore initialization."""

    def test_init_with_pool(self, mock_pool):
        """Test initialization with connection pool."""
        store = PostgresStateStore(pool=mock_pool)

        assert store.pool == mock_pool
        assert store.conflict_count == 0

    def test_init_conflict_counter(self, mock_pool):
        """Test conflict counter is initialized to zero."""
        store = PostgresStateStore(pool=mock_pool)

        assert store.conflict_count == 0


class TestBackoffCalculation:
    """Tests for exponential backoff calculation."""

    def test_backoff_delay_attempt_0(self, mock_pool):
        """Test backoff delay for first attempt."""
        store = PostgresStateStore(pool=mock_pool)

        delay = store._calculate_backoff_delay(0)

        # Delay should be BASE_DELAY_MS + jitter (0-50% of delay)
        assert delay >= BASE_DELAY_MS / 1000.0
        assert delay <= (BASE_DELAY_MS * 1.5) / 1000.0

    def test_backoff_delay_attempt_1(self, mock_pool):
        """Test backoff delay for second attempt."""
        store = PostgresStateStore(pool=mock_pool)

        delay = store._calculate_backoff_delay(1)

        expected_base = BASE_DELAY_MS * 2 / 1000.0
        assert delay >= expected_base
        assert delay <= expected_base * 1.5

    def test_backoff_delay_capped_at_max(self, mock_pool):
        """Test backoff delay is capped at MAX_DELAY_MS."""
        store = PostgresStateStore(pool=mock_pool)

        # Large attempt should cap at MAX_DELAY_MS
        delay = store._calculate_backoff_delay(10)

        max_with_jitter = (MAX_DELAY_MS * 1.5) / 1000.0
        assert delay <= max_with_jitter


class TestEnsureTableExists:
    """Tests for table creation."""

    @pytest.mark.asyncio
    async def test_ensure_table_creates_table(self, mock_pool, mock_conn):
        """Test table creation SQL is executed."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)
        await store._ensure_table_exists()

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS bandit_state" in call_args

    @pytest.mark.asyncio
    async def test_ensure_table_raises_on_error(self, mock_pool):
        """Test table creation raises StateStoreError on failure."""
        mock_pool.acquire.return_value.__aenter__.side_effect = Exception(
            "Connection error"
        )

        store = PostgresStateStore(pool=mock_pool)

        with pytest.raises(StateStoreError, match="Failed to create table"):
            await store._ensure_table_exists()


class TestGetCurrentVersion:
    """Tests for version retrieval."""

    @pytest.mark.asyncio
    async def test_get_version_existing_record(self, mock_pool, mock_conn):
        """Test getting version for existing record."""
        mock_conn.fetchrow.return_value = {"version": 5}

        store = PostgresStateStore(pool=mock_pool)
        version = await store._get_current_version(mock_conn, "router-1", "ucb1")

        assert version == 5
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_version_nonexistent_record(self, mock_pool, mock_conn):
        """Test getting version for nonexistent record."""
        mock_conn.fetchrow.return_value = None

        store = PostgresStateStore(pool=mock_pool)
        version = await store._get_current_version(mock_conn, "router-1", "ucb1")

        assert version is None


class TestSaveBanditState:
    """Tests for saving bandit state."""

    @pytest.mark.asyncio
    async def test_save_new_state_insert(self, mock_pool, mock_conn, sample_bandit_state):
        """Test saving new state creates an INSERT."""
        mock_conn.fetchrow.side_effect = [
            None,  # _get_current_version returns None (new record)
            {"version": 1},  # INSERT returns version
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            await store.save_bandit_state("router-1", "ucb1", sample_bandit_state)

        calls = mock_conn.fetchrow.call_args_list
        assert len(calls) == 2
        assert "INSERT INTO bandit_state" in calls[1][0][0]

    @pytest.mark.asyncio
    async def test_save_existing_state_update(self, mock_pool, mock_conn, sample_bandit_state):
        """Test saving existing state creates an UPDATE."""
        mock_conn.fetchrow.side_effect = [
            {"version": 3},  # _get_current_version returns 3
            {"version": 4},  # UPDATE returns new version
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            await store.save_bandit_state("router-1", "ucb1", sample_bandit_state)

        calls = mock_conn.fetchrow.call_args_list
        assert len(calls) == 2
        assert "UPDATE bandit_state" in calls[1][0][0]

    @pytest.mark.asyncio
    async def test_save_version_conflict_retries(self, mock_pool, mock_conn):
        """Test save retries on version conflict."""
        mock_conn.fetchrow.side_effect = [
            {"version": 1},  # First attempt: version 1
            None,  # First UPDATE fails (conflict)
            {"version": 2},  # Retry: version 2
            {"version": 3},  # Retry UPDATE succeeds
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 1},
            total_queries=1,
            updated_at=datetime.now(UTC),
        )

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await store.save_bandit_state("router-1", "ucb1", state)

        assert store.conflict_count == 1

    @pytest.mark.asyncio
    async def test_save_raises_after_max_retries(self, mock_pool, mock_conn):
        """Test save raises StateVersionConflictError after MAX_RETRIES."""
        mock_conn.fetchrow.side_effect = [
            {"version": i} if i % 2 == 0 else None
            for i in range(10)
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 1},
            total_queries=1,
            updated_at=datetime.now(UTC),
        )

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(
                    StateVersionConflictError, match="Version conflict persisted"
                ):
                    await store.save_bandit_state("router-1", "ucb1", state)

    @pytest.mark.asyncio
    async def test_save_raises_state_store_error(self, mock_pool):
        """Test save raises StateStoreError on database error."""
        store = PostgresStateStore(pool=mock_pool)

        state = BanditState(
            algorithm="ucb1",
            arm_ids=["model-a"],
            arm_pulls={"model-a": 1},
            total_queries=1,
            updated_at=datetime.now(UTC),
        )

        # Mock _ensure_table_exists to pass, then fail on pool.acquire
        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB error")
            with pytest.raises(StateStoreError, match="Failed to save state"):
                await store.save_bandit_state("router-1", "ucb1", state)


class TestLoadBanditState:
    """Tests for loading bandit state."""

    @pytest.mark.asyncio
    async def test_load_existing_state(self, mock_pool, mock_conn):
        """Test loading existing state returns BanditState."""
        state_dict = {
            "algorithm": "ucb1",
            "arm_ids": ["model-a", "model-b"],
            "arm_pulls": {"model-a": 5, "model-b": 3},
            "total_queries": 8,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        mock_conn.fetchrow.return_value = {"state_json": json.dumps(state_dict)}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            state = await store.load_bandit_state("router-1", "ucb1")

        assert state is not None
        assert state.algorithm == "ucb1"
        assert state.arm_ids == ["model-a", "model-b"]
        assert state.total_queries == 8

    @pytest.mark.asyncio
    async def test_load_nonexistent_state(self, mock_pool, mock_conn):
        """Test loading nonexistent state returns None."""
        mock_conn.fetchrow.return_value = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            state = await store.load_bandit_state("router-1", "nonexistent")

        assert state is None

    @pytest.mark.asyncio
    async def test_load_raises_on_error(self, mock_pool):
        """Test load raises StateStoreError on database error."""
        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB error")
            with pytest.raises(StateStoreError, match="Failed to load state"):
                await store.load_bandit_state("router-1", "ucb1")


class TestSaveHybridRouterState:
    """Tests for saving HybridRouter state."""

    @pytest.mark.asyncio
    async def test_save_hybrid_router_state_insert(self, mock_pool, mock_conn, sample_hybrid_state):
        """Test saving new hybrid router state creates INSERT."""
        mock_conn.fetchrow.side_effect = [
            None,  # No existing record
            {"version": 1},  # INSERT succeeds
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            await store.save_hybrid_router_state("router-1", sample_hybrid_state)

        calls = mock_conn.fetchrow.call_args_list
        assert len(calls) == 2
        assert "hybrid_router" in calls[1][0][0]

    @pytest.mark.asyncio
    async def test_save_hybrid_router_state_update(self, mock_pool, mock_conn, sample_hybrid_state):
        """Test saving existing hybrid router state creates UPDATE."""
        mock_conn.fetchrow.side_effect = [
            {"version": 2},  # Existing record
            {"version": 3},  # UPDATE succeeds
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            await store.save_hybrid_router_state("router-1", sample_hybrid_state)

        calls = mock_conn.fetchrow.call_args_list
        assert len(calls) == 2
        assert "UPDATE bandit_state" in calls[1][0][0]


class TestLoadHybridRouterState:
    """Tests for loading HybridRouter state."""

    @pytest.mark.asyncio
    async def test_load_hybrid_router_state_existing(self, mock_pool, mock_conn):
        """Test loading existing hybrid router state."""
        ucb1_state_dict = {
            "algorithm": "ucb1",
            "arm_ids": ["model-a"],
            "arm_pulls": {"model-a": 5},
            "total_queries": 5,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        linucb_state_dict = {
            "algorithm": "linucb",
            "arm_ids": ["model-a"],
            "arm_pulls": {"model-a": 5},
            "total_queries": 5,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        state_dict = {
            "query_count": 5,
            "current_phase": "linucb",  # RouterPhase enum value
            "ucb1_state": ucb1_state_dict,
            "linucb_state": linucb_state_dict,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        mock_conn.fetchrow.return_value = {"state_json": json.dumps(state_dict)}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            state = await store.load_hybrid_router_state("router-1")

        assert state is not None
        assert state.current_phase.value == "linucb"
        assert state.query_count == 5

    @pytest.mark.asyncio
    async def test_load_hybrid_router_state_nonexistent(self, mock_pool, mock_conn):
        """Test loading nonexistent hybrid router state returns None."""
        mock_conn.fetchrow.return_value = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            state = await store.load_hybrid_router_state("nonexistent")

        assert state is None


class TestDeleteState:
    """Tests for deleting state."""

    @pytest.mark.asyncio
    async def test_delete_state_executes_delete(self, mock_pool, mock_conn):
        """Test delete state executes DELETE SQL."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            await store.delete_state("router-1")

        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args[0][0]
        assert "DELETE FROM bandit_state" in call_args

    @pytest.mark.asyncio
    async def test_delete_state_raises_on_error(self, mock_pool):
        """Test delete raises StateStoreError on failure."""
        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB error")
            with pytest.raises(StateStoreError, match="Failed to delete state"):
                await store.delete_state("router-1")


class TestListRouterIds:
    """Tests for listing router IDs."""

    @pytest.mark.asyncio
    async def test_list_router_ids_returns_list(self, mock_pool, mock_conn):
        """Test listing router IDs returns list of strings."""
        mock_conn.fetch.return_value = [
            {"router_id": "router-1"},
            {"router_id": "router-2"},
            {"router_id": "router-3"},
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            router_ids = await store.list_router_ids()

        assert router_ids == ["router-1", "router-2", "router-3"]

    @pytest.mark.asyncio
    async def test_list_router_ids_empty(self, mock_pool, mock_conn):
        """Test listing router IDs returns empty list when no state."""
        mock_conn.fetch.return_value = []
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            router_ids = await store.list_router_ids()

        assert router_ids == []

    @pytest.mark.asyncio
    async def test_list_router_ids_raises_on_error(self, mock_pool):
        """Test list raises StateStoreError on failure."""
        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB error")
            with pytest.raises(StateStoreError, match="Failed to list router IDs"):
                await store.list_router_ids()


class TestGetStateStats:
    """Tests for state statistics."""

    @pytest.mark.asyncio
    async def test_get_state_stats_returns_dict(self, mock_pool, mock_conn):
        """Test getting state stats returns dictionary."""
        mock_conn.fetch.return_value = [
            {
                "bandit_id": "ucb1",
                "version": 5,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "state_preview": "{}",
            },
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            stats = await store.get_state_stats("router-1")

        assert stats["router_id"] == "router-1"
        assert len(stats["bandits"]) == 1
        assert stats["bandits"][0]["bandit_id"] == "ucb1"
        assert stats["bandits"][0]["version"] == 5

    @pytest.mark.asyncio
    async def test_get_state_stats_raises_on_error(self, mock_pool):
        """Test get stats raises StateStoreError on failure."""
        store = PostgresStateStore(pool=mock_pool)

        with patch.object(store, "_ensure_table_exists", new_callable=AsyncMock):
            mock_pool.acquire.return_value.__aenter__.side_effect = Exception("DB error")
            with pytest.raises(StateStoreError, match="Failed to get state stats"):
                await store.get_state_stats("router-1")


class TestStateVersionConflictError:
    """Tests for StateVersionConflictError."""

    def test_error_is_state_store_error(self):
        """Test StateVersionConflictError inherits from StateStoreError."""
        error = StateVersionConflictError("Test conflict")
        assert isinstance(error, StateStoreError)

    def test_error_message(self):
        """Test error message is preserved."""
        error = StateVersionConflictError("Version mismatch detected")
        assert str(error) == "Version mismatch detected"
