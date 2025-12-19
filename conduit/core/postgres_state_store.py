"""PostgreSQL implementation of StateStore for bandit state persistence.

Uses JSONB columns for flexible state storage with efficient querying.
Supports atomic updates and optimistic locking for safe concurrent access.

Optimistic Locking:
    When multiple replicas share the same router_id, they may attempt to update
    state concurrently. Optimistic locking prevents race conditions by:
    1. Reading current version before update
    2. Including version in WHERE clause
    3. Retrying with exponential backoff on conflict
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Any

from conduit.core.state_store import (
    BanditState,
    HybridRouterState,
    StateStore,
    StateStoreError,
)

logger = logging.getLogger(__name__)

# Optimistic locking configuration
MAX_RETRIES = 3
BASE_DELAY_MS = 50  # Base delay for exponential backoff
MAX_DELAY_MS = 500  # Maximum delay cap


class StateVersionConflictError(StateStoreError):
    """Raised when optimistic locking detects a version conflict.

    This occurs when another process updated the state between our read and write.
    The caller should retry the operation or handle the conflict.
    """

    pass


class PostgresStateStore(StateStore):
    """PostgreSQL implementation of state persistence with optimistic locking.

    Stores bandit state as JSONB in a dedicated table, supporting:
    - Atomic save/load operations
    - Optimistic locking for safe concurrent updates (multi-replica safe)
    - Automatic retry with exponential backoff on conflicts
    - Efficient JSON querying for monitoring/debugging

    Optimistic Locking:
        When saving state, the version is checked to detect concurrent updates.
        If another process modified the state between read and write, the save
        will retry with exponential backoff (up to MAX_RETRIES attempts).

        This is essential for Kubernetes deployments with multiple replicas
        sharing the same router_id (via CONDUIT_ROUTER_ID env var).

    Table Schema:
        CREATE TABLE IF NOT EXISTS bandit_state (
            id SERIAL PRIMARY KEY,
            router_id VARCHAR(255) NOT NULL,
            bandit_id VARCHAR(255) NOT NULL,
            state_json JSONB NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(router_id, bandit_id)
        );

    pgBouncer Compatibility:
        When using pgBouncer in transaction pooling mode, create the asyncpg
        pool with statement_cache_size=0 to avoid prepared statement conflicts:

            pool = await asyncpg.create_pool(
                database_url,
                statement_cache_size=0  # Required for pgBouncer
            )

    Attributes:
        pool: asyncpg connection pool
        conflict_count: Counter for version conflicts (for monitoring)
    """

    def __init__(self, pool: Any) -> None:
        """Initialize PostgreSQL state store.

        Args:
            pool: asyncpg connection pool (from Database class)
        """
        self.pool = pool
        self.conflict_count = 0  # Track conflicts for monitoring

    async def _get_current_version(
        self, conn: Any, router_id: str, bandit_id: str
    ) -> int | None:
        """Get current version for a state record.

        Args:
            conn: Database connection
            router_id: Router identifier
            bandit_id: Bandit identifier

        Returns:
            Current version number, or None if record doesn't exist
        """
        row = await conn.fetchrow(
            "SELECT version FROM bandit_state WHERE router_id = $1 AND bandit_id = $2",
            router_id,
            bandit_id,
        )
        return row["version"] if row else None

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current retry attempt (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: BASE_DELAY * 2^attempt
        delay_ms: float = min(BASE_DELAY_MS * (2**attempt), MAX_DELAY_MS)
        # Add jitter (0-50% of delay) to prevent thundering herd
        jitter_ms: float = random.uniform(0, delay_ms * 0.5)
        return (delay_ms + jitter_ms) / 1000.0

    async def _ensure_table_exists(self) -> None:
        """Create bandit_state table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS bandit_state (
            id SERIAL PRIMARY KEY,
            router_id VARCHAR(255) NOT NULL,
            bandit_id VARCHAR(255) NOT NULL,
            state_json JSONB NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(router_id, bandit_id)
        );

        CREATE INDEX IF NOT EXISTS idx_bandit_state_router_id
        ON bandit_state(router_id);
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(create_table_sql)
        except Exception as e:
            logger.error(f"Failed to create bandit_state table: {e}")
            raise StateStoreError(f"Failed to create table: {e}") from e

    async def save_bandit_state(
        self, router_id: str, bandit_id: str, state: BanditState
    ) -> None:
        """Save bandit algorithm state to PostgreSQL with optimistic locking.

        Uses version checking to prevent race conditions in multi-replica deployments.
        If a version conflict is detected (another process updated state), retries
        with exponential backoff up to MAX_RETRIES times.

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit (e.g., "ucb1", "linucb")
            state: Bandit state to persist

        Raises:
            StateStoreError: If save fails after all retries
            StateVersionConflictError: If conflicts persist after MAX_RETRIES
        """
        await self._ensure_table_exists()

        # Update timestamp
        state.updated_at = datetime.now(timezone.utc)

        # Serialize to JSON
        state_json = state.model_dump_json()

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with self.pool.acquire() as conn:
                    # Get current version (None if new record)
                    current_version = await self._get_current_version(
                        conn, router_id, bandit_id
                    )

                    if current_version is None:
                        # New record - simple INSERT
                        insert_sql = """
                        INSERT INTO bandit_state
                            (router_id, bandit_id, state_json, version, created_at, updated_at)
                        VALUES ($1, $2, $3::jsonb, 1, NOW(), NOW())
                        ON CONFLICT (router_id, bandit_id) DO NOTHING
                        RETURNING version
                        """
                        result = await conn.fetchrow(
                            insert_sql, router_id, bandit_id, state_json
                        )
                        if result is not None:
                            # Insert succeeded
                            logger.debug(
                                f"Inserted new state for {router_id}/{bandit_id}"
                            )
                            return
                        # Another process inserted first, retry as update
                        continue

                    # Existing record - UPDATE with version check (optimistic locking)
                    update_sql = """
                    UPDATE bandit_state
                    SET state_json = $3::jsonb,
                        version = version + 1,
                        updated_at = NOW()
                    WHERE router_id = $1 AND bandit_id = $2 AND version = $4
                    RETURNING version
                    """
                    result = await conn.fetchrow(
                        update_sql, router_id, bandit_id, state_json, current_version
                    )

                    if result is not None:
                        # Update succeeded
                        logger.debug(
                            f"Saved state for {router_id}/{bandit_id} "
                            f"(version {current_version} -> {result['version']})"
                        )
                        return

                    # Version conflict - another process updated the record
                    self.conflict_count += 1
                    if attempt < MAX_RETRIES:
                        delay = self._calculate_backoff_delay(attempt)
                        logger.warning(
                            f"Version conflict for {router_id}/{bandit_id}, "
                            f"retrying in {delay*1000:.0f}ms (attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise StateVersionConflictError(
                            f"Version conflict persisted after {MAX_RETRIES} retries "
                            f"for {router_id}/{bandit_id}. Total conflicts: {self.conflict_count}"
                        )

            except StateVersionConflictError:
                raise
            except Exception as e:
                logger.error(f"Failed to save bandit state: {e}")
                raise StateStoreError(f"Failed to save state: {e}") from e

    async def load_bandit_state(
        self, router_id: str, bandit_id: str
    ) -> BanditState | None:
        """Load bandit algorithm state from PostgreSQL.

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit

        Returns:
            BanditState if found, None otherwise

        Raises:
            StateStoreError: If load fails (not for missing state)
        """
        await self._ensure_table_exists()

        select_sql = """
        SELECT state_json FROM bandit_state
        WHERE router_id = $1 AND bandit_id = $2
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(select_sql, router_id, bandit_id)

            if row is None:
                logger.debug(f"No state found for {router_id}/{bandit_id}")
                return None

            # Parse JSON and create BanditState
            state_dict = json.loads(row["state_json"])
            state = BanditState(**state_dict)
            logger.debug(f"Loaded state for {router_id}/{bandit_id}")
            return state

        except Exception as e:
            logger.error(f"Failed to load bandit state: {e}")
            raise StateStoreError(f"Failed to load state: {e}") from e

    async def save_hybrid_router_state(
        self, router_id: str, state: HybridRouterState
    ) -> None:
        """Save HybridRouter state with optimistic locking.

        Stores as a single JSON document with ucb1_state and linucb_state nested.
        Uses version checking to prevent race conditions in multi-replica deployments.

        Args:
            router_id: Unique identifier for the router instance
            state: HybridRouter state to persist

        Raises:
            StateStoreError: If save fails after all retries
            StateVersionConflictError: If conflicts persist after MAX_RETRIES
        """
        await self._ensure_table_exists()

        # Update timestamp
        state.updated_at = datetime.now(timezone.utc)

        # Serialize to JSON
        state_json = state.model_dump_json()
        bandit_id = "hybrid_router"

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with self.pool.acquire() as conn:
                    # Get current version (None if new record)
                    current_version = await self._get_current_version(
                        conn, router_id, bandit_id
                    )

                    if current_version is None:
                        # New record - simple INSERT
                        insert_sql = """
                        INSERT INTO bandit_state
                            (router_id, bandit_id, state_json, version, created_at, updated_at)
                        VALUES ($1, 'hybrid_router', $2::jsonb, 1, NOW(), NOW())
                        ON CONFLICT (router_id, bandit_id) DO NOTHING
                        RETURNING version
                        """
                        result = await conn.fetchrow(insert_sql, router_id, state_json)
                        if result is not None:
                            logger.debug(
                                f"Inserted new hybrid router state for {router_id}"
                            )
                            return
                        # Another process inserted first, retry as update
                        continue

                    # Existing record - UPDATE with version check
                    update_sql = """
                    UPDATE bandit_state
                    SET state_json = $2::jsonb,
                        version = version + 1,
                        updated_at = NOW()
                    WHERE router_id = $1 AND bandit_id = 'hybrid_router' AND version = $3
                    RETURNING version
                    """
                    result = await conn.fetchrow(
                        update_sql, router_id, state_json, current_version
                    )

                    if result is not None:
                        logger.debug(
                            f"Saved hybrid router state for {router_id} "
                            f"(version {current_version} -> {result['version']})"
                        )
                        return

                    # Version conflict
                    self.conflict_count += 1
                    if attempt < MAX_RETRIES:
                        delay = self._calculate_backoff_delay(attempt)
                        logger.warning(
                            f"Version conflict for {router_id}/hybrid_router, "
                            f"retrying in {delay*1000:.0f}ms (attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise StateVersionConflictError(
                            f"Version conflict persisted after {MAX_RETRIES} retries "
                            f"for {router_id}/hybrid_router. Total conflicts: {self.conflict_count}"
                        )

            except StateVersionConflictError:
                raise
            except Exception as e:
                logger.error(f"Failed to save hybrid router state: {e}")
                raise StateStoreError(f"Failed to save state: {e}") from e

    async def load_hybrid_router_state(
        self, router_id: str
    ) -> HybridRouterState | None:
        """Load HybridRouter state from PostgreSQL.

        Args:
            router_id: Unique identifier for the router instance

        Returns:
            HybridRouterState if found, None otherwise

        Raises:
            StateStoreError: If load fails (not for missing state)
        """
        await self._ensure_table_exists()

        select_sql = """
        SELECT state_json FROM bandit_state
        WHERE router_id = $1 AND bandit_id = 'hybrid_router'
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(select_sql, router_id)

            if row is None:
                logger.debug(f"No hybrid router state found for {router_id}")
                return None

            # Parse JSON and create HybridRouterState
            state_dict = json.loads(row["state_json"])
            state = HybridRouterState(**state_dict)
            logger.debug(f"Loaded hybrid router state for {router_id}")
            return state

        except Exception as e:
            logger.error(f"Failed to load hybrid router state: {e}")
            raise StateStoreError(f"Failed to load state: {e}") from e

    async def delete_state(self, router_id: str) -> None:
        """Delete all state for a router instance.

        Args:
            router_id: Unique identifier for the router instance
        """
        await self._ensure_table_exists()

        delete_sql = """
        DELETE FROM bandit_state WHERE router_id = $1
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(delete_sql, router_id)
            logger.debug(f"Deleted all state for {router_id}")
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")
            raise StateStoreError(f"Failed to delete state: {e}") from e

    async def list_router_ids(self) -> list[str]:
        """List all router IDs with persisted state.

        Returns:
            List of router IDs
        """
        await self._ensure_table_exists()

        select_sql = """
        SELECT DISTINCT router_id FROM bandit_state ORDER BY router_id
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(select_sql)
            return [row["router_id"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list router IDs: {e}")
            raise StateStoreError(f"Failed to list router IDs: {e}") from e

    async def get_state_stats(self, router_id: str) -> dict[str, Any]:
        """Get statistics about persisted state for debugging.

        Args:
            router_id: Router instance ID

        Returns:
            Dictionary with state statistics
        """
        await self._ensure_table_exists()

        select_sql = """
        SELECT
            bandit_id,
            version,
            created_at,
            updated_at,
            jsonb_pretty(state_json) as state_preview
        FROM bandit_state
        WHERE router_id = $1
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(select_sql, router_id)

            return {
                "router_id": router_id,
                "bandits": [
                    {
                        "bandit_id": row["bandit_id"],
                        "version": row["version"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                    }
                    for row in rows
                ],
            }
        except Exception as e:
            logger.error(f"Failed to get state stats: {e}")
            raise StateStoreError(f"Failed to get state stats: {e}") from e
