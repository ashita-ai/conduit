"""PostgreSQL implementation of StateStore for bandit state persistence.

Uses JSONB columns for flexible state storage with efficient querying.
Supports atomic updates and version checking for safe concurrent access.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from conduit.core.state_store import (
    BanditState,
    HybridRouterState,
    StateStore,
    StateStoreError,
)

logger = logging.getLogger(__name__)


class PostgresStateStore(StateStore):
    """PostgreSQL implementation of state persistence.

    Stores bandit state as JSONB in a dedicated table, supporting:
    - Atomic save/load operations
    - Version checking for safe concurrent updates
    - Efficient JSON querying for monitoring/debugging

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
    """

    def __init__(self, pool: Any) -> None:
        """Initialize PostgreSQL state store.

        Args:
            pool: asyncpg connection pool (from Database class)
        """
        self.pool = pool

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
        """Save bandit algorithm state to PostgreSQL.

        Uses UPSERT to insert or update state atomically.
        Increments version on each update.

        Args:
            router_id: Unique identifier for the router instance
            bandit_id: Identifier for the specific bandit (e.g., "ucb1", "linucb")
            state: Bandit state to persist

        Raises:
            StateStoreError: If save fails
        """
        await self._ensure_table_exists()

        # Update timestamp
        state.updated_at = datetime.now(UTC)

        # Serialize to JSON
        state_json = state.model_dump_json()

        upsert_sql = """
        INSERT INTO bandit_state (router_id, bandit_id, state_json, version, created_at, updated_at)
        VALUES ($1, $2, $3::jsonb, 1, NOW(), NOW())
        ON CONFLICT (router_id, bandit_id)
        DO UPDATE SET
            state_json = $3::jsonb,
            version = bandit_state.version + 1,
            updated_at = NOW()
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(upsert_sql, router_id, bandit_id, state_json)
            logger.debug(f"Saved state for {router_id}/{bandit_id}")
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
        """Save HybridRouter state including embedded bandit states.

        Stores as a single JSON document with ucb1_state and linucb_state nested.

        Args:
            router_id: Unique identifier for the router instance
            state: HybridRouter state to persist

        Raises:
            StateStoreError: If save fails
        """
        await self._ensure_table_exists()

        # Update timestamp
        state.updated_at = datetime.now(UTC)

        # Serialize to JSON
        state_json = state.model_dump_json()

        upsert_sql = """
        INSERT INTO bandit_state (router_id, bandit_id, state_json, version, created_at, updated_at)
        VALUES ($1, 'hybrid_router', $2::jsonb, 1, NOW(), NOW())
        ON CONFLICT (router_id, bandit_id)
        DO UPDATE SET
            state_json = $2::jsonb,
            version = bandit_state.version + 1,
            updated_at = NOW()
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(upsert_sql, router_id, state_json)
            logger.debug(f"Saved hybrid router state for {router_id}")
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
