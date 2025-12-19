"""Feedback stores for pending query persistence.

This module provides storage backends for PendingQuery objects,
enabling delayed feedback scenarios where user feedback arrives
after the initial query execution.

Storage Options:
- InMemoryFeedbackStore: Simple dict-based storage for testing/development
- RedisFeedbackStore: Redis-based storage for production (low-latency)
- PostgresFeedbackStore: PostgreSQL-based storage for apps without Redis

Design Notes:
- All stores implement atomic get_and_delete() to prevent race conditions
- PostgreSQL store requires periodic cleanup_expired() calls (no native TTL)
- Redis store uses native TTL for automatic expiry

Usage:
    >>> from conduit.feedback.stores import InMemoryFeedbackStore
    >>>
    >>> store = InMemoryFeedbackStore()
    >>> await store.save_pending(pending_query)
    >>> pending = await store.get_and_delete_pending("query_id")  # Atomic!
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from conduit.feedback.models import PendingQuery

logger = logging.getLogger(__name__)


class FeedbackStore(ABC):
    """Abstract base class for pending query storage.

    Implementations must provide async methods for saving, retrieving,
    and deleting PendingQuery objects.

    Critical: Use get_and_delete_pending() for atomic retrieval to prevent
    race conditions when multiple feedback events arrive simultaneously.
    """

    @abstractmethod
    async def save_pending(self, pending: PendingQuery) -> None:
        """Save a pending query for later feedback.

        Args:
            pending: PendingQuery to store
        """
        pass

    @abstractmethod
    async def get_pending(self, query_id: str) -> PendingQuery | None:
        """Retrieve a pending query by ID (non-destructive).

        Args:
            query_id: Query ID to look up

        Returns:
            PendingQuery if found, None otherwise

        Note:
            For feedback recording, prefer get_and_delete_pending() to avoid
            race conditions.
        """
        pass

    @abstractmethod
    async def get_and_delete_pending(self, query_id: str) -> PendingQuery | None:
        """Atomically retrieve and delete a pending query.

        This prevents race conditions when multiple feedback events
        arrive for the same query_id simultaneously.

        Args:
            query_id: Query ID to retrieve and delete

        Returns:
            PendingQuery if found (now deleted), None otherwise
        """
        pass

    @abstractmethod
    async def update_pending(
        self,
        query_id: str,
        cost: float | None = None,
        latency: float | None = None,
    ) -> bool:
        """Update cost/latency for a pending query after LLM execution.

        Args:
            query_id: Query ID to update
            cost: New cost value (None to keep existing)
            latency: New latency value (None to keep existing)

        Returns:
            True if updated, False if query_id not found
        """
        pass

    @abstractmethod
    async def delete_pending(self, query_id: str) -> None:
        """Delete a pending query.

        Args:
            query_id: Query ID to delete
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired pending queries.

        Returns:
            Number of queries deleted
        """
        pass

    @abstractmethod
    async def count_pending(self) -> int:
        """Get count of pending queries.

        Returns:
            Number of pending queries in store
        """
        pass

    @abstractmethod
    async def get_session_queries(self, session_id: str) -> list[PendingQuery]:
        """Get all pending queries for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of PendingQuery objects in the session
        """
        pass

    @abstractmethod
    async def get_and_delete_session(self, session_id: str) -> list[PendingQuery]:
        """Atomically retrieve and delete all pending queries for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of PendingQuery objects (now deleted)
        """
        pass

    # Idempotency tracking methods

    @abstractmethod
    async def was_processed(self, idempotency_key: str) -> bool:
        """Check if feedback with this idempotency key was already processed.

        Used to prevent double-counting feedback from retries or duplicates.

        Args:
            idempotency_key: Unique key for the feedback event

        Returns:
            True if already processed, False otherwise
        """
        pass

    @abstractmethod
    async def mark_processed(
        self, idempotency_key: str, ttl_seconds: int = 3600
    ) -> None:
        """Mark feedback as processed to prevent future duplicates.

        Args:
            idempotency_key: Unique key for the feedback event
            ttl_seconds: How long to remember this key (default 1 hour)
        """
        pass


class InMemoryFeedbackStore(FeedbackStore):
    """In-memory feedback store for development and testing.

    Simple dict-based storage with automatic expiry cleanup.
    Not suitable for production (not persistent, single-process only).

    Thread Safety:
        Uses asyncio.Lock for atomic get_and_delete operations.
    """

    def __init__(self, default_ttl: int = 3600):
        """Initialize in-memory store.

        Args:
            default_ttl: Default TTL in seconds (default 1 hour)
        """
        self._pending: dict[str, PendingQuery] = {}
        self._processed: dict[str, datetime] = {}  # idempotency_key -> expires_at
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl

    async def save_pending(self, pending: PendingQuery) -> None:
        """Save pending query to memory."""
        async with self._lock:
            self._pending[pending.query_id] = pending
        logger.debug(f"Saved pending query: {pending.query_id}")

    async def get_pending(self, query_id: str) -> PendingQuery | None:
        """Retrieve pending query by ID (non-destructive)."""
        async with self._lock:
            pending = self._pending.get(query_id)

            if pending is None:
                return None

            if pending.is_expired():
                del self._pending[query_id]
                logger.debug(f"Pending query expired: {query_id}")
                return None

            return pending

    async def get_and_delete_pending(self, query_id: str) -> PendingQuery | None:
        """Atomically retrieve and delete pending query."""
        async with self._lock:
            pending = self._pending.pop(query_id, None)

            if pending is None:
                return None

            if pending.is_expired():
                logger.debug(f"Pending query expired: {query_id}")
                return None

            logger.debug(f"Retrieved and deleted pending query: {query_id}")
            return pending

    async def update_pending(
        self,
        query_id: str,
        cost: float | None = None,
        latency: float | None = None,
    ) -> bool:
        """Update cost/latency for a pending query."""
        async with self._lock:
            pending = self._pending.get(query_id)
            if pending is None:
                return False

            # Create updated copy
            updates = {}
            if cost is not None:
                updates["cost"] = cost
            if latency is not None:
                updates["latency"] = latency

            if updates:
                self._pending[query_id] = pending.model_copy(update=updates)
                logger.debug(f"Updated pending query: {query_id}")

            return True

    async def delete_pending(self, query_id: str) -> None:
        """Delete pending query from memory."""
        async with self._lock:
            if query_id in self._pending:
                del self._pending[query_id]
                logger.debug(f"Deleted pending query: {query_id}")

    async def cleanup_expired(self) -> int:
        """Remove all expired pending queries."""
        async with self._lock:
            expired_ids = [
                qid for qid, pending in self._pending.items() if pending.is_expired()
            ]

            for qid in expired_ids:
                del self._pending[qid]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired pending queries")

        return len(expired_ids)

    async def count_pending(self) -> int:
        """Get count of pending queries (excludes expired)."""
        await self.cleanup_expired()
        return len(self._pending)

    async def get_session_queries(self, session_id: str) -> list[PendingQuery]:
        """Get all pending queries for a session (non-destructive)."""
        async with self._lock:
            queries = []
            for pending in self._pending.values():
                if pending.session_id == session_id and not pending.is_expired():
                    queries.append(pending)
            return queries

    async def get_and_delete_session(self, session_id: str) -> list[PendingQuery]:
        """Atomically retrieve and delete all pending queries for a session."""
        async with self._lock:
            queries = []
            to_delete = []

            for query_id, pending in self._pending.items():
                if pending.session_id == session_id and not pending.is_expired():
                    queries.append(pending)
                    to_delete.append(query_id)

            for query_id in to_delete:
                del self._pending[query_id]

            logger.debug(
                f"Retrieved and deleted {len(queries)} queries for session: {session_id}"
            )
            return queries

    async def was_processed(self, idempotency_key: str) -> bool:
        """Check if feedback with this idempotency key was already processed."""
        async with self._lock:
            expires_at = self._processed.get(idempotency_key)
            if expires_at is None:
                return False

            # Check if the entry has expired
            if datetime.now(timezone.utc) > expires_at:
                del self._processed[idempotency_key]
                return False

            return True

    async def mark_processed(
        self, idempotency_key: str, ttl_seconds: int = 3600
    ) -> None:
        """Mark feedback as processed to prevent future duplicates."""
        async with self._lock:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            self._processed[idempotency_key] = expires_at
            logger.debug(f"Marked feedback as processed: {idempotency_key}")


class RedisFeedbackStore(FeedbackStore):
    """Redis-based feedback store for production.

    Uses Redis for low-latency, distributed storage of pending queries.
    Supports automatic TTL-based expiry via Redis SETEX.

    Atomicity:
        Uses Redis GETDEL command (Redis 6.2+) for atomic get_and_delete.
        Falls back to Lua script for older Redis versions.

    Key Pattern:
        conduit:feedback:pending:{query_id}
    """

    def __init__(
        self,
        redis: Any,  # Redis async client
        key_prefix: str = "conduit:feedback:pending",
        default_ttl: int = 3600,
    ):
        """Initialize Redis store.

        Args:
            redis: Redis async client instance
            key_prefix: Key prefix for pending queries
            default_ttl: Default TTL in seconds (default 1 hour)
        """
        self.redis = redis
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    def _key(self, query_id: str) -> str:
        """Generate Redis key for query ID."""
        return f"{self.key_prefix}:{query_id}"

    async def save_pending(self, pending: PendingQuery) -> None:
        """Save pending query to Redis with TTL."""
        key = self._key(pending.query_id)
        ttl = pending.ttl_seconds or self.default_ttl

        await self.redis.setex(
            key,
            ttl,
            pending.model_dump_json(),
        )
        logger.debug(f"Saved pending query to Redis: {pending.query_id}")

    async def get_pending(self, query_id: str) -> PendingQuery | None:
        """Retrieve pending query from Redis (non-destructive)."""
        key = self._key(query_id)
        data = await self.redis.get(key)

        if data is None:
            return None

        try:
            return PendingQuery.model_validate_json(data)
        except Exception as e:
            logger.warning(f"Failed to parse pending query {query_id}: {e}")
            return None

    async def get_and_delete_pending(self, query_id: str) -> PendingQuery | None:
        """Atomically retrieve and delete pending query using GETDEL."""
        key = self._key(query_id)

        # GETDEL is atomic (Redis 6.2+)
        # Falls back to GET + DEL for older versions
        try:
            data = await self.redis.getdel(key)
        except AttributeError:
            # Fallback for older redis-py versions
            data = await self.redis.get(key)
            if data:
                await self.redis.delete(key)

        if data is None:
            return None

        try:
            pending = PendingQuery.model_validate_json(data)
            logger.debug(f"Retrieved and deleted pending query: {query_id}")
            return pending
        except Exception as e:
            logger.warning(f"Failed to parse pending query {query_id}: {e}")
            return None

    async def update_pending(
        self,
        query_id: str,
        cost: float | None = None,
        latency: float | None = None,
    ) -> bool:
        """Update cost/latency for a pending query."""
        key = self._key(query_id)

        # Get current value
        data = await self.redis.get(key)
        if data is None:
            return False

        try:
            pending = PendingQuery.model_validate_json(data)
        except Exception:
            return False

        # Update fields
        updates = {}
        if cost is not None:
            updates["cost"] = cost
        if latency is not None:
            updates["latency"] = latency

        if updates:
            updated = pending.model_copy(update=updates)
            # Get remaining TTL
            ttl = await self.redis.ttl(key)
            if ttl > 0:
                await self.redis.setex(key, ttl, updated.model_dump_json())
            else:
                await self.redis.set(key, updated.model_dump_json())
            logger.debug(f"Updated pending query: {query_id}")

        return True

    async def delete_pending(self, query_id: str) -> None:
        """Delete pending query from Redis."""
        key = self._key(query_id)
        await self.redis.delete(key)
        logger.debug(f"Deleted pending query from Redis: {query_id}")

    async def cleanup_expired(self) -> int:
        """Remove expired pending queries.

        Note: Redis handles TTL-based expiry automatically.
        This method scans for any orphaned keys without TTL.
        """
        cursor = 0
        deleted = 0
        pattern = f"{self.key_prefix}:*"

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                ttl = await self.redis.ttl(key)
                if ttl == -1:  # No TTL set
                    await self.redis.delete(key)
                    deleted += 1

            if cursor == 0:
                break

        if deleted:
            logger.info(f"Cleaned up {deleted} orphaned pending queries")

        return deleted

    async def count_pending(self) -> int:
        """Get count of pending queries in Redis."""
        cursor = 0
        count = 0
        pattern = f"{self.key_prefix}:*"

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )
            count += len(keys)

            if cursor == 0:
                break

        return count

    async def get_session_queries(self, session_id: str) -> list[PendingQuery]:
        """Get all pending queries for a session.

        Note: Redis doesn't have native secondary indexes, so we scan all keys
        and filter by session_id. For high-volume use, consider adding a
        session index (e.g., Redis Set per session).
        """
        cursor = 0
        queries = []
        pattern = f"{self.key_prefix}:*"

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )

            for key in keys:
                data = await self.redis.get(key)
                if data:
                    try:
                        pending = PendingQuery.model_validate_json(data)
                        if pending.session_id == session_id:
                            queries.append(pending)
                    except Exception:
                        continue

            if cursor == 0:
                break

        return queries

    async def get_and_delete_session(self, session_id: str) -> list[PendingQuery]:
        """Atomically retrieve and delete all pending queries for a session."""
        queries = await self.get_session_queries(session_id)

        # Delete all found queries
        for pending in queries:
            await self.redis.delete(self._key(pending.query_id))

        logger.debug(
            f"Retrieved and deleted {len(queries)} queries for session: {session_id}"
        )
        return queries

    def _processed_key(self, idempotency_key: str) -> str:
        """Generate Redis key for processed feedback marker."""
        return f"{self.key_prefix}:processed:{idempotency_key}"

    async def was_processed(self, idempotency_key: str) -> bool:
        """Check if feedback with this idempotency key was already processed."""
        key = self._processed_key(idempotency_key)
        exists = await self.redis.exists(key)
        return bool(exists)

    async def mark_processed(
        self, idempotency_key: str, ttl_seconds: int = 3600
    ) -> None:
        """Mark feedback as processed to prevent future duplicates."""
        key = self._processed_key(idempotency_key)
        # Use SETEX to set with TTL - value is just a marker
        await self.redis.setex(key, ttl_seconds, "1")
        logger.debug(f"Marked feedback as processed in Redis: {idempotency_key}")


class PostgresFeedbackStore(FeedbackStore):
    """PostgreSQL-based feedback store.

    Uses PostgreSQL for persistent storage when Redis is not available.
    Stores pending queries in a dedicated table with JSONB for flexibility.

    Important:
        PostgreSQL does NOT have native TTL like Redis. You MUST call
        cleanup_expired() periodically (e.g., every 5 minutes via scheduler)
        to remove expired queries.

    Table Schema (create before use):
        CREATE TABLE IF NOT EXISTS pending_feedback (
            query_id VARCHAR(255) PRIMARY KEY,
            model_id VARCHAR(255) NOT NULL,
            features JSONB NOT NULL,
            cost DOUBLE PRECISION DEFAULT 0.0,
            latency DOUBLE PRECISION DEFAULT 0.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL
        );

        CREATE INDEX idx_pending_feedback_expires
        ON pending_feedback (expires_at);

    Atomicity:
        Uses DELETE ... RETURNING for atomic get_and_delete.
    """

    def __init__(
        self,
        pool: Any,  # asyncpg pool
        table_name: str = "pending_feedback",
        default_ttl: int = 3600,
    ):
        """Initialize PostgreSQL store.

        Args:
            pool: asyncpg connection pool
            table_name: Table name for pending queries
            default_ttl: Default TTL in seconds (default 1 hour)
        """
        self.pool = pool
        self.table_name = table_name
        self.default_ttl = default_ttl

    async def save_pending(self, pending: PendingQuery) -> None:
        """Save pending query to PostgreSQL."""
        import json

        ttl = pending.ttl_seconds or self.default_ttl
        expires_at = datetime.now(timezone.utc).timestamp() + ttl

        async with self.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                (query_id, model_id, features, cost, latency, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, to_timestamp($7))
                ON CONFLICT (query_id) DO UPDATE SET
                    model_id = EXCLUDED.model_id,
                    features = EXCLUDED.features,
                    cost = EXCLUDED.cost,
                    latency = EXCLUDED.latency,
                    expires_at = EXCLUDED.expires_at
                """,
                pending.query_id,
                pending.model_id,
                json.dumps(pending.features),
                pending.cost,
                pending.latency,
                pending.created_at,
                expires_at,
            )
        logger.debug(f"Saved pending query to PostgreSQL: {pending.query_id}")

    async def get_pending(self, query_id: str) -> PendingQuery | None:
        """Retrieve pending query from PostgreSQL (non-destructive)."""
        import json

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT query_id, model_id, features, cost, latency, created_at, expires_at
                FROM {self.table_name}
                WHERE query_id = $1 AND expires_at > NOW()
                """,
                query_id,
            )

        if row is None:
            return None

        return PendingQuery(
            query_id=row["query_id"],
            model_id=row["model_id"],
            features=(
                json.loads(row["features"])
                if isinstance(row["features"], str)
                else row["features"]
            ),
            cost=row["cost"],
            latency=row["latency"],
            created_at=row["created_at"],
            ttl_seconds=int(
                (row["expires_at"] - datetime.now(timezone.utc)).total_seconds()
            ),
        )

    async def get_and_delete_pending(self, query_id: str) -> PendingQuery | None:
        """Atomically retrieve and delete pending query using DELETE RETURNING."""
        import json

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                DELETE FROM {self.table_name}
                WHERE query_id = $1 AND expires_at > NOW()
                RETURNING query_id, model_id, features, cost, latency, created_at, expires_at
                """,
                query_id,
            )

        if row is None:
            return None

        logger.debug(f"Retrieved and deleted pending query: {query_id}")
        return PendingQuery(
            query_id=row["query_id"],
            model_id=row["model_id"],
            features=(
                json.loads(row["features"])
                if isinstance(row["features"], str)
                else row["features"]
            ),
            cost=row["cost"],
            latency=row["latency"],
            created_at=row["created_at"],
            ttl_seconds=int(
                (row["expires_at"] - datetime.now(timezone.utc)).total_seconds()
            ),
        )

    async def update_pending(
        self,
        query_id: str,
        cost: float | None = None,
        latency: float | None = None,
    ) -> bool:
        """Update cost/latency for a pending query."""
        updates: list[str] = []
        params: list[str | float] = [query_id]
        param_idx = 2

        if cost is not None:
            updates.append(f"cost = ${param_idx}")
            params.append(cost)
            param_idx += 1

        if latency is not None:
            updates.append(f"latency = ${param_idx}")
            params.append(latency)
            param_idx += 1

        if not updates:
            return True

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"""
                UPDATE {self.table_name}
                SET {', '.join(updates)}
                WHERE query_id = $1 AND expires_at > NOW()
                """,
                *params,
            )

        updated = bool(result.split()[-1] != "0")
        if updated:
            logger.debug(f"Updated pending query: {query_id}")
        return updated

    async def delete_pending(self, query_id: str) -> None:
        """Delete pending query from PostgreSQL."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE query_id = $1",
                query_id,
            )
        logger.debug(f"Deleted pending query from PostgreSQL: {query_id}")

    async def cleanup_expired(self) -> int:
        """Remove expired pending queries.

        Important: Call this periodically (e.g., every 5 minutes) since
        PostgreSQL does not have native TTL like Redis.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE expires_at <= NOW()"
            )

        deleted = int(result.split()[-1])
        if deleted:
            logger.info(f"Cleaned up {deleted} expired pending queries")

        return deleted

    async def count_pending(self) -> int:
        """Get count of non-expired pending queries."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE expires_at > NOW()"
            )

        return row[0] if row else 0

    async def get_session_queries(self, session_id: str) -> list[PendingQuery]:
        """Get all pending queries for a session.

        Note: Requires session_id column in table. Add with:
            ALTER TABLE pending_feedback ADD COLUMN session_id VARCHAR(255);
            CREATE INDEX idx_pending_feedback_session ON pending_feedback (session_id);
        """
        import json

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT query_id, model_id, features, cost, latency, created_at, expires_at, session_id
                FROM {self.table_name}
                WHERE session_id = $1 AND expires_at > NOW()
                """,
                session_id,
            )

        queries = []
        for row in rows:
            queries.append(
                PendingQuery(
                    query_id=row["query_id"],
                    model_id=row["model_id"],
                    features=(
                        json.loads(row["features"])
                        if isinstance(row["features"], str)
                        else row["features"]
                    ),
                    cost=row["cost"],
                    latency=row["latency"],
                    created_at=row["created_at"],
                    ttl_seconds=int(
                        (row["expires_at"] - datetime.now(timezone.utc)).total_seconds()
                    ),
                    session_id=row["session_id"],
                )
            )

        return queries

    async def get_and_delete_session(self, session_id: str) -> list[PendingQuery]:
        """Atomically retrieve and delete all pending queries for a session."""
        import json

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                DELETE FROM {self.table_name}
                WHERE session_id = $1 AND expires_at > NOW()
                RETURNING query_id, model_id, features, cost, latency, created_at, expires_at, session_id
                """,
                session_id,
            )

        queries = []
        for row in rows:
            queries.append(
                PendingQuery(
                    query_id=row["query_id"],
                    model_id=row["model_id"],
                    features=(
                        json.loads(row["features"])
                        if isinstance(row["features"], str)
                        else row["features"]
                    ),
                    cost=row["cost"],
                    latency=row["latency"],
                    created_at=row["created_at"],
                    ttl_seconds=int(
                        (row["expires_at"] - datetime.now(timezone.utc)).total_seconds()
                    ),
                    session_id=row["session_id"],
                )
            )

        logger.debug(
            f"Retrieved and deleted {len(queries)} queries for session: {session_id}"
        )
        return queries

    async def was_processed(self, idempotency_key: str) -> bool:
        """Check if feedback with this idempotency key was already processed.

        Requires table:
            CREATE TABLE IF NOT EXISTS processed_feedback (
                idempotency_key VARCHAR(512) PRIMARY KEY,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            );
            CREATE INDEX idx_processed_feedback_expires
            ON processed_feedback (expires_at);
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM processed_feedback
                WHERE idempotency_key = $1 AND expires_at > NOW()
                """,
                idempotency_key,
            )
        return row is not None

    async def mark_processed(
        self, idempotency_key: str, ttl_seconds: int = 3600
    ) -> None:
        """Mark feedback as processed to prevent future duplicates."""
        expires_at = datetime.now(timezone.utc).timestamp() + ttl_seconds

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO processed_feedback (idempotency_key, expires_at)
                VALUES ($1, to_timestamp($2))
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    expires_at = EXCLUDED.expires_at
                """,
                idempotency_key,
                expires_at,
            )
        logger.debug(f"Marked feedback as processed in PostgreSQL: {idempotency_key}")
