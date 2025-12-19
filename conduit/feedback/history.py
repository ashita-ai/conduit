"""Query history tracking for retry detection.

Manages recent query history in Redis for semantic similarity-based
retry detection. Stores query metadata (embeddings, timestamps, IDs)
with automatic expiration for memory efficiency.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError

from conduit.core.models import QueryFeatures


class QueryHistoryEntry(BaseModel):
    """Single query history entry.

    Attributes:
        query_id: Unique query identifier
        query_text: Original query text
        embedding: Query embedding vector
        timestamp: Unix timestamp of query
        user_id: User identifier (API key or client ID)
        model_used: Which model was selected for this query
    """

    query_id: str = Field(..., description="Query identifier")
    query_text: str = Field(..., description="Original query")
    embedding: list[float] = Field(..., description="Query embedding")
    timestamp: float = Field(..., description="Unix timestamp", ge=0.0)
    user_id: str = Field(..., description="User identifier")
    model_used: str | None = Field(None, description="Selected model")


class QueryHistoryTracker:
    """Redis-based query history tracker for retry detection.

    Stores recent queries per user with automatic expiration (5-minute TTL).
    Enables semantic similarity-based retry detection by maintaining
    query embeddings and metadata in Redis.

    Architecture:
        - Key pattern: "conduit:history:{user_id}:{query_id}"
        - User index: "conduit:history:{user_id}:index" (sorted set by timestamp)
        - TTL: 300 seconds (5 minutes) for automatic cleanup
        - Namespace: Separate from cache to avoid key collisions

    Example:
        >>> tracker = QueryHistoryTracker(redis_client)
        >>> await tracker.add_query(
        ...     query_id="q123",
        ...     query_text="What is Python?",
        ...     features=query_features,
        ...     user_id="user_abc",
        ...     model_used="gpt-4o-mini"
        ... )
        >>> recent = await tracker.get_recent_queries("user_abc", limit=10)
    """

    def __init__(
        self,
        redis: Redis[bytes] | None = None,
        ttl_seconds: int = 300,  # 5 minutes
    ):
        """Initialize query history tracker.

        Args:
            redis: Redis client instance (shared with cache service)
            ttl_seconds: Time-to-live for history entries (default 5 minutes)
        """
        self.redis = redis
        self.ttl = ttl_seconds
        self.enabled = redis is not None

    async def add_query(
        self,
        query_id: str,
        query_text: str,
        features: QueryFeatures,
        user_id: str,
        model_used: str | None = None,
    ) -> bool:
        """Add query to user's history.

        Args:
            query_id: Unique query identifier
            query_text: Original query text
            features: Query features (includes embedding)
            user_id: User identifier
            model_used: Selected model for this query

        Returns:
            True if added successfully, False on error or disabled

        Note:
            Automatically expires after TTL (5 minutes by default).
            Stores in Redis sorted set for efficient time-based retrieval.
        """
        if not self.enabled or not self.redis:
            return False

        try:
            timestamp = time.time()

            # Create history entry
            entry = QueryHistoryEntry(
                query_id=query_id,
                query_text=query_text,
                embedding=features.embedding,
                timestamp=timestamp,
                user_id=user_id,
                model_used=model_used,
            )

            # Store entry with TTL
            entry_key = f"conduit:history:{user_id}:{query_id}"
            await self.redis.setex(
                entry_key,
                self.ttl,
                entry.model_dump_json(),
            )

            # Add to user's sorted set index (score = timestamp)
            index_key = f"conduit:history:{user_id}:index"
            await self.redis.zadd(index_key, {query_id: timestamp})
            await self.redis.expire(index_key, self.ttl)

            return True

        except (ConnectionError, TimeoutError):
            # Graceful degradation - history tracking is non-critical
            return False

    async def get_recent_queries(
        self,
        user_id: str,
        limit: int = 10,
        time_window_seconds: float | None = None,
    ) -> list[QueryHistoryEntry]:
        """Retrieve recent queries for user.

        Args:
            user_id: User identifier
            limit: Maximum number of queries to return
            time_window_seconds: Optional time window (e.g., 300 for 5 minutes)

        Returns:
            List of QueryHistoryEntry objects, most recent first

        Note:
            Returns empty list on error or if tracking is disabled.
            Automatically filters expired entries based on time window.
        """
        if not self.enabled or not self.redis:
            return []

        try:
            # Get recent query IDs from sorted set
            index_key = f"conduit:history:{user_id}:index"

            if time_window_seconds:
                # Get queries within time window
                min_timestamp = time.time() - time_window_seconds
                query_ids = await self.redis.zrangebyscore(
                    index_key,
                    min_timestamp,
                    "+inf",  # Up to now
                    withscores=False,
                )
            else:
                # Get all recent queries (up to limit)
                query_ids = await self.redis.zrange(
                    index_key,
                    -limit,  # Last N items
                    -1,  # Up to end
                    withscores=False,
                )

            # Retrieve full entries
            entries: list[QueryHistoryEntry] = []
            for query_id in reversed(query_ids):  # Most recent first
                entry_key = f"conduit:history:{user_id}:{query_id.decode()}"
                entry_json = await self.redis.get(entry_key)

                if entry_json:
                    entry = QueryHistoryEntry.model_validate_json(entry_json)
                    entries.append(entry)

                if len(entries) >= limit:
                    break

            return entries

        except (ConnectionError, TimeoutError):
            return []

    async def find_similar_query(
        self,
        current_embedding: list[float],
        user_id: str,
        similarity_threshold: float = 0.85,
        time_window_seconds: float = 300.0,
    ) -> QueryHistoryEntry | None:
        """Find most similar recent query for retry detection.

        Args:
            current_embedding: Embedding of current query
            user_id: User identifier
            similarity_threshold: Minimum cosine similarity (0.85 default)
            time_window_seconds: Time window for retry consideration (5 min default)

        Returns:
            Most similar QueryHistoryEntry if above threshold, None otherwise

        Algorithm:
            1. Retrieve recent queries within time window
            2. Calculate cosine similarity with current embedding
            3. Return highest similarity above threshold

        Note:
            Cosine similarity ranges from -1 (opposite) to 1 (identical).
            Threshold of 0.85 indicates high semantic similarity.
        """
        if not self.enabled or not self.redis:
            return None

        recent_queries = await self.get_recent_queries(
            user_id=user_id,
            limit=10,  # Check last 10 queries
            time_window_seconds=time_window_seconds,
        )

        if not recent_queries:
            return None

        best_match: QueryHistoryEntry | None = None
        best_similarity = similarity_threshold

        for entry in recent_queries:
            similarity = self._cosine_similarity(current_embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        return best_match

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)

        Note:
            Assumes embeddings are already normalized (from sentence-transformers).
            For normalized vectors: cosine_similarity = dot_product
        """
        # Embeddings from sentence-transformers are normalized
        # So cosine similarity = dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return float(dot_product)

    async def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user (admin operation).

        Args:
            user_id: User identifier

        Returns:
            True if cleared successfully, False on error

        Warning:
            This is a maintenance operation. Normal expiration
            handles cleanup automatically via TTL.
        """
        if not self.enabled or not self.redis:
            return False

        try:
            # Get all query IDs from index
            index_key = f"conduit:history:{user_id}:index"
            query_ids = await self.redis.zrange(index_key, 0, -1, withscores=False)

            # Delete all entry keys
            keys_to_delete = [
                f"conduit:history:{user_id}:{qid.decode()}" for qid in query_ids
            ]
            keys_to_delete.append(index_key)

            if keys_to_delete:
                await self.redis.delete(*keys_to_delete)

            return True

        except (ConnectionError, TimeoutError):
            return False
