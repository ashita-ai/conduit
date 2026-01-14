"""Decision audit logging for compliance, debugging, and analysis.

This module provides audit trail capabilities for routing decisions:
- Captures detailed decision context (scores, features, constraints)
- Stores to PostgreSQL for querying and compliance
- Supports retention policies for data management

Use Cases:
    - Debugging: Why did Conduit select model X for query Y?
    - Compliance: Regulatory audit of AI decision-making (EU AI Act)
    - Analysis: Post-mortem investigation of routing behavior
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from conduit.core.models import QueryFeatures, RoutingDecision

logger = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    """Audit log entry for a routing decision.

    Captures complete decision context for compliance and debugging.

    Attributes:
        id: Auto-generated primary key (set by database)
        decision_id: UUID of the RoutingDecision
        query_id: UUID of the original query
        selected_model: Model ID that was selected
        fallback_chain: Ordered list of fallback models
        confidence: Decision confidence score (0-1)
        algorithm_phase: Current algorithm phase (e.g., "thompson_sampling", "linucb")
        query_count: Total queries processed by router at decision time
        arm_scores: Score breakdown for each arm at decision time
        feature_vector: Feature vector used for contextual algorithms (optional)
        constraints_applied: Any constraints that affected the decision
        reasoning: Human-readable explanation of the decision
        created_at: When the decision was made
    """

    id: int | None = None
    decision_id: str
    query_id: str
    selected_model: str
    fallback_chain: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    algorithm_phase: str
    query_count: int = Field(ge=0)
    arm_scores: dict[str, dict[str, float]]
    feature_vector: list[float] | None = None
    constraints_applied: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditQuery(BaseModel):
    """Query parameters for filtering audit log entries.

    All parameters are optional. When multiple are specified, they are ANDed.

    Attributes:
        decision_id: Filter by specific decision UUID
        query_id: Filter by specific query UUID
        selected_model: Filter by selected model
        algorithm_phase: Filter by algorithm phase
        start_time: Filter entries after this time (inclusive)
        end_time: Filter entries before this time (exclusive)
        limit: Maximum number of entries to return
        offset: Number of entries to skip (for pagination)
    """

    decision_id: str | None = None
    query_id: str | None = None
    selected_model: str | None = None
    algorithm_phase: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


@runtime_checkable
class AuditStore(Protocol):
    """Protocol for audit log storage backends.

    Implementations must provide methods for logging and querying audit entries.
    The default implementation is PostgresAuditStore.
    """

    async def log_decision(self, entry: AuditEntry) -> int:
        """Log a routing decision to the audit trail.

        Args:
            entry: Audit entry to log

        Returns:
            ID of the created audit entry
        """
        ...

    async def get_entry(self, entry_id: int) -> AuditEntry | None:
        """Get a specific audit entry by ID.

        Args:
            entry_id: Primary key of the entry

        Returns:
            AuditEntry if found, None otherwise
        """
        ...

    async def get_by_decision_id(self, decision_id: str) -> AuditEntry | None:
        """Get audit entry by decision ID.

        Args:
            decision_id: UUID of the routing decision

        Returns:
            AuditEntry if found, None otherwise
        """
        ...

    async def query(self, query: AuditQuery) -> list[AuditEntry]:
        """Query audit entries with filters.

        Args:
            query: Query parameters for filtering

        Returns:
            List of matching audit entries
        """
        ...

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete entries older than cutoff for retention policy.

        Args:
            cutoff: Delete entries with created_at before this time

        Returns:
            Number of entries deleted
        """
        ...


class PostgresAuditStore:
    """PostgreSQL implementation of AuditStore.

    Uses the decision_audit table created by the Alembic migration.
    Supports efficient querying with indexes on common filter columns.

    Attributes:
        pool: asyncpg connection pool
        retention_days: Automatic retention policy (0 = disabled)
    """

    def __init__(self, pool: Any, retention_days: int = 90) -> None:
        """Initialize PostgreSQL audit store.

        Args:
            pool: asyncpg connection pool
            retention_days: Days to retain audit entries (0 = forever)
        """
        self.pool = pool
        self.retention_days = retention_days

    async def log_decision(self, entry: AuditEntry) -> int:
        """Log a routing decision to the audit trail.

        Args:
            entry: Audit entry to log

        Returns:
            ID of the created audit entry
        """
        import json

        query = """
            INSERT INTO decision_audit (
                decision_id, query_id, selected_model, fallback_chain,
                confidence, algorithm_phase, query_count, arm_scores,
                feature_vector, constraints_applied, reasoning, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                entry.decision_id,
                entry.query_id,
                entry.selected_model,
                entry.fallback_chain,
                entry.confidence,
                entry.algorithm_phase,
                entry.query_count,
                json.dumps(entry.arm_scores),
                json.dumps(entry.feature_vector) if entry.feature_vector else None,
                json.dumps(entry.constraints_applied),
                entry.reasoning,
                entry.created_at,
            )
            return int(row["id"])

    async def get_entry(self, entry_id: int) -> AuditEntry | None:
        """Get a specific audit entry by ID."""
        query = """
            SELECT id, decision_id, query_id, selected_model, fallback_chain,
                   confidence, algorithm_phase, query_count, arm_scores,
                   feature_vector, constraints_applied, reasoning, created_at
            FROM decision_audit
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, entry_id)
            if row is None:
                return None
            return self._row_to_entry(row)

    async def get_by_decision_id(self, decision_id: str) -> AuditEntry | None:
        """Get audit entry by decision ID."""
        query = """
            SELECT id, decision_id, query_id, selected_model, fallback_chain,
                   confidence, algorithm_phase, query_count, arm_scores,
                   feature_vector, constraints_applied, reasoning, created_at
            FROM decision_audit
            WHERE decision_id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, decision_id)
            if row is None:
                return None
            return self._row_to_entry(row)

    async def query(self, audit_query: AuditQuery) -> list[AuditEntry]:
        """Query audit entries with filters."""
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if audit_query.decision_id:
            conditions.append(f"decision_id = ${param_idx}")
            params.append(audit_query.decision_id)
            param_idx += 1

        if audit_query.query_id:
            conditions.append(f"query_id = ${param_idx}")
            params.append(audit_query.query_id)
            param_idx += 1

        if audit_query.selected_model:
            conditions.append(f"selected_model = ${param_idx}")
            params.append(audit_query.selected_model)
            param_idx += 1

        if audit_query.algorithm_phase:
            conditions.append(f"algorithm_phase = ${param_idx}")
            params.append(audit_query.algorithm_phase)
            param_idx += 1

        if audit_query.start_time:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(audit_query.start_time)
            param_idx += 1

        if audit_query.end_time:
            conditions.append(f"created_at < ${param_idx}")
            params.append(audit_query.end_time)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT id, decision_id, query_id, selected_model, fallback_chain,
                   confidence, algorithm_phase, query_count, arm_scores,
                   feature_vector, constraints_applied, reasoning, created_at
            FROM decision_audit
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([audit_query.limit, audit_query.offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_entry(row) for row in rows]

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete entries older than cutoff."""
        query = "DELETE FROM decision_audit WHERE created_at < $1"

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, cutoff)
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} audit entries older than {cutoff}")
            return count

    async def apply_retention_policy(self) -> int:
        """Apply retention policy, deleting old entries.

        Returns:
            Number of entries deleted
        """
        if self.retention_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        return await self.delete_older_than(cutoff)

    def _row_to_entry(self, row: Any) -> AuditEntry:
        """Convert database row to AuditEntry."""
        return AuditEntry(
            id=row["id"],
            decision_id=row["decision_id"],
            query_id=row["query_id"],
            selected_model=row["selected_model"],
            fallback_chain=list(row["fallback_chain"]) if row["fallback_chain"] else [],
            confidence=float(row["confidence"]),
            algorithm_phase=row["algorithm_phase"],
            query_count=row["query_count"],
            arm_scores=row["arm_scores"],
            feature_vector=row["feature_vector"],
            constraints_applied=row["constraints_applied"] or {},
            reasoning=row["reasoning"],
            created_at=row["created_at"],
        )


class InMemoryAuditStore:
    """In-memory implementation for testing and development.

    Stores entries in a list without persistence. Useful for:
    - Unit testing without database
    - Development without PostgreSQL
    - Graceful degradation when DB unavailable
    """

    def __init__(self, max_entries: int = 10000) -> None:
        """Initialize in-memory store.

        Args:
            max_entries: Maximum entries to keep (oldest dropped first)
        """
        self.entries: list[AuditEntry] = []
        self.max_entries = max_entries
        self._next_id = 1

    async def log_decision(self, entry: AuditEntry) -> int:
        """Log a routing decision."""
        entry_copy = entry.model_copy()
        entry_copy.id = self._next_id
        self._next_id += 1

        self.entries.append(entry_copy)

        # Trim oldest entries if over limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

        # entry_copy.id is guaranteed to be set (not None) at this point
        assert entry_copy.id is not None
        return entry_copy.id

    async def get_entry(self, entry_id: int) -> AuditEntry | None:
        """Get entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    async def get_by_decision_id(self, decision_id: str) -> AuditEntry | None:
        """Get entry by decision ID."""
        for entry in self.entries:
            if entry.decision_id == decision_id:
                return entry
        return None

    async def query(self, audit_query: AuditQuery) -> list[AuditEntry]:
        """Query entries with filters."""
        results = []

        for entry in reversed(self.entries):  # Most recent first
            if audit_query.decision_id and entry.decision_id != audit_query.decision_id:
                continue
            if audit_query.query_id and entry.query_id != audit_query.query_id:
                continue
            if audit_query.selected_model and entry.selected_model != audit_query.selected_model:
                continue
            if audit_query.algorithm_phase and entry.algorithm_phase != audit_query.algorithm_phase:
                continue
            if audit_query.start_time and entry.created_at < audit_query.start_time:
                continue
            if audit_query.end_time and entry.created_at >= audit_query.end_time:
                continue

            results.append(entry)

            if len(results) >= audit_query.offset + audit_query.limit:
                break

        return results[audit_query.offset : audit_query.offset + audit_query.limit]

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete entries older than cutoff."""
        original_count = len(self.entries)
        self.entries = [e for e in self.entries if e.created_at >= cutoff]
        deleted = original_count - len(self.entries)
        return deleted


def create_audit_entry(
    decision: RoutingDecision,
    algorithm_phase: str,
    query_count: int,
    arm_scores: dict[str, dict[str, float]],
    constraints_applied: dict[str, Any] | None = None,
) -> AuditEntry:
    """Create an audit entry from a routing decision.

    Convenience function to construct AuditEntry from decision context.

    Args:
        decision: The routing decision to audit
        algorithm_phase: Current algorithm phase (e.g., "thompson_sampling")
        query_count: Total queries processed by router
        arm_scores: Score breakdown for each arm
        constraints_applied: Any constraints that affected the decision

    Returns:
        AuditEntry ready for logging

    Example:
        >>> entry = create_audit_entry(
        ...     decision=decision,
        ...     algorithm_phase="linucb",
        ...     query_count=1500,
        ...     arm_scores=bandit.compute_scores(features),
        ...     constraints_applied={"max_cost": 0.01}
        ... )
        >>> await audit_store.log_decision(entry)
    """
    # Extract feature vector if available (for contextual algorithms)
    feature_vector = None
    if decision.features and decision.features.embedding:
        # Include embedding + metadata features
        feature_vector = (
            decision.features.embedding
            + [
                decision.features.token_count / 1000.0,
                decision.features.complexity_score,
            ]
        )

    return AuditEntry(
        decision_id=decision.id,
        query_id=decision.query_id,
        selected_model=decision.selected_model,
        fallback_chain=decision.fallback_chain,
        confidence=decision.confidence,
        algorithm_phase=algorithm_phase,
        query_count=query_count,
        arm_scores=arm_scores,
        feature_vector=feature_vector,
        constraints_applied=constraints_applied or {},
        reasoning=decision.reasoning,
        created_at=decision.created_at,
    )
