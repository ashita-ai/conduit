"""PostgreSQL database interface using asyncpg for async connection pooling.

Supports any PostgreSQL database (self-hosted, AWS RDS, Supabase, etc.)
via standard connection string: postgresql://user:pass@host:port/database
"""

import json
import logging
import os

import asyncpg

from conduit.core.exceptions import DatabaseError
from conduit.core.models import (
    Feedback,
    ModelState,
    Query,
    Response,
    RoutingDecision,
)
from conduit.core.pricing import ModelPricing

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database interface with connection pooling.

    Uses asyncpg for high-performance async PostgreSQL connections.
    Works with any PostgreSQL database: self-hosted, AWS RDS, Google Cloud SQL,
    Supabase, Neon, or any other Postgres provider.

    Connection pooling:
        - Min connections: 5 (configurable)
        - Max connections: 20 (configurable)
        - Connection timeout: 10 seconds
        - Command timeout: 60 seconds

    Transaction boundaries:
        - Single row operations: Auto-commit
        - Multi-table saves: Explicit transactions
        - UPSERT operations: Single statement (atomic)
    """

    def __init__(
        self,
        database_url: str | None = None,
        min_size: int = 5,
        max_size: int = 20,
    ):
        """Initialize PostgreSQL connection configuration.

        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
                Format: postgresql://user:password@host:port/database
            min_size: Minimum number of connections in pool
            max_size: Maximum number of connections in pool

        Raises:
            ValueError: If database_url not provided and not in environment
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")

        if not self.database_url:
            raise ValueError(
                "Database URL must be provided or set in environment (DATABASE_URL). "
                "Format: postgresql://user:password@host:port/database"
            )

        self.min_size = min_size
        self.max_size = max_size
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create PostgreSQL connection pool."""
        if not self.database_url:
            raise DatabaseError("Database URL must be set before connecting")

        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
            )
            logger.info(
                f"PostgreSQL pool created: {self.min_size}-{self.max_size} connections"
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool and all connections."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")

    async def save_query(self, query: Query) -> str:
        """Save query and return ID.

        Args:
            query: Query to save

        Returns:
            Query ID

        Raises:
            DatabaseError: If save fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO queries (id, text, user_id, context, constraints, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    query.id,
                    query.text,
                    query.user_id,
                    json.dumps(query.context) if query.context else None,
                    (
                        json.dumps(query.constraints.model_dump())
                        if query.constraints
                        else None
                    ),
                    query.created_at,
                )

            return query.id

        except Exception as e:
            logger.error(f"Failed to save query: {e}")
            raise DatabaseError(f"Failed to save query: {e}") from e

    async def save_complete_interaction(
        self,
        routing: RoutingDecision | None,
        response: Response,
        feedback: Feedback | None = None,
    ) -> None:
        """Save routing decision, response, and optional feedback in a transaction.

        Uses explicit transaction for atomicity across multiple tables.

        Args:
            routing: Routing decision (optional - for feedback-only saves)
            response: LLM response
            feedback: Optional user feedback

        Raises:
            DatabaseError: If save fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn, conn.transaction():
                # Save routing decision (if provided)
                if routing is not None:
                    await conn.execute(
                        """
                        INSERT INTO routing_decisions
                            (id, query_id, selected_model, confidence, features, reasoning, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        routing.id,
                        routing.query_id,
                        routing.selected_model,
                        routing.confidence,
                        json.dumps(routing.features.model_dump()),
                        routing.reasoning,
                        routing.created_at,
                    )

                # Save response
                await conn.execute(
                    """
                    INSERT INTO responses
                        (id, query_id, model, text, cost, latency, tokens, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    response.id,
                    response.query_id,
                    response.model,
                    response.text,
                    response.cost,
                    response.latency,
                    response.tokens,
                    response.created_at,
                )

                # Save feedback if provided
                if feedback:
                    await conn.execute(
                        """
                        INSERT INTO feedback
                            (id, response_id, quality_score, user_rating, met_expectations, comments, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        feedback.id,
                        feedback.response_id,
                        feedback.quality_score,
                        feedback.user_rating,
                        feedback.met_expectations,
                        feedback.comments,
                        feedback.created_at,
                    )

            routing_id = routing.id if routing is not None else "none"
            logger.info(
                f"Saved complete interaction: routing={routing_id}, response={response.id}"
            )

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            raise DatabaseError(f"Failed to save interaction: {e}") from e

    async def update_model_state(self, state: ModelState) -> None:
        """Update model's state parameters using UPSERT.

        Uses INSERT ... ON CONFLICT DO UPDATE for atomic upsert.

        Args:
            state: Model state to update

        Raises:
            DatabaseError: If update fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO model_states
                        (model_id, alpha, beta, total_requests, total_cost, avg_quality, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (model_id)
                    DO UPDATE SET
                        alpha = EXCLUDED.alpha,
                        beta = EXCLUDED.beta,
                        total_requests = EXCLUDED.total_requests,
                        total_cost = EXCLUDED.total_cost,
                        avg_quality = EXCLUDED.avg_quality,
                        updated_at = EXCLUDED.updated_at
                    """,
                    state.model_id,
                    state.alpha,
                    state.beta,
                    state.total_requests,
                    state.total_cost,
                    state.avg_quality,
                    state.updated_at,
                )

            logger.debug(f"Updated model state: {state.model_id}")

        except Exception as e:
            logger.error(f"Failed to update model state: {e}")
            raise DatabaseError(f"Failed to update model state: {e}") from e

    async def get_model_states(self) -> dict[str, ModelState]:
        """Load all model states.

        Returns:
            Dictionary mapping model_id to ModelState

        Raises:
            DatabaseError: If load fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM model_states")

            states = {}
            for row in rows:
                states[row["model_id"]] = ModelState(
                    model_id=row["model_id"],
                    alpha=float(row["alpha"]),
                    beta=float(row["beta"]),
                    total_requests=int(row["total_requests"]),
                    total_cost=float(row["total_cost"]),
                    avg_quality=float(row["avg_quality"]),
                    updated_at=row["updated_at"],
                )

            logger.info(f"Loaded {len(states)} model states")
            return states

        except Exception as e:
            logger.error(f"Failed to load model states: {e}")
            raise DatabaseError(f"Failed to load model states: {e}") from e

    async def get_model_prices(self) -> dict[str, ModelPricing]:
        """Load per-model pricing information.

        Pricing is stored in the model_prices table with costs expressed
        per one million tokens.

        Returns:
            Dictionary mapping model_id to ModelPricing

        Raises:
            DatabaseError: If load fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM model_prices")

            prices: dict[str, ModelPricing] = {}
            for row in rows:
                pricing = ModelPricing(
                    model_id=row["model_id"],
                    input_cost_per_million=float(row["input_cost_per_million"]),
                    output_cost_per_million=float(row["output_cost_per_million"]),
                    cached_input_cost_per_million=(
                        float(row["cached_input_cost_per_million"])
                        if row["cached_input_cost_per_million"] is not None
                        else None
                    ),
                    source=row.get("source"),
                    snapshot_at=row.get("snapshot_at"),
                )
                prices[pricing.model_id] = pricing

            logger.info(f"Loaded {len(prices)} model price entries")
            return prices

        except Exception as e:
            logger.error(f"Failed to load model prices: {e}")
            raise DatabaseError(f"Failed to load model prices: {e}") from e

    async def get_response_by_id(self, response_id: str) -> Response | None:
        """Get response by ID.

        Args:
            response_id: Response ID

        Returns:
            Response if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        if not self.pool:
            raise DatabaseError("Database not connected")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM responses WHERE id = $1", response_id
                )

            if not row:
                return None

            return Response(
                id=row["id"],
                query_id=row["query_id"],
                model=row["model"],
                text=row["text"],
                cost=float(row["cost"]),
                latency=float(row["latency"]),
                tokens=int(row["tokens"]),
                created_at=row["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise DatabaseError(f"Failed to get response: {e}") from e

    async def fetch_latest_evaluation_metrics(
        self, time_window: str = "last_hour"
    ) -> dict[str, float | None]:
        """Fetch latest evaluation metrics for OTEL export.

        Args:
            time_window: Time window to filter metrics (last_hour, last_day, etc.)

        Returns:
            Dictionary with metric names and their latest values:
                - regret_oracle: Latest regret vs oracle baseline
                - regret_random: Latest regret vs random baseline
                - quality_trend: Latest quality trend value
                - cost_efficiency: Latest cost efficiency value
                - convergence_rate: Latest convergence rate

        Note:
            Returns None for metrics that don't exist in the table.
        """
        if not self.pool:
            raise DatabaseError("Database connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                # Fetch latest value for each metric type
                query = """
                    SELECT DISTINCT ON (metric_name)
                        metric_name,
                        metric_value
                    FROM evaluation_metrics
                    WHERE time_window = $1
                    ORDER BY metric_name, created_at DESC
                """
                rows = await conn.fetch(query, time_window)

                # Convert to dictionary
                metrics: dict[str, float | None] = {
                    "regret_oracle": None,
                    "regret_random": None,
                    "quality_trend": None,
                    "cost_efficiency": None,
                    "convergence_rate": None,
                }

                for row in rows:
                    metric_name = row["metric_name"]
                    metric_value = float(row["metric_value"])

                    # Map database metric names to our keys
                    if metric_name == "regret_vs_oracle":
                        metrics["regret_oracle"] = metric_value
                    elif metric_name == "regret_vs_random":
                        metrics["regret_random"] = metric_value
                    elif metric_name == "quality_trend":
                        metrics["quality_trend"] = metric_value
                    elif metric_name == "cost_efficiency":
                        metrics["cost_efficiency"] = metric_value
                    elif metric_name == "convergence_rate":
                        metrics["convergence_rate"] = metric_value

                return metrics

        except Exception as e:
            logger.error(f"Failed to fetch evaluation metrics: {e}")
            raise DatabaseError(f"Failed to fetch evaluation metrics: {e}") from e
