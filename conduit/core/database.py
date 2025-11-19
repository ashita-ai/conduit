"""Supabase database interface using PostgREST for REST API access.

Migration Rationale (2025-11-18):
    - Supabase free tier blocks direct PostgreSQL connections (port 5432)
    - Using supabase-py client library for REST API access instead of asyncpg
    - Maintains same interface, different underlying implementation
    - See notes/2025-11-18_database_connection_issue.md for details
"""

import json
import logging
import os
from typing import Any, cast

from supabase import AsyncClient, acreate_client

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
    """Supabase database interface via PostgREST API.

    Transaction Boundaries:
        - Single row inserts: Auto-commit (REST API atomicity)
        - Feedback loop updates: Application-level transaction emulation
        - Batch operations: Use RPC functions for server-side transactions
        - Model state updates: UPSERT for idempotent updates
        - Circuit breaker state: Auto-commit (eventual consistency)

    Note: PostgREST API doesn't support multi-statement transactions like asyncpg.
    For atomic multi-table operations, use database RPC functions (stored procedures).
    """

    def __init__(
        self, supabase_url: str | None = None, supabase_key: str | None = None
    ):
        """Initialize Supabase client configuration.

        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase anon/service key (defaults to SUPABASE_ANON_KEY env var)

        Raises:
            ValueError: If URL or key not provided and not in environment
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided or set in environment "
                "(SUPABASE_URL, SUPABASE_ANON_KEY)"
            )

        self.client: AsyncClient | None = None

    async def connect(self) -> None:
        """Create Supabase async client connection."""
        if not self.supabase_url or not self.supabase_key:
            raise DatabaseError("Supabase URL and key must be set before connecting")
        self.client = await acreate_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client connected via REST API")

    async def disconnect(self) -> None:
        """Close Supabase client connection.

        Note: supabase-py AsyncClient doesn't have explicit close() method.
        Connection cleanup relies on garbage collection. Setting client to None
        allows garbage collector to clean up resources.
        """
        if self.client:
            self.client = None
            logger.info("Supabase client connection closed")

    async def save_query(self, query: Query) -> str:
        """Save query and return ID.

        Transaction: None (single INSERT via REST API)

        Args:
            query: Query to save

        Returns:
            Query ID

        Raises:
            DatabaseError: If save fails
        """
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            # Prepare data for insertion
            data = {
                "id": query.id,
                "text": query.text,
                "user_id": query.user_id,
                "context": json.dumps(query.context) if query.context else None,
                "constraints": (
                    json.dumps(query.constraints.model_dump())
                    if query.constraints
                    else None
                ),
                "created_at": query.created_at.isoformat(),
            }

            # Insert via PostgREST
            response = await self.client.table("queries").insert(data).execute()

            if not response.data:
                raise DatabaseError("Insert returned no data")

            row = cast(dict[str, Any], response.data[0])
            return str(row["id"])

        except Exception as e:
            logger.error(f"Failed to save query: {e}")
            raise DatabaseError(f"Failed to save query: {e}") from e

    async def save_complete_interaction(
        self,
        routing: RoutingDecision | None,
        response: Response,
        feedback: Feedback | None = None,
    ) -> None:
        """Save routing decision, response, and optional feedback.

        Transaction: Application-level (PostgREST doesn't support transactions)
        Note: For true atomicity, implement as database RPC function in future

        Args:
            routing: Routing decision (optional - for feedback-only saves)
            response: LLM response
            feedback: Optional user feedback

        Raises:
            DatabaseError: If save fails
        """
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            # Save routing decision (if provided)
            if routing is not None:
                routing_data = {
                    "id": routing.id,
                    "query_id": routing.query_id,
                    "selected_model": routing.selected_model,
                    "confidence": routing.confidence,
                    "features": json.dumps(routing.features.model_dump()),
                    "reasoning": routing.reasoning,
                    "created_at": routing.created_at.isoformat(),
                }
                await self.client.table("routing_decisions").insert(routing_data).execute()  # type: ignore[arg-type]

            # Save response
            response_data = {
                "id": response.id,
                "query_id": response.query_id,
                "model": response.model,
                "text": response.text,
                "cost": response.cost,
                "latency": response.latency,
                "tokens": response.tokens,
                "created_at": response.created_at.isoformat(),
            }
            await self.client.table("responses").insert(response_data).execute()  # type: ignore[arg-type]

            # Save feedback if provided
            if feedback:
                feedback_data = {
                    "id": feedback.id,
                    "response_id": feedback.response_id,
                    "quality_score": feedback.quality_score,
                    "user_rating": feedback.user_rating,
                    "met_expectations": feedback.met_expectations,
                    "comments": feedback.comments,
                    "created_at": feedback.created_at.isoformat(),
                }
                await self.client.table("feedback").insert(feedback_data).execute()  # type: ignore[arg-type]

            routing_id = routing.id if routing is not None else "none"
            logger.info(
                f"Saved complete interaction: routing={routing_id}, response={response.id}"
            )

        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            raise DatabaseError(f"Failed to save interaction: {e}") from e

    async def update_model_state(self, state: ModelState) -> None:
        """Update model's Beta parameters.

        Transaction: None (UPSERT via PostgREST)
        Concurrency: Last-write-wins acceptable for ML updates

        Args:
            state: Model state to update

        Raises:
            DatabaseError: If update fails
        """
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            data = {
                "model_id": state.model_id,
                "alpha": state.alpha,
                "beta": state.beta,
                "total_requests": state.total_requests,
                "total_cost": state.total_cost,
                "avg_quality": state.avg_quality,
                "updated_at": state.updated_at.isoformat(),
            }

            # PostgREST upsert using on_conflict parameter
            await (
                self.client.table("model_states")
                .upsert(data, on_conflict="model_id")  # type: ignore[arg-type]
                .execute()
            )

            logger.debug(f"Updated model state: {state.model_id}")

        except Exception as e:
            logger.error(f"Failed to update model state: {e}")
            raise DatabaseError(f"Failed to update model state: {e}") from e

    async def get_model_states(self) -> dict[str, ModelState]:
        """Load all model states.

        Transaction: None (single SELECT via REST API)

        Returns:
            Dictionary mapping model_id to ModelState

        Raises:
            DatabaseError: If load fails
        """
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            response = await self.client.table("model_states").select("*").execute()

            states = {}
            for item in response.data:
                row = cast(dict[str, Any], item)
                states[str(row["model_id"])] = ModelState(
                    model_id=str(row["model_id"]),
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

        Pricing is stored in the ``model_prices`` table with costs expressed
        per one million tokens. This method converts the raw rows into
        :class:`ModelPricing` instances keyed by ``model_id``.

        Transaction: None (single SELECT via REST API)

        Returns:
            Dictionary mapping model_id to ModelPricing

        Raises:
            DatabaseError: If load fails or client is not connected
        """
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            response = await self.client.table("model_prices").select("*").execute()

            prices: dict[str, ModelPricing] = {}
            for item in response.data or []:
                row = cast(dict[str, Any], item)
                # Supabase returns timestamps as ISO 8601 strings; Pydantic will
                # handle conversion for the snapshot_at field.
                pricing = ModelPricing(
                    model_id=str(row["model_id"]),
                    input_cost_per_million=float(row["input_cost_per_million"]),
                    output_cost_per_million=float(row["output_cost_per_million"]),
                    cached_input_cost_per_million=(
                        float(row["cached_input_cost_per_million"])
                        if row.get("cached_input_cost_per_million") is not None
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
        if not self.client:
            raise DatabaseError("Database not connected")

        try:
            response = await (
                self.client.table("responses")
                .select("*")
                .eq("id", response_id)
                .execute()
            )

            if not response.data:
                return None

            row = cast(dict[str, Any], response.data[0])
            return Response(
                id=str(row["id"]),
                query_id=str(row["query_id"]),
                model=str(row["model"]),
                text=str(row["text"]),
                cost=float(row["cost"]),
                latency=float(row["latency"]),
                tokens=int(row["tokens"]),
                created_at=row["created_at"],
            )

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise DatabaseError(f"Failed to get response: {e}") from e
