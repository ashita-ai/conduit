"""Integration tests for Database using real PostgreSQL instance.

These tests require:
- DATABASE_URL environment variable (postgresql://...)
- Database schema migrated (run alembic upgrade head)
"""

import os
from datetime import datetime, timezone
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

# Skip all tests if DATABASE_URL not available
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL not configured"
)


@pytest.fixture
async def db():
    """Create connected database instance."""
    database = Database()
    await database.connect()
    yield database
    await database.disconnect()


@pytest.fixture
def sample_query():
    """Create sample query for testing."""
    return Query(
        id=f"test-query-{uuid4()}",
        text="What is 2+2?",
        user_id="test-user",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_query_with_constraints():
    """Create sample query with constraints."""
    return Query(
        id=f"test-query-constraints-{uuid4()}",
        text="Complex query",
        user_id="test-user",
        constraints=QueryConstraints(max_cost=0.01, max_latency=2.0, min_quality=0.8),
        context={"source": "test"},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_routing_decision(sample_query):
    """Create sample routing decision."""
    features = QueryFeatures(
        embedding=[0.1] * 384, token_count=10, complexity_score=0.2
    )
    return RoutingDecision(
        id=f"test-routing-{uuid4()}",
        query_id=sample_query.id,
        selected_model="gpt-4o-mini",
        confidence=0.85,
        features=features,
        reasoning="Selected for simple query",
        metadata={"attempt": 0},
    )


@pytest.fixture
def sample_response(sample_query):
    """Create sample response."""
    return Response(
        id=f"test-response-{uuid4()}",
        query_id=sample_query.id,
        model="gpt-4o-mini",
        text="4",
        cost=0.001,
        latency=0.5,
        tokens=20,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_feedback(sample_response):
    """Create sample feedback."""
    return Feedback(
        response_id=sample_response.id,
        quality_score=0.9,
        met_expectations=True,
        user_rating=5,
        comments="Great answer!",
    )


class TestDatabaseConnection:
    """Tests for database connection management."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Test connecting and disconnecting from database."""
        db = Database()

        # Initially not connected
        assert db.pool is None

        await db.connect()
        assert db.pool is not None

        await db.disconnect()
        assert db.pool is None


class TestQueryOperations:
    """Tests for query CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_query(self, db, sample_query):
        """Test saving a query."""
        query_id = await db.save_query(sample_query)

        assert query_id == sample_query.id

    @pytest.mark.asyncio
    async def test_save_query_with_constraints(self, db, sample_query_with_constraints):
        """Test saving a query with constraints and context."""
        query_id = await db.save_query(sample_query_with_constraints)

        assert query_id == sample_query_with_constraints.id


class TestCompleteInteraction:
    """Tests for saving complete interactions."""

    @pytest.mark.asyncio
    async def test_save_complete_interaction_with_routing(
        self, db, sample_query, sample_routing_decision, sample_response
    ):
        """Test saving complete interaction with routing decision."""
        # First save the query
        await db.save_query(sample_query)

        # Then save the complete interaction
        await db.save_complete_interaction(
            routing=sample_routing_decision, response=sample_response
        )

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_save_complete_interaction_without_routing(
        self, db, sample_query, sample_response
    ):
        """Test saving response without routing decision."""
        # First save the query
        await db.save_query(sample_query)

        # Save response only (no routing)
        await db.save_complete_interaction(routing=None, response=sample_response)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_save_complete_interaction_with_feedback(
        self,
        db,
        sample_query,
        sample_routing_decision,
        sample_response,
        sample_feedback,
    ):
        """Test saving complete interaction including feedback."""
        # First save the query
        await db.save_query(sample_query)

        # Save complete interaction with feedback
        await db.save_complete_interaction(
            routing=sample_routing_decision,
            response=sample_response,
            feedback=sample_feedback,
        )

        # Should not raise any errors


class TestModelStateOperations:
    """Tests for model state management."""

    @pytest.mark.asyncio
    async def test_update_model_state(self, db):
        """Test updating model state."""
        state = ModelState(model_id=f"test-model-{uuid4()}", alpha=5.0, beta=2.0)

        await db.update_model_state(state)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_update_model_state_idempotent(self, db):
        """Test updating same model state multiple times (UPSERT)."""
        model_id = f"test-model-{uuid4()}"

        # First update
        state1 = ModelState(model_id=model_id, alpha=5.0, beta=2.0)
        await db.update_model_state(state1)

        # Second update (should upsert, not fail)
        state2 = ModelState(model_id=model_id, alpha=6.0, beta=3.0)
        await db.update_model_state(state2)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_get_model_states(self, db):
        """Test retrieving model states."""
        # Create and save a unique model state
        model_id = f"test-model-{uuid4()}"
        state = ModelState(model_id=model_id, alpha=7.0, beta=4.0)
        await db.update_model_state(state)

        # Retrieve all states
        states = await db.get_model_states()

        # Should return dict with our model
        assert isinstance(states, dict)
        assert model_id in states
        assert states[model_id].alpha == 7.0
        assert states[model_id].beta == 4.0

    @pytest.mark.asyncio
    async def test_get_model_states_empty(self, db):
        """Test getting model states returns empty dict if none exist."""
        # This test assumes a fresh database or uses isolation
        states = await db.get_model_states()

        # Should return dict (possibly empty or with existing states)
        assert isinstance(states, dict)


class TestModelPricingOperations:
    """Tests for model pricing management."""

    @pytest.mark.asyncio
    async def test_get_latest_pricing(self, db):
        """Test retrieving latest model pricing."""
        prices = await db.get_latest_pricing()

        # Should return dict
        assert isinstance(prices, dict)

        # If any prices exist, validate structure
        if prices:
            for model_id, pricing in prices.items():
                assert isinstance(pricing, ModelPricing)
                assert pricing.model_id == model_id
                # Pricing can be 0 for free-tier models (e.g., codestral)
                assert pricing.input_cost_per_million >= 0
                assert pricing.output_cost_per_million >= 0


class TestResponseRetrieval:
    """Tests for response retrieval."""

    @pytest.mark.asyncio
    async def test_get_response_by_id(self, db, sample_query, sample_response):
        """Test retrieving response by ID."""
        # First save query and response
        await db.save_query(sample_query)
        await db.save_complete_interaction(routing=None, response=sample_response)

        # Retrieve the response
        retrieved = await db.get_response_by_id(sample_response.id)

        # Should find the response
        assert retrieved is not None
        assert retrieved.id == sample_response.id
        assert retrieved.query_id == sample_query.id
        assert retrieved.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_response_by_id_not_found(self, db):
        """Test retrieving non-existent response returns None."""
        retrieved = await db.get_response_by_id("nonexistent-response-id")

        assert retrieved is None


class TestErrorHandling:
    """Tests for database error handling."""

    @pytest.mark.asyncio
    async def test_save_query_without_connection_raises_error(self, sample_query):
        """Test operations without connection raise DatabaseError."""
        db = Database()
        # Don't connect

        with pytest.raises(DatabaseError) as exc_info:
            await db.save_query(sample_query)

        assert "Database not connected" in str(exc_info.value)
