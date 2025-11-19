"""Unit tests for the Router class."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from conduit.core.models import Query, RoutingDecision
from conduit.engines import Router


class TestRouter:
    """Test the high-level Router interface."""

    @pytest.fixture
    def mock_routing_engine(self):
        """Create a mock routing engine."""
        engine = AsyncMock()
        engine.route.return_value = RoutingDecision(
            query_id="test-query",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=MagicMock(),
            reasoning="Test reasoning",
            metadata={}
        )
        return engine

    def test_router_initialization_defaults(self):
        """Test Router initializes with default models."""
        router = Router()

        # Check that components are initialized
        assert router.analyzer is not None
        assert router.bandit is not None
        assert router.routing_engine is not None

        # Check default models are set
        expected_models = {
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3.5-sonnet",
            "claude-opus-4",
        }
        assert set(router.bandit.model_states.keys()) == expected_models

    def test_router_initialization_custom_models(self):
        """Test Router initializes with custom models."""
        custom_models = ["gpt-4o-mini", "claude-3.5-sonnet"]
        router = Router(models=custom_models)

        assert set(router.bandit.model_states.keys()) == set(custom_models)

    @pytest.mark.asyncio
    async def test_route_method(self):
        """Test that route method delegates to routing engine."""
        from conduit.core.models import QueryFeatures

        router = Router()
        query = Query(text="Test query")

        # Create real QueryFeatures for the mock decision
        mock_features = QueryFeatures(
            token_count=10,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
            embedding=[0.1] * 384
        )

        # Mock the routing engine's route method
        mock_decision = RoutingDecision(
            query_id=query.id,
            selected_model="gpt-4o-mini",
            confidence=0.9,
            features=mock_features,
            reasoning="Mock reasoning",
            metadata={}
        )
        router.routing_engine.route = AsyncMock(return_value=mock_decision)

        # Call route method
        result = await router.route(query)

        # Verify delegation
        router.routing_engine.route.assert_called_once_with(query)
        assert result == mock_decision
        assert result.selected_model == "gpt-4o-mini"
        assert result.confidence == 0.9

    def test_router_has_route_method(self):
        """Test that Router has the route method."""
        router = Router()
        assert hasattr(router, 'route')
        assert callable(router.route)
