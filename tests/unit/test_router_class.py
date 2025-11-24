"""Unit tests for the Router class."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from conduit.core.models import Query, RoutingDecision, QueryFeatures
from conduit.engines import Router


class TestRouter:
    """Test the high-level Router interface."""

    def test_router_initialization_defaults(self):
        """Test Router initializes with default models and hybrid router."""
        router = Router()

        # Check that components are initialized
        assert router.analyzer is not None
        assert router.hybrid_router is not None

        # Check hybrid router has both bandits
        assert router.hybrid_router.ucb1 is not None
        assert router.hybrid_router.linucb is not None

        # Check models are set
        assert len(router.hybrid_router.models) > 0

    def test_router_initialization_custom_models(self):
        """Test Router initializes with custom models."""
        custom_models = ["gpt-4o-mini", "claude-3-5-sonnet"]
        router = Router(models=custom_models)

        assert router.hybrid_router.models == custom_models
        assert len(router.hybrid_router.arms) == 2

    @pytest.mark.asyncio
    async def test_route_method(self):
        """Test that route method delegates to hybrid router."""
        router = Router()
        query = Query(text="Test query")

        # Route should delegate to hybrid_router
        result = await router.route(query)

        # Verify result structure
        assert isinstance(result, RoutingDecision)
        assert result.query_id == query.id
        assert result.selected_model in router.hybrid_router.models
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.features, QueryFeatures)

    def test_router_has_route_method(self):
        """Test that Router has the route method."""
        router = Router()
        assert hasattr(router, 'route')
        assert callable(router.route)
