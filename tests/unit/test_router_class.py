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
        assert hasattr(router, "route")
        assert callable(router.route)

    @pytest.mark.asyncio
    async def test_context_manager_basic_usage(self):
        """Test Router can be used as async context manager."""
        async with Router() as router:
            # Router should be usable inside context
            assert router is not None
            assert router.analyzer is not None
            assert router.hybrid_router is not None

            # Should be able to route queries
            query = Query(text="Test query")
            result = await router.route(query)
            assert isinstance(result, RoutingDecision)

    @pytest.mark.asyncio
    async def test_context_manager_calls_close_on_exit(self):
        """Test that context manager calls close() when exiting normally."""
        router = Router()
        router.close = AsyncMock()

        async with router:
            pass

        router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_close_on_exception(self):
        """Test that context manager calls close() even when exception raised."""
        router = Router()
        router.close = AsyncMock()

        with pytest.raises(ValueError):
            async with router:
                raise ValueError("Test exception")

        # close() should still be called despite exception
        router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """Test that __aenter__ returns the router instance."""
        router = Router()
        async with router as ctx_router:
            assert ctx_router is router

    @pytest.mark.asyncio
    async def test_context_manager_loads_state_if_persistence_enabled(self):
        """Test that context manager loads state on enter when persistence enabled."""
        mock_store = MagicMock()

        router = Router(state_store=mock_store, auto_persist=True)
        router._load_initial_state = AsyncMock()
        router.close = AsyncMock()

        async with router:
            pass

        # State should be loaded on context entry
        router._load_initial_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_skips_state_load_if_already_loaded(self):
        """Test that context manager skips state load if already loaded."""
        mock_store = MagicMock()

        router = Router(state_store=mock_store, auto_persist=True)
        router._state_loaded = True  # Already loaded
        router._load_initial_state = AsyncMock()
        router.close = AsyncMock()

        async with router:
            pass

        # State load should be skipped since already loaded
        router._load_initial_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_skips_state_load_if_no_persistence(self):
        """Test that context manager skips state load when persistence disabled."""
        router = Router()  # No state_store, auto_persist=False
        router._load_initial_state = AsyncMock()
        router.close = AsyncMock()

        async with router:
            pass

        # No state load when persistence disabled
        router._load_initial_state.assert_not_called()
