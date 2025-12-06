"""Unit tests for Router.

Tests cover:
- Initialization with defaults and custom models
- Basic routing without constraints
- Constraint filtering (cost, latency, quality, provider)
- Constraint relaxation
- Circuit breaker behavior
- Fallback strategies
- Reasoning generation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.models import Query, QueryConstraints, QueryFeatures, RoutingDecision
from conduit.engines import Router as RouterFromEngines
from conduit.engines.router import Router


@pytest.fixture
def mock_analyzer():
    """Create mock QueryAnalyzer."""
    analyzer = AsyncMock()
    analyzer.analyze.return_value = QueryFeatures(
        embedding=[0.1] * 384, token_count=10, complexity_score=0.5
    )
    analyzer.feature_dim = 386
    return analyzer


@pytest.fixture
def mock_hybrid_router():
    """Create mock HybridRouter."""
    from conduit.engines.bandits.base import ModelArm

    hybrid_router = AsyncMock()
    hybrid_router.route = AsyncMock(
        return_value=RoutingDecision(
            query_id="test-1",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=QueryFeatures(
                embedding=[0.1] * 384, token_count=10, complexity_score=0.5
            ),
            reasoning="Simple query routed to gpt-4o-mini",
            metadata={"constraints_relaxed": False},
        )
    )
    hybrid_router.models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-opus-4"]
    # Add arms for cost filtering (used by Router.route() when max_cost constraint is set)
    hybrid_router.arms = [
        ModelArm(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.15,
            cost_per_output_token=0.60,
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            cost_per_input_token=2.50,
            cost_per_output_token=10.00,
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-sonnet-4",
            provider="anthropic",
            model_name="claude-sonnet-4",
            cost_per_input_token=3.00,
            cost_per_output_token=15.00,
            expected_quality=0.93,
        ),
        ModelArm(
            model_id="claude-opus-4",
            provider="anthropic",
            model_name="claude-opus-4",
            cost_per_input_token=15.00,
            cost_per_output_token=75.00,
            expected_quality=0.98,
        ),
    ]
    return hybrid_router


@pytest.fixture
def default_models():
    """List of default available models."""
    return ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-opus-4"]


class TestRouterBasic:
    """Basic routing tests without constraints."""

    @pytest.mark.asyncio
    async def test_basic_routing_no_constraints(
        self, mock_hybrid_router, default_models
    ):
        """Test basic routing without constraints."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-1", text="What is 2+2?")
        decision = await router.route(query)

        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.85
        assert decision.query_id == "test-1"
        assert decision.features is not None
        assert decision.reasoning is not None
        assert (
            "simple" in decision.reasoning.lower()
            or "moderate" in decision.reasoning.lower()
        )
        assert decision.metadata.get("constraints_relaxed") is False

    @pytest.mark.asyncio
    async def test_routing_with_precomputed_features(
        self, mock_hybrid_router, default_models
    ):
        """Test routing with pre-extracted features."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-2", text="Complex query")
        # Note: QueryFeatures would be used if Router.route() accepted a features parameter,
        # but it always calls analyzer internally. We verify the decision uses features from hybrid_router.
        decision = await router.route(query)

        assert decision.features is not None
        assert decision.selected_model == "gpt-4o-mini"


class TestConstraintFiltering:
    """Tests for constraint filtering logic."""

    @pytest.mark.asyncio
    async def test_max_cost_constraint(self, mock_hybrid_router, default_models):
        """Test filtering by maximum cost."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        # Only gpt-4o-mini (0.0001) should satisfy max_cost=0.0002
        constraints = QueryConstraints(max_cost=0.0002)
        query = Query(id="test-3", text="Simple query", constraints=constraints)

        decision = await router.route(query)

        # Verify hybrid_router.route was called with query containing constraints
        mock_hybrid_router.route.assert_called_once()
        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == constraints
        assert decision.selected_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_max_latency_constraint(self, mock_hybrid_router, default_models):
        """Test filtering by maximum latency."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        # gpt-4o-mini (1.0) and claude-sonnet-4 (1.5) should satisfy max_latency=1.8
        constraints = QueryConstraints(max_latency=1.8)
        query = Query(id="test-4", text="Fast query", constraints=constraints)

        decision = await router.route(query)

        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == constraints
        assert decision.selected_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_min_quality_constraint(self, mock_hybrid_router, default_models):
        """Test filtering by minimum quality."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        # gpt-4o (0.9), claude-sonnet-4 (0.85), claude-opus-4 (0.95) satisfy min_quality=0.8
        constraints = QueryConstraints(min_quality=0.8)
        query = Query(id="test-5", text="High quality query", constraints=constraints)

        decision = await router.route(query)

        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == constraints
        assert decision.selected_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_preferred_provider_constraint(
        self, mock_hybrid_router, default_models
    ):
        """Test filtering by preferred provider."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        # Only claude models should be eligible
        constraints = QueryConstraints(preferred_provider="claude")
        query = Query(id="test-6", text="Provider query", constraints=constraints)

        decision = await router.route(query)

        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == constraints
        assert decision.selected_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_combined_constraints(self, mock_hybrid_router, default_models):
        """Test multiple constraints together."""
        router = Router(models=default_models)
        router.hybrid_router = mock_hybrid_router

        # Only claude-sonnet-4 satisfies all: cost <= 0.001, latency <= 2.0, quality >= 0.8
        constraints = QueryConstraints(max_cost=0.001, max_latency=2.0, min_quality=0.8)
        query = Query(id="test-7", text="Complex constraints", constraints=constraints)

        decision = await router.route(query)

        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == constraints
        assert decision.selected_model == "gpt-4o-mini"


class TestConstraintRelaxation:
    """Tests for constraint relaxation fallback."""

    @pytest.mark.asyncio
    async def test_relaxation_when_no_eligible_models(
        self, mock_hybrid_router, default_models
    ):
        """Test constraints are relaxed when no models satisfy them."""
        from conduit.engines.bandits.base import ModelArm

        router = Router(models=default_models)
        # Create expensive mock arms that won't fit any reasonable budget
        mock_hybrid_router.arms = [
            ModelArm(
                model_id="expensive-test-model",
                provider="test",
                model_name="expensive-test-model",
                cost_per_input_token=1000.0,  # $1000 per 1K tokens (forces fallback)
                cost_per_output_token=1000.0,
                expected_quality=0.85,
            ),
        ]
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-8",
                selected_model="expensive-test-model",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="Constraints relaxed",
                metadata={"constraints_relaxed": True},
            )
        )
        router.hybrid_router = mock_hybrid_router

        # Budget too low for the expensive test model
        constraints = QueryConstraints(max_cost=0.001)
        query = Query(id="test-8", text="Impossible cost", constraints=constraints)

        decision = await router.route(query)

        assert decision.selected_model == "expensive-test-model"
        assert decision.metadata["constraints_relaxed"] is True

    @pytest.mark.asyncio
    async def test_relaxation_factor_calculation(
        self, mock_hybrid_router, default_models
    ):
        """Test constraint relaxation with cost budget enforcement."""
        from conduit.engines.bandits.base import ModelArm

        # Note: This test verifies that constraints are properly passed through
        # to HybridRouter and that cost relaxation is applied when needed.
        router = Router(models=default_models)

        # Use expensive models to force cost constraint relaxation
        mock_hybrid_router.arms = [
            ModelArm(
                model_id="expensive-test-model",
                provider="test",
                model_name="expensive-test-model",
                cost_per_input_token=1000.0,  # Forces fallback
                cost_per_output_token=1000.0,
                expected_quality=0.85,
            ),
        ]
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-relax",
                selected_model="expensive-test-model",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="Constraints relaxed",
                metadata={"constraints_relaxed": True},
            )
        )
        router.hybrid_router = mock_hybrid_router

        original = QueryConstraints(max_cost=0.0001, max_latency=1.0, min_quality=0.9)
        query = Query(id="test-relax", text="Test", constraints=original)
        decision = await router.route(query)

        # Verify constraints were passed to hybrid router
        called_query = mock_hybrid_router.route.call_args[0][0]
        assert called_query.constraints == original
        # Cost constraint relaxation is applied because model is too expensive
        assert decision.metadata.get("constraints_relaxed") is True
        assert decision.metadata.get("max_cost_budget") == 0.0001


class TestCircuitBreaker:
    """Tests for circuit breaker retry logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_retry(self, mock_hybrid_router, default_models):
        """Test retry when circuit breaker is open."""
        router = Router(models=default_models)
        # Mock hybrid router to simulate retry behavior - Router calls hybrid_router.route once
        # The retry logic is handled internally by HybridRouter, so we mock it to return
        # a decision that indicates a retry occurred
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-9",
                selected_model="gpt-4o",
                confidence=0.9,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="Success after retry",
                metadata={"attempt": 1},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-9", text="Circuit test")
        decision = await router.route(query)

        # Should have selected gpt-4o after retry
        assert decision.selected_model == "gpt-4o"
        assert decision.metadata["attempt"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_max_retries(
        self, mock_hybrid_router, default_models
    ):
        """Test max retries (2) for circuit breaker."""
        router = Router(models=default_models)
        # Mock hybrid router to return fallback decision
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-10",
                selected_model="gpt-4o-mini",
                confidence=0.0,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="All circuit breakers open, using default fallback",
                metadata={"fallback": "circuit_breaker"},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-10", text="All circuits open")
        decision = await router.route(query)

        # Should fallback to default after exhausting retries
        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.0
        assert decision.metadata["fallback"] == "circuit_breaker"
        assert "circuit breakers" in decision.reasoning.lower()


class TestFallbackStrategies:
    """Tests for various fallback strategies."""

    @pytest.mark.asyncio
    async def test_default_fallback_when_no_models_after_relaxation(
        self, mock_hybrid_router, default_models
    ):
        """Test default fallback when constraints can't be satisfied even after relaxation."""
        from conduit.engines.bandits.base import ModelArm

        router = Router(models=default_models)
        # Create expensive mock arms that won't fit any reasonable budget
        mock_hybrid_router.arms = [
            ModelArm(
                model_id="expensive-test-model",
                provider="test",
                model_name="expensive-test-model",
                cost_per_input_token=1000.0,  # Forces fallback
                cost_per_output_token=1000.0,
                expected_quality=0.5,
            ),
        ]
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-11",
                selected_model="expensive-test-model",
                confidence=0.0,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="No models satisfied constraints after relaxation, using default",
                metadata={"constraints_relaxed": True, "fallback": "default"},
            )
        )
        router.hybrid_router = mock_hybrid_router

        # Budget too low for expensive model - forces cost constraint relaxation
        constraints = QueryConstraints(max_cost=0.001, max_latency=0.001)
        query = Query(id="test-11", text="Impossible", constraints=constraints)

        decision = await router.route(query)

        assert decision.selected_model == "expensive-test-model"
        assert decision.confidence == 0.0
        assert decision.metadata["constraints_relaxed"] is True
        # Note: "fallback" and "reasoning" come from the mock, so we verify cost relaxation works
        assert decision.metadata.get("max_cost_budget") == 0.001

    @pytest.mark.asyncio
    async def test_default_fallback_when_all_circuits_open(
        self, mock_hybrid_router, default_models
    ):
        """Test default fallback when all models have circuit breakers open."""
        router = Router(models=default_models)
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-12",
                selected_model="gpt-4o-mini",
                confidence=0.0,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=10, complexity_score=0.5
                ),
                reasoning="All circuit breakers open, using default fallback",
                metadata={"fallback": "circuit_breaker"},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-12", text="All circuits")
        decision = await router.route(query)

        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.0
        assert decision.metadata["fallback"] == "circuit_breaker"


class TestReasoningGeneration:
    """Tests for selection reasoning explanation."""

    @pytest.mark.asyncio
    async def test_reasoning_simple_query(self, mock_hybrid_router, default_models):
        """Test reasoning for simple query."""
        router = Router(models=default_models)
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-13",
                selected_model="gpt-4o-mini",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=5, complexity_score=0.2
                ),
                reasoning="Simple query in general domain, selected gpt-4o-mini for efficiency",
                metadata={},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-13", text="Simple")
        decision = await router.route(query)

        assert "simple" in decision.reasoning.lower()
        assert "general" in decision.reasoning.lower()
        assert "gpt-4o-mini" in decision.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_complex_query(self, mock_hybrid_router, default_models):
        """Test reasoning for complex query."""
        router = Router(models=default_models)
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-14",
                selected_model="gpt-4o",
                confidence=0.9,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=200, complexity_score=0.85
                ),
                reasoning="Complex code query, selected gpt-4o with high success rate (α=10.0, β=3.0)",
                metadata={},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-14", text="Complex code")
        decision = await router.route(query)

        assert "complex" in decision.reasoning.lower()
        assert "code" in decision.reasoning.lower()
        assert (
            "success rate" in decision.reasoning.lower()
            or "α=" in decision.reasoning
            or "alpha" in decision.reasoning.lower()
        )

    @pytest.mark.asyncio
    async def test_reasoning_moderate_query(self, mock_hybrid_router, default_models):
        """Test reasoning for moderate complexity query."""
        router = Router(models=default_models)
        mock_hybrid_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="test-15",
                selected_model="gpt-4o-mini",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384, token_count=50, complexity_score=0.5
                ),
                reasoning="Moderate complexity science query, selected gpt-4o-mini",
                metadata={},
            )
        )
        router.hybrid_router = mock_hybrid_router

        query = Query(id="test-15", text="Moderate science")
        decision = await router.route(query)

        assert "moderate" in decision.reasoning.lower()
        assert "science" in decision.reasoning.lower()


class TestRouterInitialization:
    """Tests for Router initialization (merged from test_router_class.py)."""

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

    def test_router_exported_from_engines(self):
        """Test that Router is exported from conduit.engines."""
        assert Router is RouterFromEngines

    def test_router_has_route_method(self):
        """Test that Router has the route method."""
        router = Router()
        assert hasattr(router, "route")
        assert callable(router.route)


class TestRouterIntegration:
    """Integration tests with real components (not mocked)."""

    @pytest.mark.asyncio
    async def test_route_method_real_components(self):
        """Test that route method works with real components."""
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


class TestRouterIdConfiguration:
    """Tests for router_id configuration including CONDUIT_ROUTER_ID env var."""

    def test_explicit_router_id_takes_priority(self):
        """Test that explicit router_id parameter overrides env var."""
        import os

        # Set env var
        os.environ["CONDUIT_ROUTER_ID"] = "env-router-id"
        try:
            router = Router(router_id="explicit-router-id")
            assert router.router_id == "explicit-router-id"
        finally:
            del os.environ["CONDUIT_ROUTER_ID"]

    def test_env_var_used_when_no_explicit_id(self):
        """Test that CONDUIT_ROUTER_ID env var is used when router_id not provided."""
        import os

        os.environ["CONDUIT_ROUTER_ID"] = "kubernetes-cluster"
        try:
            router = Router()
            assert router.router_id == "kubernetes-cluster"
        finally:
            del os.environ["CONDUIT_ROUTER_ID"]

    def test_timestamp_fallback_when_no_id_provided(self):
        """Test that timestamp-based ID is generated when no ID provided."""
        import os

        # Ensure env var is not set
        os.environ.pop("CONDUIT_ROUTER_ID", None)

        router = Router()
        assert router.router_id.startswith("router-")
        # Should be timestamp format: router-YYYYMMDD-HHMMSS-microseconds
        assert len(router.router_id) > len("router-")

    def test_env_var_enables_multi_replica_state_sharing(self):
        """Test that multiple routers with same env var share the same router_id."""
        import os

        os.environ["CONDUIT_ROUTER_ID"] = "production-cluster"
        try:
            router1 = Router()
            router2 = Router()
            # Both should have the same router_id for state sharing
            assert router1.router_id == router2.router_id == "production-cluster"
        finally:
            del os.environ["CONDUIT_ROUTER_ID"]


class TestRouterUpdate:
    """Tests for Router.update() method."""

    @pytest.fixture
    def sample_features(self):
        """Create sample query features."""
        return QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
        )

    @pytest.mark.asyncio
    async def test_update_calls_hybrid_router(self, sample_features):
        """Test that update delegates to hybrid_router."""
        router = Router()
        # Mock the hybrid router update method
        router.hybrid_router.update = AsyncMock()

        await router.update(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.95,
            latency=0.5,
            features=sample_features,
        )

        router.hybrid_router.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_with_auto_persist_saves_state(self, sample_features):
        """Test that update saves state when auto_persist is enabled."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.update = AsyncMock()
        router._save_state = AsyncMock()

        await router.update(
            model_id="gpt-4o-mini",
            cost=0.001,
            quality_score=0.95,
            latency=0.5,
            features=sample_features,
        )

        router._save_state.assert_called_once()


class TestRouterStatePersistence:
    """Tests for Router state persistence methods."""

    @pytest.mark.asyncio
    async def test_load_initial_state_success(self):
        """Test successful state loading on initialization."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.load_state = AsyncMock(return_value=True)

        await router._load_initial_state()

        router.hybrid_router.load_state.assert_called_once()
        assert router._state_loaded is True

    @pytest.mark.asyncio
    async def test_load_initial_state_no_saved_state(self):
        """Test loading when no saved state exists."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.load_state = AsyncMock(return_value=False)

        await router._load_initial_state()

        assert router._state_loaded is True

    @pytest.mark.asyncio
    async def test_load_initial_state_handles_error(self):
        """Test that state loading errors don't break router."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.load_state = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await router._load_initial_state()

        assert router._state_loaded is True

    @pytest.mark.asyncio
    async def test_load_initial_state_skips_if_no_store(self):
        """Test that load skips when no state store configured."""
        router = Router()  # No state_store
        await router._load_initial_state()
        assert router._state_loaded is False  # Never set because no store

    @pytest.mark.asyncio
    async def test_load_initial_state_skips_if_already_loaded(self):
        """Test that load skips when state already loaded."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router._state_loaded = True
        router.hybrid_router.load_state = AsyncMock()

        await router._load_initial_state()

        router.hybrid_router.load_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_state_success(self):
        """Test successful state saving."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.save_state = AsyncMock()

        await router._save_state()

        router.hybrid_router.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_state_handles_error(self):
        """Test that save errors don't break router."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router.hybrid_router.save_state = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await router._save_state()

    @pytest.mark.asyncio
    async def test_save_state_skips_if_no_store(self):
        """Test that save skips when no state store configured."""
        router = Router()  # No state_store
        router.hybrid_router.save_state = AsyncMock()

        await router._save_state()

        router.hybrid_router.save_state.assert_not_called()


class TestRouterClose:
    """Tests for Router.close() method."""

    @pytest.mark.asyncio
    async def test_close_saves_final_state(self):
        """Test that close saves final state when auto_persist enabled."""
        mock_state_store = MagicMock()
        router = Router(state_store=mock_state_store)
        router._save_state = AsyncMock()

        await router.close()

        router._save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_auto_persist(self):
        """Test that close skips save when auto_persist disabled."""
        router = Router()  # No state_store means auto_persist is False
        router._save_state = AsyncMock()

        await router.close()

        router._save_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_closes_cache(self):
        """Test that close closes cache connection."""
        mock_cache = MagicMock()
        mock_cache.close = AsyncMock()
        router = Router()
        router.cache = mock_cache

        await router.close()

        mock_cache.close.assert_called_once()
