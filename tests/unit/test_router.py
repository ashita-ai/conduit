"""Unit tests for Router."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from conduit.core.models import (
    Query,
    QueryConstraints,
    QueryFeatures,
    RoutingDecision,
)
from conduit.engines.router import Router


@pytest.fixture
def mock_analyzer():
    """Create mock QueryAnalyzer."""
    analyzer = AsyncMock()
    analyzer.analyze.return_value = QueryFeatures(
        embedding=[0.1] * 384,
        token_count=10,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )
    return analyzer


@pytest.fixture
def mock_bandit():
    """Create mock ContextualBandit."""
    bandit = MagicMock()
    bandit.select_model.return_value = "gpt-4o-mini"
    bandit.get_confidence.return_value = 0.85

    # Mock model state for reasoning
    mock_state = MagicMock()
    mock_state.mean_success_rate = 0.75
    mock_state.alpha = 10.0
    mock_state.beta = 3.0
    bandit.get_model_state.return_value = mock_state

    return bandit


@pytest.fixture
def default_models():
    """List of default available models."""
    return ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4", "claude-opus-4"]


class TestRouterBasic:
    """Basic routing tests without constraints."""

    @pytest.mark.asyncio
    async def test_basic_routing_no_constraints(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test basic routing without constraints."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        query = Query(id="test-1", text="What is 2+2?")
        decision = await router.route(query)

        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.85
        assert decision.query_id == "test-1"
        assert decision.features is not None
        assert decision.reasoning is not None
        assert "simple" in decision.reasoning.lower() or "moderate" in decision.reasoning.lower()
        assert decision.metadata.get("constraints_relaxed") is False

    @pytest.mark.asyncio
    async def test_routing_with_precomputed_features(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test routing with pre-extracted features."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        query = Query(id="test-2", text="Complex query")
        features = QueryFeatures(
            embedding=[0.2] * 384,
            token_count=100,
            complexity_score=0.8,
            domain="code",
            domain_confidence=0.9,
        )

        decision = await router.route(query, features=features)

        # Should not call analyzer since features provided
        mock_analyzer.analyze.assert_not_called()
        assert decision.features == features
        assert decision.selected_model == "gpt-4o-mini"


class TestConstraintFiltering:
    """Tests for constraint filtering logic."""

    @pytest.mark.asyncio
    async def test_max_cost_constraint(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test filtering by maximum cost."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Only gpt-4o-mini (0.0001) should satisfy max_cost=0.0002
        constraints = QueryConstraints(max_cost=0.0002)
        query = Query(id="test-3", text="Simple query", constraints=constraints)

        mock_bandit.select_model.return_value = "gpt-4o-mini"
        decision = await router.route(query)

        # Verify bandit was called with filtered models
        call_args = mock_bandit.select_model.call_args
        eligible_models = call_args.kwargs["models"]
        assert "gpt-4o-mini" in eligible_models
        assert "gpt-4o" not in eligible_models  # Too expensive
        assert "claude-opus-4" not in eligible_models  # Too expensive

    @pytest.mark.asyncio
    async def test_max_latency_constraint(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test filtering by maximum latency."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # gpt-4o-mini (1.0) and claude-sonnet-4 (1.5) should satisfy max_latency=1.8
        constraints = QueryConstraints(max_latency=1.8)
        query = Query(id="test-4", text="Fast query", constraints=constraints)

        mock_bandit.select_model.return_value = "gpt-4o-mini"
        decision = await router.route(query)

        call_args = mock_bandit.select_model.call_args
        eligible_models = call_args.kwargs["models"]
        assert "gpt-4o-mini" in eligible_models
        assert "claude-sonnet-4" in eligible_models
        assert "gpt-4o" not in eligible_models  # Too slow (2.0)
        assert "claude-opus-4" not in eligible_models  # Too slow (3.0)

    @pytest.mark.asyncio
    async def test_min_quality_constraint(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test filtering by minimum quality."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # gpt-4o (0.9), claude-sonnet-4 (0.85), claude-opus-4 (0.95) satisfy min_quality=0.8
        constraints = QueryConstraints(min_quality=0.8)
        query = Query(id="test-5", text="High quality query", constraints=constraints)

        mock_bandit.select_model.return_value = "gpt-4o"
        decision = await router.route(query)

        call_args = mock_bandit.select_model.call_args
        eligible_models = call_args.kwargs["models"]
        assert "gpt-4o" in eligible_models
        assert "claude-sonnet-4" in eligible_models
        assert "claude-opus-4" in eligible_models
        assert "gpt-4o-mini" not in eligible_models  # Too low quality (0.7)

    @pytest.mark.asyncio
    async def test_preferred_provider_constraint(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test filtering by preferred provider."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Only claude models should be eligible
        constraints = QueryConstraints(preferred_provider="claude")
        query = Query(id="test-6", text="Provider query", constraints=constraints)

        mock_bandit.select_model.return_value = "claude-sonnet-4"
        decision = await router.route(query)

        call_args = mock_bandit.select_model.call_args
        eligible_models = call_args.kwargs["models"]
        assert "claude-sonnet-4" in eligible_models
        assert "claude-opus-4" in eligible_models
        assert "gpt-4o-mini" not in eligible_models
        assert "gpt-4o" not in eligible_models

    @pytest.mark.asyncio
    async def test_combined_constraints(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test multiple constraints together."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Only claude-sonnet-4 satisfies all: cost <= 0.001, latency <= 2.0, quality >= 0.8
        constraints = QueryConstraints(
            max_cost=0.001,
            max_latency=2.0,
            min_quality=0.8,
        )
        query = Query(id="test-7", text="Complex constraints", constraints=constraints)

        mock_bandit.select_model.return_value = "claude-sonnet-4"
        decision = await router.route(query)

        call_args = mock_bandit.select_model.call_args
        eligible_models = call_args.kwargs["models"]
        assert "claude-sonnet-4" in eligible_models
        assert len(eligible_models) >= 1


class TestConstraintRelaxation:
    """Tests for constraint relaxation fallback."""

    @pytest.mark.asyncio
    async def test_relaxation_when_no_eligible_models(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test constraints are relaxed when no models satisfy them."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Impossible constraint: cost <= 0.00001 (no model satisfies)
        constraints = QueryConstraints(max_cost=0.00001)
        query = Query(id="test-8", text="Impossible cost", constraints=constraints)

        mock_bandit.select_model.return_value = "gpt-4o-mini"
        decision = await router.route(query)

        assert decision.selected_model == "gpt-4o-mini"
        assert decision.metadata["constraints_relaxed"] is True

    @pytest.mark.asyncio
    async def test_relaxation_factor_calculation(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test constraint relaxation increases by 20%."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        original = QueryConstraints(max_cost=0.0001, max_latency=1.0, min_quality=0.9)
        relaxed = router._relax_constraints(original, factor=0.2)

        # Max cost and latency should increase by 20%
        assert relaxed.max_cost == pytest.approx(0.0001 * 1.2)
        assert relaxed.max_latency == pytest.approx(1.0 * 1.2)
        # Min quality should decrease by 20%
        assert relaxed.min_quality == pytest.approx(0.9 * 0.8)


class TestCircuitBreaker:
    """Tests for circuit breaker retry logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_retry(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test retry when circuit breaker is open."""
        # gpt-4o-mini has circuit breaker OPEN
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
            circuit_breaker_states={"gpt-4o-mini": True},
        )

        # First selection returns gpt-4o-mini (circuit open), should retry with gpt-4o
        mock_bandit.select_model.side_effect = ["gpt-4o-mini", "gpt-4o"]
        query = Query(id="test-9", text="Circuit test")

        decision = await router.route(query)

        # Should have selected gpt-4o after retry
        assert decision.selected_model == "gpt-4o"
        assert decision.metadata["attempt"] == 1  # One retry
        assert mock_bandit.select_model.call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_max_retries(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test max retries (2) for circuit breaker."""
        # All models have circuit breaker OPEN
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
            circuit_breaker_states={
                "gpt-4o-mini": True,
                "gpt-4o": True,
                "claude-sonnet-4": True,
            },
        )

        mock_bandit.select_model.side_effect = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"]
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
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test default fallback when constraints can't be satisfied even after relaxation."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Impossible constraints even after relaxation
        constraints = QueryConstraints(max_cost=0.000001, max_latency=0.001)
        query = Query(id="test-11", text="Impossible", constraints=constraints)

        decision = await router.route(query)

        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.0
        assert decision.metadata["constraints_relaxed"] is True
        assert decision.metadata["fallback"] == "default"
        assert "no models satisfied constraints" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_default_fallback_when_all_circuits_open(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test default fallback when all models have circuit breakers open."""
        # All models have circuit breaker OPEN
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
            circuit_breaker_states={model: True for model in default_models},
        )

        mock_bandit.select_model.return_value = "gpt-4o-mini"
        query = Query(id="test-12", text="All circuits")

        decision = await router.route(query)

        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.0
        assert decision.metadata["fallback"] == "circuit_breaker"


class TestReasoningGeneration:
    """Tests for selection reasoning explanation."""

    @pytest.mark.asyncio
    async def test_reasoning_simple_query(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test reasoning for simple query."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Simple query (complexity < 0.3)
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=5,
            complexity_score=0.2,
            domain="general",
            domain_confidence=0.7,
        )

        query = Query(id="test-13", text="Simple")
        decision = await router.route(query, features=features)

        assert "simple" in decision.reasoning.lower()
        assert "general" in decision.reasoning.lower()
        assert "gpt-4o-mini" in decision.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_complex_query(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test reasoning for complex query."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Complex query (complexity >= 0.7)
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=200,
            complexity_score=0.85,
            domain="code",
            domain_confidence=0.95,
        )

        query = Query(id="test-14", text="Complex code")
        decision = await router.route(query, features=features)

        assert "complex" in decision.reasoning.lower()
        assert "code" in decision.reasoning.lower()
        assert "success rate" in decision.reasoning.lower()
        # Should include bandit statistics
        assert "α=" in decision.reasoning or "alpha" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_reasoning_moderate_query(
        self, mock_analyzer, mock_bandit, default_models
    ):
        """Test reasoning for moderate complexity query."""
        router = Router(
            bandit=mock_bandit,
            analyzer=mock_analyzer,
            models=default_models,
        )

        # Moderate query (0.3 <= complexity < 0.7)
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="science",
            domain_confidence=0.8,
        )

        query = Query(id="test-15", text="Moderate science")
        decision = await router.route(query, features=features)

        assert "moderate" in decision.reasoning.lower()
        assert "science" in decision.reasoning.lower()
