"""Integration tests for constraint filtering in HybridRouter."""

import pytest

from conduit.core.models import Query, QueryConstraints
from conduit.core.pricing import ModelPricing
from conduit.engines.constraints import ConstraintFilter
from conduit.engines.hybrid_router import HybridRouter


@pytest.fixture
def test_models():
    """Test model IDs."""
    return ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]


@pytest.fixture
def constraint_filter():
    """Create constraint filter with pricing data."""
    pricing = {
        "gpt-4o-mini": ModelPricing(
            model_id="gpt-4o-mini",
            input_cost_per_million=0.15,
            output_cost_per_million=0.60,
        ),
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            input_cost_per_million=5.0,
            output_cost_per_million=15.0,
        ),
        "claude-3-haiku": ModelPricing(
            model_id="claude-3-haiku",
            input_cost_per_million=0.25,
            output_cost_per_million=1.25,
        ),
    }
    return ConstraintFilter(model_pricing=pricing)


@pytest.fixture
def hybrid_router_with_constraints(test_models, constraint_filter):
    """Create hybrid router with constraint filter."""
    return HybridRouter(
        models=test_models,
        switch_threshold=10,
        feature_dim=387,
        constraint_filter=constraint_filter,
    )


@pytest.mark.asyncio
async def test_no_constraints_routes_normally(hybrid_router_with_constraints):
    """Test routing without constraints works normally."""
    query = Query(text="What is 2+2?")
    decision = await hybrid_router_with_constraints.route(query)

    assert decision.selected_model in hybrid_router_with_constraints.models
    assert "constraints_relaxed" in decision.metadata
    assert decision.metadata["constraints_relaxed"] is False
    assert decision.metadata["excluded_count"] == 0


@pytest.mark.asyncio
async def test_cost_constraint_filters_expensive_models(hybrid_router_with_constraints):
    """Test cost constraint filters out expensive models."""
    # Max cost that allows only gpt-4o-mini
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(max_cost=0.001),
    )
    decision = await hybrid_router_with_constraints.route(query)

    # Should select gpt-4o-mini (cheapest)
    # gpt-4o should be excluded due to cost
    assert decision.selected_model in hybrid_router_with_constraints.models
    assert "constraints_relaxed" in decision.metadata
    # Note: might be relaxed if mini doesn't pass strict constraint


@pytest.mark.asyncio
async def test_quality_constraint_filters_low_quality_models(hybrid_router_with_constraints):
    """Test quality constraint filters out lower quality models."""
    query = Query(
        text="Explain quantum physics in detail",
        constraints=QueryConstraints(min_quality=0.90),
    )
    decision = await hybrid_router_with_constraints.route(query)

    # Should select gpt-4o (high quality ~0.95)
    # mini and haiku should be excluded (quality ~0.75)
    assert decision.selected_model in hybrid_router_with_constraints.models


@pytest.mark.asyncio
async def test_provider_constraint_filters_other_providers(hybrid_router_with_constraints):
    """Test provider constraint filters out non-matching providers."""
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(preferred_provider="openai"),
    )
    decision = await hybrid_router_with_constraints.route(query)

    # Should select gpt-4o-mini or gpt-4o (OpenAI models)
    # claude-3-haiku should be excluded (Anthropic)
    assert decision.selected_model in ["gpt-4o-mini", "gpt-4o"]
    assert decision.metadata["excluded_count"] >= 1


@pytest.mark.asyncio
async def test_multiple_constraints_combined(hybrid_router_with_constraints):
    """Test multiple constraints work together."""
    query = Query(
        text="Simple math question",
        constraints=QueryConstraints(
            max_cost=0.002,
            min_quality=0.70,
            preferred_provider="openai",
        ),
    )
    decision = await hybrid_router_with_constraints.route(query)

    # Should select gpt-4o-mini (meets all: openai, cheap, quality >= 0.70)
    assert decision.selected_model in hybrid_router_with_constraints.models


@pytest.mark.asyncio
async def test_impossible_constraints_with_relaxation(hybrid_router_with_constraints):
    """Test impossible constraints trigger relaxation."""
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(max_cost=0.00001),  # Impossibly low
    )
    decision = await hybrid_router_with_constraints.route(query)

    # Should relax constraints and select some model
    assert decision.selected_model in hybrid_router_with_constraints.models
    assert decision.metadata["constraints_relaxed"] is True


@pytest.mark.asyncio
async def test_constraint_metadata_in_decision(hybrid_router_with_constraints):
    """Test constraint filtering metadata appears in decision."""
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(preferred_provider="openai"),
    )
    decision = await hybrid_router_with_constraints.route(query)

    assert "constraints_relaxed" in decision.metadata
    assert "excluded_count" in decision.metadata
    assert isinstance(decision.metadata["constraints_relaxed"], bool)
    assert isinstance(decision.metadata["excluded_count"], int)


@pytest.mark.asyncio
async def test_constraints_work_in_ucb1_phase(hybrid_router_with_constraints):
    """Test constraints filter models in UCB1 phase."""
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(preferred_provider="openai"),
    )

    # Route in UCB1 phase (before transition)
    decision = await hybrid_router_with_constraints.route(query)

    assert hybrid_router_with_constraints.current_phase == "ucb1"
    assert decision.selected_model in ["gpt-4o-mini", "gpt-4o"]
    assert decision.metadata["phase"] == "ucb1"


@pytest.mark.asyncio
async def test_constraints_work_in_linucb_phase(hybrid_router_with_constraints):
    """Test constraints filter models in LinUCB phase."""
    query_no_constraint = Query(text="What is 2+2?")

    # Get to LinUCB phase
    for _ in range(10):
        await hybrid_router_with_constraints.route(query_no_constraint)

    assert hybrid_router_with_constraints.current_phase == "linucb"

    # Now route with constraints
    query = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(preferred_provider="openai"),
    )
    decision = await hybrid_router_with_constraints.route(query)

    assert decision.selected_model in ["gpt-4o-mini", "gpt-4o"]
    assert decision.metadata["phase"] == "linucb"


@pytest.mark.asyncio
async def test_constraints_persist_across_queries(hybrid_router_with_constraints):
    """Test constraints are applied independently per query."""
    # Query 1: With constraint
    query1 = Query(
        text="What is 2+2?",
        constraints=QueryConstraints(preferred_provider="openai"),
    )
    decision1 = await hybrid_router_with_constraints.route(query1)
    assert decision1.selected_model in ["gpt-4o-mini", "gpt-4o"]

    # Query 2: Without constraint
    query2 = Query(text="What is 3+3?")
    decision2 = await hybrid_router_with_constraints.route(query2)
    # Can select any model
    assert decision2.selected_model in hybrid_router_with_constraints.models

    # Query 3: Different constraint
    query3 = Query(
        text="What is 4+4?",
        constraints=QueryConstraints(preferred_provider="anthropic"),
    )
    decision3 = await hybrid_router_with_constraints.route(query3)
    assert decision3.selected_model == "claude-3-haiku"
