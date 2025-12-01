"""Tests for conduit_litellm routing strategy.

These tests verify the ConduitRoutingStrategy integration with LiteLLM,
including async context handling and deployment selection.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

# Skip all tests if LiteLLM not installed
pytest.importorskip("litellm")

from conduit_litellm.strategy import ConduitRoutingStrategy
from conduit_litellm.utils import extract_query_text, extract_model_ids, validate_litellm_model_list
from conduit.core.models import Query, RoutingDecision, QueryFeatures


@pytest.fixture
def mock_litellm_router():
    """Create mock LiteLLM router with model_list."""
    router = Mock()
    router.model_list = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "gpt-4o-mini"},
            "litellm_params": {"model": "gpt-4o-mini"}
        },
        {
            "model_name": "claude-3",
            "model_info": {"id": "claude-3-haiku"},
            "litellm_params": {"model": "claude-3-haiku"}
        }
    ]
    return router


@pytest.fixture
def mock_conduit_router():
    """Create mock Conduit router."""
    router = AsyncMock()
    router.route = AsyncMock(return_value=RoutingDecision(
        query_id="test_query",
        selected_model="gpt-4o-mini",
        confidence=0.85,
        features=QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5
        ),
        reasoning="Test routing"
    ))
    return router


@pytest.mark.asyncio
async def test_async_get_available_deployment(mock_litellm_router, mock_conduit_router):
    """Test async deployment selection works correctly."""
    strategy = ConduitRoutingStrategy(conduit_router=mock_conduit_router)
    strategy._router = mock_litellm_router
    strategy._initialized = True

    messages = [{"role": "user", "content": "Hello world"}]
    deployment = await strategy.async_get_available_deployment(
        model="gpt-4",
        messages=messages
    )

    assert deployment["model_info"]["id"] == "gpt-4o-mini"
    mock_conduit_router.route.assert_called_once()


@pytest.mark.asyncio
async def test_sync_in_async_context_no_runtime_error(mock_litellm_router, mock_conduit_router):
    """Test sync method works in async context without RuntimeError.

    This is the fix for Issue #31: Previously, calling sync method in an
    async context would raise RuntimeError due to loop.run_until_complete().
    Now it should work by running in a separate thread.
    """
    strategy = ConduitRoutingStrategy(conduit_router=mock_conduit_router)
    strategy._router = mock_litellm_router
    strategy._initialized = True

    messages = [{"role": "user", "content": "Hello world"}]

    # This should NOT raise RuntimeError
    deployment = strategy.get_available_deployment(
        model="gpt-4",
        messages=messages
    )

    assert deployment["model_info"]["id"] == "gpt-4o-mini"
    mock_conduit_router.route.assert_called_once()


def test_sync_without_event_loop(mock_litellm_router, mock_conduit_router):
    """Test sync method works when no event loop is running."""
    strategy = ConduitRoutingStrategy(conduit_router=mock_conduit_router)
    strategy._router = mock_litellm_router
    strategy._initialized = True

    messages = [{"role": "user", "content": "Hello world"}]

    deployment = strategy.get_available_deployment(
        model="gpt-4",
        messages=messages
    )

    assert deployment["model_info"]["id"] == "gpt-4o-mini"
    mock_conduit_router.route.assert_called_once()


@pytest.mark.asyncio
async def test_initialization_from_litellm(mock_litellm_router):
    """Test automatic initialization from LiteLLM model_list."""
    strategy = ConduitRoutingStrategy()  # Hybrid routing always enabled

    # Mock router initialization
    await strategy._initialize_from_litellm(mock_litellm_router)

    assert strategy._initialized
    assert strategy.conduit_router is not None
    assert strategy._router == mock_litellm_router


def test_extract_query_text_from_messages():
    """Test query text extraction from messages."""
    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "What is 2+2?"}
    ]

    text = extract_query_text(messages=messages, input_data=None)
    assert text == "What is 2+2?"


def test_extract_query_text_from_input():
    """Test query text extraction from input string."""
    text = extract_query_text(messages=None, input_data="Calculate the sum")
    assert text == "Calculate the sum"


def test_extract_query_text_from_list():
    """Test query text extraction from list input."""
    text = extract_query_text(messages=None, input_data=["query1", "query2"])
    assert text == "query1 query2"


@pytest.mark.asyncio
async def test_fallback_to_model_group(mock_litellm_router, mock_conduit_router):
    """Test fallback when Conduit selects model not in model_list."""
    # Configure Conduit to select a model not in LiteLLM's list
    mock_conduit_router.route = AsyncMock(return_value=RoutingDecision(
        query_id="test_query",
        selected_model="unknown-model",
        confidence=0.85,
        features=QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5
        ),
        reasoning="Test routing"
    ))

    strategy = ConduitRoutingStrategy(conduit_router=mock_conduit_router)
    strategy._router = mock_litellm_router
    strategy._initialized = True

    messages = [{"role": "user", "content": "Hello"}]
    deployment = await strategy.async_get_available_deployment(
        model="gpt-4",
        messages=messages
    )

    # Should fallback to model_name match
    assert deployment["model_name"] == "gpt-4"


def test_setup_strategy_helper(mock_litellm_router):
    """Test setup_strategy helper method."""
    strategy = ConduitRoutingStrategy()  # Hybrid routing always enabled

    ConduitRoutingStrategy.setup_strategy(mock_litellm_router, strategy)

    assert strategy._router == mock_litellm_router
    mock_litellm_router.set_custom_routing_strategy.assert_called_once_with(strategy)


def test_extract_model_ids_with_standard_litellm_format():
    """Test extract_model_ids works with standard LiteLLM format (no model_info.id)."""
    # Standard LiteLLM format without model_info.id
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4"}
        },
        {
            "model_name": "claude-3",
            "litellm_params": {"model": "anthropic/claude-3-opus"}
        }
    ]
    
    model_ids = extract_model_ids(model_list)
    
    # Should auto-generate IDs from model_name
    assert model_ids == ["gpt-4", "claude-3"]


def test_extract_model_ids_with_explicit_id():
    """Test extract_model_ids works with explicit model_info.id."""
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4"},
            "model_info": {"id": "gpt-4-openai"}
        }
    ]
    
    model_ids = extract_model_ids(model_list)
    
    # Should use explicit ID
    assert model_ids == ["gpt-4-openai"]


def test_extract_model_ids_from_litellm_params():
    """Test extract_model_ids generates ID from litellm_params.model when model_name missing."""
    model_list = [
        {
            "litellm_params": {"model": "openai/gpt-4o-mini"}
        }
    ]
    
    model_ids = extract_model_ids(model_list)
    
    # Should normalize litellm_params.model (remove provider prefix)
    assert model_ids == ["gpt-4o-mini"]


def test_validate_litellm_model_list_standard_format():
    """Test validation accepts standard LiteLLM format without model_info."""
    model_list = [
        {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4"}
        }
    ]
    
    # Should not raise ValueError
    validate_litellm_model_list(model_list)
