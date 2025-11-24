"""Unit tests for Arbiter integration with conduit_litellm."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.models import Query, QueryFeatures, Response
from conduit.evaluation.arbiter_evaluator import ArbiterEvaluator


# Test fixture for evaluator
@pytest.fixture
def mock_evaluator():
    """Create mock ArbiterEvaluator."""
    evaluator = MagicMock(spec=ArbiterEvaluator)
    evaluator.evaluate_async = AsyncMock(return_value=0.85)
    return evaluator


# Test fixture for LiteLLM response
@pytest.fixture
def litellm_response():
    """Create mock LiteLLM response object."""
    response = MagicMock()
    response.id = "resp-123"
    response.choices = [MagicMock(message=MagicMock(content="Test response"))]

    # Mock usage data (token counts)
    response.usage = MagicMock()
    response.usage.total_tokens = 150
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 100

    # Mock cost
    response._hidden_params = {"response_cost": 0.0001}

    return response


# Test fixture for router with analyzer
@pytest.fixture
def mock_router():
    """Create mock Conduit router."""
    router = MagicMock()
    router.hybrid_router = MagicMock()
    router.hybrid_router.update = AsyncMock()

    # Mock analyzer
    router.analyzer = MagicMock()
    router.analyzer.analyze = AsyncMock(
        return_value=QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
    )

    # Mock LinUCB bandit arms for validation (matches feedback.py validation logic)
    mock_linucb_bandit = MagicMock()
    mock_linucb_bandit.arms = {
        "gpt-4o-mini": MagicMock(model_id="gpt-4o-mini"),
    }
    router.hybrid_router.linucb_bandit = mock_linucb_bandit

    return router


class TestArbiterIntegration:
    """Tests for Arbiter evaluator integration."""

    @pytest.mark.asyncio
    async def test_feedback_logger_with_evaluator(
        self, mock_router, mock_evaluator, litellm_response
    ):
        """Test ConduitFeedbackLogger calls evaluator when provided."""
        from conduit_litellm.feedback import ConduitFeedbackLogger

        # Create feedback logger with evaluator
        logger = ConduitFeedbackLogger(mock_router, evaluator=mock_evaluator)

        # Simulate LiteLLM success callback
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        await logger.async_log_success_event(
            kwargs=kwargs,
            response_obj=litellm_response,
            start_time=1.0,
            end_time=2.5,
        )

        # Give async task time to start
        await asyncio.sleep(0.1)

        # Verify evaluator was called
        assert mock_evaluator.evaluate_async.called
        call_args = mock_evaluator.evaluate_async.call_args

        # Check Response object
        response_obj = call_args[0][0]
        assert isinstance(response_obj, Response)
        assert response_obj.text == "Test response"
        assert response_obj.model == "gpt-4o-mini"
        assert response_obj.cost == 0.0001
        assert response_obj.latency == 1.5  # end_time - start_time
        assert response_obj.tokens == 150

        # Check Query object
        query_obj = call_args[0][1]
        assert isinstance(query_obj, Query)
        assert query_obj.text == "Hello"

    @pytest.mark.asyncio
    async def test_feedback_logger_without_evaluator(
        self, mock_router, litellm_response
    ):
        """Test ConduitFeedbackLogger works without evaluator."""
        from conduit_litellm.feedback import ConduitFeedbackLogger

        # Create feedback logger WITHOUT evaluator
        logger = ConduitFeedbackLogger(mock_router, evaluator=None)

        # Simulate LiteLLM success callback
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Should not raise error
        await logger.async_log_success_event(
            kwargs=kwargs,
            response_obj=litellm_response,
            start_time=1.0,
            end_time=2.5,
        )

        # Verify bandit update was still called
        assert mock_router.hybrid_router.update.called

    @pytest.mark.asyncio
    async def test_strategy_initialization_with_evaluator(self, mock_evaluator):
        """Test ConduitRoutingStrategy accepts evaluator parameter."""
        from conduit_litellm.strategy import ConduitRoutingStrategy

        # Create strategy with evaluator
        strategy = ConduitRoutingStrategy(
            use_hybrid=True,
            evaluator=mock_evaluator,
        )

        # Verify evaluator was stored
        assert strategy.evaluator is mock_evaluator

        # Verify evaluator was NOT passed to Router config
        assert "evaluator" not in strategy.conduit_config

    @pytest.mark.asyncio
    async def test_evaluator_token_extraction(
        self, mock_router, mock_evaluator
    ):
        """Test token count extraction from LiteLLM response."""
        from conduit_litellm.feedback import ConduitFeedbackLogger

        # Create response with token usage
        response = MagicMock()
        response.id = "resp-456"
        response.choices = [MagicMock(message=MagicMock(content="Response text"))]
        response.usage = MagicMock()
        response.usage.total_tokens = 250
        response._hidden_params = {"response_cost": 0.0002}

        logger = ConduitFeedbackLogger(mock_router, evaluator=mock_evaluator)

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test"}],
        }

        await logger.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=1.0,
            end_time=3.0,
        )

        await asyncio.sleep(0.1)

        # Verify token count was extracted
        assert mock_evaluator.evaluate_async.called
        response_obj = mock_evaluator.evaluate_async.call_args[0][0]
        assert response_obj.tokens == 250

    @pytest.mark.asyncio
    async def test_evaluator_missing_tokens(
        self, mock_router, mock_evaluator
    ):
        """Test handles missing token usage gracefully."""
        from conduit_litellm.feedback import ConduitFeedbackLogger

        # Create response WITHOUT usage data
        response = MagicMock()
        response.id = "resp-789"
        response.choices = [MagicMock(message=MagicMock(content="Text"))]
        response._hidden_params = {"response_cost": 0.0001}
        # Explicitly set usage to None to indicate missing data
        response.usage = None

        logger = ConduitFeedbackLogger(mock_router, evaluator=mock_evaluator)

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Test"}],
        }

        # Should not raise error
        await logger.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=1.0,
            end_time=2.0,
        )

        await asyncio.sleep(0.1)

        # Verify evaluator was called with tokens=0
        assert mock_evaluator.evaluate_async.called
        response_obj = mock_evaluator.evaluate_async.call_args[0][0]
        assert response_obj.tokens == 0
