"""Unit tests for RoutingService."""

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, Mock, patch

from conduit.api.service import RoutingService
from conduit.core.models import (
    Feedback,
    ModelState,
    Query,
    QueryConstraints,
    QueryFeatures,
    Response,
    RoutingDecision,
    RoutingResult,
)


class TestResult(BaseModel):
    """Test result type for structured outputs."""

    answer: str
    confidence: float


@pytest.fixture
def mock_database():
    """Mock database interface."""
    db = AsyncMock()
    db.save_query = AsyncMock()
    db.save_complete_interaction = AsyncMock()
    db.update_model_state = AsyncMock()
    db.get_response_by_id = AsyncMock()
    return db


@pytest.fixture
def mock_analyzer():
    """Mock query analyzer."""
    analyzer = Mock()
    return analyzer


@pytest.fixture
def mock_bandit():
    """Mock contextual bandit."""
    bandit = Mock()
    bandit.update = Mock()
    bandit.get_model_state = Mock(
        return_value=ModelState(
            model_id="gpt-4o-mini",
            alpha=2.0,
            beta=1.0,
            total_requests=10,
            total_cost=0.05,
            avg_quality=0.85,
        )
    )
    return bandit


@pytest.fixture
def mock_executor():
    """Mock model executor."""
    executor = AsyncMock()
    return executor


@pytest.fixture
def mock_router():
    """Mock routing engine."""
    router = AsyncMock()
    router.models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"]
    return router


@pytest.fixture
def service(mock_database, mock_analyzer, mock_bandit, mock_executor, mock_router):
    """Create RoutingService instance with mocks."""
    return RoutingService(
        database=mock_database,
        analyzer=mock_analyzer,
        bandit=mock_bandit,
        executor=mock_executor,
        router=mock_router,
    )


class TestComplete:
    """Tests for RoutingService.complete()."""

    @pytest.mark.asyncio
    async def test_complete_basic(self, service, mock_database, mock_router, mock_executor):
        """Test basic completion flow."""
        # Setup mocks
        mock_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="query-123",
                selected_model="gpt-4o-mini",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384,
                    token_count=10,
                    complexity_score=0.3,
                    domain="general",
                    domain_confidence=0.7,
                ),
                reasoning="Selected gpt-4o-mini for low complexity query",
            )
        )

        mock_executor.execute = AsyncMock(
            return_value=Response(
                query_id="query-123",
                model="gpt-4o-mini",
                text='{"answer": "4", "confidence": 0.95}',
                cost=0.001,
                latency=0.5,
                tokens=20,
            )
        )

        # Execute
        result = await service.complete(
            prompt="What is 2+2?",
            result_type=TestResult,
            user_id="test-user",
        )

        # Verify
        assert isinstance(result, RoutingResult)
        assert result.model == "gpt-4o-mini"
        assert "answer" in result.data
        assert result.metadata["cost"] == 0.001
        assert result.metadata["routing_confidence"] == 0.85

        # Verify interactions
        mock_database.save_query.assert_called_once()
        mock_router.route.assert_called_once()
        mock_executor.execute.assert_called_once()
        mock_database.save_complete_interaction.assert_called_once()
        mock_database.update_model_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_constraints(
        self, service, mock_database, mock_router, mock_executor
    ):
        """Test completion with routing constraints."""
        # Setup routing decision
        mock_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="query-456",
                selected_model="gpt-4o-mini",
                confidence=0.9,
                features=QueryFeatures(
                    embedding=[0.1] * 384,
                    token_count=5,
                    complexity_score=0.2,
                    domain="general",
                    domain_confidence=0.8,
                ),
                reasoning="Selected cheap model due to cost constraint",
            )
        )

        mock_executor.execute = AsyncMock(
            return_value=Response(
                query_id="query-456",
                model="gpt-4o-mini",
                text='{"answer": "Paris", "confidence": 0.99}',
                cost=0.0005,
                latency=0.3,
                tokens=10,
            )
        )

        # Execute with constraints
        result = await service.complete(
            prompt="What is the capital of France?",
            result_type=TestResult,
            constraints={"max_cost": 0.001, "min_quality": 0.7},
        )

        # Verify constraints were passed
        call_args = mock_database.save_query.call_args
        query = call_args[0][0]
        assert query.constraints is not None
        assert query.constraints.max_cost == 0.001
        assert query.constraints.min_quality == 0.7

    @pytest.mark.asyncio
    async def test_complete_default_result_type(
        self, service, mock_router, mock_executor
    ):
        """Test completion uses default result type when none provided."""
        mock_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="query-789",
                selected_model="gpt-4o-mini",
                confidence=0.8,
                features=QueryFeatures(
                    embedding=[0.1] * 384,
                    token_count=8,
                    complexity_score=0.25,
                    domain="general",
                    domain_confidence=0.75,
                ),
                reasoning="Default routing",
            )
        )

        mock_executor.execute = AsyncMock(
            return_value=Response(
                query_id="query-789",
                model="gpt-4o-mini",
                text='{"content": "Hello, world!"}',
                cost=0.0008,
                latency=0.4,
                tokens=15,
            )
        )

        # Execute without result_type (should use DefaultResult)
        result = await service.complete(prompt="Say hello")

        assert isinstance(result, RoutingResult)
        mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_execution_error(self, service, mock_router, mock_executor):
        """Test handling of execution errors."""
        from conduit.core.exceptions import ExecutionError

        mock_router.route = AsyncMock(
            return_value=RoutingDecision(
                query_id="query-error",
                selected_model="gpt-4o-mini",
                confidence=0.85,
                features=QueryFeatures(
                    embedding=[0.1] * 384,
                    token_count=10,
                    complexity_score=0.3,
                    domain="general",
                    domain_confidence=0.7,
                ),
                reasoning="Test routing",
            )
        )

        # Simulate execution failure
        mock_executor.execute = AsyncMock(
            side_effect=ExecutionError("Model API error")
        )

        # Should propagate ExecutionError
        with pytest.raises(ExecutionError, match="Model API error"):
            await service.complete(prompt="Test error")


class TestSubmitFeedback:
    """Tests for RoutingService.submit_feedback()."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, service, mock_database, mock_bandit):
        """Test successful feedback submission."""
        # Setup mock response
        mock_database.get_response_by_id = AsyncMock(
            return_value=Response(
                id="response-123",
                query_id="query-123",
                model="gpt-4o-mini",
                text='{"answer": "test"}',
                cost=0.001,
                latency=0.5,
                tokens=20,
            )
        )

        # Submit feedback
        feedback = await service.submit_feedback(
            response_id="response-123",
            quality_score=0.9,
            met_expectations=True,
            user_rating=5,
            comments="Great response!",
        )

        # Verify feedback object
        assert isinstance(feedback, Feedback)
        assert feedback.response_id == "response-123"
        assert feedback.quality_score == 0.9
        assert feedback.met_expectations is True
        assert feedback.user_rating == 5
        assert feedback.comments == "Great response!"

        # Verify interactions
        mock_database.get_response_by_id.assert_called_once_with("response-123")
        mock_database.save_complete_interaction.assert_called_once()
        mock_bandit.update.assert_called_once_with(
            model="gpt-4o-mini", reward=0.9, query_id="query-123"
        )
        mock_database.update_model_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_feedback_response_not_found(self, service, mock_database):
        """Test feedback submission with non-existent response."""
        # Setup mock to return None
        mock_database.get_response_by_id = AsyncMock(return_value=None)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Response invalid-id not found"):
            await service.submit_feedback(
                response_id="invalid-id",
                quality_score=0.8,
                met_expectations=True,
            )

    @pytest.mark.asyncio
    async def test_submit_feedback_minimal(self, service, mock_database):
        """Test feedback with only required fields."""
        mock_database.get_response_by_id = AsyncMock(
            return_value=Response(
                id="response-456",
                query_id="query-456",
                model="claude-sonnet-4",
                text='{"answer": "test"}',
                cost=0.002,
                latency=0.8,
                tokens=30,
            )
        )

        feedback = await service.submit_feedback(
            response_id="response-456",
            quality_score=0.75,
            met_expectations=False,
        )

        assert feedback.user_rating is None
        assert feedback.comments is None
        assert feedback.quality_score == 0.75


class TestGetStats:
    """Tests for RoutingService.get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats_placeholder(self, service):
        """Test stats returns placeholder data."""
        stats = await service.get_stats()

        assert isinstance(stats, dict)
        assert "total_queries" in stats
        assert "total_cost" in stats
        assert "avg_latency" in stats
        assert "model_distribution" in stats
        assert "quality_metrics" in stats


class TestGetModels:
    """Tests for RoutingService.get_models()."""

    @pytest.mark.asyncio
    async def test_get_models(self, service, mock_router):
        """Test getting available models."""
        models = await service.get_models()

        assert isinstance(models, list)
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "claude-sonnet-4" in models
