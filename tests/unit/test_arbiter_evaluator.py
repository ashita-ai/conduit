"""Tests for Arbiter evaluation integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.database import Database
from conduit.core.models import Query, Response
from conduit.evaluation.arbiter_evaluator import ArbiterEvaluator


@pytest.fixture
def test_db():
    """Mock database for testing."""
    db = MagicMock(spec=Database)
    db.save_complete_interaction = AsyncMock()
    return db


@pytest.fixture
def test_evaluator(test_db):
    """Create evaluator with test configuration."""
    return ArbiterEvaluator(
        db=test_db,
        sample_rate=0.1,
        daily_budget=10.0,
        model="gpt-4o-mini",
    )


@pytest.fixture
def test_query():
    """Sample query for testing."""
    return Query(
        text="What is the capital of France?",
        user_id="test-user",
    )


@pytest.fixture
def test_response():
    """Sample response for testing."""
    return Response(
        id="resp-123",
        query_id="query-456",
        model="gpt-4o-mini",
        text="The capital of France is Paris.",
        cost=0.0001,
        latency=0.5,
        tokens=100,
    )


class TestArbiterEvaluatorInit:
    """Test evaluator initialization."""

    def test_init_with_defaults(self, test_db):
        """Test initialization with default parameters."""
        evaluator = ArbiterEvaluator(test_db)

        assert evaluator.db == test_db
        assert evaluator.sample_rate == 0.1
        assert evaluator.daily_budget == 10.0
        assert evaluator.model == "gpt-4o-mini"
        assert len(evaluator.evaluators) == 2  # semantic + factuality

    def test_init_with_custom_config(self, test_db):
        """Test initialization with custom configuration."""
        evaluator = ArbiterEvaluator(
            test_db,
            sample_rate=0.2,
            daily_budget=20.0,
            model="gpt-4o",
        )

        assert evaluator.sample_rate == 0.2
        assert evaluator.daily_budget == 20.0
        assert evaluator.model == "gpt-4o"

    def test_init_with_invalid_sample_rate(self, test_db):
        """Test initialization fails with invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate must be in"):
            ArbiterEvaluator(test_db, sample_rate=1.5)

        with pytest.raises(ValueError, match="sample_rate must be in"):
            ArbiterEvaluator(test_db, sample_rate=-0.1)


class TestEvaluationSampling:
    """Test evaluation sampling logic."""

    @pytest.mark.asyncio
    async def test_should_evaluate_respects_sample_rate(self, test_evaluator):
        """Test sampling rate is respected."""
        # With 10% sampling, approximately 10% should return True
        results = [await test_evaluator.should_evaluate() for _ in range(1000)]
        true_count = sum(results)

        # Allow 20% variance (80-120 out of 1000)
        assert 80 <= true_count <= 120, f"Expected ~100, got {true_count}"

    @pytest.mark.asyncio
    async def test_should_evaluate_with_zero_sample_rate(self, test_db):
        """Test 0% sampling never evaluates."""
        evaluator = ArbiterEvaluator(test_db, sample_rate=0.0)
        results = [await evaluator.should_evaluate() for _ in range(100)]

        assert sum(results) == 0

    @pytest.mark.asyncio
    async def test_should_evaluate_with_full_sample_rate(self, test_db):
        """Test 100% sampling always evaluates."""
        evaluator = ArbiterEvaluator(test_db, sample_rate=1.0)
        results = [await evaluator.should_evaluate() for _ in range(100)]

        assert sum(results) == 100


class TestAsyncEvaluation:
    """Test async evaluation execution."""

    @pytest.mark.asyncio
    @patch("conduit.evaluation.arbiter_evaluator.evaluate")
    async def test_evaluate_async_success(
        self, mock_evaluate, test_evaluator, test_response, test_query, test_db
    ):
        """Test successful evaluation."""
        # Mock Arbiter evaluate result
        mock_result = MagicMock()
        mock_result.overall_score = 0.95
        mock_result.interactions = [MagicMock(cost_usd=0.0001)]
        mock_evaluate.return_value = mock_result

        # Force sampling to always evaluate
        test_evaluator.sample_rate = 1.0

        # Run evaluation
        score = await test_evaluator.evaluate_async(test_response, test_query)

        # Verify score returned
        assert score == 0.95

        # Verify Arbiter was called with correct params
        mock_evaluate.assert_called_once()
        call_kwargs = mock_evaluate.call_args[1]
        assert call_kwargs["output"] == test_response.text
        assert call_kwargs["reference"] == test_query.text
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["evaluators"] == ["semantic", "factuality"]

        # Verify feedback was saved to database
        test_db.save_complete_interaction.assert_called_once()
        call_args = test_db.save_complete_interaction.call_args
        feedback = call_args[1]["feedback"]
        assert feedback.response_id == test_response.id
        assert feedback.quality_score == 0.95
        assert "Arbiter eval" in feedback.comments

    @pytest.mark.asyncio
    async def test_evaluate_async_skipped_by_sampling(
        self, test_evaluator, test_response, test_query
    ):
        """Test evaluation skipped due to sampling."""
        # Force sampling to never evaluate
        test_evaluator.sample_rate = 0.0

        score = await test_evaluator.evaluate_async(test_response, test_query)

        # Should return None when skipped
        assert score is None

    @pytest.mark.asyncio
    @patch("conduit.evaluation.arbiter_evaluator.evaluate")
    async def test_evaluate_async_handles_errors_gracefully(
        self, mock_evaluate, test_evaluator, test_response, test_query, test_db
    ):
        """Test evaluation failures don't crash routing."""
        # Mock evaluation failure
        mock_evaluate.side_effect = Exception("Evaluation API error")

        # Force sampling
        test_evaluator.sample_rate = 1.0

        # Should not raise exception
        score = await test_evaluator.evaluate_async(test_response, test_query)

        # Should return None on failure
        assert score is None

        # Database should not be called
        test_db.save_complete_interaction.assert_not_called()

    @pytest.mark.asyncio
    @patch("conduit.evaluation.arbiter_evaluator.evaluate")
    async def test_evaluate_async_fire_and_forget(
        self, mock_evaluate, test_evaluator, test_response, test_query
    ):
        """Test evaluation runs in background without blocking."""
        # Mock slow evaluation (100ms)
        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(0.1)
            result = MagicMock()
            result.overall_score = 0.9
            result.interactions = [MagicMock(cost_usd=0.0001)]
            return result

        mock_evaluate.side_effect = slow_evaluate
        test_evaluator.sample_rate = 1.0

        # Create task (fire-and-forget)
        task = asyncio.create_task(
            test_evaluator.evaluate_async(test_response, test_query)
        )

        # Should return immediately without waiting
        # (task is running in background)
        assert not task.done()

        # Wait for task to complete
        score = await task
        assert score == 0.9


class TestConfiguration:
    """Test evaluator configuration management."""

    def test_get_config(self, test_evaluator):
        """Test configuration retrieval."""
        config = test_evaluator.get_config()

        assert config["sample_rate"] == 0.1
        assert config["daily_budget"] == 10.0
        assert config["model"] == "gpt-4o-mini"
        assert config["evaluators"] == ["semantic", "factuality"]
