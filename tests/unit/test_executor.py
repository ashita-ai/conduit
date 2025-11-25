"""Unit tests for ModelExecutor LLM execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from conduit.core.exceptions import ExecutionError
from conduit.core.pricing import ModelPricing
from conduit.core.models import Response
from conduit.engines.executor import ModelExecutor


class TestResult(BaseModel):
    """Test result type for executor tests."""

    answer: str
    confidence: float


class MockUsage:
    """Mock PydanticAI Usage object."""

    def __init__(self, request_tokens: int = 100, response_tokens: int = 50):
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.total_tokens = request_tokens + response_tokens


class MockAgentResult:
    """Mock PydanticAI Agent result (v1.20+ API)."""

    def __init__(
        self,
        data: BaseModel,
        usage: MockUsage,
        text: str = "Test response",
    ):
        self.output = data  # v1.20+ uses .output instead of .data()
        self._usage = usage
        self._text = text

    def usage(self) -> MockUsage:
        return self._usage

    def output_text(self) -> str:
        return self._text


class TestModelExecutor:
    """Tests for ModelExecutor."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful LLM execution returns Response."""
        executor = ModelExecutor()

        # Mock agent result
        test_data = TestResult(answer="Test answer", confidence=0.95)
        mock_usage = MockUsage(request_tokens=100, response_tokens=50)
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        # Mock Agent.run()
        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o-mini",
                prompt="What is 2+2?",
                result_type=TestResult,
                query_id="test-query-123",
                timeout=60.0,
            )

            # Verify Response object
            assert isinstance(response, Response)
            assert response.query_id == "test-query-123"
            assert response.model == "gpt-4o-mini"
            assert '"answer":"Test answer"' in response.text
            assert response.cost > 0.0
            assert response.latency > 0.0
            assert response.tokens == 150

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout raises ExecutionError."""
        executor = ModelExecutor()

        # Mock Agent.run() to timeout
        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.side_effect = asyncio.TimeoutError("Timeout")
            MockAgent.return_value = mock_agent

            with pytest.raises(ExecutionError) as exc_info:
                await executor.execute(
                    model="gpt-4o",
                    prompt="Complex query",
                    result_type=TestResult,
                    query_id="test-timeout",
                    timeout=1.0,
                )

            assert "exceeded timeout" in str(exc_info.value)
            assert "1.0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """Test execution error raises ExecutionError with chaining."""
        executor = ModelExecutor()

        # Mock Agent.run() to raise exception
        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.side_effect = ValueError("Invalid API key")
            MockAgent.return_value = mock_agent

            with pytest.raises(ExecutionError) as exc_info:
                await executor.execute(
                    model="claude-sonnet-4",
                    prompt="Test query",
                    result_type=TestResult,
                    query_id="test-error",
                )

            assert "failed" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_cost_calculation_with_usage_object(self):
        """Test cost calculation with Usage object."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)
        mock_usage = MockUsage(request_tokens=1000, response_tokens=500)
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o-mini",
                prompt="Test",
                result_type=TestResult,
                query_id="test-cost",
            )

            # gpt-4o-mini pricing: input=0.00015, output=0.0006 per 1K tokens
            # Cost = (1000/1000 * 0.00015) + (500/1000 * 0.0006)
            # Cost = 0.00015 + 0.0003 = 0.00045
            assert response.cost == pytest.approx(0.00045, rel=0.01)

    @pytest.mark.asyncio
    async def test_cost_calculation_with_dict_fallback(self):
        """Test cost calculation with dict-based usage."""
        executor = ModelExecutor()

        # Create mock result with dict-based usage
        test_data = TestResult(answer="Answer", confidence=0.9)
        mock_result = MagicMock()
        mock_result.output = test_data  # v1.20+ uses .output attribute
        mock_result.output_text.return_value = '{"answer": "Answer", "confidence": 0.9}'

        # Mock usage() to return a dict
        usage_dict = {"request_tokens": 2000, "response_tokens": 1000, "total_tokens": 3000}
        mock_result.usage.return_value = usage_dict

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o",
                prompt="Test",
                result_type=TestResult,
                query_id="test-dict-cost",
            )

            # gpt-4o pricing: input=0.0025, output=0.01 per 1K tokens
            # Cost = (2000/1000 * 0.0025) + (1000/1000 * 0.01)
            # Cost = 0.005 + 0.01 = 0.015
            assert response.cost == pytest.approx(0.015, rel=0.01)

    @pytest.mark.asyncio
    async def test_agent_caching(self):
        """Test agent instances are cached per model+result_type."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)
        mock_usage = MockUsage()
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            # First call should create agent
            await executor.execute(
                model="gpt-4o-mini",
                prompt="First query",
                result_type=TestResult,
                query_id="query-1",
            )

            # Second call with same model+type should reuse agent
            await executor.execute(
                model="gpt-4o-mini",
                prompt="Second query",
                result_type=TestResult,
                query_id="query-2",
            )

            # Agent should be created only once
            assert MockAgent.call_count == 1

            # Third call with different model should create new agent
            await executor.execute(
                model="gpt-4o",
                prompt="Third query",
                result_type=TestResult,
                query_id="query-3",
            )

            # Agent should be created twice now
            assert MockAgent.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_calculation_with_alternative_token_fields(self):
        """Test cost calculation handles input_tokens/output_tokens fields."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)

        # Mock usage object with alternative field names
        class AlternativeUsage:
            def __init__(self):
                self.input_tokens = 500
                self.output_tokens = 250
                self.total_tokens = 750

        mock_result = MagicMock()
        mock_result.output = test_data  # v1.20+ uses .output attribute
        mock_result.output_text.return_value = '{"answer": "Answer", "confidence": 0.9}'
        mock_result.usage.return_value = AlternativeUsage()

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="claude-3.5-sonnet",
                prompt="Test",
                result_type=TestResult,
                query_id="test-alt-tokens",
            )

            # claude-3.5-sonnet uses _default pricing from conduit.yaml: input=1.00, output=3.00 per 1M tokens
            # Cost = (500 * 1.00/1M) + (250 * 3.00/1M)
            # Cost = 0.0005 + 0.00075 = 0.00125
            assert response.cost == pytest.approx(0.00125, rel=0.01)

    @pytest.mark.asyncio
    async def test_cost_calculation_fallback_to_zero(self):
        """Test cost calculation returns 0.0 for unextractable usage."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)

        # Mock usage object with no recognizable fields
        mock_result = MagicMock()
        mock_result.output = test_data  # v1.20+ uses .output attribute
        mock_result.output_text.return_value = '{"answer": "Answer", "confidence": 0.9}'
        mock_result.usage.return_value = "invalid usage format"

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o-mini",
                prompt="Test",
                result_type=TestResult,
                query_id="test-fallback",
            )

            # Should fallback to 0.0 cost
            assert response.cost == 0.0

    @pytest.mark.asyncio
    async def test_cost_calculation_handles_model_dump_exception(self):
        """Test cost calculation handles exception when model_dump() fails."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)

        # Create usage object that raises exception on model_dump()
        class BrokenUsage:
            def model_dump(self):
                raise RuntimeError("model_dump failed")

        mock_result = MagicMock()
        mock_result.output = test_data  # v1.20+ uses .output attribute
        mock_result.output_text.return_value = '{"answer": "Answer", "confidence": 0.9}'
        mock_result.usage.return_value = BrokenUsage()

        with patch("conduit.engines.executor.Agent") as MockAgent:
            with patch("conduit.engines.executor.logger") as mock_logger:
                mock_agent = AsyncMock()
                mock_agent.run.return_value = mock_result
                MockAgent.return_value = mock_agent

                response = await executor.execute(
                    model="gpt-4o-mini",
                    prompt="Test",
                    result_type=TestResult,
                    query_id="test-exception",
                )

                # Should fallback to 0.0 cost
                assert response.cost == 0.0

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                assert "Could not extract tokens" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_cost_calculation_uses_database_pricing_when_available(self):
        """Test cost calculation prefers database-backed pricing over fallback table."""
        # Database-backed pricing: 0.001 input, 0.004 output per 1M tokens
        pricing = {
            "custom-model": ModelPricing(
                model_id="custom-model",
                input_cost_per_million=0.001,
                output_cost_per_million=0.004,
                cached_input_cost_per_million=None,
                source="test",
                snapshot_at=None,
            )
        }
        executor = ModelExecutor(pricing=pricing)

        test_data = TestResult(answer="Answer", confidence=0.9)
        mock_usage = MockUsage(request_tokens=1_000_000, response_tokens=500_000)
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="custom-model",
                prompt="Test",
                result_type=TestResult,
                query_id="test-db-pricing",
            )

            # Cost per token: input=1e-9, output=4e-9
            # Cost = 1_000_000 * 1e-9 + 500_000 * 4e-9 = 0.001 + 0.002 = 0.003
            assert response.cost == pytest.approx(0.003, rel=0.01)

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test latency is measured correctly."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Answer", confidence=0.9)
        mock_usage = MockUsage()
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()

            # Add small delay to simulate LLM call
            async def delayed_run(*args, **kwargs):
                await asyncio.sleep(0.1)
                return mock_result

            mock_agent.run = delayed_run
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o-mini",
                prompt="Test",
                result_type=TestResult,
                query_id="test-latency",
            )

            # Latency should be at least 100ms
            assert response.latency >= 0.1
            assert response.latency < 1.0  # Should complete quickly

    @pytest.mark.asyncio
    async def test_response_json_serialization(self):
        """Test response text is properly JSON-serialized."""
        executor = ModelExecutor()

        test_data = TestResult(answer="Test answer with \"quotes\"", confidence=0.95)
        mock_usage = MockUsage()
        mock_result = MockAgentResult(data=test_data, usage=mock_usage)

        with patch("conduit.engines.executor.Agent") as MockAgent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = mock_result
            MockAgent.return_value = mock_agent

            response = await executor.execute(
                model="gpt-4o-mini",
                prompt="Test",
                result_type=TestResult,
                query_id="test-json",
            )

            # Verify response text is valid JSON
            import json

            parsed = json.loads(response.text)
            assert parsed["answer"] == 'Test answer with "quotes"'
            assert parsed["confidence"] == 0.95
