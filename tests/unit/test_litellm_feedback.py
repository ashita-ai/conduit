"""Tests for LiteLLM feedback integration (ConduitFeedbackLogger).

Tests automatic feedback capture from LiteLLM responses and bandit updates.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from conduit.core.models import QueryFeatures
from conduit.engines.bandits.base import BanditFeedback

# Try to import conduit_litellm components
pytest.importorskip("litellm", reason="litellm not installed")

from conduit_litellm.feedback import ConduitFeedbackLogger


@pytest.fixture
def mock_router():
    """Create mock Conduit router with analyzer and bandit."""
    router = Mock()

    # Mock analyzer with async analyze method
    analyzer = Mock()
    analyzer.analyze = AsyncMock(
        return_value=QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
    )
    router.analyzer = analyzer

    # Mock bandit with async update method and arms
    bandit = Mock()
    bandit.update = AsyncMock()
    bandit.arms = {
        "gpt-4o-mini": Mock(),
        "claude-3-haiku": Mock(),
        "gemini-pro": Mock(),
    }
    router.bandit = bandit
    router.hybrid_router = None

    return router


@pytest.fixture
def mock_hybrid_router():
    """Create mock Conduit router with HybridRouter."""
    router = Mock()

    # Mock analyzer
    analyzer = Mock()
    analyzer.analyze = AsyncMock(
        return_value=QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
            domain="general",
            domain_confidence=0.8,
        )
    )
    router.analyzer = analyzer

    # Mock hybrid_router with async update method
    hybrid = Mock()
    hybrid.update = AsyncMock()
    # Mock linucb_bandit with arms
    linucb = Mock()
    linucb.arms = {
        "gpt-4o-mini": Mock(),
        "claude-3-haiku": Mock(),
    }
    hybrid.linucb_bandit = linucb
    router.hybrid_router = hybrid
    router.bandit = None

    return router


@pytest.fixture
def feedback_logger(mock_router):
    """Create ConduitFeedbackLogger instance."""
    return ConduitFeedbackLogger(mock_router)


@pytest.fixture
def litellm_kwargs():
    """Create sample LiteLLM request kwargs."""
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ],
    }


@pytest.fixture
def litellm_response():
    """Create mock LiteLLM response object."""
    response = Mock()
    response.id = "resp_123"
    response._hidden_params = {"response_cost": 0.00015}
    response.response_cost = 0.00015  # Fallback attribute

    # Mock choices structure for text extraction
    message = Mock()
    message.content = "4"  # Sample response content
    choice = Mock()
    choice.message = message
    response.choices = [choice]

    return response


class TestConduitFeedbackLogger:
    """Tests for ConduitFeedbackLogger class."""

    def test_init(self, mock_router):
        """Test logger initialization."""
        logger = ConduitFeedbackLogger(mock_router)

        assert logger.router is mock_router
        assert logger.router.analyzer is not None

    @pytest.mark.asyncio
    async def test_async_log_success_event(
        self, feedback_logger, mock_router, litellm_kwargs, litellm_response
    ):
        """Test successful completion logging and bandit update."""
        start_time = 1000.0
        end_time = 1001.5

        # Call success event handler
        await feedback_logger.async_log_success_event(
            litellm_kwargs, litellm_response, start_time, end_time
        )

        # Verify analyzer was called to extract features
        mock_router.analyzer.analyze.assert_called_once_with("What is 2+2?")

        # Verify bandit update was called
        mock_router.bandit.update.assert_called_once()

        # Verify feedback structure
        call_args = mock_router.bandit.update.call_args
        feedback = call_args[0][0]  # First positional arg
        features = call_args[0][1]  # Second positional arg

        assert isinstance(feedback, BanditFeedback)
        assert feedback.model_id == "gpt-4o-mini"
        assert feedback.cost == 0.00015
        # Quality is estimated from content (short response "4" gets lower score)
        assert 0.5 <= feedback.quality_score <= 0.6  # Estimated quality
        assert feedback.latency == 1.5  # end_time - start_time
        assert feedback.success is True
        assert feedback.metadata["source"] == "litellm"
        assert feedback.metadata["response_id"] == "resp_123"

        assert isinstance(features, QueryFeatures)
        assert features.token_count == 50

    @pytest.mark.asyncio
    async def test_async_log_failure_event(
        self, feedback_logger, mock_router, litellm_kwargs, litellm_response
    ):
        """Test failed completion logging with low quality score."""
        start_time = 1000.0
        end_time = 1002.0

        # Call failure event handler
        await feedback_logger.async_log_failure_event(
            litellm_kwargs, litellm_response, start_time, end_time
        )

        # Verify bandit update was called
        mock_router.bandit.update.assert_called_once()

        # Verify feedback has low quality
        call_args = mock_router.bandit.update.call_args
        feedback = call_args[0][0]

        assert feedback.model_id == "gpt-4o-mini"
        assert feedback.quality_score == 0.1  # Failure = low quality
        assert feedback.latency == 2.0
        assert feedback.success is False
        assert "error" in feedback.metadata

    @pytest.mark.asyncio
    async def test_hybrid_router_update(
        self, mock_hybrid_router, litellm_kwargs, litellm_response
    ):
        """Test feedback updates HybridRouter when in hybrid mode."""
        logger = ConduitFeedbackLogger(mock_hybrid_router)

        await logger.async_log_success_event(
            litellm_kwargs, litellm_response, 1000.0, 1001.0
        )

        # Verify hybrid_router.update was called (not bandit.update)
        mock_hybrid_router.hybrid_router.update.assert_called_once()
        assert mock_hybrid_router.bandit is None  # Not used in hybrid mode

    @pytest.mark.asyncio
    async def test_cost_unavailable_skips_feedback(
        self, feedback_logger, mock_router, litellm_kwargs
    ):
        """Test that unavailable cost skips feedback (prevents cost=0.0 corruption)."""
        response = Mock()
        response._hidden_params = {}
        # Remove fallback
        if hasattr(response, "response_cost"):
            delattr(response, "response_cost")

        await feedback_logger.async_log_success_event(
            litellm_kwargs, response, 1000.0, 1001.0
        )

        # Analyzer should still be called
        mock_router.analyzer.analyze.assert_called_once()
        # But bandit update should NOT be called
        mock_router.bandit.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_model_id_skips_feedback(
        self, feedback_logger, mock_router, litellm_response
    ):
        """Test that unknown model_id skips feedback."""
        kwargs = {
            "model": "unknown",
            "messages": [{"role": "user", "content": "test"}],
        }

        await feedback_logger.async_log_success_event(
            kwargs, litellm_response, 1000.0, 1001.0
        )

        # Analyzer called but bandit not updated
        mock_router.analyzer.analyze.assert_called_once()
        mock_router.bandit.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_not_in_arms_skips_feedback(
        self, feedback_logger, mock_router, litellm_response
    ):
        """Test that model not in arms skips feedback."""
        kwargs = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "test"}],
        }

        await feedback_logger.async_log_success_event(
            kwargs, litellm_response, 1000.0, 1001.0
        )

        # Analyzer called but bandit not updated
        mock_router.analyzer.analyze.assert_called_once()
        mock_router.bandit.update.assert_not_called()

    def test_extract_query_text_messages(self, feedback_logger):
        """Test query text extraction from messages format."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User query here"},
            ]
        }

        text = feedback_logger._extract_query_text(kwargs)
        assert text == "User query here"

    def test_extract_query_text_input_string(self, feedback_logger):
        """Test query text extraction from input string."""
        kwargs = {"input": "Direct input text"}

        text = feedback_logger._extract_query_text(kwargs)
        assert text == "Direct input text"

    def test_extract_query_text_input_list(self, feedback_logger):
        """Test query text extraction from input list."""
        kwargs = {"input": ["First part", "Second part"]}

        text = feedback_logger._extract_query_text(kwargs)
        assert text == "First part Second part"

    def test_extract_query_text_empty(self, feedback_logger):
        """Test query text extraction returns empty string when no text."""
        kwargs = {}

        text = feedback_logger._extract_query_text(kwargs)
        assert text == ""

    def test_extract_cost_hidden_params(self, feedback_logger):
        """Test cost extraction from _hidden_params."""
        response = Mock()
        response._hidden_params = {"response_cost": 0.00025}

        cost = feedback_logger._extract_cost(response)
        assert cost == 0.00025

    def test_extract_cost_fallback_attribute(self, feedback_logger):
        """Test cost extraction from fallback attribute."""
        response = Mock()
        response._hidden_params = {}  # Empty hidden params
        response.response_cost = 0.00035

        cost = feedback_logger._extract_cost(response)
        assert cost == 0.00035

    def test_extract_cost_unavailable(self, feedback_logger):
        """Test cost extraction returns None when unavailable."""
        response = Mock()
        response._hidden_params = {}
        # Remove fallback attribute
        if hasattr(response, "response_cost"):
            delattr(response, "response_cost")

        cost = feedback_logger._extract_cost(response)
        assert cost is None

    def test_validate_model_id_exists(self, feedback_logger):
        """Test model validation for existing model."""
        assert feedback_logger._validate_model_id("gpt-4o-mini") is True
        assert feedback_logger._validate_model_id("claude-3-haiku") is True

    def test_validate_model_id_not_exists(self, feedback_logger):
        """Test model validation for non-existent model."""
        assert feedback_logger._validate_model_id("nonexistent") is False

    def test_get_available_model_ids(self, feedback_logger):
        """Test getting available model IDs."""
        model_ids = feedback_logger._get_available_model_ids()
        assert set(model_ids) == {"gpt-4o-mini", "claude-3-haiku", "gemini-pro"}

    @pytest.mark.asyncio
    async def test_no_bandit_available(self, litellm_kwargs, litellm_response):
        """Test graceful handling when no bandit is available."""
        router = Mock()
        router.analyzer = Mock()
        router.analyzer.analyze = AsyncMock(
            return_value=QueryFeatures(
                embedding=[0.1] * 384,
                token_count=50,
                complexity_score=0.5,
                domain="general",
                domain_confidence=0.8,
            )
        )
        router.hybrid_router = None
        router.bandit = None

        logger = ConduitFeedbackLogger(router)

        # Should not raise, just log warning
        await logger.async_log_success_event(
            litellm_kwargs, litellm_response, 1000.0, 1001.0
        )

    @pytest.mark.asyncio
    async def test_empty_query_text_skips_feedback(
        self, feedback_logger, mock_router, litellm_response
    ):
        """Test that empty query text skips feedback update."""
        kwargs = {"model": "gpt-4o-mini", "messages": []}  # No messages

        await feedback_logger.async_log_success_event(
            kwargs, litellm_response, 1000.0, 1001.0
        )

        # Analyzer should not be called
        mock_router.analyzer.analyze.assert_not_called()
        # Bandit update should not be called
        mock_router.bandit.update.assert_not_called()

    def test_sync_log_success_event_no_loop(
        self, feedback_logger, litellm_kwargs, litellm_response
    ):
        """Test sync wrapper when no event loop running."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run") as mock_run:
                feedback_logger.log_success_event(
                    litellm_kwargs, litellm_response, 1000.0, 1001.0
                )

                # Verify asyncio.run was called
                mock_run.assert_called_once()

    def test_sync_log_success_event_with_loop(
        self, feedback_logger, litellm_kwargs, litellm_response
    ):
        """Test sync wrapper when event loop already running."""
        mock_loop = Mock()
        mock_loop.create_task = Mock()

        with patch("asyncio.get_running_loop", return_value=mock_loop):
            feedback_logger.log_success_event(
                litellm_kwargs, litellm_response, 1000.0, 1001.0
            )

            # Verify task was created
            mock_loop.create_task.assert_called_once()


class TestFeedbackIntegration:
    """Integration tests for complete feedback loop."""

    @pytest.mark.asyncio
    async def test_feedback_updates_bandit_state(
        self, mock_router, litellm_kwargs, litellm_response
    ):
        """Test that feedback actually updates bandit internal state."""
        # Create logger
        logger = ConduitFeedbackLogger(mock_router)

        # Simulate multiple successes
        for i in range(3):
            await logger.async_log_success_event(
                litellm_kwargs, litellm_response, 1000.0 + i, 1001.0 + i
            )

        # Verify bandit.update was called 3 times
        assert mock_router.bandit.update.call_count == 3

        # Verify all calls had different latencies
        latencies = [
            call[0][0].latency for call in mock_router.bandit.update.call_args_list
        ]
        assert latencies == [1.0, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_feedback_handles_different_models(
        self, mock_router, litellm_response
    ):
        """Test feedback works with different model IDs."""
        logger = ConduitFeedbackLogger(mock_router)

        models = ["gpt-4o-mini", "claude-3-haiku", "gemini-pro"]

        for model in models:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": f"Test for {model}"}],
            }

            await logger.async_log_success_event(
                kwargs, litellm_response, 1000.0, 1001.0
            )

        # Verify 3 updates
        assert mock_router.bandit.update.call_count == 3

        # Verify different model IDs
        model_ids = [
            call[0][0].model_id for call in mock_router.bandit.update.call_args_list
        ]
        assert model_ids == models
