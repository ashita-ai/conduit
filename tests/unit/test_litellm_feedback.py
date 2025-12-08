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
from conduit_litellm.utils import extract_query_text


@pytest.fixture
def mock_router():
    """Create mock Conduit router with analyzer and bandit."""
    router = Mock()

    # Mock analyzer with async analyze method
    analyzer = Mock()
    analyzer.analyze = AsyncMock(
        return_value=QueryFeatures(
            embedding=[0.1] * 384, token_count=50, complexity_score=0.5
        )
    )
    router.analyzer = analyzer

    # Mock bandit with async update method and arms
    # Note: Arms use MAPPED Conduit IDs (after map_litellm_to_conduit translation)
    # LiteLLM "gpt-4o-mini" → Conduit "o4-mini"
    # LiteLLM "claude-3-haiku" → stays "claude-3-haiku" (no mapping)
    # LiteLLM "gemini-pro" → Conduit "gemini-2.5-pro"
    bandit = Mock()
    bandit.update = AsyncMock()
    bandit.arms = {
        "o4-mini": Mock(),
        "claude-3-haiku": Mock(),
        "gemini-2.5-pro": Mock(),
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
            embedding=[0.1] * 384, token_count=50, complexity_score=0.5
        )
    )
    router.analyzer = analyzer

    # Mock hybrid_router with async update method
    hybrid = Mock()
    hybrid.update = AsyncMock()
    # Mock linucb with arms (using MAPPED Conduit IDs)
    # HybridRouter uses "linucb" attribute (not "linucb_bandit")
    linucb = Mock()
    linucb.arms = {
        "o4-mini": Mock(),
        "claude-3-haiku": Mock(),
    }
    hybrid.linucb = linucb
    router.hybrid_router = hybrid
    router.bandit = None

    return router


@pytest.fixture
def litellm_to_conduit_map():
    """Mapping from LiteLLM model names to Conduit model IDs.

    This simulates what ConduitRoutingStrategy would register.
    """
    return {
        "gpt-4o-mini": "o4-mini",
        "claude-3-haiku": "claude-3-haiku",
        "gemini-pro": "gemini-2.5-pro",
    }


@pytest.fixture
def feedback_logger(mock_router, litellm_to_conduit_map):
    """Create ConduitFeedbackLogger instance with proper mappings."""
    return ConduitFeedbackLogger(
        mock_router,
        litellm_to_conduit_map=litellm_to_conduit_map,
    )


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
        # Model ID is MAPPED from LiteLLM "gpt-4o-mini" to Conduit "o4-mini"
        assert feedback.model_id == "o4-mini"
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

        # Model ID is MAPPED from LiteLLM "gpt-4o-mini" to Conduit "o4-mini"
        assert feedback.model_id == "o4-mini"
        assert feedback.quality_score == 0.1  # Failure = low quality
        assert feedback.latency == 2.0
        assert feedback.success is False
        assert "error" in feedback.metadata

    @pytest.mark.asyncio
    async def test_hybrid_router_update(
        self,
        mock_hybrid_router,
        litellm_kwargs,
        litellm_response,
        litellm_to_conduit_map,
    ):
        """Test feedback updates HybridRouter when in hybrid mode."""
        logger = ConduitFeedbackLogger(
            mock_hybrid_router,
            litellm_to_conduit_map=litellm_to_conduit_map,
        )

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

    def test_extract_query_text_messages(self):
        """Test query text extraction from messages format."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query here"},
        ]

        text = extract_query_text(messages=messages, input_data=None)
        assert text == "User query here"

    def test_extract_query_text_input_string(self):
        """Test query text extraction from input string."""
        text = extract_query_text(messages=None, input_data="Direct input text")
        assert text == "Direct input text"

    def test_extract_query_text_input_list(self):
        """Test query text extraction from input list."""
        text = extract_query_text(
            messages=None, input_data=["First part", "Second part"]
        )
        assert text == "First part Second part"

    def test_extract_query_text_empty(self):
        """Test query text extraction returns empty string when no text."""
        text = extract_query_text(messages=None, input_data=None)
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

    def test_resolve_model_id_exact_match(self, feedback_logger):
        """Test model resolution with exact mapping match."""
        from conduit_litellm.model_registry import MatchSource

        # "gpt-4o-mini" is mapped to "o4-mini" in the litellm_to_conduit_map
        result = feedback_logger._resolve_model_id("gpt-4o-mini")
        assert result.success
        assert result.model_id == "o4-mini"
        assert result.source == MatchSource.EXACT
        assert result.confidence == 1.0

    def test_resolve_model_id_not_found(self, feedback_logger):
        """Test model resolution for unmapped model returns failure."""
        from conduit_litellm.model_registry import MatchSource

        result = feedback_logger._resolve_model_id("nonexistent-model-xyz")
        assert not result.success
        assert result.source == MatchSource.FAILED
        assert result.confidence == 0.0

    def test_get_available_model_ids(self, feedback_logger):
        """Test getting available model IDs (MAPPED Conduit IDs)."""
        model_ids = feedback_logger._get_available_model_ids()
        assert set(model_ids) == {"o4-mini", "claude-3-haiku", "gemini-2.5-pro"}

    @pytest.mark.asyncio
    async def test_no_bandit_available(self, litellm_kwargs, litellm_response):
        """Test graceful handling when no bandit is available."""
        router = Mock()
        router.analyzer = Mock()
        router.analyzer.analyze = AsyncMock(
            return_value=QueryFeatures(
                embedding=[0.1] * 384, token_count=50, complexity_score=0.5
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
        self, mock_router, litellm_kwargs, litellm_response, litellm_to_conduit_map
    ):
        """Test that feedback actually updates bandit internal state."""
        # Create logger with mappings
        logger = ConduitFeedbackLogger(
            mock_router,
            litellm_to_conduit_map=litellm_to_conduit_map,
        )

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
        self, mock_router, litellm_response, litellm_to_conduit_map
    ):
        """Test feedback works with different model IDs."""
        logger = ConduitFeedbackLogger(
            mock_router,
            litellm_to_conduit_map=litellm_to_conduit_map,
        )

        # LiteLLM model IDs used in requests
        litellm_models = ["gpt-4o-mini", "claude-3-haiku", "gemini-pro"]
        # Expected Conduit model IDs after mapping
        expected_mapped = ["o4-mini", "claude-3-haiku", "gemini-2.5-pro"]

        for model in litellm_models:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": f"Test for {model}"}],
            }

            await logger.async_log_success_event(
                kwargs, litellm_response, 1000.0, 1001.0
            )

        # Verify 3 updates
        assert mock_router.bandit.update.call_count == 3

        # Verify model IDs are MAPPED from LiteLLM to Conduit format
        model_ids = [
            call[0][0].model_id for call in mock_router.bandit.update.call_args_list
        ]
        assert model_ids == expected_mapped


class TestQualityEstimation:
    """Tests for quality estimation methods."""

    def test_estimate_quality_normal_response(self, feedback_logger):
        """Test quality estimation for normal response."""
        query = "What is the capital of France?"
        response = "The capital of France is Paris."

        quality = feedback_logger._estimate_quality(query, response)

        # Should get reasonable quality score
        assert 0.7 <= quality <= 0.95

    def test_estimate_quality_empty_response(self, feedback_logger):
        """Test quality estimation for empty response."""
        quality = feedback_logger._estimate_quality("test query", "")
        assert quality == 0.1  # empty_quality from config

    def test_estimate_quality_whitespace_only(self, feedback_logger):
        """Test quality estimation for whitespace-only response."""
        quality = feedback_logger._estimate_quality("test query", "   \n\t  ")
        assert quality == 0.1  # empty_quality from config

    def test_estimate_quality_short_response(self, feedback_logger):
        """Test quality estimation penalizes very short responses."""
        query = "Explain quantum mechanics in detail"
        response = "QM"  # Very short

        quality = feedback_logger._estimate_quality(query, response)

        # Should have penalty for short response
        assert quality < 0.8

    def test_estimate_quality_repetitive_response(self, feedback_logger):
        """Test quality estimation penalizes repetitive content."""
        query = "Write a story"
        response = "hello hello hello hello hello hello hello hello hello hello"

        quality = feedback_logger._estimate_quality(query, response)

        # Should have penalty for repetition
        assert quality <= 0.7

    def test_estimate_quality_no_keyword_overlap(self, feedback_logger):
        """Test quality estimation penalizes irrelevant responses."""
        query = "What is Python programming?"
        response = "The weather is nice today in California."

        quality = feedback_logger._estimate_quality(query, response)

        # Should have penalty for no keyword overlap
        assert quality < 0.8

    def test_has_repetition_detects_loops(self, feedback_logger):
        """Test repetition detection catches stuck models."""
        text = "hello world " * 10

        assert feedback_logger._has_repetition(text) is True

    def test_has_repetition_short_text(self, feedback_logger):
        """Test repetition detection handles short text."""
        text = "hi"

        assert feedback_logger._has_repetition(text) is False

    def test_has_repetition_normal_text(self, feedback_logger):
        """Test repetition detection passes normal text."""
        text = "This is a normal response with varied content and no unusual patterns."

        assert feedback_logger._has_repetition(text) is False

    def test_keyword_overlap_full_overlap(self, feedback_logger):
        """Test keyword overlap with matching keywords."""
        query = "python programming language"
        response = "Python is a programming language used for many applications."

        overlap = feedback_logger._keyword_overlap(query, response)

        assert overlap > 0.5  # Should have good overlap

    def test_keyword_overlap_no_overlap(self, feedback_logger):
        """Test keyword overlap with no matching keywords."""
        query = "quantum physics"
        response = "The weather is sunny today."

        overlap = feedback_logger._keyword_overlap(query, response)

        assert overlap == 0.0

    def test_keyword_overlap_empty_query(self, feedback_logger):
        """Test keyword overlap handles empty query."""
        overlap = feedback_logger._keyword_overlap("", "Some response text")

        assert overlap == 0.0

    def test_keyword_overlap_stopwords_ignored(self, feedback_logger):
        """Test keyword overlap ignores stopwords."""
        query = "the a an is are"  # All stopwords
        response = "the a an is are"

        overlap = feedback_logger._keyword_overlap(query, response)

        assert overlap == 0.0  # Stopwords filtered out


class TestResponseTextExtraction:
    """Tests for response text extraction."""

    def test_extract_response_text_chat_completion(self, feedback_logger):
        """Test extracting text from chat completion format."""
        response = Mock()
        message = Mock()
        message.content = "This is the response"
        choice = Mock()
        choice.message = message
        response.choices = [choice]

        text = feedback_logger._extract_response_text(response)

        assert text == "This is the response"

    def test_extract_response_text_none_content(self, feedback_logger):
        """Test extracting text when content is None."""
        response = Mock()
        message = Mock()
        message.content = None
        choice = Mock()
        choice.message = message
        response.choices = [choice]

        text = feedback_logger._extract_response_text(response)

        assert text == ""

    def test_extract_response_text_text_attribute(self, feedback_logger):
        """Test extracting text from text attribute."""
        response = Mock(spec=["text"])
        response.text = "Direct text response"

        text = feedback_logger._extract_response_text(response)

        assert text == "Direct text response"

    def test_extract_response_text_content_attribute(self, feedback_logger):
        """Test extracting text from content attribute."""
        response = Mock(spec=["content"])
        response.content = "Content response"

        text = feedback_logger._extract_response_text(response)

        assert text == "Content response"

    def test_extract_response_text_empty_choices(self, feedback_logger):
        """Test extracting text with empty choices list."""
        response = Mock(spec=["choices"])
        response.choices = []

        text = feedback_logger._extract_response_text(response)

        assert text == ""

    def test_extract_response_text_no_attributes(self, feedback_logger):
        """Test extracting text when no known attributes exist."""
        response = Mock(spec=[])  # No relevant attributes

        text = feedback_logger._extract_response_text(response)

        assert text == ""


class TestLatencyHandling:
    """Tests for latency calculation edge cases."""

    @pytest.mark.asyncio
    async def test_latency_with_datetime_objects(
        self, feedback_logger, mock_router, litellm_response
    ):
        """Test latency calculation handles datetime objects."""
        from datetime import datetime, timedelta

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
        }

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2.5)

        await feedback_logger.async_log_success_event(
            kwargs, litellm_response, start_time, end_time
        )

        # Verify latency was calculated correctly
        call_args = mock_router.bandit.update.call_args
        feedback = call_args[0][0]
        assert 2.4 <= feedback.latency <= 2.6

    @pytest.mark.asyncio
    async def test_latency_with_float_timestamps(
        self, feedback_logger, mock_router, litellm_response
    ):
        """Test latency calculation handles float timestamps."""
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "test"}],
        }

        await feedback_logger.async_log_success_event(
            kwargs, litellm_response, 1000.0, 1003.5
        )

        call_args = mock_router.bandit.update.call_args
        feedback = call_args[0][0]
        assert feedback.latency == 3.5
