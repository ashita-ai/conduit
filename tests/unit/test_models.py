"""Unit tests for core data models."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from conduit.core.models import (
    MAX_QUERY_TEXT_BYTES,
    Feedback,
    ModelState,
    Query,
    QueryConstraints,
    QueryFeatures,
    Response,
    RoutingDecision,
    RoutingResult,
)


class TestQueryConstraints:
    """Tests for QueryConstraints model."""

    def test_valid_constraints(self):
        """Test valid constraint creation."""
        constraints = QueryConstraints(
            max_cost=0.01, max_latency=2.0, min_quality=0.7, preferred_provider="openai"
        )
        assert constraints.max_cost == 0.01
        assert constraints.max_latency == 2.0
        assert constraints.min_quality == 0.7
        assert constraints.preferred_provider == "openai"

    def test_optional_constraints(self):
        """Test constraints with None values."""
        constraints = QueryConstraints()
        assert constraints.max_cost is None
        assert constraints.max_latency is None
        assert constraints.min_quality is None
        assert constraints.preferred_provider is None

    def test_invalid_quality_range(self):
        """Test min_quality must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            QueryConstraints(min_quality=1.5)

        with pytest.raises(ValidationError):
            QueryConstraints(min_quality=-0.1)


class TestQuery:
    """Tests for Query model."""

    def test_valid_query(self):
        """Test valid query creation."""
        query = Query(text="What is the capital of France?", user_id="user123")
        assert query.text == "What is the capital of France?"
        assert query.user_id == "user123"
        assert query.id is not None
        assert isinstance(query.created_at, datetime)

    def test_empty_text_validation(self):
        """Test empty text is rejected."""
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            Query(text="")

    def test_whitespace_stripped(self):
        """Test whitespace is stripped from text."""
        query = Query(text="  What is AI?  ")
        assert query.text == "What is AI?"

    def test_query_with_constraints(self):
        """Test query with constraints."""
        constraints = QueryConstraints(max_cost=0.01)
        query = Query(text="Test query", constraints=constraints)
        assert query.constraints is not None
        assert query.constraints.max_cost == 0.01

    def test_text_size_limit_exceeded(self):
        """Test query text exceeding 100KB limit is rejected."""
        # Create text just over the limit (100KB + 1 byte)
        large_text = "a" * (MAX_QUERY_TEXT_BYTES + 1)
        with pytest.raises(ValidationError, match="exceeds maximum size"):
            Query(text=large_text)

    def test_text_at_size_limit_accepted(self):
        """Test query text exactly at 100KB limit is accepted."""
        # Create text exactly at the limit
        max_text = "a" * MAX_QUERY_TEXT_BYTES
        query = Query(text=max_text)
        assert len(query.text.encode("utf-8")) == MAX_QUERY_TEXT_BYTES

    def test_null_bytes_removed(self):
        """Test null bytes are stripped from query text."""
        query = Query(text="Hello\x00World")
        assert "\x00" not in query.text
        assert query.text == "HelloWorld"

    def test_control_characters_removed(self):
        """Test C0 control characters (except tab/newline) are removed."""
        # \x01 (SOH), \x02 (STX), \x1f (US) should be removed
        # \x09 (tab), \x0a (LF), \x0d (CR) should be kept
        query = Query(text="Hello\x01\x02World\x09Tab\x0aNewline\x0dCR\x1fEnd")
        assert "\x01" not in query.text
        assert "\x02" not in query.text
        assert "\x1f" not in query.text
        assert "\x09" in query.text  # Tab preserved
        assert "\x0a" in query.text  # Newline preserved
        assert "\x0d" in query.text  # CR preserved
        assert query.text == "HelloWorld\tTab\nNewline\rCREnd"

    def test_whitespace_only_rejected(self):
        """Test whitespace-only text is rejected after stripping."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            Query(text="   \t\n   ")

    def test_unicode_text_accepted(self):
        """Test valid Unicode text is accepted."""
        query = Query(text="Hello 世界 🌍 مرحبا")
        assert query.text == "Hello 世界 🌍 مرحبا"

    def test_multibyte_unicode_size_limit(self):
        """Test size limit is enforced in bytes, not characters."""
        # Each emoji is 4 bytes in UTF-8
        # 25,001 emojis = 100,004 bytes > 100KB limit
        emoji_count = (MAX_QUERY_TEXT_BYTES // 4) + 1
        large_emoji_text = "🌍" * emoji_count
        with pytest.raises(ValidationError, match="exceeds maximum size"):
            Query(text=large_emoji_text)


class TestQueryFeatures:
    """Tests for QueryFeatures model."""

    def test_valid_features(self):
        """Test valid feature creation."""
        embedding = [0.1] * 384
        features = QueryFeatures(
            embedding=embedding, token_count=10, complexity_score=0.5
        )
        assert len(features.embedding) == 384
        assert features.token_count == 10
        assert features.complexity_score == 0.5

    def test_variable_embedding_length(self):
        """Test embedding accepts variable dimensions for different models."""
        # Should work with different embedding dimensions
        features_small = QueryFeatures(
            embedding=[0.1] * 100,  # Small embedding (e.g., distilbert)
            token_count=10,
            complexity_score=0.5,
        )
        assert len(features_small.embedding) == 100

        features_large = QueryFeatures(
            embedding=[0.1] * 768,  # Large embedding (e.g., BERT)
            token_count=10,
            complexity_score=0.5,
        )
        assert len(features_large.embedding) == 768

    def test_complexity_score_range(self):
        """Test complexity_score must be 0.0-1.0."""
        embedding = [0.1] * 384
        with pytest.raises(ValidationError):
            QueryFeatures(
                embedding=embedding,
                token_count=10,
                complexity_score=1.5,  # Out of range
            )


class TestRoutingDecision:
    """Tests for RoutingDecision model."""

    def test_valid_decision(self):
        """Test valid routing decision creation."""
        embedding = [0.1] * 384
        features = QueryFeatures(
            embedding=embedding, token_count=10, complexity_score=0.5
        )
        decision = RoutingDecision(
            query_id="query123",
            selected_model="gpt-4o-mini",
            confidence=0.87,
            features=features,
            reasoning="Simple query, low cost model selected",
        )
        assert decision.query_id == "query123"
        assert decision.selected_model == "gpt-4o-mini"
        assert decision.confidence == 0.87
        assert decision.features == features


class TestResponse:
    """Tests for Response model."""

    def test_valid_response(self):
        """Test valid response creation."""
        response = Response(
            query_id="query123",
            model="gpt-4o-mini",
            text='{"answer": "Paris"}',
            cost=0.0002,
            latency=1.2,
            tokens=15,
        )
        assert response.query_id == "query123"
        assert response.model == "gpt-4o-mini"
        assert response.cost == 0.0002
        assert response.latency == 1.2
        assert response.tokens == 15

    def test_negative_cost_rejected(self):
        """Test negative cost is rejected."""
        with pytest.raises(ValidationError):
            Response(
                query_id="query123",
                model="gpt-4o-mini",
                text="result",
                cost=-0.01,  # Invalid
                latency=1.0,
                tokens=10,
            )


class TestFeedback:
    """Tests for Feedback model."""

    def test_valid_feedback(self):
        """Test valid feedback creation."""
        feedback = Feedback(
            response_id="resp123",
            quality_score=0.95,
            user_rating=5,
            met_expectations=True,
            comments="Perfect answer",
        )
        assert feedback.response_id == "resp123"
        assert feedback.quality_score == 0.95
        assert feedback.user_rating == 5
        assert feedback.met_expectations is True
        assert feedback.comments == "Perfect answer"

    def test_optional_rating_and_comments(self):
        """Test feedback without rating and comments."""
        feedback = Feedback(
            response_id="resp123", quality_score=0.8, met_expectations=True
        )
        assert feedback.user_rating is None
        assert feedback.comments is None

    def test_rating_range_validation(self):
        """Test user_rating must be 1-5."""
        with pytest.raises(ValidationError):
            Feedback(
                response_id="resp123",
                quality_score=0.8,
                user_rating=6,  # Out of range
                met_expectations=True,
            )


class TestModelState:
    """Tests for ModelState model."""

    def test_valid_model_state(self):
        """Test valid model state creation."""
        state = ModelState(
            model_id="gpt-4o-mini",
            alpha=10.5,
            beta=5.2,
            total_requests=100,
            total_cost=1.25,
            avg_quality=0.87,
        )
        assert state.model_id == "gpt-4o-mini"
        assert state.alpha == 10.5
        assert state.beta == 5.2
        assert state.total_requests == 100
        assert state.total_cost == 1.25
        assert state.avg_quality == 0.87

    def test_default_values(self):
        """Test default values for new model."""
        state = ModelState(model_id="new-model")
        assert state.alpha == 1.0
        assert state.beta == 1.0
        assert state.total_requests == 0
        assert state.total_cost == 0.0
        assert state.avg_quality == 0.0

    def test_mean_success_rate_calculation(self):
        """Test mean success rate property."""
        state = ModelState(model_id="test", alpha=10.0, beta=5.0)
        expected_mean = 10.0 / (10.0 + 5.0)
        assert state.mean_success_rate == pytest.approx(expected_mean)

    def test_variance_calculation(self):
        """Test variance property."""
        state = ModelState(model_id="test", alpha=10.0, beta=5.0)
        ab = 15.0
        expected_variance = (10.0 * 5.0) / (ab * ab * (ab + 1))
        assert state.variance == pytest.approx(expected_variance)


class TestRoutingResult:
    """Tests for RoutingResult model."""

    def test_from_response_and_routing(self):
        """Test creating RoutingResult from Response and RoutingDecision."""
        # Create response
        response = Response(
            id="resp123",
            query_id="query123",
            model="gpt-4o-mini",
            text='{"answer": "Paris", "confidence": 0.99}',
            cost=0.0002,
            latency=1.2,
            tokens=15,
        )

        # Create routing decision
        embedding = [0.1] * 384
        features = QueryFeatures(
            embedding=embedding, token_count=10, complexity_score=0.3
        )
        routing = RoutingDecision(
            query_id="query123",
            selected_model="gpt-4o-mini",
            confidence=0.87,
            features=features,
            reasoning="Simple query",
        )

        # Create result
        result = RoutingResult.from_response(response, routing)

        assert result.id == "resp123"
        assert result.query_id == "query123"
        assert result.model == "gpt-4o-mini"
        assert result.data == {"answer": "Paris", "confidence": 0.99}
        assert result.metadata["cost"] == 0.0002
        assert result.metadata["latency"] == 1.2
        assert result.metadata["tokens"] == 15
        assert result.metadata["routing_confidence"] == 0.87
        assert result.metadata["reasoning"] == "Simple query"
