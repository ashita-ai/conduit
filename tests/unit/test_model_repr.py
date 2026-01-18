"""Tests for __repr__ methods on core Pydantic models.

Verifies that all core models have concise, useful repr output
for debugging purposes (Issue #212).
"""

import pytest

from conduit.core.models import Query, QueryFeatures, Response, RoutingDecision
from conduit.engines.bandits.base import BanditFeedback, ModelArm
from conduit.feedback.models import FeedbackEvent, PendingQuery


class TestQueryRepr:
    """Tests for Query.__repr__."""

    def test_short_text(self) -> None:
        """Query with short text shows full text."""
        query = Query(text="Hello world")
        repr_str = repr(query)
        assert "Query(" in repr_str
        assert "Hello world" in repr_str
        assert len(repr_str) < 80

    def test_long_text_truncated(self) -> None:
        """Query with long text is truncated with ellipsis."""
        long_text = "This is a very long query text that exceeds thirty characters"
        query = Query(text=long_text)
        repr_str = repr(query)
        assert "..." in repr_str
        assert len(repr_str) < 80

    def test_includes_id_prefix(self) -> None:
        """Query repr includes first 8 chars of ID."""
        query = Query(text="Test query")
        repr_str = repr(query)
        assert query.id[:8] in repr_str


class TestQueryFeaturesRepr:
    """Tests for QueryFeatures.__repr__."""

    def test_shows_dimensions_and_tokens(self) -> None:
        """QueryFeatures shows embedding dimensions and token count."""
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
        )
        repr_str = repr(features)
        assert "QueryFeatures(" in repr_str
        assert "dims=384" in repr_str
        assert "tokens=50" in repr_str
        assert len(repr_str) < 80


class TestRoutingDecisionRepr:
    """Tests for RoutingDecision.__repr__."""

    def test_shows_model_and_confidence(self) -> None:
        """RoutingDecision shows selected model and confidence."""
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.5,
        )
        decision = RoutingDecision(
            query_id="test-query",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=features,
            reasoning="High quality expected",
        )
        repr_str = repr(decision)
        assert "RoutingDecision(" in repr_str
        assert "gpt-4o-mini" in repr_str
        assert "conf=0.85" in repr_str
        assert len(repr_str) < 80


class TestResponseRepr:
    """Tests for Response.__repr__."""

    def test_shows_model_cost_latency(self) -> None:
        """Response shows model, cost, and latency."""
        response = Response(
            query_id="test-query",
            model="gpt-4o-mini",
            text="Hello!",
            cost=0.0001,
            latency=1.234,
            tokens=100,
        )
        repr_str = repr(response)
        assert "Response(" in repr_str
        assert "gpt-4o-mini" in repr_str
        assert "$0.0001" in repr_str
        assert "1.23s" in repr_str
        assert len(repr_str) < 80


class TestModelArmRepr:
    """Tests for ModelArm.__repr__."""

    def test_shows_model_id_and_provider(self) -> None:
        """ModelArm shows model_id and provider."""
        arm = ModelArm(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
        )
        repr_str = repr(arm)
        assert "ModelArm(" in repr_str
        assert "gpt-4o-mini" in repr_str
        assert "provider='openai'" in repr_str
        assert len(repr_str) < 80


class TestBanditFeedbackRepr:
    """Tests for BanditFeedback.__repr__."""

    def test_shows_model_id_and_quality(self) -> None:
        """BanditFeedback shows model_id and quality score."""
        feedback = BanditFeedback(
            model_id="gpt-4o-mini",
            cost=0.0001,
            quality_score=0.95,
            latency=1.2,
        )
        repr_str = repr(feedback)
        assert "BanditFeedback(" in repr_str
        assert "gpt-4o-mini" in repr_str
        assert "q=0.95" in repr_str
        assert len(repr_str) < 80


class TestFeedbackEventRepr:
    """Tests for FeedbackEvent.__repr__."""

    def test_shows_query_id_and_signal_type(self) -> None:
        """FeedbackEvent shows query_id prefix and signal_type."""
        event = FeedbackEvent(
            query_id="12345678-abcd-efgh-ijkl-mnopqrstuvwx",
            signal_type="thumbs",
            payload={"value": "up"},
        )
        repr_str = repr(event)
        assert "FeedbackEvent(" in repr_str
        assert "12345678" in repr_str
        assert "thumbs" in repr_str
        assert len(repr_str) < 80


class TestPendingQueryRepr:
    """Tests for PendingQuery.__repr__."""

    def test_shows_query_id_and_model(self) -> None:
        """PendingQuery shows query_id prefix and model_id."""
        pending = PendingQuery(
            query_id="12345678-abcd-efgh-ijkl-mnopqrstuvwx",
            model_id="gpt-4o-mini",
            features={"embedding": [0.1] * 384, "token_count": 50},
        )
        repr_str = repr(pending)
        assert "PendingQuery(" in repr_str
        assert "12345678" in repr_str
        assert "gpt-4o-mini" in repr_str
        assert len(repr_str) < 80
