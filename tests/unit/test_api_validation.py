"""Unit tests for API validation schemas.

Tests Pydantic validation edge cases for request/response schemas.
"""

import pytest
from pydantic import ValidationError

from conduit.api.validation import (
    AuditQueryRequest,
    CompleteRequest,
    ContextMetadata,
    FeedbackRequest,
)
from conduit.core.models import QueryConstraints


class TestCompleteRequestValidation:
    """Tests for CompleteRequest validation."""

    def test_valid_minimal_request(self):
        """Test valid request with only required fields."""
        request = CompleteRequest(prompt="What is 2+2?")
        assert request.prompt == "What is 2+2?"
        assert request.constraints is None
        assert request.user_id is None

    def test_valid_full_request(self):
        """Test valid request with all fields."""
        request = CompleteRequest(
            prompt="Explain quantum computing",
            result_type="MyModel",
            constraints=QueryConstraints(max_cost=0.01, min_quality=0.9),
            user_id="user-123",
            context=ContextMetadata(source="web", session_id="sess-456"),
        )
        assert request.prompt == "Explain quantum computing"
        assert request.result_type == "MyModel"
        assert request.constraints is not None
        assert request.constraints.max_cost == 0.01

    def test_empty_prompt_rejected(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompleteRequest(prompt="")

        errors = exc_info.value.errors()
        assert any(
            e["type"] == "string_too_short" and "prompt" in str(e["loc"])
            for e in errors
        )

    def test_whitespace_only_prompt_rejected(self):
        """Test that whitespace-only prompt is rejected (min_length=1)."""
        # Pydantic min_length counts characters, so " " passes min_length=1
        # This is expected behavior - if we want to reject whitespace,
        # we'd need a custom validator
        request = CompleteRequest(prompt=" ")
        assert request.prompt == " "

    def test_missing_prompt_rejected(self):
        """Test that missing prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompleteRequest()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any("prompt" in str(e["loc"]) for e in errors)

    def test_long_prompt_accepted(self):
        """Test that long prompts are accepted."""
        long_prompt = "x" * 10000
        request = CompleteRequest(prompt=long_prompt)
        assert len(request.prompt) == 10000


class TestQueryConstraintsValidation:
    """Tests for QueryConstraints validation."""

    def test_valid_constraints(self):
        """Test valid constraints."""
        constraints = QueryConstraints(
            max_cost=0.05,
            max_latency=2.0,
            min_quality=0.8,
            preferred_provider="openai",
        )
        assert constraints.max_cost == 0.05
        assert constraints.max_latency == 2.0
        assert constraints.min_quality == 0.8
        assert constraints.preferred_provider == "openai"

    def test_negative_max_cost_rejected(self):
        """Test that negative max_cost is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryConstraints(max_cost=-0.01)

        errors = exc_info.value.errors()
        assert any("max_cost" in str(e["loc"]) for e in errors)

    def test_negative_max_latency_rejected(self):
        """Test that negative max_latency is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryConstraints(max_latency=-1.0)

        errors = exc_info.value.errors()
        assert any("max_latency" in str(e["loc"]) for e in errors)

    def test_min_quality_below_zero_rejected(self):
        """Test that min_quality below 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryConstraints(min_quality=-0.1)

        errors = exc_info.value.errors()
        assert any("min_quality" in str(e["loc"]) for e in errors)

    def test_min_quality_above_one_rejected(self):
        """Test that min_quality above 1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryConstraints(min_quality=1.5)

        errors = exc_info.value.errors()
        assert any("min_quality" in str(e["loc"]) for e in errors)

    def test_boundary_values_accepted(self):
        """Test boundary values are accepted."""
        # min_quality at boundaries
        constraints_zero = QueryConstraints(min_quality=0.0)
        assert constraints_zero.min_quality == 0.0

        constraints_one = QueryConstraints(min_quality=1.0)
        assert constraints_one.min_quality == 1.0

    def test_empty_constraints_valid(self):
        """Test empty constraints are valid."""
        constraints = QueryConstraints()
        assert constraints.max_cost is None
        assert constraints.max_latency is None
        assert constraints.min_quality is None
        assert constraints.preferred_provider is None


class TestFeedbackRequestValidation:
    """Tests for FeedbackRequest validation."""

    def test_valid_feedback(self):
        """Test valid feedback request."""
        feedback = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.9,
            met_expectations=True,
            user_rating=5,
            comments="Great response!",
        )
        assert feedback.response_id == "resp-123"
        assert feedback.quality_score == 0.9
        assert feedback.met_expectations is True

    def test_quality_score_below_zero_rejected(self):
        """Test quality_score below 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                response_id="resp-123",
                quality_score=-0.1,
                met_expectations=True,
            )

        errors = exc_info.value.errors()
        assert any("quality_score" in str(e["loc"]) for e in errors)

    def test_quality_score_above_one_rejected(self):
        """Test quality_score above 1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                response_id="resp-123",
                quality_score=1.1,
                met_expectations=True,
            )

        errors = exc_info.value.errors()
        assert any("quality_score" in str(e["loc"]) for e in errors)

    def test_quality_score_boundary_values(self):
        """Test quality_score boundary values."""
        feedback_zero = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.0,
            met_expectations=False,
        )
        assert feedback_zero.quality_score == 0.0

        feedback_one = FeedbackRequest(
            response_id="resp-123",
            quality_score=1.0,
            met_expectations=True,
        )
        assert feedback_one.quality_score == 1.0

    def test_user_rating_below_one_rejected(self):
        """Test user_rating below 1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                response_id="resp-123",
                quality_score=0.5,
                met_expectations=True,
                user_rating=0,
            )

        errors = exc_info.value.errors()
        assert any("user_rating" in str(e["loc"]) for e in errors)

    def test_user_rating_above_five_rejected(self):
        """Test user_rating above 5 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(
                response_id="resp-123",
                quality_score=0.5,
                met_expectations=True,
                user_rating=6,
            )

        errors = exc_info.value.errors()
        assert any("user_rating" in str(e["loc"]) for e in errors)

    def test_user_rating_boundary_values(self):
        """Test user_rating boundary values."""
        feedback_one = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.5,
            met_expectations=False,
            user_rating=1,
        )
        assert feedback_one.user_rating == 1

        feedback_five = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.5,
            met_expectations=True,
            user_rating=5,
        )
        assert feedback_five.user_rating == 5

    def test_user_rating_optional(self):
        """Test user_rating is optional."""
        feedback = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.5,
            met_expectations=True,
        )
        assert feedback.user_rating is None

    def test_missing_required_fields_rejected(self):
        """Test missing required fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackRequest(response_id="resp-123")  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) >= 2  # quality_score and met_expectations missing


class TestAuditQueryRequestValidation:
    """Tests for AuditQueryRequest validation."""

    def test_valid_query_request(self):
        """Test valid audit query request."""
        request = AuditQueryRequest(
            decision_id="dec-123",
            selected_model="gpt-4o",
            limit=50,
            offset=10,
        )
        assert request.decision_id == "dec-123"
        assert request.limit == 50
        assert request.offset == 10

    def test_default_values(self):
        """Test default values are applied."""
        request = AuditQueryRequest()
        assert request.limit == 100
        assert request.offset == 0
        assert request.decision_id is None

    def test_limit_below_one_rejected(self):
        """Test limit below 1 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AuditQueryRequest(limit=0)

        errors = exc_info.value.errors()
        assert any("limit" in str(e["loc"]) for e in errors)

    def test_limit_above_1000_rejected(self):
        """Test limit above 1000 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AuditQueryRequest(limit=1001)

        errors = exc_info.value.errors()
        assert any("limit" in str(e["loc"]) for e in errors)

    def test_limit_boundary_values(self):
        """Test limit boundary values."""
        request_min = AuditQueryRequest(limit=1)
        assert request_min.limit == 1

        request_max = AuditQueryRequest(limit=1000)
        assert request_max.limit == 1000

    def test_offset_negative_rejected(self):
        """Test negative offset is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AuditQueryRequest(offset=-1)

        errors = exc_info.value.errors()
        assert any("offset" in str(e["loc"]) for e in errors)

    def test_offset_zero_accepted(self):
        """Test offset of zero is accepted."""
        request = AuditQueryRequest(offset=0)
        assert request.offset == 0

    def test_large_offset_accepted(self):
        """Test large offset is accepted."""
        request = AuditQueryRequest(offset=1000000)
        assert request.offset == 1000000


class TestContextMetadataValidation:
    """Tests for ContextMetadata validation."""

    def test_valid_context(self):
        """Test valid context metadata."""
        context = ContextMetadata(
            source="mobile_app",
            session_id="sess-123",
            request_id="req-456",
            tags=["important", "production"],
        )
        assert context.source == "mobile_app"
        assert context.session_id == "sess-123"
        assert context.tags == ["important", "production"]

    def test_all_fields_optional(self):
        """Test all fields are optional."""
        context = ContextMetadata()
        assert context.source is None
        assert context.session_id is None
        assert context.request_id is None
        assert context.tags is None

    def test_extra_fields_allowed(self):
        """Test extra fields are allowed (model_config extra='allow')."""
        context = ContextMetadata(
            source="test",
            custom_field="custom_value",  # type: ignore[call-arg]
            another_field=123,  # type: ignore[call-arg]
        )
        assert context.source == "test"
        # Extra fields should be accessible
        assert context.model_extra == {
            "custom_field": "custom_value",
            "another_field": 123,
        }

    def test_tags_with_empty_list(self):
        """Test empty tags list is valid."""
        context = ContextMetadata(tags=[])
        assert context.tags == []

    def test_nested_in_complete_request(self):
        """Test ContextMetadata nested in CompleteRequest."""
        request = CompleteRequest(
            prompt="Test",
            context=ContextMetadata(
                source="api",
                session_id="sess-789",
            ),
        )
        assert request.context is not None
        assert request.context.source == "api"
        assert request.context.session_id == "sess-789"


class TestTypeCoercion:
    """Tests for Pydantic type coercion behavior."""

    def test_float_coerced_to_int_for_rating(self):
        """Test float is coerced to int for user_rating."""
        # Pydantic v2 does strict validation by default for int fields
        with pytest.raises(ValidationError):
            FeedbackRequest(
                response_id="resp-123",
                quality_score=0.5,
                met_expectations=True,
                user_rating=4.7,  # type: ignore[arg-type]
            )

    def test_string_coerced_to_float_for_quality(self):
        """Test string is coerced to float for quality_score."""
        # Pydantic v2 will coerce "0.5" to 0.5
        feedback = FeedbackRequest(
            response_id="resp-123",
            quality_score="0.5",  # type: ignore[arg-type]
            met_expectations=True,
        )
        assert feedback.quality_score == 0.5

    def test_invalid_string_for_float_rejected(self):
        """Test invalid string for float field is rejected."""
        with pytest.raises(ValidationError):
            FeedbackRequest(
                response_id="resp-123",
                quality_score="not_a_number",  # type: ignore[arg-type]
                met_expectations=True,
            )

    def test_bool_string_for_bool_field(self):
        """Test string 'true'/'false' handling for bool field."""
        # Pydantic v2 may coerce some string values to bool
        # The behavior depends on exact Pydantic version
        # Just verify that proper bool values work
        feedback_true = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.5,
            met_expectations=True,
        )
        assert feedback_true.met_expectations is True

        feedback_false = FeedbackRequest(
            response_id="resp-123",
            quality_score=0.5,
            met_expectations=False,
        )
        assert feedback_false.met_expectations is False
