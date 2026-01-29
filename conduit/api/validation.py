"""Request and response validation schemas for Conduit API."""

from typing import Any

from pydantic import BaseModel, Field

from conduit.core.models import QueryConstraints


class ContextMetadata(BaseModel):
    """Optional context metadata for routing requests.

    Provides structured fields for common context information while
    allowing arbitrary extra fields for application-specific data.
    All fields are optional to maximize flexibility.
    """

    source: str | None = Field(
        None,
        description="Request source identifier",
        json_schema_extra={"example": "web_app"},
    )
    session_id: str | None = Field(
        None,
        description="Session identifier",
        json_schema_extra={"example": "session_123"},
    )
    request_id: str | None = Field(
        None,
        description="External request ID",
        json_schema_extra={"example": "req_456"},
    )
    tags: list[str] | None = Field(
        None,
        description="Optional tags for categorization",
        json_schema_extra={"example": ["beta-user", "experimentation"]},
    )

    model_config = {"extra": "allow"}


class CompleteRequest(BaseModel):
    """Request schema for POST /v1/complete."""

    prompt: str = Field(
        ...,
        description="User query/prompt",
        min_length=1,
        json_schema_extra={"example": "What is quantum computing?"},
    )
    result_type: str | None = Field(
        None,
        description="Pydantic model name for structured output (optional)",
        json_schema_extra={"example": "AnswerSchema"},
    )
    constraints: QueryConstraints | None = Field(
        None,
        description="Optional routing constraints (max_cost, max_latency, min_quality, preferred_provider)",
        json_schema_extra={
            "example": {
                "max_cost": 0.01,
                "min_quality": 0.8,
                "preferred_provider": "gpt-4o",
            },
        },
    )
    user_id: str | None = Field(
        None,
        description="User identifier for tracking",
        json_schema_extra={"example": "user_12334"},
    )
    context: ContextMetadata | None = Field(
        None,
        description="Additional context metadata",
        json_schema_extra={
            "example": {
                "source": "web_app",
                "session_id": "session_123",
                "tags": ["beta-user"],
            }
        },
    )


class CompleteResponse(BaseModel):
    """Response schema for POST /v1/complete."""

    id: str = Field(
        ...,
        description="Response ID",
        json_schema_extra={"example": "resp_789"},
    )
    query_id: str = Field(..., description="Query ID")
    model: str = Field(..., description="Model used for completion")
    data: dict[str, Any] = Field(..., description="Structured response data")
    metadata: dict[str, Any] = Field(
        ...,
        description="Routing metadata (cost, latency, tokens, confidence, reasoning)",
    )


class FeedbackRequest(BaseModel):
    """Request schema for POST /v1/feedback."""

    response_id: str = Field(
        ...,
        description="Response ID to provide feedback for",
        json_schema_extra={"example": "response_456"},
    )
    quality_score: float = Field(
        ...,
        description="Quality score (0.0-1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92},
    )
    user_rating: int | None = Field(
        None,
        description="User rating (1-5 stars)",
        ge=1,
        le=5,
        json_schema_extra={"example": 5},
    )
    met_expectations: bool = Field(
        ...,
        description="Whether response met user expectations",
        json_schema_extra={"example": True},
    )
    comments: str | None = Field(
        None,
        description="Optional user comments",
        json_schema_extra={"example": "Clear and insightful feedback."},
    )


class FeedbackResponse(BaseModel):
    """Response schema for POST /v1/feedback."""

    id: str = Field(
        ...,
        description="Feedback ID",
        json_schema_extra={"example": "feedback_568"},
    )
    response_id: str = Field(..., description="Response ID")
    message: str = Field(..., description="Confirmation message")


class StatsResponse(BaseModel):
    """Response schema for GET /v1/stats."""

    total_queries: int = Field(..., description="Total queries processed")
    total_cost: float = Field(..., description="Total cost in dollars")
    avg_latency: float = Field(..., description="Average latency in seconds")
    model_distribution: dict[str, int] = Field(..., description="Query count per model")
    quality_metrics: dict[str, float] = Field(
        ..., description="Quality metrics (avg_quality, success_rate)"
    )


class ModelsResponse(BaseModel):
    """Response schema for GET /v1/models."""

    models: list[str] = Field(..., description="Available model IDs")
    default_model: str = Field(..., description="Default fallback model")


class HealthResponse(BaseModel):
    """Response schema for health endpoints."""

    status: str = Field(..., description="Health status (healthy, unhealthy)")
    timestamp: str = Field(..., description="ISO timestamp")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    code: str | None = Field(None, description="Error code")
    context: dict[str, Any] | None = Field(None, description="Additional error context")


# Audit API schemas


class AuditQueryRequest(BaseModel):
    """Request schema for GET /v1/audit query parameters."""

    decision_id: str | None = Field(None, description="Filter by decision ID")
    query_id: str | None = Field(None, description="Filter by query ID")
    selected_model: str | None = Field(None, description="Filter by selected model")
    algorithm_phase: str | None = Field(None, description="Filter by algorithm phase")
    start_time: str | None = Field(
        None, description="Filter entries after this ISO timestamp"
    )
    end_time: str | None = Field(
        None, description="Filter entries before this ISO timestamp"
    )
    limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum entries to return"
    )
    offset: int = Field(default=0, ge=0, description="Number of entries to skip")


class AuditEntryResponse(BaseModel):
    """Response schema for a single audit entry."""

    id: int = Field(..., description="Audit entry ID")
    decision_id: str = Field(..., description="Routing decision ID")
    query_id: str = Field(..., description="Query ID")
    selected_model: str = Field(..., description="Selected model")
    fallback_chain: list[str] = Field(..., description="Fallback models")
    confidence: float = Field(..., description="Decision confidence (0-1)")
    algorithm_phase: str = Field(..., description="Algorithm phase at decision time")
    query_count: int = Field(..., description="Router query count at decision time")
    arm_scores: dict[str, dict[str, float]] = Field(
        ..., description="Score breakdown for each model arm"
    )
    feature_vector: list[float] | None = Field(
        None, description="Feature vector (for contextual algorithms)"
    )
    constraints_applied: dict[str, Any] = Field(
        default_factory=dict, description="Constraints that affected the decision"
    )
    reasoning: str | None = Field(None, description="Decision reasoning")
    created_at: str = Field(..., description="ISO timestamp of decision")


class AuditListResponse(BaseModel):
    """Response schema for GET /v1/audit list endpoint."""

    entries: list[AuditEntryResponse] = Field(..., description="Audit entries")
    total: int = Field(..., description="Total entries matching query (before limit)")
    limit: int = Field(..., description="Limit applied")
    offset: int = Field(..., description="Offset applied")
