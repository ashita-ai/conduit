"""Request and response validation schemas for Conduit API."""

from typing import Any

from pydantic import BaseModel, Field


class CompleteRequest(BaseModel):
    """Request schema for POST /v1/complete."""

    prompt: str = Field(..., description="User query/prompt", min_length=1)
    result_type: str | None = Field(
        None, description="Pydantic model name for structured output (optional)"
    )
    constraints: dict[str, Any] | None = Field(
        None,
        description="Optional routing constraints (max_cost, max_latency, min_quality, preferred_provider)",
    )
    user_id: str | None = Field(None, description="User identifier for tracking")
    context: dict[str, Any] | None = Field(
        None, description="Additional context metadata"
    )


class CompleteResponse(BaseModel):
    """Response schema for POST /v1/complete."""

    id: str = Field(..., description="Response ID")
    query_id: str = Field(..., description="Query ID")
    model: str = Field(..., description="Model used for completion")
    data: dict[str, Any] = Field(..., description="Structured response data")
    metadata: dict[str, Any] = Field(
        ...,
        description="Routing metadata (cost, latency, tokens, confidence, reasoning)",
    )


class FeedbackRequest(BaseModel):
    """Request schema for POST /v1/feedback."""

    response_id: str = Field(..., description="Response ID to provide feedback for")
    quality_score: float = Field(
        ..., description="Quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    user_rating: int | None = Field(
        None, description="User rating (1-5 stars)", ge=1, le=5
    )
    met_expectations: bool = Field(
        ..., description="Whether response met user expectations"
    )
    comments: str | None = Field(None, description="Optional user comments")


class FeedbackResponse(BaseModel):
    """Response schema for POST /v1/feedback."""

    id: str = Field(..., description="Feedback ID")
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
    detail: str | None = Field(None, description="Error details")
    code: str | None = Field(None, description="Error code")
