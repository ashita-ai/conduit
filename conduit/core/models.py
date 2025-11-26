"""Core data models for Conduit routing system.

This module defines Pydantic models for queries, routing decisions,
responses, feedback, and ML model state.
"""

import json
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class UserPreferences(BaseModel):
    """User routing preferences for reward optimization.

    Controls how Conduit balances quality, cost, and latency when
    selecting models. Uses predefined presets for simplicity.

    Presets:
    - balanced: Default (0.7 quality, 0.2 cost, 0.1 latency)
    - quality: Maximize quality (0.8 quality, 0.1 cost, 0.1 latency)
    - cost: Minimize cost (0.4 quality, 0.5 cost, 0.1 latency)
    - speed: Minimize latency (0.4 quality, 0.1 cost, 0.5 latency)

    Example:
        >>> preferences = UserPreferences(optimize_for="cost")
        >>> query = Query(text="Simple math", preferences=preferences)
    """

    optimize_for: Literal["balanced", "quality", "cost", "speed"] = Field(
        default="balanced", description="Routing optimization priority"
    )


class QueryConstraints(BaseModel):
    """Constraints for routing decisions."""

    max_cost: float | None = Field(None, description="Maximum cost in dollars", ge=0.0)
    max_latency: float | None = Field(
        None, description="Maximum latency in seconds", ge=0.0
    )
    min_quality: float | None = Field(
        None, description="Minimum quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    preferred_provider: str | None = Field(
        None, description="Preferred LLM provider (openai, anthropic, google, groq)"
    )


class Query(BaseModel):
    """User query to be routed to an LLM."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique query ID")
    text: str = Field(..., description="Query text", min_length=1)
    user_id: str | None = Field(None, description="User identifier")
    context: dict[str, Any] | None = Field(
        None, description="Additional context metadata"
    )
    constraints: QueryConstraints | None = Field(
        None, description="Routing constraints"
    )
    preferences: UserPreferences = Field(
        default_factory=UserPreferences, description="User routing preferences"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Query creation timestamp",
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()


class QueryFeatures(BaseModel):
    """Extracted features from query for routing decision."""

    embedding: list[float] = Field(
        ..., description="Semantic embedding vector (dimension depends on model)"
    )
    token_count: int = Field(..., description="Approximate token count", ge=0)
    complexity_score: float = Field(
        ..., description="Complexity score (0.0-1.0)", ge=0.0, le=1.0
    )
    domain: str = Field(..., description="Query domain classification")
    domain_confidence: float = Field(
        ..., description="Domain classification confidence", ge=0.0, le=1.0
    )


class RoutingDecision(BaseModel):
    """ML-powered routing decision for a query.

    Confidence Score Interpretation (Strategic Decision 2025-11-18):
    - Probabilistic guarantee: 95%+ of queries should have confidence >= 0.6
    - High confidence (>0.8): Strong historical evidence for this query type
    - Medium confidence (0.5-0.8): Moderate evidence, reasonable prediction
    - Low confidence (<0.5): Limited data, exploratory routing
    - Zero confidence (0.0): Fallback/default routing (no ML prediction)

    Note: We provide probabilistic guarantees, not deterministic promises.
    See notes/2025-11-18_business_panel_analysis.md for strategic rationale.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Decision ID")
    query_id: str = Field(..., description="Associated query ID")
    selected_model: str = Field(..., description="Selected LLM model")
    confidence: float = Field(
        ...,
        description="Thompson sampling confidence (0.0-1.0): >0.8=high, 0.5-0.8=medium, <0.5=low, 0.0=fallback",
        ge=0.0,
        le=1.0,
    )
    features: QueryFeatures = Field(..., description="Extracted query features")
    reasoning: str = Field(..., description="Explanation of routing decision")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional routing metadata (constraints_relaxed, fallback, attempt)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Decision timestamp",
    )


class Response(BaseModel):
    """LLM response to a query."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique response ID"
    )
    query_id: str = Field(..., description="Associated query ID")
    model: str = Field(..., description="Model that generated response")
    text: str = Field(..., description="Response text (JSON for structured outputs)")
    cost: float = Field(..., description="Cost in dollars", ge=0.0)
    latency: float = Field(..., description="Latency in seconds", ge=0.0)
    tokens: int = Field(..., description="Total tokens used", ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp",
    )


class Feedback(BaseModel):
    """Explicit user feedback on response quality.

    Part of dual feedback system:
    - Explicit: User-provided ratings (this model)
    - Implicit: System-observed signals (ImplicitFeedback) [Phase 2]
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique feedback ID"
    )
    response_id: str = Field(..., description="Associated response ID")
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
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feedback timestamp",
    )


class ImplicitFeedback(BaseModel):
    """Implicit feedback signals from system observation.

    Part of dual feedback system - captures user behavior signals
    without requiring explicit feedback submission. Improves antifragility
    of learning algorithm by reducing dependency on user participation.

    Implements "Observability Trinity":
    - Error detection: Model failures and quality issues
    - Latency tracking: Response times and user patience
    - Retry detection: Repeated queries indicating dissatisfaction

    Phase: 2 (implemented)
    Strategic rationale: See notes/2025-11-18_business_panel_analysis.md
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique signal ID"
    )
    query_id: str = Field(..., description="Associated query ID")
    model_id: str = Field(..., description="Model that generated response")
    timestamp: float = Field(
        ..., description="Unix timestamp of signal capture", ge=0.0
    )

    # Error signals
    error_occurred: bool = Field(
        default=False, description="Model returned error or invalid response"
    )
    error_type: str | None = Field(None, description="Type of error if occurred")

    # Latency signals
    latency_seconds: float = Field(
        ..., description="Actual response time in seconds", ge=0.0
    )
    latency_accepted: bool = Field(
        default=True, description="User waited for response (did not timeout)"
    )
    latency_tolerance: str = Field(
        default="high",
        description="Categorized user patience (high/medium/low)",
    )

    # Retry signals
    retry_detected: bool = Field(
        default=False, description="User re-submitted semantically similar query"
    )
    retry_delay_seconds: float | None = Field(
        None, description="Time between original and retry query", ge=0.0
    )
    similarity_score: float | None = Field(
        None, description="Cosine similarity to previous query (0-1)", ge=0.0, le=1.0
    )
    original_query_id: str | None = Field(
        None, description="ID of query being retried (if retry detected)"
    )


class ModelState(BaseModel):
    """Thompson Sampling state for a model (Beta distribution)."""

    model_id: str = Field(..., description="Model identifier")
    alpha: float = Field(default=1.0, description="Beta distribution α parameter", gt=0)
    beta: float = Field(default=1.0, description="Beta distribution β parameter", gt=0)
    total_requests: int = Field(default=0, description="Total requests to this model")
    total_cost: float = Field(default=0.0, description="Total cost accumulated", ge=0.0)
    avg_quality: float = Field(
        default=0.0, description="Average quality score", ge=0.0, le=1.0
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    @property
    def mean_success_rate(self) -> float:
        """Expected success rate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of Beta distribution."""
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))


class RoutingResult(BaseModel):
    """Complete routing result returned to client."""

    id: str = Field(..., description="Response ID")
    query_id: str = Field(..., description="Query ID")
    model: str = Field(..., description="Model used")
    data: dict[str, Any] = Field(..., description="Structured response data")
    metadata: dict[str, Any] = Field(..., description="Routing metadata")

    @classmethod
    def from_response(
        cls, response: Response, routing: RoutingDecision
    ) -> "RoutingResult":
        """Create RoutingResult from Response and RoutingDecision."""
        return cls(
            id=response.id,
            query_id=response.query_id,
            model=response.model,
            data=json.loads(response.text),
            metadata={
                "cost": response.cost,
                "latency": response.latency,
                "tokens": response.tokens,
                "routing_confidence": routing.confidence,
                "reasoning": routing.reasoning,
            },
        )
