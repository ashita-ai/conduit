"""Unit tests for API routes."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from conduit.api.app import create_app
from conduit.api.service import RoutingService
from conduit.core.models import Feedback, RoutingResult


@pytest.fixture
def mock_service():
    """Create mock RoutingService."""
    service = AsyncMock(spec=RoutingService)
    # Add database attribute for health checks
    service.database = MagicMock()
    service.database.client = MagicMock()
    service.database.get_model_states = AsyncMock(return_value={})
    return service


@pytest.fixture
def client(mock_service):
    """Create test client with mocked service."""
    from fastapi import FastAPI

    from conduit.api.routes import create_routes

    # Create a minimal FastAPI app without the complex lifespan
    app = FastAPI(
        title="Conduit Test",
        version="test",
    )

    # Add routes with our mocked service
    router = create_routes(mock_service)
    app.include_router(router)

    # Create test client
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestCompleteEndpoint:
    """Tests for POST /v1/complete."""

    def test_complete_success(self, client, mock_service):
        """Test successful completion request."""
        # Setup mock
        mock_service.complete = AsyncMock(
            return_value=RoutingResult(
                id="response-123",
                query_id="query-123",
                model="gpt-4o-mini",
                data={"answer": "4", "confidence": 0.95},
                metadata={
                    "cost": 0.001,
                    "latency": 0.5,
                    "tokens": 20,
                    "routing_confidence": 0.85,
                    "reasoning": "Selected gpt-4o-mini for simple query",
                },
            )
        )

        # Make request
        response = client.post(
            "/v1/complete",
            json={"prompt": "What is 2+2?", "user_id": "test-user"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-4o-mini"
        assert "answer" in data["data"]
        assert data["metadata"]["cost"] == 0.001

    def test_complete_with_constraints(self, client, mock_service):
        """Test completion with routing constraints."""
        mock_service.complete = AsyncMock(
            return_value=RoutingResult(
                id="response-456",
                query_id="query-456",
                model="gpt-4o-mini",
                data={"content": "Test"},
                metadata={
                    "cost": 0.0005,
                    "latency": 0.3,
                    "tokens": 10,
                    "routing_confidence": 0.9,
                    "reasoning": "Cost-optimized routing",
                },
            )
        )

        response = client.post(
            "/v1/complete",
            json={
                "prompt": "Test query",
                "constraints": {"max_cost": 0.001, "min_quality": 0.7},
            },
        )

        assert response.status_code == 200
        # Verify constraints were passed to service (now as QueryConstraints model)
        call_args = mock_service.complete.call_args
        constraints = call_args.kwargs["constraints"]
        assert constraints.max_cost == 0.001

    def test_complete_missing_prompt(self, client):
        """Test completion with missing prompt."""
        response = client.post("/v1/complete", json={})

        assert response.status_code == 422  # Validation error

    def test_complete_routing_error(self, client, mock_service):
        """Test handling of routing errors."""
        from conduit.core.exceptions import RoutingError

        mock_service.complete = AsyncMock(side_effect=RoutingError("Routing failed"))

        response = client.post(
            "/v1/complete",
            json={"prompt": "Test"},
        )

        assert response.status_code == 400  # RoutingError returns Bad Request
        data = response.json()["detail"]
        assert data["code"] == "ROUTING_FAILED"
        assert data["error"] == "Routing failed"

    def test_complete_execution_error(self, client, mock_service):
        """Test handling of execution errors."""
        from conduit.core.exceptions import ExecutionError

        mock_service.complete = AsyncMock(
            side_effect=ExecutionError("Model API failed")
        )

        response = client.post("/v1/complete", json={"prompt": "Test"})

        assert response.status_code == 500
        data = response.json()["detail"]
        assert data["code"] == "EXECUTION_FAILED"
        assert data["error"] == "LLM execution failed"


class TestFeedbackEndpoint:
    """Tests for POST /v1/feedback."""

    def test_feedback_success(self, client, mock_service):
        """Test successful feedback submission."""
        mock_service.submit_feedback = AsyncMock(
            return_value=Feedback(
                response_id="response-123",
                quality_score=0.9,
                met_expectations=True,
                user_rating=5,
                comments="Great!",
            )
        )

        response = client.post(
            "/v1/feedback",
            json={
                "response_id": "response-123",
                "quality_score": 0.9,
                "met_expectations": True,
                "user_rating": 5,
                "comments": "Great!",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response_id"] == "response-123"
        assert "message" in data  # FeedbackResponse includes confirmation message
        assert data["id"] is not None

    def test_feedback_minimal(self, client, mock_service):
        """Test feedback with only required fields."""
        mock_service.submit_feedback = AsyncMock(
            return_value=Feedback(
                response_id="response-456",
                quality_score=0.75,
                met_expectations=False,
            )
        )

        response = client.post(
            "/v1/feedback",
            json={
                "response_id": "response-456",
                "quality_score": 0.75,
                "met_expectations": False,
            },
        )

        assert response.status_code == 200

    def test_feedback_invalid_score(self, client):
        """Test feedback with invalid quality score."""
        response = client.post(
            "/v1/feedback",
            json={
                "response_id": "response-123",
                "quality_score": 1.5,  # Invalid: > 1.0
                "met_expectations": True,
            },
        )

        assert response.status_code == 422  # Validation error

    def test_feedback_response_not_found(self, client, mock_service):
        """Test feedback for non-existent response."""
        mock_service.submit_feedback = AsyncMock(
            side_effect=ValueError("Response not found")
        )

        response = client.post(
            "/v1/feedback",
            json={
                "response_id": "invalid-id",
                "quality_score": 0.8,
                "met_expectations": True,
            },
        )

        assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for GET /v1/stats."""

    def test_stats_success(self, client, mock_service):
        """Test getting statistics."""
        mock_service.get_stats = AsyncMock(
            return_value={
                "total_queries": 100,
                "total_cost": 0.50,
                "avg_latency": 0.75,
                "model_distribution": {"gpt-4o-mini": 80, "gpt-4o": 20},
                "quality_metrics": {"avg_quality": 0.85, "success_rate": 0.95},
            }
        )

        response = client.get("/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 100
        assert "model_distribution" in data


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    def test_models_success(self, client, mock_service):
        """Test getting available models."""
        mock_service.get_models = AsyncMock(
            return_value=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4"]
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "gpt-4o-mini" in data["models"]


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_live(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_ready(self, client):
        """Test readiness probe with database check."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_startup(self, client):
        """Test startup probe."""
        response = client.get("/health/startup")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestErrorHandling:
    """Tests for general error handling."""

    def test_stats_error(self, client, mock_service):
        """Test stats endpoint error handling."""
        mock_service.get_stats = AsyncMock(side_effect=Exception("Database error"))

        response = client.get("/v1/stats")

        assert response.status_code == 500

    def test_models_error(self, client, mock_service):
        """Test models endpoint error handling."""
        mock_service.get_models = AsyncMock(side_effect=Exception("Config error"))

        response = client.get("/v1/models")

        assert response.status_code == 500

    def test_complete_unexpected_error(self, client, mock_service):
        """Test complete endpoint handles unexpected exceptions."""
        mock_service.complete = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        response = client.post("/v1/complete", json={"prompt": "Test query"})

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_feedback_unexpected_error(self, client, mock_service):
        """Test feedback endpoint handles unexpected exceptions."""
        mock_service.submit_feedback = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        response = client.post(
            "/v1/feedback",
            json={
                "response_id": "test-id",
                "quality_score": 0.9,
                "met_expectations": True,
            },
        )

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
