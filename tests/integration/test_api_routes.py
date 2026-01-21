"""Integration tests for FastAPI routes.

Tests all API endpoints with proper authentication, validation,
and error handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from conduit.api.app import create_app
from conduit.core.models import (
    Query,
    QueryConstraints,
    QueryFeatures,
    Response,
    RoutingDecision,
    RoutingResult,
)


@pytest.fixture
def test_client():
    """Create test client without lifespan events."""
    # Create app without lifespan to avoid database connection
    from fastapi import FastAPI

    from conduit.api.middleware import setup_middleware

    app = FastAPI(title="Test Conduit")
    setup_middleware(app)

    # Mock service for testing
    mock_service = MagicMock()
    mock_service.complete = AsyncMock(
        return_value=RoutingResult(
            id="test-response-id",
            query_id="test-query-id",
            model="gpt-4o-mini",
            data={"content": "Test response"},
            metadata={
                "cost": 0.0001,
                "latency": 1.5,
                "tokens": 100,
                "routing_confidence": 0.85,
                "reasoning": "Test routing decision",
            },
        )
    )

    # Import and configure routes with mock service
    from conduit.api.routes import create_routes

    api_router = create_routes(mock_service)
    app.include_router(api_router)

    return TestClient(app), mock_service


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_live(self, test_client):
        """Test liveness probe returns 200."""
        client, _ = test_client
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_startup(self, test_client):
        """Test startup probe returns 200."""
        client, _ = test_client
        response = client.get("/health/startup")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_ready_without_database(self, test_client):
        """Test readiness probe fails without database."""
        client, mock_service = test_client

        # Mock database client as None (not connected)
        mock_service.database.client = None

        response = client.get("/health/ready")

        # Should return 503 Service Unavailable
        assert response.status_code == 503


class TestCompleteEndpoint:
    """Test /v1/complete endpoint."""

    def test_complete_success(self, test_client):
        """Test successful completion request."""
        client, _ = test_client

        request_data = {
            "prompt": "What is 2+2?",
            "user_id": "test-user",
        }

        response = client.post("/v1/complete", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-4o-mini"
        assert "metadata" in data
        assert data["metadata"]["cost"] == 0.0001

    def test_complete_with_constraints(self, test_client):
        """Test completion with routing constraints."""
        client, _ = test_client

        request_data = {
            "prompt": "Complex analysis task",
            "constraints": {
                "max_cost": 0.001,
                "max_latency": 5.0,
                "min_quality": 0.7,
            },
        }

        response = client.post("/v1/complete", json=request_data)

        assert response.status_code == 200

    def test_complete_empty_prompt(self, test_client):
        """Test completion with empty prompt fails validation."""
        client, _ = test_client

        request_data = {
            "prompt": "",
        }

        response = client.post("/v1/complete", json=request_data)

        # FastAPI validation should reject empty prompt
        assert response.status_code == 422

    def test_complete_missing_prompt(self, test_client):
        """Test completion without prompt fails validation."""
        client, _ = test_client

        request_data = {}

        response = client.post("/v1/complete", json=request_data)

        # FastAPI validation should require prompt
        assert response.status_code == 422


class TestFeedbackEndpoint:
    """Test /v1/feedback endpoint."""

    def test_feedback_success(self, test_client):
        """Test successful feedback submission."""
        client, mock_service = test_client

        # Mock feedback submission
        from conduit.core.models import Feedback

        mock_service.submit_feedback = AsyncMock(
            return_value=Feedback(
                id="test-feedback-id",
                response_id="test-response-id",
                quality_score=0.9,
                met_expectations=True,
            )
        )

        request_data = {
            "response_id": "test-response-id",
            "quality_score": 0.9,
            "met_expectations": True,
            "user_rating": 5,
        }

        response = client.post("/v1/feedback", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["response_id"] == "test-response-id"
        assert "message" in data

    def test_feedback_invalid_quality_score(self, test_client):
        """Test feedback with invalid quality score fails validation."""
        client, _ = test_client

        request_data = {
            "response_id": "test-response-id",
            "quality_score": 1.5,  # Out of 0.0-1.0 range
            "met_expectations": True,
        }

        response = client.post("/v1/feedback", json=request_data)

        # FastAPI validation should reject score > 1.0
        assert response.status_code == 422

    def test_feedback_invalid_rating(self, test_client):
        """Test feedback with invalid user rating fails validation."""
        client, _ = test_client

        request_data = {
            "response_id": "test-response-id",
            "quality_score": 0.8,
            "met_expectations": True,
            "user_rating": 6,  # Out of 1-5 range
        }

        response = client.post("/v1/feedback", json=request_data)

        # FastAPI validation should reject rating > 5
        assert response.status_code == 422


class TestStatsEndpoint:
    """Test /v1/stats endpoint."""

    def test_stats_success(self, test_client):
        """Test successful stats retrieval."""
        client, mock_service = test_client

        # Mock stats response
        mock_service.get_stats = AsyncMock(
            return_value={
                "total_queries": 100,
                "total_cost": 0.05,
                "avg_latency": 1.2,
                "model_distribution": {
                    "gpt-4o-mini": 80,
                    "gpt-4o": 20,
                },
                "quality_metrics": {
                    "avg_quality": 0.85,
                    "success_rate": 0.98,
                },
            }
        )

        response = client.get("/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 100
        assert "model_distribution" in data
        assert "quality_metrics" in data


class TestMetricsEndpoint:
    """Test /metrics Prometheus endpoint."""

    def test_metrics_endpoint(self, test_client):
        client, mock_service = test_client

        # Mock stats response (same source as /v1/stats)
        mock_service.get_stats = AsyncMock(
            return_value={
                "total_queries": 1523,
                "total_cost": 12.45,
                "avg_latency": 0.234,
                "model_distribution": {
                    "gpt-4o-mini": 892,
                    "claude-3-haiku": 631,
                },
            }
        )

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

        body = response.text

        assert "conduit_queries_total 1523" in body
        assert "conduit_cost_dollars_total 12.45" in body
        assert "conduit_latency_seconds 0.234" in body
        assert 'conduit_model_queries_total{model="gpt-4o-mini"} 892' in body
        assert 'conduit_model_queries_total{model="claude-3-haiku"} 631' in body

    def test_metrics_endpoint_error(self, test_client):
        client, mock_service = test_client

        mock_service.get_stats = AsyncMock(side_effect=RuntimeError("DB error"))

        response = client.get("/metrics")

        assert response.status_code == 500


class TestModelsEndpoint:
    """Test /v1/models endpoint."""

    def test_models_list(self, test_client):
        """Test models list retrieval."""
        client, mock_service = test_client

        # Mock models response
        mock_service.get_models = AsyncMock(
            return_value=["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"]
        )

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 3
        assert (
            data["default_model"] == "claude-haiku-4-5-20251001"
        )  # First model from conduit.yaml priors
