"""Integration tests for middleware components.

Tests authentication, rate limiting, and request size limits.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conduit.api.auth import AuthenticationMiddleware
from conduit.api.ratelimit import RateLimitMiddleware
from conduit.api.sizelimit import RequestSizeLimitMiddleware


@pytest.fixture
def simple_app():
    """Create simple FastAPI app for testing middleware."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}

    @app.post("/test")
    async def test_post():
        return {"message": "success"}

    @app.get("/v1/protected")
    async def protected_endpoint():
        return {"message": "protected resource"}

    @app.get("/health/live")
    async def health():
        return {"status": "healthy"}

    return app


class TestAuthenticationMiddleware:
    """Test API key authentication middleware."""

    def test_auth_disabled_allows_all(self, simple_app):
        """Test authentication disabled allows all requests."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to disable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = False
            mock_settings.api_key = "test-key"

            response = client.get("/v1/protected")
            assert response.status_code == 200

    def test_auth_required_missing_header(self, simple_app):
        """Test authentication required rejects missing header."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to enable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key-12345"

            response = client.get("/v1/protected")
            assert response.status_code == 401
            assert response.json()["error"] == "authentication_required"

    def test_auth_valid_key_allowed(self, simple_app):
        """Test valid API key allows access."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to enable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key-12345"

            headers = {"Authorization": "Bearer test-key-12345"}
            response = client.get("/v1/protected", headers=headers)
            assert response.status_code == 200

    def test_auth_invalid_key_rejected(self, simple_app):
        """Test invalid API key rejects access."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to enable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key-12345"

            headers = {"Authorization": "Bearer wrong-key"}
            response = client.get("/v1/protected", headers=headers)
            assert response.status_code == 401
            assert response.json()["error"] == "invalid_api_key"

    def test_auth_health_check_exempt(self, simple_app):
        """Test health checks exempt from authentication."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to enable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key-12345"

            # Health check should succeed without auth
            response = client.get("/health/live")
            assert response.status_code == 200

    def test_auth_malformed_header(self, simple_app):
        """Test malformed Authorization header rejected."""
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])

        client = TestClient(simple_app)

        # Mock settings to enable auth
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key-12345"

            # Missing "Bearer" prefix
            headers = {"Authorization": "test-key-12345"}
            response = client.get("/v1/protected", headers=headers)
            assert response.status_code == 401
            assert response.json()["error"] == "invalid_auth_format"


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    def test_ratelimit_redis_unavailable_fails_open(self, simple_app):
        """Test rate limiter fails open when Redis unavailable."""
        # Mock Redis connection to fail
        with patch("conduit.api.ratelimit.Redis") as MockRedis:
            MockRedis.from_url.side_effect = Exception("Connection failed")

            simple_app.add_middleware(
                RateLimitMiddleware, rate_limit=10, window_seconds=60
            )

            client = TestClient(simple_app)
            response = client.get("/test")

            # Should allow request (fail open)
            assert response.status_code == 200

    def test_ratelimit_health_check_exempt(self, simple_app):
        """Test health checks exempt from rate limiting."""
        simple_app.add_middleware(
            RateLimitMiddleware, rate_limit=10, window_seconds=60
        )

        client = TestClient(simple_app)
        response = client.get("/health/live")

        # Health check should always succeed
        assert response.status_code == 200

    def test_ratelimit_headers_present(self, simple_app):
        """Test rate limit headers added to response."""
        simple_app.add_middleware(
            RateLimitMiddleware, rate_limit=10, window_seconds=60
        )
        client = TestClient(simple_app)
        # Mock rate limit check to avoid Redis dependency
        with patch(
            "conduit.api.ratelimit.RateLimitMiddleware._check_rate_limit",
            new_callable=AsyncMock,
        ) as mock_check:
            # allowed=True, current_count=3, retry_after=0
            mock_check.return_value = (True, 3, 0)

            response = client.get("/test")

            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers



class TestRequestSizeLimitMiddleware:
    """Test request size limiting middleware."""

    def test_sizelimit_get_request_exempt(self, simple_app):
        """Test GET requests exempt from size limits."""
        simple_app.add_middleware(RequestSizeLimitMiddleware, max_size=100)

        client = TestClient(simple_app)
        response = client.get("/test")

        assert response.status_code == 200

    def test_sizelimit_small_request_allowed(self, simple_app):
        """Test small requests allowed."""
        simple_app.add_middleware(RequestSizeLimitMiddleware, max_size=1000)

        client = TestClient(simple_app)
        data = {"message": "small payload"}
        response = client.post("/test", json=data)

        assert response.status_code == 200

    def test_sizelimit_large_request_rejected(self, simple_app):
        """Test oversized requests rejected."""
        simple_app.add_middleware(RequestSizeLimitMiddleware, max_size=50)

        client = TestClient(simple_app)

        # Create payload larger than limit
        large_data = {"message": "x" * 100}

        # Set Content-Length header manually
        response = client.post(
            "/test",
            json=large_data,
            headers={"Content-Length": "200"},
        )

        # Should reject with 413 Payload Too Large
        assert response.status_code == 413
        assert response.json()["error"] == "request_too_large"

    def test_sizelimit_health_check_exempt(self, simple_app):
        """Test health checks exempt from size limits."""
        simple_app.add_middleware(RequestSizeLimitMiddleware, max_size=10)

        client = TestClient(simple_app)
        response = client.get("/health/live")

        assert response.status_code == 200


class TestMiddlewareOrdering:
    """Test middleware execution order."""

    def test_middleware_stack(self, simple_app):
        """Test full middleware stack in correct order."""
        # Add middleware in application order (reverse of execution)
        simple_app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])
        simple_app.add_middleware(RateLimitMiddleware, rate_limit=100)
        simple_app.add_middleware(RequestSizeLimitMiddleware, max_size=10000)

        client = TestClient(simple_app)

        # Mock auth settings
        with patch("conduit.api.auth.settings") as mock_settings:
            mock_settings.api_require_auth = True
            mock_settings.api_key = "test-key"

            # Request with valid auth should pass all middleware
            headers = {"Authorization": "Bearer test-key"}
            response = client.get("/v1/protected", headers=headers)

            # If all middleware passes, should reach endpoint
            assert response.status_code == 200
