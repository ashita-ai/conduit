"""Unit tests for conduit.api.sizelimit module.

Tests for request size limiting middleware.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request
from fastapi.responses import JSONResponse

from conduit.api.sizelimit import RequestSizeLimitMiddleware


class TestRequestSizeLimitMiddleware:
    """Tests for RequestSizeLimitMiddleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/route"
        request.headers = {}
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create a mock call_next function."""
        async def call_next(request):
            return MagicMock(status_code=200)
        return call_next

    def test_init_default_size(self, mock_app):
        """Test middleware initialization with default size."""
        with patch("conduit.api.sizelimit.settings") as mock_settings:
            mock_settings.api_max_request_size = 10000
            middleware = RequestSizeLimitMiddleware(mock_app)
            assert middleware.max_size == 10000

    def test_init_custom_size(self, mock_app):
        """Test middleware initialization with custom size."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=5000)
        assert middleware.max_size == 5000

    @pytest.mark.asyncio
    async def test_get_request_skipped(self, mock_app, mock_request, mock_call_next):
        """Test GET requests are not size checked."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.method = "GET"

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_skipped(self, mock_app, mock_request, mock_call_next):
        """Test health check requests are not size checked."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.method = "POST"
        mock_request.url.path = "/health/ready"

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_within_limit(self, mock_app, mock_request, mock_call_next):
        """Test request within size limit passes through."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=1000)
        mock_request.headers = {"Content-Length": "500"}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_exceeds_limit(self, mock_app, mock_request, mock_call_next):
        """Test request exceeding size limit returns 413."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.headers = {"Content-Length": "500"}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_request_at_exact_limit(self, mock_app, mock_request, mock_call_next):
        """Test request at exact size limit passes through."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=500)
        mock_request.headers = {"Content-Length": "500"}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_request_no_content_length(self, mock_app, mock_request, mock_call_next):
        """Test request without Content-Length header passes through."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.headers = {}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_content_length(self, mock_app, mock_request, mock_call_next):
        """Test request with invalid Content-Length passes through."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.headers = {"Content-Length": "invalid"}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_413_response_format(self, mock_app, mock_request, mock_call_next):
        """Test 413 response includes correct error details."""
        middleware = RequestSizeLimitMiddleware(mock_app, max_size=100)
        mock_request.headers = {"Content-Length": "500"}

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 413
        # Body contains error info (JSONResponse body is bytes)
        body = response.body.decode()
        assert "request_too_large" in body
        assert "100" in body  # max_size
        assert "500" in body  # actual_size
