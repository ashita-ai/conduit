"""Unit tests for authentication middleware.

Tests cover:
- Authentication disabled (bypass)
- Health check exemptions
- Missing Authorization header
- Invalid Authorization format
- Invalid API key
- Valid API key
- Protected vs unprotected paths
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with auth enabled."""
        with patch("conduit.api.auth.settings") as mock:
            mock.api_key = "test-api-key-12345"
            mock.api_require_auth = True
            yield mock

    @pytest.fixture
    def mock_settings_disabled(self):
        """Create mock settings with auth disabled."""
        with patch("conduit.api.auth.settings") as mock:
            mock.api_key = "test-api-key-12345"
            mock.api_require_auth = False
            yield mock

    @pytest.fixture
    def mock_settings_no_key(self):
        """Create mock settings with auth enabled but no key."""
        with patch("conduit.api.auth.settings") as mock:
            mock.api_key = None
            mock.api_require_auth = True
            yield mock

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def mock_request(self):
        """Create mock request with protected path."""
        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        return request

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = MagicMock()
        response.status_code = 200
        return response

    @pytest.fixture
    def mock_call_next(self, mock_response):
        """Create mock call_next that returns mock response."""
        call_next = AsyncMock(return_value=mock_response)
        return call_next

    def test_init_with_defaults(self, mock_settings, mock_app):
        """Test initialization with default protected prefixes."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        assert middleware.protected_prefixes == ["/v1/"]
        assert middleware.api_key == "test-api-key-12345"
        assert middleware.require_auth is True

    def test_init_with_custom_prefixes(self, mock_settings, mock_app):
        """Test initialization with custom protected prefixes."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(
            mock_app, protected_prefixes=["/api/", "/admin/"]
        )

        assert middleware.protected_prefixes == ["/api/", "/admin/"]

    def test_init_warning_when_no_key_configured(self, mock_settings_no_key, mock_app):
        """Test warning logged when auth required but no key configured."""
        from conduit.api.auth import AuthenticationMiddleware

        with patch("conduit.api.auth.logger") as mock_logger:
            middleware = AuthenticationMiddleware(mock_app)
            mock_logger.warning.assert_called_once()
            assert "no API key configured" in mock_logger.warning.call_args[0][0]

        assert middleware.api_key is None
        assert middleware.require_auth is True

    @pytest.mark.asyncio
    async def test_dispatch_auth_disabled(
        self, mock_settings_disabled, mock_app, mock_request, mock_call_next
    ):
        """Test that requests pass through when auth is disabled."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        response = await middleware.dispatch(mock_request, mock_call_next)

        mock_call_next.assert_called_once_with(mock_request)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_health_check_exempt(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test that health check endpoints are exempt from auth."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/health/ready"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_called_once_with(request)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_unprotected_path_allowed(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test that unprotected paths pass through without auth."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/docs"  # Not under /v1/
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_called_once_with(request)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_missing_authorization_header(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test 401 returned when Authorization header is missing."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        response = await middleware.dispatch(mock_request, mock_call_next)

        mock_call_next.assert_not_called()
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.body is not None

    @pytest.mark.asyncio
    async def test_dispatch_invalid_auth_format_no_bearer(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test 401 returned when Authorization format is invalid (no Bearer)."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="Basic sometoken")

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_not_called()
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_dispatch_invalid_auth_format_malformed(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test 401 returned when Authorization format is malformed."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer")  # No token

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_not_called()
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_dispatch_invalid_api_key(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test 401 returned when API key is invalid."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer wrong-api-key")

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_not_called()
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_dispatch_valid_api_key(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test request passes through with valid API key."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer test-api-key-12345")

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_called_once_with(request)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_api_key_not_configured_rejects_all(
        self, mock_settings_no_key, mock_app, mock_call_next
    ):
        """Test all protected requests rejected when no API key configured."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="Bearer any-key")

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_not_called()
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_dispatch_case_insensitive_bearer(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test that 'bearer' is case-insensitive."""
        from conduit.api.auth import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app)

        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value="BEARER test-api-key-12345")

        response = await middleware.dispatch(request, mock_call_next)

        mock_call_next.assert_called_once_with(request)
        assert response.status_code == 200
