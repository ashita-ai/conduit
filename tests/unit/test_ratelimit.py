"""Unit tests for rate limiting middleware.

Tests cover:
- Initialization with defaults and custom values
- Health check exemptions
- Redis unavailable (fail-open)
- Rate limit exceeded
- Rate limit within bounds
- User identification (API key vs IP)
- Redis connection errors
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        with patch("conduit.api.ratelimit.settings") as mock:
            mock.redis_url = "redis://localhost:6379"
            mock.api_rate_limit = 100
            yield mock

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/v1/route"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        return request

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        return response

    @pytest.fixture
    def mock_call_next(self, mock_response):
        """Create mock call_next that returns mock response."""
        call_next = AsyncMock(return_value=mock_response)
        return call_next

    def test_init_with_defaults(self, mock_settings, mock_app):
        """Test initialization with default values from settings."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            assert middleware.rate_limit == 100
            assert middleware.window_seconds == 60
            assert middleware.redis_url == "redis://localhost:6379"

    def test_init_with_custom_values(self, mock_settings, mock_app):
        """Test initialization with custom values."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(
                mock_app,
                redis_url="redis://custom:6380",
                rate_limit=50,
                window_seconds=30,
            )

            assert middleware.rate_limit == 50
            assert middleware.window_seconds == 30
            assert middleware.redis_url == "redis://custom:6380"

    def test_init_redis_failure(self, mock_settings, mock_app):
        """Test initialization handles Redis connection failure gracefully."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = Exception("Connection failed")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            assert middleware.redis is None

    @pytest.mark.asyncio
    async def test_dispatch_health_check_exempt(
        self, mock_settings, mock_app, mock_call_next
    ):
        """Test that health check endpoints bypass rate limiting."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            request = MagicMock()
            request.url.path = "/health/ready"

            response = await middleware.dispatch(request, mock_call_next)

            mock_call_next.assert_called_once_with(request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_redis_unavailable_fail_open(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test that requests pass through when Redis is unavailable (fail-open)."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = Exception("Connection failed")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_within_rate_limit(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test request passes through when within rate limit."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline execution
            mock_pipe = MagicMock()
            mock_pipe.zremrangebyscore = MagicMock()
            mock_pipe.zcount = MagicMock()
            mock_pipe.execute = AsyncMock(return_value=[0, 5])  # 5 requests in window
            mock_redis.pipeline.return_value = mock_pipe
            mock_redis.zadd = AsyncMock()
            mock_redis.expire = AsyncMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)
            assert response.status_code == 200
            # Check rate limit headers were added
            assert "X-RateLimit-Limit" in response.headers
            assert response.headers["X-RateLimit-Limit"] == "100"

    @pytest.mark.asyncio
    async def test_dispatch_rate_limit_exceeded(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test 429 returned when rate limit exceeded."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline execution with 100+ requests
            mock_pipe = MagicMock()
            mock_pipe.zremrangebyscore = MagicMock()
            mock_pipe.zcount = MagicMock()
            mock_pipe.execute = AsyncMock(return_value=[0, 100])  # At limit
            mock_redis.pipeline.return_value = mock_pipe

            # Mock oldest entry for retry-after calculation
            import time

            mock_redis.zrange = AsyncMock(
                return_value=[(b"old", time.time() - 30)]  # 30 seconds ago
            )

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_not_called()
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_dispatch_redis_connection_error(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test fail-open when Redis connection error occurs during dispatch."""
        from redis.exceptions import ConnectionError

        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline to raise connection error
            mock_redis.pipeline.side_effect = ConnectionError("Connection lost")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_redis_timeout_error(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test fail-open when Redis timeout error occurs."""
        from redis.exceptions import TimeoutError

        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline to raise timeout error
            mock_redis.pipeline.side_effect = TimeoutError("Timeout")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_unexpected_error(
        self, mock_settings, mock_app, mock_request, mock_call_next
    ):
        """Test fail-open when unexpected error occurs."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline to raise unexpected error
            mock_redis.pipeline.side_effect = RuntimeError("Unexpected")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            response = await middleware.dispatch(mock_request, mock_call_next)

            mock_call_next.assert_called_once_with(mock_request)
            assert response.status_code == 200


class TestRateLimitUserIdentification:
    """Tests for user identification in rate limiting."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        with patch("conduit.api.ratelimit.settings") as mock:
            mock.redis_url = "redis://localhost:6379"
            mock.api_rate_limit = 100
            yield mock

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    def test_get_user_id_from_api_key(self, mock_settings, mock_app):
        """Test user identification from API key."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            request = MagicMock()
            request.headers = MagicMock()
            request.headers.get = MagicMock(
                side_effect=lambda key: (
                    "Bearer test-api-key" if key == "Authorization" else None
                )
            )

            user_id = middleware._get_user_id(request)

            # Should be hashed API key (16 chars)
            assert len(user_id) == 16
            assert user_id != "test-api-key"

    def test_get_user_id_from_x_forwarded_for(self, mock_settings, mock_app):
        """Test user identification from X-Forwarded-For header."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            request = MagicMock()
            request.headers = MagicMock()
            request.headers.get = MagicMock(
                side_effect=lambda key: (
                    "10.0.0.1, 10.0.0.2, 10.0.0.3" if key == "X-Forwarded-For" else None
                )
            )

            user_id = middleware._get_user_id(request)

            # Should be first IP in chain
            assert user_id == "10.0.0.1"

    def test_get_user_id_from_client_host(self, mock_settings, mock_app):
        """Test user identification from direct client host."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            request = MagicMock()
            request.headers = MagicMock()
            request.headers.get = MagicMock(return_value=None)
            request.client = MagicMock()
            request.client.host = "192.168.1.100"

            user_id = middleware._get_user_id(request)

            assert user_id == "192.168.1.100"

    def test_get_user_id_no_client(self, mock_settings, mock_app):
        """Test user identification when client is None."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = MagicMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            request = MagicMock()
            request.headers = MagicMock()
            request.headers.get = MagicMock(return_value=None)
            request.client = None

            user_id = middleware._get_user_id(request)

            assert user_id == "unknown"


class TestRateLimitCheckRateLimit:
    """Tests for _check_rate_limit method."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        with patch("conduit.api.ratelimit.settings") as mock:
            mock.redis_url = "redis://localhost:6379"
            mock.api_rate_limit = 100
            yield mock

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_redis(self, mock_settings, mock_app):
        """Test _check_rate_limit returns allowed when no Redis."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = Exception("Connection failed")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            allowed, count, retry = await middleware._check_rate_limit("test-user")

            assert allowed is True
            assert count == 0
            assert retry == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, mock_settings, mock_app):
        """Test _check_rate_limit when under limit."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline
            mock_pipe = MagicMock()
            mock_pipe.zremrangebyscore = MagicMock()
            mock_pipe.zcount = MagicMock()
            mock_pipe.execute = AsyncMock(return_value=[0, 50])  # 50 requests
            mock_redis.pipeline.return_value = mock_pipe
            mock_redis.zadd = AsyncMock()
            mock_redis.expire = AsyncMock()

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            allowed, count, retry = await middleware._check_rate_limit("test-user")

            assert allowed is True
            assert count == 51  # Current count + 1
            assert retry == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_with_oldest(self, mock_settings, mock_app):
        """Test _check_rate_limit when exceeded with oldest entry."""
        import time

        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline - at limit
            mock_pipe = MagicMock()
            mock_pipe.zremrangebyscore = MagicMock()
            mock_pipe.zcount = MagicMock()
            mock_pipe.execute = AsyncMock(return_value=[0, 100])
            mock_redis.pipeline.return_value = mock_pipe

            # Mock oldest entry (45 seconds ago)
            oldest_time = time.time() - 45
            mock_redis.zrange = AsyncMock(return_value=[(b"entry", oldest_time)])

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            allowed, count, retry = await middleware._check_rate_limit("test-user")

            assert allowed is False
            assert count == 100
            # Retry after should be roughly 15-16 seconds (60 - 45 + 1)
            assert 15 <= retry <= 17

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_no_oldest(self, mock_settings, mock_app):
        """Test _check_rate_limit when exceeded with no oldest entry."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.from_url.return_value = mock_redis

            # Mock pipeline - at limit
            mock_pipe = MagicMock()
            mock_pipe.zremrangebyscore = MagicMock()
            mock_pipe.zcount = MagicMock()
            mock_pipe.execute = AsyncMock(return_value=[0, 100])
            mock_redis.pipeline.return_value = mock_pipe

            # Mock empty oldest entry
            mock_redis.zrange = AsyncMock(return_value=[])

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            allowed, count, retry = await middleware._check_rate_limit("test-user")

            assert allowed is False
            assert count == 100
            assert retry == 60  # Default to window_seconds


class TestRateLimitClose:
    """Tests for close method."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        with patch("conduit.api.ratelimit.settings") as mock:
            mock.redis_url = "redis://localhost:6379"
            mock.api_rate_limit = 100
            yield mock

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_close_with_redis(self, mock_settings, mock_app):
        """Test close method closes Redis connection."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis.close = AsyncMock()
            mock_redis_class.from_url.return_value = mock_redis

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            await middleware.close()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_redis(self, mock_settings, mock_app):
        """Test close method handles no Redis gracefully."""
        with patch("conduit.api.ratelimit.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = Exception("Connection failed")

            from conduit.api.ratelimit import RateLimitMiddleware

            middleware = RateLimitMiddleware(mock_app)

            # Should not raise
            await middleware.close()
