"""Rate limiting middleware using Redis sliding window algorithm.

This module implements distributed rate limiting using Redis for tracking
request counts per user. Uses sliding window algorithm for accurate rate limiting.

Algorithm:
    - Sliding window counter using sorted sets (ZSET)
    - Window size: 1 minute (60 seconds)
    - Limit: Configurable (default 100 requests/minute/user)
    - User identification: API key or IP address fallback

Performance:
    - O(log N) for Redis operations (ZADD, ZREMRANGEBYSCORE, ZCOUNT)
    - Minimal memory footprint (old entries auto-expire)
"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError
from starlette.middleware.base import BaseHTTPMiddleware

from conduit.core.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based rate limiting middleware with sliding window.

    Implements distributed rate limiting across API instances using Redis.
    Falls back to allowing requests if Redis unavailable (fail-open design).

    Example:
        >>> app.add_middleware(
        ...     RateLimitMiddleware,
        ...     redis_url="redis://localhost:6379",
        ...     rate_limit=100  # requests per minute
        ... )
    """

    def __init__(
        self,
        app: Callable[..., Any],
        redis_url: str | None = None,
        rate_limit: int | None = None,
        window_seconds: int = 60,
    ):
        """Initialize rate limiting middleware.

        Args:
            app: ASGI application
            redis_url: Redis connection URL (defaults to settings)
            rate_limit: Requests per window (defaults to settings)
            window_seconds: Time window in seconds (default 60 = 1 minute)
        """
        super().__init__(app)
        self.redis_url = redis_url or settings.redis_url
        self.rate_limit = rate_limit or settings.api_rate_limit
        self.window_seconds = window_seconds
        self.redis: Redis[bytes] | None = None

        # Initialize Redis client
        try:
            self.redis = Redis.from_url(
                self.redis_url,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=False,
                max_connections=10,
                decode_responses=False,
            )
            logger.info(f"Rate limiter initialized with {self.rate_limit}/min limit")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for rate limiting: {e}")
            self.redis = None

    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Response]]
    ) -> Response:
        """Process request and enforce rate limits.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response or 429 Too Many Requests

        Note:
            Fails open (allows requests) if Redis unavailable.
            Health checks are exempt from rate limiting.
        """
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health/"):
            return await call_next(request)

        # Fail open if Redis unavailable
        if not self.redis:
            return await call_next(request)

        # Identify user (API key from auth or IP address fallback)
        user_id = self._get_user_id(request)

        try:
            # Check and increment rate limit
            allowed, current_count, retry_after = await self._check_rate_limit(user_id)

            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {user_id}: "
                    f"{current_count}/{self.rate_limit} requests in {self.window_seconds}s"
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit exceeded: {self.rate_limit} requests per minute",
                        "retry_after": retry_after,
                        "limit": self.rate_limit,
                        "current": current_count,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.rate_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    },
                )

            # Rate limit passed, add headers and continue
            response = await call_next(request)

            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.rate_limit - current_count)
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time()) + self.window_seconds
            )

            return response

        except (ConnectionError, TimeoutError) as e:
            # Redis connection failed, fail open (allow request)
            logger.error(f"Redis connection error in rate limiter: {e}")
            return await call_next(request)

        except Exception as e:
            # Unexpected error, fail open
            logger.error(f"Unexpected error in rate limiter: {e}")
            return await call_next(request)

    async def _check_rate_limit(self, user_id: str) -> tuple[bool, int, int]:
        """Check if user is within rate limit using sliding window.

        Args:
            user_id: User identifier (API key or IP)

        Returns:
            Tuple of (allowed, current_count, retry_after_seconds)

        Algorithm:
            1. Remove expired entries (older than window)
            2. Count requests in current window
            3. If under limit, add current request and allow
            4. If over limit, deny and calculate retry time
        """
        if not self.redis:
            return (True, 0, 0)

        now = time.time()
        window_start = now - self.window_seconds
        key = f"conduit:ratelimit:{user_id}"

        # Redis pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Remove entries older than window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        pipe.zcount(key, window_start, now)

        # Execute pipeline
        results = await pipe.execute()
        current_count = int(results[1])

        # Check if under limit
        if current_count < self.rate_limit:
            # Add current request with score = timestamp
            await self.redis.zadd(key, {str(now): now})
            # Set expiry to window + buffer (cleanup)
            await self.redis.expire(key, self.window_seconds + 10)

            return (True, current_count + 1, 0)

        # Over limit, calculate retry time
        # Get oldest request in window to determine when it expires
        oldest_entries = await self.redis.zrange(key, 0, 0, withscores=True)

        if oldest_entries:
            oldest_timestamp = float(oldest_entries[0][1])
            retry_after = int(oldest_timestamp + self.window_seconds - now) + 1
        else:
            retry_after = self.window_seconds

        return (False, current_count, retry_after)

    def _get_user_id(self, request: Request) -> str:
        """Extract user identifier from request.

        Args:
            request: HTTP request

        Returns:
            User identifier (API key hash or IP address)

        Note:
            Uses API key if authenticated, otherwise falls back to IP address.
            API keys are hashed for privacy in Redis keys.
        """
        # Try to get API key from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.split()[1]
            # Hash API key for privacy (don't store raw keys in Redis)
            import hashlib

            return hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Fallback to IP address
        # Check X-Forwarded-For for proxy/load balancer scenarios
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take first IP in chain (original client)
            return forwarded_for.split(",")[0].strip()

        # Direct connection IP
        client_host = request.client.host if request.client else "unknown"
        return client_host

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self.redis:
            await self.redis.close()
            logger.info("Rate limiter Redis connection closed")
