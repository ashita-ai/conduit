"""API authentication middleware for API key validation.

This module implements bearer token authentication using API keys.
Authentication is optional and controlled via configuration.

Security Design:
    - API key passed in Authorization header: "Bearer <key>"
    - Constant-time comparison to prevent timing attacks
    - Configurable endpoints requiring authentication
    - Graceful bypass when authentication disabled
"""

import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from conduit.core.config import settings

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.

    Validates API keys for protected endpoints using bearer token format.
    Supports configurable authentication requirements and endpoint exclusions.

    Example:
        >>> app.add_middleware(
        ...     AuthenticationMiddleware,
        ...     protected_prefixes=["/v1/"]
        ... )
    """

    def __init__(
        self,
        app: Callable[..., Any],
        protected_prefixes: list[str] | None = None,
    ):
        """Initialize authentication middleware.

        Args:
            app: ASGI application
            protected_prefixes: List of path prefixes requiring auth (default: ["/v1/"])
        """
        super().__init__(app)
        self.protected_prefixes = protected_prefixes or ["/v1/"]
        self.api_key = settings.api_key
        self.require_auth = settings.api_require_auth

        if self.require_auth and not self.api_key:
            logger.warning(
                "API authentication required but no API key configured. "
                "All protected requests will be rejected."
            )

    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Response]]
    ) -> Response:
        """Process request and validate authentication if required.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response from handler or 401 Unauthorized

        Note:
            Health check endpoints (/health/*) are always exempt from authentication.
        """
        # Skip authentication if disabled
        if not self.require_auth:
            return await call_next(request)

        # Always allow health checks (Kubernetes probes)
        if request.url.path.startswith("/health/"):
            return await call_next(request)

        # Check if path requires authentication
        requires_auth = any(
            request.url.path.startswith(prefix) for prefix in self.protected_prefixes
        )

        if not requires_auth:
            return await call_next(request)

        # Extract and validate API key
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            logger.warning(f"Missing Authorization header for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_required",
                    "message": "Missing Authorization header",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse bearer token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(
                f"Invalid Authorization header format for {request.url.path}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "invalid_auth_format",
                    "message": "Authorization header must be 'Bearer <token>'",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        provided_key = parts[1]

        # Validate API key using constant-time comparison
        if not self.api_key or not secrets.compare_digest(provided_key, self.api_key):
            logger.warning(f"Invalid API key for {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "invalid_api_key",
                    "message": "Invalid API key",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Authentication successful
        return await call_next(request)
