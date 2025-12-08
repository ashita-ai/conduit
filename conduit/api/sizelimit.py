"""Request size limiting middleware for DoS prevention.

This module implements maximum request body size enforcement to prevent
denial-of-service attacks via large payloads.

Security Design:
    - Enforces configurable max request size (default 10KB)
    - Prevents memory exhaustion from large prompts
    - Returns 413 Payload Too Large for oversized requests
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from conduit.core.config import settings

logger = logging.getLogger(__name__)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce maximum request body size.

    Prevents DoS attacks via excessively large request payloads.
    Configurable size limit with clear error messages.

    Example:
        >>> app.add_middleware(
        ...     RequestSizeLimitMiddleware,
        ...     max_size=10_000  # 10KB
        ... )
    """

    def __init__(
        self,
        app: Callable[..., Any],
        max_size: int | None = None,
    ):
        """Initialize request size limit middleware.

        Args:
            app: ASGI application
            max_size: Maximum request body size in bytes (defaults to settings)
        """
        super().__init__(app)
        self.max_size = max_size or settings.api_max_request_size
        logger.info(f"Request size limit initialized: {self.max_size} bytes")

    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Response]]
    ) -> Response:
        """Process request and enforce size limits.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response or 413 Payload Too Large

        Note:
            GET requests and health checks are exempt from size limits.
        """
        # Skip size check for GET requests (no body)
        if request.method == "GET":
            return await call_next(request)

        # Skip for health checks
        if request.url.path.startswith("/health/"):
            return await call_next(request)

        # Check Content-Length header
        content_length = request.headers.get("Content-Length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    logger.warning(
                        f"Request body too large: {size} bytes > {self.max_size} bytes "
                        f"for {request.url.path}"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "error": "request_too_large",
                            "message": f"Request body exceeds maximum size of {self.max_size} bytes",
                            "max_size": self.max_size,
                            "actual_size": size,
                        },
                    )
            except ValueError:
                # Invalid Content-Length, let request proceed (will fail elsewhere)
                logger.warning(f"Invalid Content-Length header: {content_length}")

        # Size limit passed
        return await call_next(request)
