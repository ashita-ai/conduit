"""Middleware for FastAPI application."""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from conduit.api.auth import AuthenticationMiddleware
from conduit.api.ratelimit import RateLimitMiddleware
from conduit.api.sizelimit import RequestSizeLimitMiddleware
from conduit.core.config import settings

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log HTTP requests and responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log request and response details."""
        start_time = time.time()

        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Log response
        latency = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - Latency: {latency:.3f}s"
        )

        return response


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware in correct order.

    Middleware Order (applied bottom-to-top):
        1. CORS - handles cross-origin requests
        2. Request size limit - prevents DoS via large payloads
        3. Rate limiting - prevents abuse
        4. Authentication - validates API keys
        5. Logging - records request/response
    """
    # CORS (outermost)
    setup_cors(app)

    # Security middleware (applied in reverse order)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthenticationMiddleware, protected_prefixes=["/v1/"])
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)

    logger.info("All middleware configured successfully")
