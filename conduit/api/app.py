"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from conduit.api.middleware import setup_middleware
from conduit.api.routes import create_routes
from conduit.api.service import RoutingService
from conduit.core.config import settings
from conduit.core.database import Database
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router
from conduit.observability import setup_telemetry, shutdown_telemetry

logger = logging.getLogger(__name__)

# Global service instance
_service: RoutingService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan (startup/shutdown)."""
    global _service

    # Startup
    logger.info("Starting Conduit API server...")

    # Initialize database
    database = Database()
    await database.connect()

    # Initialize components (Router now handles analyzer + hybrid routing internally)
    router = Router(
        models=settings.default_models,
        embedding_provider_type=settings.embedding_provider,
        embedding_model=settings.embedding_model if settings.embedding_model else None,
        embedding_api_key=(
            settings.embedding_api_key if settings.embedding_api_key else None
        ),
    )

    # Load pricing information (if available) and pass to executor.
    # If loading fails, the executor will fall back to built-in pricing.
    try:
        model_prices = await database.get_latest_pricing()
        logger.info(f"Loaded {len(model_prices)} model prices from database")
    except Exception as e:
        logger.warning(f"Failed to load model prices, using fallback pricing: {e}")
        model_prices = {}

    executor = ModelExecutor()

    # Create service
    _service = RoutingService(
        database=database,
        router=router,
        executor=executor,
    )

    # Register routes
    api_router = create_routes(_service)
    app.include_router(api_router)

    logger.info("Conduit API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Conduit API server...")
    await database.disconnect()
    shutdown_telemetry()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Conduit",
        description="ML-powered LLM routing system for cost/latency/quality optimization",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Setup OpenTelemetry instrumentation
    setup_telemetry(app)

    # Setup middleware
    setup_middleware(app)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: object, exc: Exception) -> JSONResponse:
        """Handle unhandled exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app
