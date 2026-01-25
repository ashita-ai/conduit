"""FastAPI application factory with graceful shutdown support."""

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
from conduit.core.lifecycle import LifecycleManager
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router
from conduit.observability import setup_telemetry, shutdown_telemetry

logger = logging.getLogger(__name__)

# Global service instance
_service: RoutingService | None = None

# Global lifecycle manager for graceful shutdown
_lifecycle_manager: LifecycleManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan (startup/shutdown) with graceful shutdown support.

    Startup sequence:
        1. Initialize database connection pool
        2. Initialize router with ML components
        3. Load pricing information
        4. Create routing service
        5. Setup lifecycle manager with signal handlers
        6. Register API routes

    Shutdown sequence (via LifecycleManager):
        1. Stop accepting new requests
        2. Drain in-flight requests (30s timeout)
        3. Persist bandit state to database
        4. Close database connection pool
        5. Shutdown telemetry
    """
    global _service, _lifecycle_manager

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

    # Setup lifecycle manager for graceful shutdown
    _lifecycle_manager = LifecycleManager(
        router=router,
        database=database,
        shutdown_timeout=30.0,
        on_shutdown_complete=_on_shutdown_complete,
    )

    # Install signal handlers for SIGTERM/SIGINT
    try:
        _lifecycle_manager.install_signal_handlers()
    except Exception as e:
        # Signal handlers may fail in some environments (e.g., tests, non-main thread)
        logger.warning(f"Could not install signal handlers: {e}")

    # Store lifecycle manager in app state for middleware access
    app.state.lifecycle_manager = _lifecycle_manager

    # Register routes
    api_router = create_routes(_service)
    app.include_router(api_router)

    logger.info("Conduit API server started successfully")

    yield

    # Shutdown via lifecycle manager
    logger.info("Shutting down Conduit API server...")

    # If shutdown wasn't triggered by signal, do it now
    if _lifecycle_manager and not _lifecycle_manager.shutdown_requested:
        await _lifecycle_manager.shutdown()
    elif _lifecycle_manager is None:
        # Fallback if lifecycle manager wasn't created
        await router.close()
        await database.disconnect()

    shutdown_telemetry()
    logger.info("Shutdown complete")


async def _on_shutdown_complete() -> None:
    """Callback invoked after graceful shutdown completes."""
    logger.info("All shutdown tasks completed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI app with graceful shutdown support
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


def get_lifecycle_manager() -> LifecycleManager | None:
    """Get the global lifecycle manager instance.

    Returns:
        LifecycleManager if initialized, None otherwise
    """
    return _lifecycle_manager
