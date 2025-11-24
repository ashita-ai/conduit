"""Factory for creating RoutingService instances."""

import logging

from pydantic import BaseModel

# Import RoutingService here to avoid circular dependency
from conduit.api.service import RoutingService
from conduit.core.config import settings
from conduit.core.database import Database
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router

logger = logging.getLogger(__name__)


async def create_service(
    database: Database | None = None,
    default_result_type: type[BaseModel] | None = None,
) -> RoutingService:
    """Create a RoutingService instance with all dependencies.

    Args:
        database: Optional database instance (creates new if None)
        default_result_type: Optional default Pydantic model for responses

    Returns:
        Configured RoutingService instance
    """
    # Initialize database if not provided
    if database is None:
        database = Database()
        await database.connect()

    # Initialize components (Router now handles analyzer + hybrid routing internally)
    router = Router(
        models=settings.default_models,
        embedding_provider_type=settings.embedding_provider,
        embedding_model=settings.embedding_model if settings.embedding_model else None,
        embedding_api_key=settings.embedding_api_key if settings.embedding_api_key else None,
    )
    executor = ModelExecutor()

    # Note: Model state loading removed - HybridRouter doesn't support load_states()
    # TODO: Implement state persistence for HybridRouter (UCB1 + LinUCB)

    # Create service
    service = RoutingService(
        database=database,
        router=router,
        executor=executor,
        default_result_type=default_result_type,
    )

    return service

