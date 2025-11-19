"""Factory for creating RoutingService instances."""

import logging

from pydantic import BaseModel

from conduit.core.config import settings
from conduit.core.database import Database
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import RoutingEngine

# Import RoutingService here to avoid circular dependency
from conduit.api.service import RoutingService

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

    # Initialize components
    analyzer = QueryAnalyzer(embedding_model=settings.embedding_model)
    bandit = ContextualBandit(models=settings.default_models)
    executor = ModelExecutor()
    router = RoutingEngine(
        bandit=bandit,
        analyzer=analyzer,
        models=settings.default_models,
    )

    # Load model states from database if available
    try:
        states = await database.get_model_states()
        bandit.load_states(states)
        logger.info(f"Loaded {len(states)} model states from database")
    except Exception as e:
        logger.warning(f"Failed to load model states (continuing without): {e}")

    # Create service
    service = RoutingService(
        database=database,
        analyzer=analyzer,
        bandit=bandit,
        executor=executor,
        router=router,
        default_result_type=default_result_type,
    )

    return service

