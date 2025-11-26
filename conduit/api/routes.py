"""FastAPI route handlers for Conduit API."""

import logging

from fastapi import APIRouter, HTTPException, status

from conduit.api.service import RoutingService
from conduit.api.validation import (
    CompleteRequest,
    CompleteResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelsResponse,
    StatsResponse,
)
from conduit.core.config import get_fallback_model
from conduit.core.exceptions import ExecutionError, RoutingError

logger = logging.getLogger(__name__)


def create_routes(service: RoutingService) -> APIRouter:
    """Create and configure API routes.

    Args:
        service: Routing service instance

    Returns:
        Configured APIRouter
    """
    # Create new router instance
    api_router = APIRouter()

    # POST /v1/complete - Route and execute LLM query
    @api_router.post(
        "/v1/complete",
        response_model=CompleteResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def complete(request: CompleteRequest) -> CompleteResponse:
        """Route and execute LLM query."""
        try:
            # Resolve result_type. For now, use a simple dict-based result type
            # until a result type registry is implemented.
            from pydantic import BaseModel

            class DictResult(BaseModel):
                """Default result type for unstructured responses."""

                content: str

            # Use provided result_type or default
            result_type: type[BaseModel] | None = None
            result_type = DictResult

            result = await service.complete(
                prompt=request.prompt,
                result_type=result_type,
                constraints=request.constraints,
                user_id=request.user_id,
                context=request.context,
            )

            return CompleteResponse(
                id=result.id,
                query_id=result.query_id,
                model=result.model,
                data=result.data,
                metadata=result.metadata,
            )

        except RoutingError as e:
            logger.error(f"Routing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except ExecutionError as e:
            logger.error(f"Execution error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM execution failed: {str(e)}",
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error in complete: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    # POST /v1/feedback - Submit quality feedback
    @api_router.post(
        "/v1/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_200_OK,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Submit feedback for a response."""
        try:
            feedback_obj = await service.submit_feedback(
                response_id=request.response_id,
                quality_score=request.quality_score,
                met_expectations=request.met_expectations,
                user_rating=request.user_rating,
                comments=request.comments,
            )

            return FeedbackResponse(
                id=feedback_obj.id,
                response_id=feedback_obj.response_id,
                message="Feedback submitted successfully",
            )

        except ValueError as e:
            logger.error(f"Feedback error: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error in feedback: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    # GET /v1/stats - Analytics and metrics
    @api_router.get(
        "/v1/stats",
        response_model=StatsResponse,
        status_code=status.HTTP_200_OK,
    )
    async def stats() -> StatsResponse:
        """Get routing statistics."""
        try:
            stats_data = await service.get_stats()
            return StatsResponse(**stats_data)
        except Exception as e:
            logger.exception(f"Unexpected error in stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    # GET /v1/models - List available models
    @api_router.get(
        "/v1/models",
        response_model=ModelsResponse,
        status_code=status.HTTP_200_OK,
    )
    async def models() -> ModelsResponse:
        """List available models."""
        try:
            model_list = await service.get_models()
            return ModelsResponse(
                models=model_list,
                default_model=get_fallback_model(),
            )
        except Exception as e:
            logger.exception(f"Unexpected error in models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    # GET /health/live - Liveness probe
    @api_router.get(
        "/health/live",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
    )
    async def health_live() -> HealthResponse:
        """Liveness probe for Kubernetes."""
        from datetime import datetime, timezone

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # GET /health/ready - Readiness probe
    @api_router.get(
        "/health/ready",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
    )
    async def health_ready() -> HealthResponse:
        """Readiness probe for Kubernetes.

        Validates database connectivity before marking ready.
        """
        from datetime import datetime, timezone

        # Check database connectivity
        try:
            if service.database.pool:
                # Simple database check - attempt to fetch model states
                await service.database.get_model_states()
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database not connected",
                )
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database unhealthy: {str(e)}",
            ) from e

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # GET /health/startup - Startup probe
    @api_router.get(
        "/health/startup",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
    )
    async def health_startup() -> HealthResponse:
        """Startup probe for Kubernetes."""
        from datetime import datetime, timezone

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    return api_router
