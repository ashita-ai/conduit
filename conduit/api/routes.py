"""FastAPI route handlers for Conduit API."""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from conduit.api.service import RoutingService
from conduit.api.validation import (
    AuditEntryResponse,
    AuditListResponse,
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
                detail={
                    "error": e.message,
                    "code": e.code,
                    "context": e.details or None,
                },
            ) from e
        except ExecutionError as e:
            logger.error(f"Execution error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "LLM execution failed",
                    "code": e.code,
                    "context": e.details or None,
                },
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

    # GET /metrics - Prometheus metrics endpoint
    @api_router.get("/metrics")
    async def metrics() -> Response:
        """Export metrics in Prometheus format."""

        def _escape_label_value(value: str) -> str:
            """Escape Prometheus label value per exposition format spec."""
            return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        try:
            stats = await service.get_stats()

            total_queries = stats.get("total_queries", 0)
            total_cost = stats.get("total_cost", 0.0)
            avg_latency = stats.get("avg_latency", 0.0)
            model_distribution = stats.get("model_distribution", {})

            lines: list[str] = []

            lines.append(
                "# HELP conduit_queries_total Total number of queries processed"
            )
            lines.append("# TYPE conduit_queries_total counter")
            lines.append(f"conduit_queries_total {total_queries}")

            lines.append("# HELP conduit_cost_dollars_total Total cost in dollars")
            lines.append("# TYPE conduit_cost_dollars_total counter")
            lines.append(f"conduit_cost_dollars_total {total_cost:.6f}")

            lines.append("# HELP conduit_latency_seconds Average latency in seconds")
            lines.append("# TYPE conduit_latency_seconds gauge")
            lines.append(f"conduit_latency_seconds {avg_latency:.6f}")

            lines.append("# HELP conduit_model_queries_total Queries per model")
            lines.append("# TYPE conduit_model_queries_total counter")
            for model, count in model_distribution.items():
                lines.append(
                    f'conduit_model_queries_total{{model="{_escape_label_value(model)}"}} {count}'
                )

            return Response(
                content="\n".join(lines) + "\n",
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        except Exception as e:
            logger.exception(f"Unexpected error in metrics endpoint: {e}")
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

    # GET /v1/audit - Query audit log
    @api_router.get(
        "/v1/audit",
        response_model=AuditListResponse,
        status_code=status.HTTP_200_OK,
        responses={
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def audit_list(
        decision_id: str | None = None,
        query_id: str | None = None,
        selected_model: str | None = None,
        algorithm_phase: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditListResponse:
        """Query decision audit log.

        Returns audit entries for routing decisions with optional filters.
        Used for debugging, compliance, and analysis.
        """
        from datetime import datetime

        from conduit.observability.audit import AuditQuery

        try:
            # Check if audit store is available (accessed via router)
            audit_store = service.router.audit_store
            if not audit_store:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Audit logging not enabled",
                )

            # Parse datetime filters
            start_dt = None
            end_dt = None
            if start_time:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            if end_time:
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            # Build query
            query = AuditQuery(
                decision_id=decision_id,
                query_id=query_id,
                selected_model=selected_model,
                algorithm_phase=algorithm_phase,
                start_time=start_dt,
                end_time=end_dt,
                limit=min(limit, 1000),
                offset=offset,
            )

            # Execute query
            entries = await audit_store.query(query)

            # Convert to response format
            response_entries = [
                AuditEntryResponse(
                    id=entry.id or 0,
                    decision_id=entry.decision_id,
                    query_id=entry.query_id,
                    selected_model=entry.selected_model,
                    fallback_chain=entry.fallback_chain,
                    confidence=entry.confidence,
                    algorithm_phase=entry.algorithm_phase,
                    query_count=entry.query_count,
                    arm_scores=entry.arm_scores,
                    feature_vector=entry.feature_vector,
                    constraints_applied=entry.constraints_applied,
                    reasoning=entry.reasoning,
                    created_at=entry.created_at.isoformat(),
                )
                for entry in entries
            ]

            return AuditListResponse(
                entries=response_entries,
                total=len(
                    response_entries
                ),  # Simplified - would need count query for true total
                limit=query.limit,
                offset=query.offset,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in audit list: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    # GET /v1/audit/{decision_id} - Get specific audit entry
    @api_router.get(
        "/v1/audit/{decision_id}",
        response_model=AuditEntryResponse,
        status_code=status.HTTP_200_OK,
        responses={
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def audit_get(decision_id: str) -> AuditEntryResponse:
        """Get audit entry for a specific routing decision.

        Args:
            decision_id: UUID of the routing decision
        """
        try:
            # Check if audit store is available (accessed via router)
            audit_store = service.router.audit_store
            if not audit_store:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Audit logging not enabled",
                )

            entry = await audit_store.get_by_decision_id(decision_id)

            if entry is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Audit entry not found for decision: {decision_id}",
                )

            return AuditEntryResponse(
                id=entry.id or 0,
                decision_id=entry.decision_id,
                query_id=entry.query_id,
                selected_model=entry.selected_model,
                fallback_chain=entry.fallback_chain,
                confidence=entry.confidence,
                algorithm_phase=entry.algorithm_phase,
                query_count=entry.query_count,
                arm_scores=entry.arm_scores,
                feature_vector=entry.feature_vector,
                constraints_applied=entry.constraints_applied,
                reasoning=entry.reasoning,
                created_at=entry.created_at.isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in audit get: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    return api_router
