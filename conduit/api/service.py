"""Service layer for routing operations."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from conduit.core.database import Database
from conduit.core.exceptions import ExecutionError
from conduit.core.models import (
    Feedback,
    Query,
    QueryConstraints,
    RoutingResult,
)
from conduit.engines.bandits.base import BanditFeedback
from conduit.engines.executor import ModelExecutor
from conduit.engines.router import Router

if TYPE_CHECKING:
    from conduit.evaluation import ArbiterEvaluator

logger = logging.getLogger(__name__)


class RoutingService:
    """Service for handling routing requests."""

    def __init__(
        self,
        database: Database,
        router: Router,
        executor: ModelExecutor,
        default_result_type: type[BaseModel] | None = None,
        evaluator: "ArbiterEvaluator | None" = None,
    ):
        """Initialize routing service.

        Args:
            database: Database interface
            router: Router with hybrid routing (UCB1→LinUCB)
            executor: LLM executor
            default_result_type: Default Pydantic model for structured output
            evaluator: Optional Arbiter evaluator for quality assessment
        """
        self.database = database
        self.router = router
        self.executor = executor
        self.default_result_type = default_result_type
        self.evaluator = evaluator

    async def complete(
        self,
        prompt: str,
        result_type: type[BaseModel] | None = None,
        constraints: QueryConstraints | dict[str, Any] | None = None,
        user_id: str | None = None,
        context: BaseModel | dict[str, Any] | None = None,
    ) -> RoutingResult:
        """Route and execute LLM query.

        Args:
            prompt: User query/prompt
            result_type: Pydantic model for structured output
            constraints: Optional routing constraints
            user_id: User identifier
            context: Additional context

        Returns:
            RoutingResult with response and metadata

        Raises:
            RoutingError: If routing fails
            ExecutionError: If LLM execution fails
        """
        # Use default result type if not specified
        if result_type is None:
            result_type = self.default_result_type

        if result_type is None:
            # Create a simple default result type
            from pydantic import BaseModel

            class DefaultResult(BaseModel):
                """Default result type for unstructured responses."""

                content: str

            result_type = DefaultResult

        # Parse constraints (accept both typed model and dict for backward compatibility)
        query_constraints = None
        if constraints:
            if isinstance(constraints, QueryConstraints):
                query_constraints = constraints
            else:
                query_constraints = QueryConstraints(**constraints)

        # Convert context to dict if it's a Pydantic model
        context_dict: dict[str, Any] | None = None
        if context:
            if isinstance(context, BaseModel):
                context_dict = context.model_dump(exclude_none=True)
            else:
                context_dict = context

        # Create query
        query = Query(
            text=prompt,
            user_id=user_id,
            context=context_dict,
            constraints=query_constraints,
        )

        # Save query to database
        await self.database.save_query(query)

        # Route query
        routing = await self.router.route(query)

        # Execute LLM call
        try:
            response = await self.executor.execute(
                model=routing.selected_model,
                prompt=prompt,
                result_type=result_type,
                query_id=query.id,
            )
        except ExecutionError as e:
            logger.error(f"Execution failed for query {query.id}: {e}")
            raise

        # Save complete interaction
        await self.database.save_complete_interaction(
            routing=routing, response=response
        )

        # Fire-and-forget async evaluation (doesn't block routing)
        if self.evaluator:
            asyncio.create_task(self.evaluator.evaluate_async(response, query))

        # Update bandit with feedback using new BanditFeedback API
        # Until explicit/implicit feedback is wired in, use conservative estimates
        feedback = BanditFeedback(
            model_id=routing.selected_model,
            cost=response.cost,
            quality_score=0.8,  # Conservative default until explicit feedback
            latency=response.latency,
            success=True,
        )
        await self.router.hybrid_router.update(feedback, routing.features)

        # Return result
        return RoutingResult.from_response(response, routing)

    async def submit_feedback(
        self,
        response_id: str,
        quality_score: float,
        met_expectations: bool,
        user_rating: int | None = None,
        comments: str | None = None,
    ) -> Feedback:
        """Submit feedback for a response.

        Args:
            response_id: Response ID
            quality_score: Quality score (0.0-1.0)
            met_expectations: Whether response met expectations
            user_rating: Optional user rating (1-5)
            comments: Optional comments

        Returns:
            Feedback object

        Raises:
            ValueError: If response_id not found
        """
        # Get response to verify it exists
        response = await self.database.get_response_by_id(response_id)
        if response is None:
            raise ValueError(f"Response {response_id} not found")

        # Create feedback
        feedback = Feedback(
            response_id=response_id,
            quality_score=quality_score,
            user_rating=user_rating,
            met_expectations=met_expectations,
            comments=comments,
        )

        # Save feedback
        await self.database.save_complete_interaction(
            routing=None, response=response, feedback=feedback
        )

        # Update bandit with actual feedback using new BanditFeedback API
        # Need to regenerate features from original query for bandit update
        query_obj = await self.database.get_query_by_id(response.query_id)
        if query_obj:
            features = await self.router.analyzer.analyze(query_obj.text)
            bandit_feedback = BanditFeedback(
                model_id=response.model,
                cost=response.cost,
                quality_score=quality_score,  # Use actual user rating
                latency=response.latency,
                success=met_expectations,
            )
            await self.router.hybrid_router.update(bandit_feedback, features)
        else:
            logger.warning(
                f"Could not find query {response.query_id} for feedback update"
            )

        return feedback

    async def get_stats(self) -> dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dictionary with statistics
        """
        # Statistics aggregation is implemented at the database layer. This
        # method delegates to Database so that Supabase RPC functions or views
        # can evolve independently of the API surface.
        # For now, return placeholder values until database-backed stats are added.
        return {
            "total_queries": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
            "model_distribution": {},
            "quality_metrics": {"avg_quality": 0.0, "success_rate": 0.0},
        }

    async def get_models(self) -> list[str]:
        """Get available models.

        Returns:
            List of model IDs
        """
        return self.router.hybrid_router.models
