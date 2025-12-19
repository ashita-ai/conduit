"""Async LLM-as-judge evaluation using Arbiter framework.

Evaluates routing quality using semantic similarity and factual correctness.
Runs in background without blocking routing, stores results in feedback table.
"""

import logging
import random
from datetime import date, datetime, timezone
from typing import Any

# Lazy import for optional arbiter dependency
try:
    from arbiter_ai import evaluate
except ImportError:
    evaluate = None  # type: ignore[assignment]

from conduit.core.config import get_arbiter_model
from conduit.core.database import Database
from conduit.core.models import Feedback, Query, Response

logger = logging.getLogger(__name__)


class ArbiterEvaluator:
    """Async LLM-as-judge evaluator using Arbiter framework.

    Evaluates query-response pairs using multiple criteria:
    - Semantic similarity between query and response
    - Factual correctness of response

    Features:
    - Async fire-and-forget evaluation (doesn't block routing)
    - Configurable sampling rate (default 10%)
    - Automatic cost tracking via Arbiter
    - Graceful error handling (failures don't crash routing)
    - Budget limits to control evaluation costs

    Example:
        >>> evaluator = ArbiterEvaluator(db, sample_rate=0.1)
        >>> # Fire and forget - doesn't block
        >>> asyncio.create_task(evaluator.evaluate_async(response, query))
    """

    def __init__(
        self,
        db: Database,
        sample_rate: float = 0.1,
        daily_budget: float = 10.0,
        model: str | None = None,
    ):
        """Initialize Arbiter evaluator.

        Args:
            db: Database instance for storing feedback
            sample_rate: Fraction of responses to evaluate (0.0-1.0)
            daily_budget: Maximum daily spend on evaluations (USD)
            model: Model to use for evaluation (defaults to conduit.yaml arbiter model)

        Raises:
            ValueError: If sample_rate not in [0.0, 1.0]
            ImportError: If arbiter package is not installed
        """
        if evaluate is None:
            raise ImportError(
                "ArbiterEvaluator requires the 'arbiter' package. "
                "Install with: pip install conduit[arbiter] or pip install arbiter-ai"
            )

        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {sample_rate}")

        self.db = db
        self.sample_rate = sample_rate
        self.daily_budget = daily_budget
        self.model = model if model is not None else get_arbiter_model()

        # Evaluator names for quality assessment
        self.evaluators = [
            "semantic",  # Query-response similarity
            "factuality",  # Factual correctness
        ]

        # Cost tracking state (resets daily at midnight UTC)
        self._daily_cost: float = 0.0
        self._cost_reset_date: date = datetime.now(timezone.utc).date()
        self._evaluation_count: int = 0
        self._total_cost: float = 0.0  # Lifetime total

        logger.info(
            f"Arbiter evaluator initialized: "
            f"sample_rate={sample_rate:.1%}, "
            f"daily_budget=${daily_budget:.2f}, "
            f"model={model}"
        )

    def _check_and_reset_daily_budget(self) -> None:
        """Reset daily cost counter if date has changed (midnight UTC).

        Called before budget checks to ensure we're tracking the current day.
        """
        today = datetime.now(timezone.utc).date()
        if today > self._cost_reset_date:
            logger.info(
                f"Daily budget reset: previous day spent ${self._daily_cost:.4f}, "
                f"evaluated {self._evaluation_count} responses"
            )
            self._daily_cost = 0.0
            self._evaluation_count = 0
            self._cost_reset_date = today

    def _track_cost(self, cost: float) -> None:
        """Track evaluation cost for budget management.

        Args:
            cost: Cost of the evaluation in USD
        """
        self._daily_cost += cost
        self._total_cost += cost
        self._evaluation_count += 1

    async def should_evaluate(self) -> bool:
        """Check if we should evaluate this response.

        Considers:
        1. Random sampling (self.sample_rate)
        2. Daily budget limit (self.daily_budget)

        Returns:
            True if should evaluate, False otherwise
        """
        # Random sampling first (cheapest check)
        if random.random() > self.sample_rate:
            return False

        # Reset daily budget if date changed
        self._check_and_reset_daily_budget()

        # Budget check
        if self._daily_cost >= self.daily_budget:
            logger.debug(
                f"Budget exhausted: ${self._daily_cost:.4f} >= ${self.daily_budget:.2f}"
            )
            return False

        return True

    async def evaluate_async(self, response: Response, query: Query) -> float | None:
        """Evaluate response quality asynchronously.

        This is a fire-and-forget async operation that:
        1. Runs evaluation with multiple criteria
        2. Stores results in feedback table
        3. Never blocks routing (errors are logged, not raised)

        Args:
            response: Response to evaluate
            query: Original query

        Returns:
            Overall quality score (0.0-1.0) if evaluation succeeds, None if skipped

        Note:
            This should be called with asyncio.create_task() to run in background
        """
        try:
            # Check sampling and budget
            if not await self.should_evaluate():
                return None

            # Run evaluation with Arbiter
            result = await evaluate(
                output=response.text,
                reference=query.text,
                evaluators=self.evaluators,
                model=self.model,
            )

            # Extract overall score (average of all evaluators)
            overall_score = result.overall_score

            # Extract cost from first interaction (evaluation LLM call)
            eval_cost = (
                result.interactions[0].cost if result.interactions[0].cost else 0.0
            )
            cost_float = float(eval_cost) if eval_cost is not None else 0.0

            # Track cost for budget management
            self._track_cost(cost_float)

            # Store in feedback table for bandit learning
            cost_str = f"{cost_float:.6f}"

            feedback = Feedback(
                response_id=response.id,
                quality_score=overall_score,
                user_rating=None,  # Arbiter doesn't collect user ratings
                met_expectations=(overall_score >= 0.7),  # Threshold for expectations
                comments=f"Arbiter eval: ${cost_str} (semantic + factuality)",
            )

            await self.db.save_complete_interaction(
                routing=None, response=response, feedback=feedback
            )

            logger.info(
                f"Evaluation complete: response={response.id[:8]}, "
                f"score={overall_score:.3f}, "
                f"cost=${cost_str}"
            )

            return overall_score

        except Exception as e:
            # Never crash routing due to evaluation failures
            logger.warning(f"Evaluation failed for response {response.id[:8]}: {e}")
            return None

    def get_config(self) -> dict[str, Any]:
        """Get current evaluator configuration.

        Returns:
            Configuration dictionary with all settings
        """
        return {
            "sample_rate": self.sample_rate,
            "daily_budget": self.daily_budget,
            "model": self.model,
            "evaluators": self.evaluators,  # Already a list of strings
        }

    def get_cost_stats(self) -> dict[str, Any]:
        """Get current cost tracking statistics.

        Returns:
            Dictionary with cost tracking information:
            - daily_cost: Current day's spend (USD)
            - daily_budget: Configured daily limit (USD)
            - budget_remaining: Remaining budget for today (USD)
            - budget_utilization: Percentage of daily budget used
            - evaluation_count: Number of evaluations today
            - total_cost: Lifetime total spend (USD)
            - cost_reset_date: Date of last budget reset (ISO format)

        Example:
            >>> stats = evaluator.get_cost_stats()
            >>> print(f"Budget: ${stats['budget_remaining']:.2f} remaining")
            Budget: $8.50 remaining
        """
        # Ensure we have current day's data
        self._check_and_reset_daily_budget()

        budget_remaining = max(0.0, self.daily_budget - self._daily_cost)
        utilization = (
            (self._daily_cost / self.daily_budget * 100)
            if self.daily_budget > 0
            else 0.0
        )

        return {
            "daily_cost": self._daily_cost,
            "daily_budget": self.daily_budget,
            "budget_remaining": budget_remaining,
            "budget_utilization": utilization,
            "evaluation_count": self._evaluation_count,
            "total_cost": self._total_cost,
            "cost_reset_date": self._cost_reset_date.isoformat(),
        }
