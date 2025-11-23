"""Async LLM-as-judge evaluation using Arbiter framework.

Evaluates routing quality using semantic similarity and factual correctness.
Runs in background without blocking routing, stores results in feedback table.
"""

import logging
import os
import random
from typing import Optional

from arbiter import evaluate

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
        model: str = "gpt-4o-mini",
    ):
        """Initialize Arbiter evaluator.

        Args:
            db: Database instance for storing feedback
            sample_rate: Fraction of responses to evaluate (0.0-1.0)
            daily_budget: Maximum daily spend on evaluations (USD)
            model: Model to use for evaluation (cheap model recommended)

        Raises:
            ValueError: If sample_rate not in [0.0, 1.0]
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {sample_rate}")

        self.db = db
        self.sample_rate = sample_rate
        self.daily_budget = daily_budget
        self.model = model

        # Evaluator names for quality assessment
        self.evaluators = [
            "semantic",  # Query-response similarity
            "factuality",  # Factual correctness
        ]

        logger.info(
            f"Arbiter evaluator initialized: "
            f"sample_rate={sample_rate:.1%}, "
            f"daily_budget=${daily_budget:.2f}, "
            f"model={model}"
        )

    async def should_evaluate(self) -> bool:
        """Check if we should evaluate this response.

        Considers:
        1. Random sampling (self.sample_rate)
        2. Daily budget limit

        Returns:
            True if should evaluate, False otherwise
        """
        # Random sampling
        if random.random() > self.sample_rate:
            return False

        # Budget check (TODO: implement cost tracking)
        # For now, just use sampling rate
        return True

    async def evaluate_async(
        self, response: Response, query: Query
    ) -> Optional[float]:
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

            # Store in feedback table for bandit learning
            feedback = Feedback(
                response_id=response.id,
                quality_score=overall_score,
                met_expectations=(overall_score >= 0.7),  # Threshold for expectations
                comments=f"Arbiter eval: ${result.interactions[0].cost_usd:.6f} (semantic + factuality)",
            )

            await self.db.save_complete_interaction(
                routing=None, response=response, feedback=feedback
            )

            logger.info(
                f"Evaluation complete: response={response.id[:8]}, "
                f"score={overall_score:.3f}, "
                f"cost=${result.interactions[0].cost_usd:.6f}"
            )

            return overall_score

        except Exception as e:
            # Never crash routing due to evaluation failures
            logger.warning(f"Evaluation failed for response {response.id[:8]}: {e}")
            return None

    def get_config(self) -> dict:
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
