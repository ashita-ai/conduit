"""LLM execution via PydanticAI with timeout and error handling."""

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import Agent

from conduit.core.config import get_default_pricing, get_fallback_pricing
from conduit.core.exceptions import ExecutionError
from conduit.core.models import Response
from conduit.core.pricing import ModelPricing

logger = logging.getLogger(__name__)


class ModelExecutor:
    """Execute LLM calls via PydanticAI."""

    def __init__(self, pricing: dict[str, ModelPricing] | None = None) -> None:
        """Initialize executor with agent cache and pricing.

        Args:
            pricing: Optional mapping of model_id to ModelPricing loaded from
                the database. If not provided, a built-in approximate pricing
                table will be used as a fallback.
        """
        self.clients: dict[str, Agent[Any, Any]] = {}
        self.pricing: dict[str, ModelPricing] = pricing or {}

    async def execute(
        self,
        model: str,
        prompt: str,
        result_type: type[BaseModel],
        query_id: str,
        timeout: float = 60.0,
    ) -> Response:
        """Execute LLM call with selected model and timeout.

        Args:
            model: Model ID (e.g., "gpt-4o-mini")
            prompt: User query
            result_type: Pydantic model for structured output
            query_id: For tracking
            timeout: Maximum execution time in seconds (default 60s)

        Returns:
            Response with result and metadata

        Raises:
            ExecutionError: If model call fails or times out
            asyncio.TimeoutError: If execution exceeds timeout

        Timeout Strategy:
            - Default: 60s for all models
            - Fast models (mini): 30s recommended
            - Premium models (opus): 90s recommended
            - Configurable per-query via constraints.max_latency
        """
        start_time = time.time()

        # Get or create PydanticAI agent
        agent = self._get_agent(model, result_type)

        # Execute with timeout and automatic retry
        try:
            result = await asyncio.wait_for(agent.run(prompt), timeout=timeout)
            latency = time.time() - start_time

            # Extract cost from interaction
            usage = result.usage()
            cost = self._compute_cost(usage, model)

            # Extract total tokens (handle both object and dict)
            if hasattr(usage, "total_tokens"):
                total_tokens = usage.total_tokens
            elif isinstance(usage, dict):
                total_tokens = usage.get("total_tokens", 0)
            else:
                total_tokens = 0

            return Response(
                id=str(uuid4()),
                query_id=query_id,
                model=model,
                # PydanticAI v1.20+ uses .output instead of .data()
                text=result.output.model_dump_json(),  # type: ignore[attr-defined]
                cost=cost,
                latency=latency,
                tokens=total_tokens,
            )

        except asyncio.TimeoutError as e:
            latency = time.time() - start_time
            logger.error(
                f"Execution timeout for {model} after {latency:.2f}s (limit: {timeout}s)"
            )
            raise ExecutionError(
                f"Model {model} exceeded timeout of {timeout}s",
                details={"latency": latency, "timeout": timeout},
            ) from e
        except Exception as e:
            logger.error(f"Execution failed for {model}: {e}")
            raise ExecutionError(f"Model {model} failed: {e}") from e

    def _get_agent(self, model: str, result_type: type[BaseModel]) -> Agent:
        """Get cached or create new PydanticAI agent.

        Args:
            model: Model identifier
            result_type: Pydantic model for structured output

        Returns:
            Configured PydanticAI agent
        """
        cache_key = f"{model}_{result_type.__name__}"

        if cache_key not in self.clients:
            # PydanticAI v1.20+ uses output_type instead of result_type
            self.clients[cache_key] = Agent(model=model, output_type=result_type)
            logger.debug(f"Created new agent for {cache_key}")

        return self.clients[cache_key]

    def _compute_cost(self, usage: Any, model: str) -> float:
        """Compute cost based on token usage and model pricing.

        Args:
            usage: Token usage from PydanticAI (Usage object or dict)
            model: Model identifier

        Returns:
            Cost in dollars

        Pricing strategy:
            1. If database-backed pricing is available for the model, use it.
            2. Otherwise, fall back to a built-in approximate pricing table.
        """
        # Prefer database-backed pricing when available
        db_pricing = self.pricing.get(model)

        if db_pricing is not None:
            input_cost_per_token = db_pricing.input_cost_per_token
            output_cost_per_token = db_pricing.output_cost_per_token
        else:
            # Load fallback pricing from conduit.yaml
            # Note: These are approximate. Database pricing should be preferred.
            fallback_pricing = get_fallback_pricing()
            default_pricing = get_default_pricing()

            model_pricing = fallback_pricing.get(model, default_pricing)

            input_cost_per_token = model_pricing["input"] / 1_000_000.0
            output_cost_per_token = model_pricing["output"] / 1_000_000.0

        # Handle Usage object (has attributes) or dict (has .get method)
        if hasattr(usage, "request_tokens") or hasattr(usage, "input_tokens"):
            # Usage object with attributes (check both naming conventions)
            input_tokens = getattr(usage, "request_tokens", 0) or getattr(
                usage, "input_tokens", 0
            )
            output_tokens = getattr(usage, "response_tokens", 0) or getattr(
                usage, "output_tokens", 0
            )
        elif isinstance(usage, dict):
            # Dict with keys
            input_tokens = usage.get("request_tokens", 0) or usage.get(
                "input_tokens", 0
            )
            output_tokens = usage.get("response_tokens", 0) or usage.get(
                "output_tokens", 0
            )
        else:
            # Fallback: try to convert to dict
            try:
                usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
                input_tokens = usage_dict.get("request_tokens", 0) or usage_dict.get(
                    "input_tokens", 0
                )
                output_tokens = usage_dict.get("response_tokens", 0) or usage_dict.get(
                    "output_tokens", 0
                )
            except Exception:
                logger.warning(f"Could not extract tokens from usage object: {usage}")
                return 0.0

        cost = (input_tokens * input_cost_per_token) + (
            output_tokens * output_cost_per_token
        )

        return float(cost)
