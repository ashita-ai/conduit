"""LLM execution via PydanticAI with timeout and error handling."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import Agent

from conduit.core.exceptions import ExecutionError
from conduit.core.models import Response, RoutingDecision
from conduit.core.pricing import compute_cost

logger = logging.getLogger(__name__)


class AllModelsFailedError(ExecutionError):
    """Raised when all models (primary + fallbacks) fail."""

    def __init__(self, message: str, errors: list[tuple[str, Exception]]) -> None:
        """Initialize with message and list of (model_id, error) tuples."""
        super().__init__(message)
        self.errors = errors


@dataclass
class ExecutionResult:
    """Result of LLM execution, potentially with fallback.

    Attributes:
        response: The successful Response from the model
        model_used: The model that actually produced the response
        was_fallback: True if a fallback model was used (primary failed)
        original_model: The originally selected model (may differ from model_used)
        failed_models: List of models that failed before success
    """

    response: Response
    model_used: str
    was_fallback: bool = False
    original_model: str = ""
    failed_models: list[str] = field(default_factory=list)


class ModelExecutor:
    """Execute LLM calls via PydanticAI."""

    def __init__(self) -> None:
        """Initialize executor with agent cache.

        Pricing is handled automatically via LiteLLM's bundled model_cost database.
        """
        self.clients: dict[str, Agent[Any, Any]] = {}

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

    async def execute_with_fallback(
        self,
        decision: RoutingDecision,
        prompt: str,
        result_type: type[BaseModel],
        timeout: float = 60.0,
        max_fallbacks: int = 3,
    ) -> ExecutionResult:
        """Execute LLM call with automatic fallback on failure.

        Tries the selected model first, then falls back to models in
        the fallback_chain if the primary model fails.

        Args:
            decision: RoutingDecision with selected_model and fallback_chain
            prompt: User query
            result_type: Pydantic model for structured output
            timeout: Maximum execution time per model in seconds
            max_fallbacks: Maximum number of fallback attempts

        Returns:
            ExecutionResult with response and metadata about fallback usage

        Raises:
            AllModelsFailed: If all models (primary + fallbacks) fail

        Example:
            >>> decision = await router.route(query)
            >>> result = await executor.execute_with_fallback(
            ...     decision=decision,
            ...     prompt=query.text,
            ...     result_type=MyOutput,
            ... )
            >>> if result.was_fallback:
            ...     print(f"Used fallback: {result.model_used}")
        """
        # Build list of models to try: selected + fallbacks
        models_to_try = [decision.selected_model] + decision.fallback_chain[
            :max_fallbacks
        ]

        failed_models: list[str] = []
        errors: list[tuple[str, Exception]] = []

        for model in models_to_try:
            try:
                response = await self.execute(
                    model=model,
                    prompt=prompt,
                    result_type=result_type,
                    query_id=decision.query_id,
                    timeout=timeout,
                )

                was_fallback = model != decision.selected_model

                if was_fallback:
                    logger.info(
                        f"Fallback succeeded: {model} (original: {decision.selected_model}, "
                        f"failed: {failed_models})"
                    )

                return ExecutionResult(
                    response=response,
                    model_used=model,
                    was_fallback=was_fallback,
                    original_model=decision.selected_model,
                    failed_models=failed_models,
                )

            except (ExecutionError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"Model {model} failed: {e}, trying next fallback "
                    f"({len(models_to_try) - len(failed_models) - 1} remaining)"
                )
                failed_models.append(model)
                errors.append((model, e))
                continue

        # All models failed
        error_summary = "; ".join(f"{m}: {e}" for m, e in errors)
        raise AllModelsFailedError(
            f"All {len(models_to_try)} models failed. Errors: {error_summary}",
            errors=errors,
        )

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

        Uses LiteLLM's bundled model_cost database for accurate, up-to-date pricing.

        Args:
            usage: Token usage from PydanticAI (Usage object or dict)
            model: Model identifier

        Returns:
            Cost in dollars
        """
        # Extract token counts from Usage object or dict
        if hasattr(usage, "request_tokens") or hasattr(usage, "input_tokens"):
            # Usage object with attributes (check both naming conventions)
            input_tokens = getattr(usage, "request_tokens", 0) or getattr(
                usage, "input_tokens", 0
            )
            output_tokens = getattr(usage, "response_tokens", 0) or getattr(
                usage, "output_tokens", 0
            )
            cache_read_tokens = getattr(usage, "cache_read_tokens", 0) or 0
        elif isinstance(usage, dict):
            # Dict with keys
            input_tokens = usage.get("request_tokens", 0) or usage.get(
                "input_tokens", 0
            )
            output_tokens = usage.get("response_tokens", 0) or usage.get(
                "output_tokens", 0
            )
            cache_read_tokens = usage.get("cache_read_tokens", 0) or 0
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
                cache_read_tokens = usage_dict.get("cache_read_tokens", 0) or 0
            except Exception:
                logger.warning(f"Could not extract tokens from usage object: {usage}")
                return 0.0

        return compute_cost(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            model_id=model,
            cache_read_tokens=cache_read_tokens or 0,
        )
